import torch
from transformers import AutoModelForCausalLM
# import sys
# work_dir = '...'
# sys.path.append(work_dir)

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torchvision
import json
import argparse
import copy
import random
from typing import List, Dict
import gc
from pathlib import Path
import torch.multiprocessing as mp
from tqdm import tqdm
import subprocess
import torch.distributed as dist
import time
from utils.reward_hps import HPSv2
from utils.reward_git import GIT
from utils.reward_gdino import GDino
from utils.reward_orm import ORM
from transformers import (

    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,

    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_all(42)

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = os.popen(f'scontrol show hostname {node_list} | head -n1').read().strip()
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['RANK'] = str(proc_id)
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        args.rank = proc_id
        args.world_size = ntasks
        args.gpu = proc_id % num_gpus
    else:
        args.rank = 0
        args.world_size = 1
        args.gpu = 0

    torch.cuda.set_device(args.gpu)

    # if args.world_size > 1:
    #     dist.init_process_group(
    #         backend="nccl", 
    #         init_method="env://", 
    #         world_size=args.world_size, 
    #         rank=args.rank
    #     )
    # dist.barrier()

    setup_for_distributed(args.rank == 0)

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    prompt_text: str,
    pure_prompt: str,
    save_dir: str,
    reward_funcs,
    reward_processing_classes,
    temperature: float = 1,
    num_generation: int = 9,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    new_generations_image: int = 10,
    conversation: List[Dict[str, str]] = None,
    example: List[Dict] = None,             
    accelerator: Accelerator = None,
    device: str = "cuda"
):  
    if not isinstance(example,list):
        example = [example]
        
    prompt_inputs = vl_chat_processor.tokenizer(
            text=[prompt],
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=True
    )
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    prompt_ids = prompt_ids.repeat_interleave(num_generation, dim=0).to('cuda')
    prompt_mask = prompt_mask.repeat_interleave(num_generation, dim=0).to('cuda')
    input_embeds = mmgpt.language_model.get_input_embeddings()(prompt_ids)

    def compute_entropy(logits):
        log_probs = logits.log_softmax(dim=-1)
        token_probs = log_probs.exp()    
        entropy_token = -(token_probs * log_probs).sum(dim=-1)
        V = logits.size(-1)  # codebook size
        #print('codebook_size',V)
        entropy_token = entropy_token / torch.log(torch.tensor(V, device=logits.device, dtype=logits.dtype))
        return entropy_token

    del prompt_inputs
    torch.cuda.empty_cache()

    # TODO: if num_generations is too large, we need to split it into multiple batches
    if num_generation > 10:
        total_generations = []
        entropy_generations = []
        for i in range(prompt_ids.shape[0] // num_generation):
            current_input_embeds = input_embeds[i*num_generation: (i+1)*num_generation]
            current_attn_mask = prompt_mask[i*num_generation: (i+1)*num_generation]
            prompt_completion_ids = mmgpt.language_model.generate(
                inputs_embeds=current_input_embeds,
                attention_mask=current_attn_mask,
                pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
                bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
                eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=True,
                use_cache=True,
            )
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_ids
            completion_ids = prompt_completion_ids
            is_eos = completion_ids == vl_chat_processor.tokenizer.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            prompt_all_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            input_embeds_used = mmgpt.language_model.get_input_embeddings()(prompt_all_ids)
            attention_mask_used = torch.cat([prompt_mask, completion_mask], dim=1)
            hidden_states = mmgpt.language_model(inputs_embeds=input_embeds_used, attention_mask=attention_mask_used, output_hidden_states=True).hidden_states 
            last_hidden_states = hidden_states[-1]
            #print('last_hidden_states',last_hidden_states)
            text_logits = mmgpt.language_model.lm_head(last_hidden_states) 
            entropy_generations.append(compute_entropy(text_logits))
            total_generations.append(prompt_completion_ids)
        prompt_completion_ids = torch.cat(total_generations, dim=0)
        entropy_text = torch.cat(entropy_generations,dim=0).float()
        del current_input_embeds, current_attn_mask,input_embeds_used,attention_mask_used
        del hidden_states, last_hidden_states, text_logits, completion_ids
        torch.cuda.empty_cache()
        #print('entropy_text',entropy_text.shape)
    else: # if num_generations == 1, we directly generate all for the batch data
        prompt_completion_ids = mmgpt.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=prompt_mask,
            pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
            bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
            eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=True,
            use_cache=True,
        )
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_ids
        completion_ids = prompt_completion_ids
        is_eos = completion_ids == vl_chat_processor.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        prompt_all_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        input_embeds_used = mmgpt.language_model.get_input_embeddings()(prompt_all_ids)
        attention_mask_used = torch.cat([prompt_mask, completion_mask], dim=1)
        hidden_states = mmgpt.language_model(inputs_embeds=input_embeds_used, attention_mask=attention_mask_used, output_hidden_states=True).hidden_states 
        last_hidden_states = hidden_states[-1]
        #print('last_hidden_states',last_hidden_states)
        text_logits = mmgpt.language_model.lm_head(last_hidden_states) 
        entropy_text = compute_entropy(text_logits).float()
        #print('entropy_text',entropy_text.shape)
        del hidden_states, last_hidden_states, text_logits, completion_ids,input_embeds_used,attention_mask_used
        torch.cuda.empty_cache()
    
    entropy_text_mean = entropy_text.mean(dim=1)
    # entropy_text_mean = entropy_text_mean.mean(dim=0)
    entropy_text_std = entropy_text.std(dim=1)
    # entropy_text_std = entropy_text_std.mean(dim=0)

    prompt_length = prompt_ids.size(1)
    prompt_ids = prompt_ids
    completion_ids = prompt_completion_ids

    del input_embeds, prompt_mask, entropy_text
    torch.cuda.empty_cache()

    image_gen_prompt_list = []
    
    prompt = vl_chat_processor.tokenizer.decode(prompt_ids[0].cpu().tolist(), skip_special_tokens=True)
    #('=============inner generation prompt',prompt,'=========================\n')
    for i in range(completion_ids.shape[0]):
        answer = vl_chat_processor.tokenizer.decode(completion_ids[i].cpu().tolist(), skip_special_tokens=True)
        image_gen_prompt = f"{prompt_text}. {answer}"

        conversation = [
            {
                "role": "User",
                "content": image_gen_prompt,
            },
            {"role": "Assistant", "content": ""},
        ]
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )

        #print(f"Prompt {i}: {sft_format}\Semantic-CoT {i}: {answer}")
        image_gen_prompt_list.append(sft_format)

    del prompt_completion_ids, completion_ids
    torch.cuda.empty_cache()

    prompt_inputs = vl_chat_processor.tokenizer(
        text=image_gen_prompt_list,
        return_tensors="pt",
        padding=True,
        padding_side="right",
        add_special_tokens=True,
    ) # {'input_ids', 'attention_mask'}

    prompt_ids, attention_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
    prompt_ids = prompt_ids.to('cuda')
    attention_mask = attention_mask.to('cuda')
    # attention_mask = torch.ones_like(attention_mask)
    # # add image start token at the end
    image_start_token_id = vl_chat_processor.tokenizer.encode(vl_chat_processor.image_start_tag)[1]
    prompt_ids = torch.cat([prompt_ids, prompt_ids.new_full((prompt_ids.size(0), 1), image_start_token_id)], dim=1)
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))], dim=1)
    
    # prompt_ids = prompt_ids.repeat_interleave(num_generation, dim=0).to('cuda')
    # attention_mask = attention_mask.repeat_interleave(num_generation, dim=0).to('cuda')

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(prompt_ids)
    pad_input_embeds = mmgpt.language_model.get_input_embeddings()(prompt_ids.new_full((1, 1), vl_chat_processor.pad_id))
    total_generated_tokens_img = []

    del prompt_inputs, image_gen_prompt_list
    torch.cuda.empty_cache()

    inputs_embeds = inputs_embeds.repeat_interleave(new_generations_image, dim=0)
    attention_mask = attention_mask.repeat_interleave(new_generations_image, dim=0)
    pad_input_embeds = pad_input_embeds.repeat_interleave(new_generations_image, dim=0)

    # Currently only one image generation (since the diversity is low)
    batch_size = min(4, new_generations_image) 
    entropy_generations = []
    for j in range(0, inputs_embeds.shape[0], batch_size):
        end_idx = min(j + batch_size, inputs_embeds.shape[0])
        batch_inputs_embeds = inputs_embeds[j:end_idx]
        batch_attention_mask = attention_mask[j:end_idx]
        batch_pad_input_embeds = pad_input_embeds[:end_idx-j]
        
        # prepare the conditional and unconditional input
        cond_inputs_embeds = batch_inputs_embeds
        cond_attention_mask = batch_attention_mask
        uncond_inputs_embeds = cond_inputs_embeds.clone()
        uncond_inputs_embeds[:, 1:-1] = batch_pad_input_embeds
        
        inputs_embeds_img = torch.repeat_interleave(cond_inputs_embeds, 2, dim=0)
        inputs_embeds_img[1::2] = uncond_inputs_embeds
        attention_mask_img = torch.repeat_interleave(cond_attention_mask, 2, dim=0)
        attention_mask_img[1::2] = torch.ones_like(attention_mask_img[1::2])
        # import pdb; pdb.set_trace()

        split_size = 2*new_generations_image
        for jj in range(0, inputs_embeds_img.shape[0], split_size):
            #print(f"Generating image {jj}")
            start = jj
            end = min(jj + split_size, inputs_embeds_img.shape[0])
            generated_tokens = torch.zeros(((end-start)//2, image_token_num_per_image), dtype=torch.int64).cuda()
            cur_inputs_embeds_img = inputs_embeds_img[start: end]
            cur_attention_mask_img = attention_mask_img[start: end]

            entropy_img_batch = []
            for k in range(image_token_num_per_image):
                outputs = mmgpt.language_model.model(
                    inputs_embeds=cur_inputs_embeds_img, 
                    use_cache=True, 
                    past_key_values=outputs.past_key_values if k != 0 else None, 
                    attention_mask=cur_attention_mask_img
                )
                
                hidden_states = outputs.last_hidden_state
                logits = mmgpt.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                
                logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)

                token_entropy_img = compute_entropy(logits)
                entropy_img_batch.append(token_entropy_img.unsqueeze(1))

                probs = torch.softmax(logits / temperature, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, k] = next_token.squeeze(dim=-1)

                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
                cur_inputs_embeds_img = img_embeds.unsqueeze(dim=1)
                cur_attention_mask_img = torch.cat([cur_attention_mask_img, cur_attention_mask_img.new_ones((cur_attention_mask_img.shape[0], 1), dtype=torch.int)], dim=1)

            total_generated_tokens_img.append(generated_tokens)
            entropy_generations.append(torch.cat(entropy_img_batch, dim=1))

            del batch_inputs_embeds, batch_attention_mask, batch_pad_input_embeds
            del cond_inputs_embeds, cond_attention_mask, uncond_inputs_embeds,entropy_img_batch
            del inputs_embeds_img, attention_mask_img, generated_tokens
            torch.cuda.empty_cache()

    entropy_imag = torch.cat(entropy_generations,dim=0).float()
    # print('entropy_imag',entropy_imag.shape)
    entropy_imag_mean = entropy_imag.mean(dim=1)
    # entropy_imag_mean = entropy_imag_mean.mean(dim=0)
    entropy_imag_std = entropy_imag.std(dim=1)
    # entropy_imag_std =entropy_imag_std.mean(dim=0)

    del entropy_imag
    torch.cuda.empty_cache()

    total_generated_tokens_img = torch.cat(total_generated_tokens_img, dim=0)

    image_batch_size = 8   
    num_samples = total_generated_tokens_img.shape[0]

    decoded_list = []

    for i in range(0, num_samples, image_batch_size):
        max_index = min(num_samples,i+image_batch_size)
        chunk = total_generated_tokens_img[i:max_index]

        dec = mmgpt.gen_vision_model.decode_code(
            chunk.to(dtype=torch.int),
            shape=[chunk.shape[0], 8, img_size // patch_size, img_size // patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

        decoded_list.append(dec)
        del chunk, dec
        torch.cuda.empty_cache()

    visual_img = np.concatenate(decoded_list, axis=0)
    images = [Image.fromarray(visual_img[idx]) for idx in range(visual_img.shape[0])]
    # dec = mmgpt.gen_vision_model.decode_code(total_generated_tokens_img.to(dtype=torch.int), shape=[num_generation*new_generations_image, 8, img_size//patch_size, img_size//patch_size])
    # dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    # dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    
    # visual_img = np.zeros((total_generated_tokens_img.shape[0], img_size, img_size, 3), dtype=np.uint8)
    # visual_img[:, :, :] = dec
    # images = [Image.fromarray(visual_img[idx]) for idx in range(visual_img.shape[0])]

    prompts = [input["raw_prompt"] for input in example for _ in range(num_generation) for __ in range(new_generations_image)]
    rewards_per_func = torch.zeros(len(images), len(reward_funcs), device=device)
    
    for i, (reward_func, reward_processing_class) in enumerate(
        zip(reward_funcs, reward_processing_classes)
    ):
        #print('example[0].keys():',example[0].keys())
        reward_kwargs = {key: [] for key in example[0].keys() if key not in ["prompt", "completion"]}
        
        # 填充reward_kwargs
        for key in reward_kwargs:
            for input_example in example:
                # repeate the key for (num_generation * new_generations_image) times
                reward_kwargs[key].extend([input_example[key]] * num_generation * new_generations_image)
        
        batch_size = 4  
        for batch_start in range(0, len(images), batch_size):
            batch_end = min(batch_start + batch_size, len(images))
            batch_images = images[batch_start:batch_end]
            batch_prompts = prompts[batch_start:batch_end]
            batch_kwargs = {k: v[batch_start:batch_end] for k, v in reward_kwargs.items()}
            
            output_reward_func = reward_func(
                prompts=batch_prompts, 
                images=batch_images, 
                **batch_kwargs
            )
            rewards_per_func[batch_start:batch_end, i] = torch.tensor(
                output_reward_func, dtype=torch.float32, device=device
            )
            
            del batch_images, batch_prompts, batch_kwargs
            torch.cuda.empty_cache()
    
    rewards_per_func = rewards_per_func.sum(dim=1)

    os.makedirs(save_dir, exist_ok=True)

    mark = []
    counter = 0  

    for batch_idx, dec in enumerate(decoded_list):
        batch_size = dec.shape[0]
        for i in range(batch_size):
            mark.append(counter // new_generations_image)
            save_path = os.path.join(save_dir, f"{pure_prompt}_{counter:06d}.png")
            PIL.Image.fromarray(dec[i]).save(save_path)
            counter += 1

    del images, prompts, decoded_list
    torch.cuda.empty_cache()
    # mark = []
    # for i in range(num_generation*new_generations_image):
    #     mark.append(i//new_generations_image)
    #     save_path = os.path.join(save_dir, f"{pure_prompt}_{i:06d}.png")
    #     #save_path = os.path.join(save_dir, f"{pure_prompt}.png")
    #     PIL.Image.fromarray(dec[i]).save(save_path)

    # del images, prompts, dec
    # torch.cuda.empty_cache()
    
    #mark = [[i]*new_generations_image for i in range(num_generation)]
    return rewards_per_func, entropy_text_mean, entropy_imag_mean, entropy_text_std, entropy_imag_std,mark


def split_prompts(prompts, world_size):
    chunk_size = (len(prompts) + world_size - 1) // world_size
    return [prompts[i * chunk_size: (i + 1) * chunk_size] for i in range(world_size)]


reward_funcs_registry = {
    "hps": 'hps',
    'hps_compare': 'hps_compare',
    'git': 'git',
    'gdino': 'gdino',
    'orm': 'orm',
    'unify': 'unify',
}



def main(args):
    accelerator = Accelerator()
    device = accelerator.device
    
    with open(args.data_path, 'r') as f:
        all_data = [json.loads(line) for line in f if line.strip()]
    
    #print('all_data',all_data)
    data_splits = np.array_split(all_data, accelerator.num_processes)
    local_data = data_splits[accelerator.process_index]
    
    reward_funcs_name_group = args.reward_funcs.split()
    #print('+++++++=reward_funcs_name_group:',reward_funcs_name_group)
    
    reward_funcs_name = [reward_funcs_registry[func] for func in reward_funcs_name_group]
    reward_funcs = []
    for func_name in reward_funcs_name:
        if func_name == 'hps':
            reward_func = HPSv2(args)
        elif func_name == 'git':
            reward_func = GIT(args)
        elif func_name == 'gdino':
            reward_func = GDino(args)
        elif func_name == 'orm':
            reward_func = ORM(args)
        else:
            print('error in  reward funcs')
        # reward_func.name = func_name
        reward_funcs.append(reward_func)

    reward_processing_classes = [None] * len(reward_funcs)

    for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
        if isinstance(reward_func, PreTrainedModel):
            if reward_processing_class is None:
                reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
            if reward_processing_class.pad_token_id is None:
                reward_processing_class.pad_token = reward_processing_class.eos_token
            # The reward model computes the reward for the latest non-padded token in the input sequence.
            # So it's important to set the pad token ID to the padding token ID of the processing class.
            reward_func.config.pad_token_id = reward_processing_class.pad_token_id
            reward_processing_classes[i] = reward_processing_class
    
    for i, reward_func in enumerate(reward_funcs):
        if isinstance(reward_func, PreTrainedModel):
            reward_funcs[i] = accelerator.prepare_model(reward_func, evaluation_mode=True)
        elif isinstance(reward_func, (HPSv2, GDino, GIT)):
            reward_func.load_to_device(accelerator.device)
        elif isinstance(reward_func, ORM):
            reward_func.load_to_device(accelerator.device)
            reward_func.accelerator = accelerator
            reward_func.model = accelerator.prepare_model(reward_func.model, evaluation_mode=True)
        elif hasattr(reward_func, "to"):
            reward_func.to(accelerator.device)
            if hasattr(reward_func, "eval"):
                reward_func.eval()
        else:
            raise TypeError(f"Unsupported reward_func type: {type(reward_func)}")

    # Load model & processor on current GPU
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16
    )
    vl_gpt = accelerator.prepare_model(vl_gpt, evaluation_mode=True)
    vl_gpt.eval()
    #print('************************\nvl_gpt:\n',vl_gpt)

    with open(args.reasoning_prompt_path, 'r') as f:
        cot_prompt = f.read().strip()

    # entropy_imag_mean_group = []
    # entropy_imag_std_group = []
    # entropy_imag_sample_mean_group = []
    # entropy_imag_sample_std_group = []

    # entropy_text_mean_group = []
    # entropy_text_std_group = []
    # entropy_text_sample_mean_group = []
    # entropy_text_sample_std_group = []
    
    def make_conversation(example):
        def make_detection_prompt(nouns):
            if len(nouns) == 0:
                return '', []
            
            token_spans = []
            pointer = 0
            for noun in nouns:
                n_split = noun.strip().split(" ")
                if len(n_split) == 1:
                    length = len(n_split[0])
                    token_spans.append([[pointer, pointer + length]])
                    pointer += length + 3 # on the blank space after the noun
                else: # multiple words
                    beg_len = len(n_split[0])
                    total_length = len(noun)
                    end_len = len(n_split[-1])
                    token_spans.append([[pointer, pointer + beg_len], [pointer + total_length - end_len, pointer + total_length]])
                    pointer += total_length + 3 # on the blank space after the noun
            text_prompt = ' . '.join(nouns) + "." # need to end with '.
            return text_prompt, token_spans
        
        # make detection prompt
        if 'nouns' in example and example['nouns'] is not None:
            det_text_prompt, det_token_spans = make_detection_prompt(example['nouns'])
        else:
            det_text_prompt = ''
            det_token_spans = []
        det_prompt_dict = {
            'text_prompt': det_text_prompt,
            'token_spans': det_token_spans,
        }
        # make vqa prompt
        if 'attr_nouns' in example and example['attr_nouns'] is not None:
            questions = [f"{attr_noun}?" for attr_noun in example['attr_nouns']]
            vqa_prompt = {'questions': questions}
        else:
            vqa_prompt = {'questions': []}  # Changed from None to empty list

        return {
            "prompt": [
                {"role": "User", "content": cot_prompt.format(example["prompt"])},
                {"role": "Assistant", "content": ""},
            ],
            'raw_prompt': example["prompt"],
            'det_prompt': det_prompt_dict,
            'task_type': example['task_type'],
        }

    # def process_example_for_inference(example):
    #     return {**example, **make_conversation(example)}

    def process_example_for_inference(example): 
        original = example.copy() 
        new_fields = make_conversation(example) 
        original.update(new_fields) 
        return original

    rank_save_path = os.path.join(args.mark_dir, f"results_rank_new{accelerator.process_index}.jsonl")
    os.makedirs(args.mark_dir, exist_ok=True)

    with open(rank_save_path, "w", encoding="utf-8") as fout:
        for example in tqdm(local_data, desc=f"Processing on {accelerator.process_index}"):
            example = process_example_for_inference(example)
            #print('++++++example+++++++',example)
            pure_prompt = example['raw_prompt']
            prompt = copy.deepcopy(pure_prompt)
            prompt_text = copy.deepcopy(prompt)

            conversation = example['prompt']
            # conversation = [
            #     {"role": "User", "content": cot_prompt.format(prompt)},
            #     {"role": "Assistant", "content": ""}
            # ]

            system_prompt = "You are a helpful assistant that receives an image prompt and generate a visualization of the prompt."
            sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=vl_chat_processor.sft_format,
                system_prompt=system_prompt
            )
            prompt = sft_format
            rewards, entropy_text_mean, entropy_imag_mean, entropy_text_std, entropy_imag_std, mark = generate(
                    vl_gpt,
                    vl_chat_processor,
                    prompt,
                    prompt_text,
                    pure_prompt,
                    args.save_dir,
                    reward_funcs,
                    reward_processing_classes,
                    num_generation=args.num_generation,
                    new_generations_image=args.new_generations_image,
                    conversation=conversation,
                    example=example,
                    accelerator=accelerator,
                    device=device
                    )
            # print('rewards',rewards.shape,'entropy_text_mean',entropy_text_mean.shape,'entropy_imag_mean',entropy_imag_mean.shape)
            for i in range(args.num_generation):
                for j in range(args.new_generations_image):
                    result_item = {
                        "raw_prompt": pure_prompt,
                        "mark": mark[i*args.new_generations_image+j],
                        "rewards": rewards[i*args.new_generations_image+j].item(),
                        "entropy_text_mean": entropy_text_mean[i].item(),
                        "entropy_text_std": entropy_text_std[i].item(),
                        "entropy_imag_mean": entropy_imag_mean[i*args.new_generations_image+j].item(),
                        "entropy_imag_std": entropy_imag_std[i*args.new_generations_image+j].item()
                    }
                    fout.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            # result_item = {
            #             "raw_prompt": pure_prompt,
            #             "rewards": rewards.item(),
            #             "entropy_text_mean": entropy_text_mean.item(),
            #             "entropy_imag_mean": entropy_imag_mean.item(),
            #             "entropy_text_std": entropy_text_std.item(),
            #             "entropy_imag_std": entropy_imag_std.item()
            #         }
            # fout.write(json.dumps(result_item, ensure_ascii=False) + "\n")

    accelerator.wait_for_everyone()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--reasoning_prompt_path", type=str, required=True,help="Path to the cot reasoning prompt")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to the output image directory")
    parser.add_argument("--mark_dir", type=str, required=True, help="Path to the output data directory")
    parser.add_argument("--num_generation", type=int, default=10,help="number of different cot prompts")
    parser.add_argument("--new_generations_image", type=int, default=10,help="number of generations for each cot prompt")
    parser.add_argument("--hps_ckpt_path", type=str, required=True,help="../../../weight/reward_weight/HPS_v2.1_compressed.pt")
    parser.add_argument("--git_ckpt_path", type=str, required=True,help="../../../weight/reward_weight/git-large-vqav2")
    parser.add_argument("--gdino_ckpt_path", type=str, required=True,help="../../../weight/reward_weight/groundingdino_swint_ogc.pth")
    parser.add_argument("--gdino_config_path", type=str, required=True,help="./utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--orm_ckpt_path", type=str, required=True,help="../../../weight/reward_weight/ORM-T2I-R1")
    parser.add_argument("--reward_funcs", type=str, default="hps git gdino orm")
    args = parser.parse_args()
    main(args)
