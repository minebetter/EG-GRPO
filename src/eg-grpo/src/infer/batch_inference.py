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
    setup_for_distributed(args.rank == 0)

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    prompt_text: str,
    pure_prompt: str,
    save_dir: str,
    temperature: float = 1,
    num_generation: int = 9,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    conversation: List[Dict[str, str]] = None,
):  

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

    # TODO: if num_generations is too large, we need to split it into multiple batches
    if num_generation > 20:
        total_generations = []
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
            total_generations.append(prompt_completion_ids)
        prompt_completion_ids = torch.cat(total_generations, dim=0)
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

    image_gen_prompt_list = []
    
    prompt = vl_chat_processor.tokenizer.decode(prompt_ids[0].cpu().tolist(), skip_special_tokens=True)
    #('=============inner generation prompt',prompt,'=========================\n')
    for i in range(completion_ids.shape[0]):
        answer = vl_chat_processor.tokenizer.decode(completion_ids[i].cpu().tolist(), skip_special_tokens=True)
        image_gen_prompt = f"{prompt_text}. {answer}"
        # print('image_gen_prompt:',image_gen_prompt)
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

        print(f"Prompt {i}: {sft_format}\Semantic-CoT {i}: {answer}")
        image_gen_prompt_list.append(sft_format)

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

    # Currently only one image generation (since the diversity is low)
    for j in range(inputs_embeds.shape[0] // num_generation):
        # Make cond and uncond inputs embeds and attention mask
        cond_inputs_embeds = inputs_embeds[j*num_generation: (j+1)*num_generation]
        cond_attention_mask = attention_mask[j*num_generation: (j+1)*num_generation]
        uncond_inputs_embeds = cond_inputs_embeds.clone()
        uncond_inputs_embeds[:, 1:-1] = pad_input_embeds
        
        inputs_embeds_img = torch.repeat_interleave(cond_inputs_embeds, 2, dim=0)
        inputs_embeds_img[1::2] = uncond_inputs_embeds
        attention_mask_img = torch.repeat_interleave(cond_attention_mask, 2, dim=0)
        attention_mask_img[1::2] = torch.ones_like(attention_mask_img[1::2])
        # import pdb; pdb.set_trace()

        split_size = 2 * num_generation
        for jj in range(0, inputs_embeds_img.shape[0], split_size):
            #print(f"Generating image {jj}")
            start = jj
            end = min(jj + split_size, inputs_embeds_img.shape[0])
            generated_tokens = torch.zeros(((end-start)//2, image_token_num_per_image), dtype=torch.int64).cuda()
            cur_inputs_embeds_img = inputs_embeds_img[start: end]
            cur_attention_mask_img = attention_mask_img[start: end]

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
                probs = torch.softmax(logits / temperature, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, k] = next_token.squeeze(dim=-1)

                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
                cur_inputs_embeds_img = img_embeds.unsqueeze(dim=1)
                cur_attention_mask_img = torch.cat([cur_attention_mask_img, cur_attention_mask_img.new_ones((cur_attention_mask_img.shape[0], 1), dtype=torch.int)], dim=1)

            print(generated_tokens.shape)
            total_generated_tokens_img.append(generated_tokens)

    total_generated_tokens_img = torch.cat(total_generated_tokens_img, dim=0)

    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[num_generation, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)


    os.makedirs(save_dir, exist_ok=True)

    # save generations as pngs
    for i in range(num_generation):
        #save_path = os.path.join(save_dir, f"{pure_prompt}_{i:06d}.png")
        save_path = os.path.join(save_dir, f"{pure_prompt}.png")
        PIL.Image.fromarray(dec[i]).save(save_path)


def split_prompts(prompts, world_size):
    chunk_size = (len(prompts) + world_size - 1) // world_size
    return [prompts[i * chunk_size: (i + 1) * chunk_size] for i in range(world_size)]


def main(args):
    init_distributed_mode(args)
    # Load prompts
    with open(args.data_path, 'r') as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    random.shuffle(all_prompts)

    prompt_chunks = split_prompts(all_prompts, args.world_size)
    sub_prompts = prompt_chunks[args.rank]

    # Load model & processor on current GPU
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map={"": args.gpu},  # one model for each gpu
        torch_dtype=torch.bfloat16
    )
    vl_gpt.eval()

    with open(args.reasoning_prompt_path, 'r') as f:
        cot_prompt = f.read().strip()

    for idx, prompt in enumerate(tqdm(sub_prompts, desc=f"GPU{args.gpu} [{args.rank}]")):
        pure_prompt = copy.deepcopy(prompt)
        prompt_text = copy.deepcopy(prompt)

        conversation = [
            {"role": "User", "content": cot_prompt.format(prompt)},
            {"role": "Assistant", "content": ""}
        ]

        system_prompt = "You are a helpful assistant that receives an image prompt and generates a visualization of the prompt."
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt=system_prompt
        )
        prompt = sft_format
        generate(
            vl_gpt,
            vl_chat_processor,
            prompt,
            prompt_text,
            pure_prompt,
            args.save_dir,
            num_generation=args.num_generation,
            conversation=conversation
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--reasoning_prompt_path", type=str,required=True, help="Path to the cot reasoning prompt")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--num_generation", type=int, default=1,help="Number of generations per prompt")
    args = parser.parse_args()
    main(args)
