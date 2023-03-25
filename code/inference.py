import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from colorama import Fore, Style

import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, set_seed

def set_seeds(seed):
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def generate_prompt(instruction, input=None):
    # sorry about the formatting disaster gotta move fast
    if input:
        return f"""下方是一個關於任務的指令，以及一個提供與任務相關之資訊的輸入。請撰寫一個能適當地完成該任務指令需求的回覆。
### 指令:
{instruction}

### 輸入:
{input}

### 回覆:"""
    else:
        return f"""下方是一個關於任務的指令。請撰寫一個能適當地完成該任務指令需求的回覆。
### 輸入:
{instruction}

### 回覆:"""

def evaluate(instruction, generation_config, input=None):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256,
    )
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        print(f"{Fore.GREEN}回覆:{Style.RESET_ALL}")
        print(output.split("### 回覆:")[1].strip() + '\n')

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="decapoda-research/llama-7b-hf")
    parser.add_argument("--ckpt_name", type=str, default='../model/7b-tw_plus_en_ins-6_epoch')

    parser.add_argument("--cache_dir", type=str, default="../cache")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.65)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_config()
    # set_seeds(args.seed)
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir
    )
    model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit=True,
        # device_map="auto",
        device_map={'': 0},
        cache_dir=args.cache_dir
    )
    # load from checkpoint
    model = PeftModel.from_pretrained(model, args.ckpt_name, device_map={'': 0})

    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
   
    while(True):
        evaluate(input(f"\n{'-'*10}\n{Fore.BLUE}指令: {Style.RESET_ALL}"), generation_config)