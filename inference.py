import os

import sys
sys.path.append('/local/home/yuxi_xie/Projects/COrAL')
from coral.models import load_pretrained_models
from coral.models import AutoModelForOA
from coral.utils import seed_everything, jsonlines_dump, jsonlines_load
from coral.math_eval import extract_answer, math_equal

import time
import argparse
from tqdm import tqdm
from datasets import load_dataset
from calflops.calculate_pipline import CalFlopsPipline

import torch


def prepare_input(question: str, tokenizer, system: str = '', task: str = 'default'):
    if task == 'tldr':
        prompt = f"""Below is a post from Reddit. Write a summary of the content.\n\n### Content:\n{question}\n\n### Summary:"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:"""
    
    inputs = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors='pt',
    )
    return inputs['input_ids'], inputs['attention_mask'].bool(), prompt


def load_and_process_tldr(N=-1):
    raw_data = load_dataset('CarperAI/openai_summarize_tldr', split='test')
    if N < len(raw_data) and N > 0:
        raw_data = random.sample(list(raw_data), N)
    
    data = []
    for dt in raw_data:
        if N > 0 and len(data) >= N: break
        
        dt['prompt'] = dt['prompt'].strip().replace('TL;DR:', '').strip().replace('Summary:', '').strip()
        dt['response'] = dt['label'].strip().replace('TL;DR:', '').strip().replace('Summary:', '').strip()
        
        data.append({
            'question': dt["prompt"],
            'response': dt['response'],
        })
    
    return data


def load_and_process_MATH(N=-1):
    raw_data = load_dataset('HuggingFaceH4/MATH-500', split='test')
    if N < len(raw_data) and N > 0:
        raw_data = random.sample(list(raw_data), N)
    
    data = []
    for dt in raw_data:
        if N > 0 and len(data) >= N: break
        
        data.append({
            'question': dt["problem"],
            'level': dt['level'],
            'answer': dt['answer'],
            'response': dt['solution'],
        })
    
    return data


def load_and_process_GSM8K(N=-1):
    raw_data = load_dataset('openai/gsm8k', 'main')['test']
    if N < len(raw_data) and N > 0:
        raw_data = random.sample(list(raw_data), N)
    
    data = []
    for dt in raw_data:
        if N > 0 and len(data) >= N: break
        
        data.append({
            'question': dt["question"],
            'answer': dt['answer'].split('#### ')[-1].strip(),
            'response': dt['answer'],
        })
    
    return data


def load_model_tokenizer(model_path="/share/edc/home/yuxi_xie/coral/stage3-mixmath/checkpoint-47567"):
    oa_model, tokenizer = load_pretrained_models(
        model_path,
        dtype=torch.bfloat16,
        auto_model_type=AutoModelForOA,
        auto_device_mapping=True,
    )
    oa_model.eval()
    return oa_model, tokenizer


@torch.no_grad()
def conduct_inference(
    input_ids, 
    model,
    tokenizer,
    max_length=512,
    occurance_threshold=8,
    block_size=16,
    forward_size=4,
    backward_size=8,
    eval_forward_size=4,
    eval_backward_size=8,
    skip_verify=True,
    force_repeat=True,
    left2right=False,
    calculate_flop=False,
    verbal=False,
    use_cache=True,
):
    if calculate_flop:
        calculate_flops_pipline = CalFlopsPipline(
            model=model,
            include_backPropagation=False,
            compute_bp_factor=2,
        )
        calculate_flops_pipline.start_flops_calculate(ignore_list=None)
    stime = time.time()
    
    traces, output_ids = model.oa_generate(
        input_ids=input_ids.to(model.device),
        tokenizer=tokenizer,
        block_size=block_size,
        forward_size=forward_size,
        backward_size=backward_size,
        eval_forward_size=eval_forward_size,
        eval_backward_size=eval_backward_size,
        skip_verify=skip_verify,
        force_repeat=force_repeat,
        left2right=left2right,
        occurance_threshold=occurance_threshold,
        max_length=max_length,
        verbal=verbal,
        use_cache=use_cache,
    )
    
    duration = time.time() - stime
    flops = None if not calculate_flop else calculate_flops_pipline.get_total_flops()
    if calculate_flop:
        calculate_flops_pipline.end_flops_calculate()
    
    traces = [x[0] for x in traces]
    gen_ids = output_ids[0, input_ids.size(-1):].cpu().tolist()
    
    return gen_ids, traces, flops, duration


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--out_filename', type=str, required=True)
    parser.add_argument('--out_directory', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    # model
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=512)
    # data
    parser.add_argument('--data_name', type=str, default='gsm8k')
    # next token
    parser.add_argument('--left2right', action='store_true')
    # oa
    parser.add_argument('--block_size', type=int, default=16)
    parser.add_argument('--forward_size', type=int, default=4)
    parser.add_argument('--backward_size', type=int, default=8)
    parser.add_argument('--eval_forward_size', type=int, default=4)
    parser.add_argument('--eval_backward_size', type=int, default=8)
    # oa w/o verification
    parser.add_argument('--not_force_repeat', action='store_true')
    parser.add_argument('--skip_verify', action='store_true')
    parser.add_argument('--occurance_threshold', type=int, default=8)
    parser.add_argument('--verbal', action='store_true')
    
    args = parser.parse_args()
    seed_everything(seed=args.seed)
    
    if args.left2right:
        ftype = '_next_token'
    else:
        ftype = f'_B{args.block_size}_f{args.forward_size}_b{args.backward_size}'
        if args.not_force_repeat:
            ftype += '_norepeat'
        if args.skip_verify:
            ftype += f'_noverify_o{args.occurance_threshold}'
        else:
            ftype += f'_verify_f{args.eval_forward_size}_b{args.eval_backward_size}_o{args.occurance_threshold}'
    output_fname = os.path.join(
        args.out_directory, args.out_filename,
    ).replace('.jsonl', f'{ftype}.jsonl')
    results = []
    if os.path.exists(output_fname):
        results = jsonlines_load(output_fname)
    
    model, tokenizer = load_model_tokenizer(model_path=args.model_path)
    if args.data_name.startswith('gsm8k'):
        dataset = load_and_process_GSM8K()
    elif args.data_name.startswith('math'):
        dataset = load_and_process_MATH()
    elif args.data_name.startswith('tldr'):
        dataset = load_and_process_tldr()
    
    idx = -1
    is_correct, tflops, speed, iterations, tokens = [], [], [], [], []
    for dt in tqdm(dataset, disable=True):
        idx += 1
        if args.data_name != 'tldr' and len(is_correct) < len(results):
            generated = results[idx]['generated']
            prediction = extract_answer(generated)
            gen_ids = results[idx]['gen_ids_in_traces'][-1]
            
            is_correct.append(math_equal(prediction, dt['answer']))
            tflops.append(results[idx]['flops'] / 1e12)
            speed.append(len(gen_ids) / results[idx]['duration'])
            iterations.append(len(results[idx]['gen_ids_in_traces']))
            tokens.append(len(gen_ids))
            continue
        
        input_ids, *_ = prepare_input(dt['question'], tokenizer, task=args.data_name)
        gen_ids, traces, flops, duration = conduct_inference(
            input_ids, 
            model,
            tokenizer,
            max_length=args.max_length,
            occurance_threshold=args.occurance_threshold,
            block_size=args.block_size,
            forward_size=args.forward_size,
            backward_size=args.backward_size,
            eval_forward_size=args.eval_forward_size,
            eval_backward_size=args.eval_backward_size,
            skip_verify=args.skip_verify,
            force_repeat=not args.not_force_repeat,
            left2right=args.left2right,
            calculate_flop=True,
            verbal=args.verbal,
            use_cache=True,
        )
        
        generated = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        if args.data_name == 'tldr':
            tflops.append(flops / 1e12)
            speed.append(len(gen_ids) / duration)
            iterations.append(len(traces))
            tokens.append(len(gen_ids))
            print((
                ' * computation: {cost:.3f} TFLOPS ({num}/{total_num} samples)\n'
                ' * speed: {speed:.3f} tokens per second\n'
                ' * iteration: {iteration:.1f} iterations\n'
                ' * tokens: {token:.1f} tokens\n'
            ).format(
                num=len(results),
                total_num=len(dataset),
                cost=sum(tflops) / max(1, len(tflops)),
                speed=sum(speed) / max(1, len(speed)),
                iteration=sum(iterations) / max(1, len(iterations)),
                token=sum(tokens) / max(1, len(tokens)),
            ))
        else:
            prediction = extract_answer(generated)
            
            is_correct.append(math_equal(prediction, dt['answer']))
            tflops.append(flops / 1e12)
            speed.append(len(gen_ids) / duration)
            iterations.append(len(traces))
            tokens.append(len(gen_ids))
            print((
                ' * accuracy: {accuracy}% ({num}/{total_num} samples)\n'
                ' * computation: {cost:.3f} TFLOPS\n'
                ' * speed: {speed:.3f} tokens per second\n'
                ' * iteration: {iteration:.1f} iterations\n'
                ' * tokens: {token:.1f} tokens\n'
            ).format(
                accuracy=sum(is_correct) / max(1, len(is_correct)) * 100,
                num=len(is_correct),
                total_num=len(dataset),
                cost=sum(tflops) / max(1, len(tflops)),
                speed=sum(speed) / max(1, len(speed)),
                iteration=sum(iterations) / max(1, len(iterations)),
                token=sum(tokens) / max(1, len(tokens)),
            ))
        
        rst = {
            'question': dt['question'],
            'answer': dt['answer'] if 'answer' in dt else dt['response'],
            'generated': generated,
            'gen_ids_in_traces': traces,
            'duration': duration,
            'flops': flops,
        }
        results.append(rst)
        
        jsonlines_dump(output_fname, [rst], mode='a')
        
    