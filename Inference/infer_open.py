import json
import os
from transformers import AutoTokenizer
import argparse
from vllm import LLM, SamplingParams
import jsonlines
import logging
from tqdm import tqdm
from pathlib import Path
import math
import pickle
from datetime import datetime


def generate_text_template(question, tokenizer):
    system_prompt = (
        "You are an expert in graph reasoning problems. "
        "Please solve the following problem and output the answer in middle brackets. "
        "Do not use code to solve the problem."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    if "Qwen3" in args.LLM:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking
        )
    else:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


def batch_data(data_list, batch_size=1):
    n = math.ceil(len(data_list) / batch_size)
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1) * batch_size
        batch_data.append(data_list[start:end])
    last_start = (n-1) * batch_size
    last_end = len(data_list)
    batch_data.append(data_list[last_start:last_end])
    return batch_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='******LLM inference on causal data******')
    parser.add_argument("--gpu_num", type=int, default=4, help="GPU number")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for inference")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Max tokens for inference")
    parser.add_argument("--task", type=str, default="MKC", help="task name")
    parser.add_argument("--difficulty", type=str, default="easy", help="difficulty",
                        choices=["easy", "medium", "hard", "challenge"])
    parser.add_argument("--LLM", type=str, default="Qwen3-8B", help="Reasoning LLM")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for inference")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p for inference")
    parser.add_argument("--n", type=int, default=8, help="Number of samples for each question")
    parser.add_argument("--min_p", type=float, default=0, help="Min p for inference")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192, help="Max number of batched tokens")
    parser.add_argument("--enable_thinking", type=bool, default=True, help="Enable thinking for inference")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for inference")
    parser.add_argument("--end_index", type=int, default=50, help="End index for inference")
    args = parser.parse_args()

    # if args.gpu_num == 4:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"  # "0,1,2,3"
    # elif args.gpu_num == 2:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

    # change this to your own path
    LLM_to_path = {
        "Distill_Qwen_32B": "/path/to/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "Qwen3_0.6B": "/path/to/models/Qwen/Qwen3-0___6B",
        "QWQ-32B": "/path/to/models/Qwen/QwQ-32B",
        "Qwen3-32B": "/path/to/models/Qwen/Qwen3-32B",
        "Qwen3-32B-no-thinking": "/path/to/models/Qwen/Qwen3-32B",
        "Qwen3-14B-no-thinking": "/path/to/models/Qwen/Qwen3-14B",
        "Qwen2.5-32B": "/path/to/models/Qwen/Qwen2___5-32B-Instruct",
        "Qwen2.5-7B": "/path/to/models/Qwen/Qwen2___5-7B-Instruct",
        "Qwen3-14B": "/path/to/models/Qwen/Qwen3-14B",
        "Qwen3-8B": "/path/to/models/Qwen/Qwen3-8B",
        "Qwen3-8B-no-thinking": "/path/to/models/Qwen/Qwen3-8B",
        "Qwen3-14B-no-thinking": "/path/to/models/Qwen/Qwen3-14B",
        "OpenThinker-32B": "/path/to/models/open-thoughts/OpenThinker-32B",
        "OpenThinker-7B": "/path/to/models/open-thoughts/OpenThinker-7B",
        "Skywork-OR1-32B": "/path/to/models/Skywork/Skywork-OR1-32B",
        "Qwen3-4B": "/path/to/models/Qwen/Qwen3-4B",
        "Gemma-3-27b": "/path/to/models/LLM-Research/gemma-3-27b-it",
        "Llama-3.3-70B": "/path/to/models/LLM-Research/Llama-3___3-70B-Instruct",
        "Distill_Qwen_14B": "/path/to/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "Distill_Qwen_7B": "/path/to/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "Distill_Llama_70B": "/path/to/models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "Light-R1-32B": "/path/to/models/360zhinao/Light-R1-32B",
        "Light-R1-7B-DS": "/path/to/models/360zhinao/Light-R1-7B-DS",
        "Skywork-OR1-7B-Preview": "/path/to/models/Skywork/Skywork-OR1-7B-Preview",
        "Qwen3-235B": "/path/to/models/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
        "Qwen3-235B-no-thinking": "/path/to/models/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    }

    if "no-thinking" in args.LLM:
        args.enable_thinking = False

    date = datetime.now().strftime("%Y-%m-%d")
    save_dir = f"/path/to/Inference/final_results_32B/{args.LLM}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if os.path.exists(f"{save_dir}/{args.task}-{args.difficulty}.json"):
        print(f"File {save_dir}/{args.task}-{args.difficulty}.json already exists")
        exit(0)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        min_p=args.min_p,
        n=args.n
    )

    if "Qwen3-235B" in args.LLM:
        args.n = 3
        args.end_index = 30

    print('=' * 50)
    print('RUNTIME PARAMETERS:')
    print(f'LLM: {args.LLM}')
    print(f'GPU_NUM: {args.gpu_num}')
    print(f'BATCH_SIZE: {args.batch_size}')
    print(f'MAX_TOKENS: {args.max_tokens}')
    print(f'TEMPERATURE: {args.temperature}')
    print(f'TOP_P: {args.top_p}')
    print(f'N_SAMPLES: {args.n}')
    print(f'ENABLE_THINKING: {args.enable_thinking}')
    print(f'START_INDEX: {args.start_index}')
    print(f'END_INDEX: {args.end_index}')
    print('=' * 50)
    print('SAMPLING PARAMS:', sampling_params)
    print('=' * 50)

    enable_expert = True if "Qwen3-235B" in args.LLM else False

    llm_config = {
        'model': LLM_to_path[args.LLM],
        'tensor_parallel_size': args.gpu_num,
        'gpu_memory_utilization': 0.8,
        'enable_expert_parallel': enable_expert,
        'max_num_seqs': args.batch_size,
    }
    print('VLLM CONFIG:', llm_config)
    print('=' * 50)

    llm = LLM(**llm_config)
    tokenizer = AutoTokenizer.from_pretrained(LLM_to_path[args.LLM])
    temp = 0

    data_path = f"/path/to/data_generation/dataset/{args.task}_{args.difficulty}.pkl"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input file not found: {data_path}")

    with open(data_path, 'rb') as f:
        datas = pickle.load(f)


    datas = datas[args.start_index:args.end_index]

    print('loading datas')
    input_texts = []
    for data in datas:
        id = data["id"]
        question = data["problem_text"]
        template_text = generate_text_template(question, tokenizer)
        input_texts.append((template_text, id))

    print(f'input_texts length is {len(input_texts)}\n')
    batch_inputs = batch_data(input_texts[temp:], batch_size=args.batch_size)
    print(f'total samples are: {len(input_texts)}')
    logging.info(f'total samples are: {len(input_texts)}')

    with jsonlines.open(f"{save_dir}/{args.task}-{args.difficulty}.json", mode='w') as wf:
        for batch_input in tqdm(batch_inputs):
            template_texts = [item[0] for item in batch_input]
            ids = [item[1] for item in batch_input]
            completions = llm.generate(template_texts, sampling_params)
            # completions: List[List[RequestOutput]], each question corresponds to N outputs

            for idx, (qid, outputs) in enumerate(zip(ids, completions)):
                res = {"id": qid}
                for j, out in enumerate(outputs.outputs):
                    res[f"predict{j+1}"] = out.text
                # padding predictions if less than expected (rare)
                for k in range(len(outputs.outputs), args.n):
                    res[f"predict{k+1}"] = ""
                wf.write(res)