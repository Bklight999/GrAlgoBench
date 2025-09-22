import json
import os
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import argparse
from vllm import LLM, SamplingParams
import jsonlines
import logging
from tqdm import tqdm
from pathlib import Path
import re
import pandas as pd
from examples import get_examples
import math
import json
import string
import pickle
from modelscope.hub.file_download import model_file_download
from modelscope import snapshot_download
import datetime
import sys
sys.path.append('/path/to/data_generation')
from tasks import *
import yaml


def generate_text_template(args, question, answer, tokenizer):
    with open(f"/path/to/configs/{args.task_name}.yaml", "r") as f:
        config = yaml.safe_load(f)
        system_prompt = config['system_prompt']
        user_prompt = config['user_prompt']
        user_prompt = user_prompt.format(question=question, answer=answer)
    


    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    return text


def batch_data(data_list, batch_size=1):
    n = math.ceil(len(data_list) / batch_size)
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = len(data_list)
    batch_data.append(data_list[last_start:last_end])
    return batch_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='******LLM inference on causal data******')
    parser.add_argument("--model_path", type=str, default="path/to/model", help="loading paths of LLM")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    parser.add_argument("--max_tokens", type=int, default=8000, help="Max tokens for inference")
    parser.add_argument("--task_name", type=str, default="MST", help="task name")
    parser.add_argument("--difficulty", type=str, default="easy", help="difficulty")
    parser.add_argument("--gpu_num", type=int, default=2, help="gpu number")
    parser.add_argument("--response_generated_from_what_model", type=str, default="Qwen2.5-32B", help="response generated from what model")
    args = parser.parse_args()
    start_time = datetime.datetime.now()

    classname = args.task_name + '_Task'
    task = globals()[classname]('/path/to/dataset')
    task.load_dataset(args.difficulty)

    save_dir = f"/path/to/final_results"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    output_dir = Path(f"{save_dir}/{args.response_generated_from_what_model}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=args.max_tokens) 
    print('sampleing =====', sampling_params)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.gpu_num, gpu_memory_utilization=0.85, max_num_seqs = args.batch_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    temp = 0

    input_texts = [] 
    original_datas = []
    results_dict = {}

    answer_path = f'/path/to/results_for_algorithm/{args.response_generated_from_what_model}/{args.task_name}-{args.difficulty}.json'
    problem_path = f'/path/to/dataset/{args.task_name}_{args.difficulty}.pkl'
    
    if not os.path.exists(answer_path):
        raise FileNotFoundError(f"Input file not found: {answer_path}")
    
    with open(problem_path, 'rb') as f:
        problems = pickle.load(f)
    
    answers = []
    with open(answer_path, 'r', encoding='utf-8') as f:
        for line in f:
            answers.append(json.loads(line))
    
    
    
    problems = problems[:len(answers)]
    
        
    print('loading problems')

    all_evaluations = []  # List to store all evaluations for each data point

    for i,(problem,answer) in enumerate(zip(problems,answers)):
        question = problem['problem_text']
        data_evaluations = {"id": i, "predict": {}}
        if "Gemini" in args.response_generated_from_what_model:
            predict_num = 3
        else:
            predict_num = 8
        for j in range(predict_num):
            if "Gemini" in args.response_generated_from_what_model:
                k = j
            else:
                k = j+1
            #pdb.set_trace()
            ans = answer[f'predict{k}']
            data_evaluations["predict"][str(k)]={}
            data_evaluations["predict"][str(k)]["content"] = ans
            data_evaluations["predict"][str(k)]["label"] = ""
            data_evaluations["predict"][str(k)]["correctness"] = task.check_solution(i, ans)
            template_text = generate_text_template(args, question, ans, tokenizer)
            input_texts.append((template_text, i, k))
        all_evaluations.append(data_evaluations)
    
    print(f'input_texts length is {len(input_texts)}\n')

    batch_inputs = batch_data(input_texts[temp:], batch_size=args.batch_size)

    print(f'total samples are: {len(input_texts)}')
    logging.info(f'total samples are: {len(input_texts)}')
    ind = temp

    algorithm_dict = {"enumeration": 0, "search": 0, "greedy": 0, "not_mentioned": 0}
    correct_dict = {"enumeration": 0, "search": 0, "greedy": 0, "not_mentioned": 0}

    with jsonlines.open(f"{save_dir}/{args.response_generated_from_what_model}/{args.task_name}-{args.difficulty}.json", mode='w') as wf:
        for batch_input in tqdm(batch_inputs):
            # Extract just the template texts for the model
            template_texts = [item[0] for item in batch_input]
            data_indices = [item[1] for item in batch_input]
            predict_indices = [item[2] for item in batch_input]
            completions = llm.generate(template_texts, sampling_params)
            
            for i, (template, data_idx, predict_idx, output) in enumerate(zip(template_texts, data_indices, predict_indices, completions)):
                predict = output.outputs[0].text
                #pdb.set_trace()
                all_evaluations[data_idx]["predict"][str(predict_idx)]["label"] = predict
                if predict == "a":
                    algorithm_dict["enumeration"] += 1
                    if all_evaluations[data_idx]["predict"][str(predict_idx)]["correctness"][1] == True:
                        correct_dict["enumeration"] += 1
                elif predict == "b":
                    algorithm_dict["search"] += 1
                    if all_evaluations[data_idx]["predict"][str(predict_idx)]["correctness"][1] == True:
                        correct_dict["search"] += 1
                elif predict == "c":
                    algorithm_dict["greedy"] += 1
                    if all_evaluations[data_idx]["predict"][str(predict_idx)]["correctness"][1] == True:
                        correct_dict["greedy"] += 1
                elif predict == "d":
                    algorithm_dict["not_mentioned"] += 1
                    if all_evaluations[data_idx]["predict"][str(predict_idx)]["correctness"][1] == True:
                        correct_dict["not_mentioned"] += 1
                # all_evaluations[data_idx]["predict"][str(predict_idx)]["correctness"] = task.check_solution(data_idx, predict)
        # Write all evaluations to file
        for evaluations in all_evaluations:
            wf.write({"evaluations": evaluations})

    for key in correct_dict:
        correct_dict[key] = correct_dict[key] / algorithm_dict[key] if algorithm_dict[key] != 0 else 0
    for key in algorithm_dict:
        algorithm_dict[key] = algorithm_dict[key] / (len(problems)*predict_num)


    print(f"algorithm_dict: {algorithm_dict}")
    print(f"correct_dict: {correct_dict}")
    # save to txt
    with open(f"{save_dir}/{args.response_generated_from_what_model}/{args.task_name}-{args.difficulty}.txt", "w") as f:
        f.write(f"algorithm_dict: {algorithm_dict}\n")
        f.write(f"correct_dict: {correct_dict}\n")
