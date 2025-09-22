import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from vllm import LLM, SamplingParams
import logging
from tqdm import tqdm
from pathlib import Path
import re
import pandas as pd
from examples import get_examples
import math
import string
import pickle
from modelscope.hub.file_download import model_file_download
from modelscope import snapshot_download
import datetime
import sys

sys.path.append('/path/to/GrAlgoBench/data_generation')
from tasks import *
from tasks.base import *


def generate_reformated_response(response):
    parts = response.split("\n\n")
    reformated_response = []
    for i, part in enumerate(parts):
        part = f"<<{i}>> " + part
        reformated_response.append(part)
    return "\n\n".join(reformated_response)


def generate_final_response(response, split_principles):
    pattern = r"<<(?P<start>.*?)>> - <<(?P<end>.*?)>> \[(?P<description>.*?)\]"
    matches = re.findall(pattern, split_principles)

    matches.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 0)
    result_parts = []
    current_pos = 0
    section_num = 0
    
    for match in matches:
        start, end, description = match
        start_pattern = f"<<{start}>>"
        end_pattern = f"<<{end}>>"
        
        start_pos = response.find(start_pattern, current_pos)
        end_pos = response.find(end_pattern, current_pos)
        
        if start_pos != -1 and end_pos != -1:
            section_num += 1
            if start_pos > current_pos:
                result_parts.append(response[current_pos:start_pos].strip())
            result_parts.append(f"[Section {section_num}]: {description}\n")
            content_start = start_pos + len(start_pattern)
            content_end = end_pos
            content_between = response[content_start:content_end]
            tag_pattern = r"<<\d+>>"
            clean_content = re.sub(tag_pattern, '', content_between)
            if clean_content:
                result_parts.append(clean_content)
            current_pos = end_pos + len(end_pattern)

    if current_pos < len(response):
        remaining = response[current_pos:].strip()
        if remaining:
            result_parts.append(remaining)
    
    final_response = "\n\n".join(result_parts)
    final_response = re.sub(r"<<.*?>>", "", final_response)
    return final_response


def generate_text_template(args, question, answer, tokenizer):
    prompt = f"""
Here is the problem and the response:

Problem: {question}

Response: {answer}

Present your output in this format: <<start tag>> - <<end tag>> [Brief description]. The end tag should be the same as or later than the start tag of the section, and the start tag of the next section should follow directly after the end tag of the previous section. \n\n Do not output any other text or explanation.
"""
    
    system_prompt = f"""
You are an intelligent assistant. Given a graph problem and its response, your task is to review the response, which is divided into multiple parts with each step labeled using tags. After reading through the steps, you should group them into distinct sections, where each section represents a complete and logical problem-solving attempt or process.

Specific instructions:

1. Each section should be a standalone and complete problem-solving approach or effort.

2. For each section, include both the starting and ending tags (the ending tag should not be earlier than the starting tag). Additionally, provide a brief summary or title of the section.

3. Present your output in this format: <<start tag>> - <<end tag>> [Brief description]. The end tag should be the same as or later than the start tag of the section, and the start tag of the next section should follow directly after the end tag of the previous section.\n\n Do not output any other text or explanation.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
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
    for i in range(n - 1):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data.append(data_list[start:end])
    last_start = (n - 1) * batch_size
    last_end = len(data_list)
    batch_data.append(data_list[last_start:last_end])
    return batch_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM inference on causal data')
    parser.add_argument("--model_path", type=str, default="/path/to/models/Qwen/Qwen2.5-72B-Instruct", help="Loading path of the LLM")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    parser.add_argument("--max_tokens", type=int, default=8000, help="Max tokens for inference")
    parser.add_argument("--task_name", type=str, default="MKC", help="Task name")
    parser.add_argument("--difficulty", type=str, default="hard", help="Difficulty")
    parser.add_argument("--response_generated_from_what_model", type=str, default="Qwen3-32B", help="Name of the model that generated responses")
    args = parser.parse_args()

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    save_dir = f"/path/to/GrAlgoBench/error_analysis/reformatted_response/{date}"

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(f"{save_dir}/{args.response_generated_from_what_model}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=args.max_tokens) 
    print('sampling =====', sampling_params)
    llm = LLM(model=args.model_path, tensor_parallel_size=4, gpu_memory_utilization=0.9, max_num_seqs=args.batch_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    temp = 0

    input_texts = [] 
    results_dict = {}

    data_path = f'/path/to/GrAlgoBench/Inference/final_results/{args.response_generated_from_what_model}/{args.task_name}-{args.difficulty}.json'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input file not found: {data_path}")
    
    datas = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            datas.append(json.loads(line))
    
    with open(f"/path/to/GrAlgoBench/data_generation/dataset/{args.task_name}_{args.difficulty}.pkl", "rb") as f:
        problem_and_gt = pickle.load(f)
    
    dataset_loc = '/path/to/GrAlgoBench/data_generation/dataset'
    task = globals()[args.task_name + '_Task'](dataset_loc)
    task.load_dataset(args.difficulty)

    print('loading datas')

    for i, (data, qa) in enumerate(zip(datas, problem_and_gt)):
        question = qa["problem_text"]
        gt = qa["exact_answer"]

        for j in range(8):
            k = j + 1
            _, correctness = task.check_solution(i, data[f"predict{k}"])
            if correctness == 1:
                continue
            original_answer = data[f"predict{k}"]
            if args.response_generated_from_what_model not in ["Qwen2.5-32B", "Qwen3-32B-no-thinking"]:
                original_answer = original_answer.split('</think>')[0]
            answer = generate_reformated_response(original_answer)
            template_text = generate_text_template(args, question, answer, tokenizer)
            input_texts.append((question, template_text, gt, answer, original_answer))

    print(f'input_texts length is {len(input_texts)}\n')

    batch_inputs = batch_data(input_texts[temp:], batch_size=args.batch_size)
    print(f'total samples are: {len(input_texts)}')
    logging.info(f'total samples are: {len(input_texts)}')

    reformated_responses = []
    for batch_input in tqdm(batch_inputs):  
        questions = [item[0] for item in batch_input]
        template_texts = [item[1] for item in batch_input]
        gts = [item[2] for item in batch_input]
        answers = [item[3] for item in batch_input]
        original_answers = [item[4] for item in batch_input]
        completions = llm.generate(template_texts, sampling_params)
        
        for i, (question, gt, answer, original_answer, output) in enumerate(zip(questions, gts, answers, original_answers, completions)):
            predict = output.outputs[0].text
            reformated_responses.append(
                {
                    "id": len(reformated_responses),
                    "problem": question,
                    "answer": gt,
                    "original_response": original_answer,
                    "reformated_response": generate_final_response(answer, predict)
                }
            )

    with open(f"{save_dir}/{args.response_generated_from_what_model}/{args.task_name}-{args.difficulty}.json", "w") as f:
        json.dump(reformated_responses, f, indent=4)