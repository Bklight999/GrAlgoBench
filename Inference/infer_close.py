from openai import OpenAI
from collections import defaultdict
import pickle
import argparse
import json
import datetime
from time import sleep
import concurrent.futures
import json
import time
from tqdm import tqdm
from argparse import Namespace
import jsonlines
import os
from pathlib import Path
from datetime import datetime
import pdb

llm_to_api = {
    "deepseek_v3": "deepseek-chat",
    "deepseek_r1": "deepseek-reasoner",
    "qwen3-235B-thinking": "",
    "qwen3-235B-non-thinking": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "gpt_4.1_mini": "gpt-4.1-mini",
    "o4_mini": "o4-mini-2025-04-16-high",
    "gemini2.5": "gemini-2.5-pro",
    "gpt5_mini": "gpt-5-mini",
    "gpt5": "gpt-5",
    "gemini2.5_pro": "gemini-2.5-pro",
    "gpt_oss_20b": "gpt-oss-20b",
    "gpt_oss_120b": "gpt-oss-120b",
    "gemini2.5_thinking": "gemini-2.5-pro",
    "gemini2.5_non_thinking": "gemini-2.5-pro-nothinking"
}


def process_data(i, data, llm, client, llm_to_api, response_dict, args, save_dir, all_results, sampling_times):
    global error_knt
    system_prompt = """You are an expert in graph reasoning problems. 
Please solve the following problem and output the answer in middle brackets. 
Do not use code to solve the problem."""

    question = data["problem_text"]

    try:
        if "deepseek" in llm:
            for i in range(sampling_times):
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                    model=llm_to_api[llm],
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=8192 if llm == 'deepseek_v3' else 32768,
                    n=1
                )
                response_dict[f"predict{i}"] = chat_completion.choices[0].message.content
        elif "oss" or "gemini" in llm:
            for i in range(sampling_times):
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": question},
                    ],
                    model=llm_to_api[llm],
                )
                response_dict[f"predict{i}"] = chat_completion.choices[0].message.content
        elif "gpt" in llm:
            for i in range(sampling_times):
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                    model=llm_to_api[llm],
                )
                response_dict[f"predict{i}"] = chat_completion.choices[0].message.content
        else:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": question},
                ],
                model=llm_to_api[llm],
                temperature=0.6,
                top_p=0.95,
                max_tokens=32768,
                n=sampling_times
            )
            for i in range(sampling_times):
                response_dict[f"predict{i}"] = chat_completion.choices[i].message.content

    except Exception as e:
        print('Call API failed! ', e)
        time.sleep(1)
        for i in range(sampling_times):
            response_dict[f"predict{i}"] = 'Error!'
        error_knt += 1

    all_results.append(response_dict)


def main(args, datas, llm, client, llm_to_api, response_dict, save_dir, all_result, sampling_times):
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for i in range(args.st, args.ed):
            data = datas[i]
            response_dict = {"id": data["id"]}
            for i in range(sampling_times):
                response_dict[f"predict{i}"] = ""
            print(response_dict)
            if args.resume:
                done_flag = False
                for result in all_results:
                    if response_dict["id"] == result["id"] and result["predict0"] != "Error!":
                        done_flag = True
                        break
                    elif response_dict["id"] == result["id"] and result["predict0"] == "Error!":
                        all_results.remove(result)
                        break
                if done_flag:
                    continue
            futures.append(executor.submit(process_data, i, data, llm, client, llm_to_api,
                                           response_dict, args, save_dir, all_results, sampling_times))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()

    time.sleep(args.sleep)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='gpt5_mini', help='llm model name')
    parser.add_argument('--task', type=str, default='MST', help='task name')
    parser.add_argument('--difficulty', type=str, default='hard', help='problem difficulty')
    parser.add_argument('--resume', type=bool, default=False, help='resume from last checkpoint')
    parser.add_argument('--sleep', type=int, default=5, help='sleep seconds between API calls')
    parser.add_argument('--st', type=int, default=0, help='start index')
    parser.add_argument('--ed', type=int, default=0, help='end index')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers in the thread pool')
    parser.add_argument('--sampling_times', type=int, default=4, help='Number of sampling times')

    args = parser.parse_args()
    error_knt = 0
    
    response_dict = defaultdict(dict)

    llm = args.llm
    cnt = 0

    # API clients configured by llm type (keys and endpoints shown here as placeholders)
    if "deepseek" in args.llm:
        client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key="sk-your_api_key_here"
        )
    elif "oss" in args.llm:
        client = OpenAI(
            base_url="https://api2.aigcbest.top/v1",
            api_key="sk-your_api_key_here"
        )
    elif "gpt" in args.llm:
        client = OpenAI(
            base_url="https://35.aigcbest.top/v1",
            api_key="sk-your_api_key_here"
        )
    elif "gemini" in args.llm:
        client = OpenAI(
            base_url="https://api2.aigcbest.top/v1",
            api_key="sk-your_api_key_here"
        )
    else:
        client = OpenAI(
            base_url="https://api2.aigcbest.top/v1",
            api_key="sk-your_api_key_here"
        )
   
    # input and output paths
    data_path = f'/path/to/data_generation/dataset_0830/{args.task}_{args.difficulty}.pkl'
    date = datetime.now().strftime("%Y-%m-%d")
    save_dir = f"/path/to/Inference/results_qianduoduo_{date}/{args.llm}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    with open(data_path, 'rb') as f:
        datas = pickle.load(f)
    print(len(datas))
    args.ed = len(datas)
    all_results = []

    if args.resume and os.path.exists(f"{save_dir}/{args.task}-{args.difficulty}.json"):
        with jsonlines.open(f"{save_dir}/{args.task}-{args.difficulty}.json", mode='r') as reader:
            all_results = list(reader)

    main(args, datas, llm, client, llm_to_api, response_dict, save_dir, all_results, args.sampling_times)
    
    print('error_knt:', error_knt)

    all_results.sort(key=lambda x: x['id'])

    with jsonlines.open(f"{save_dir}/{args.task}-{args.difficulty}.json", mode='w') as writer:
        for item in all_results:
            writer.write(item)