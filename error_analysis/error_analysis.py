from openai import OpenAI
from collections import defaultdict
import pickle
import argparse
import json
from time import sleep
import concurrent.futures
import time
from tqdm import tqdm
from argparse import Namespace
import jsonlines
import os
from pathlib import Path
from datetime import datetime

llm_to_api = {
    "gpt4": "gpt-4o",
    "mini": "gpt-4o-mini",
    "gpt": "gpt-3.5-turbo-0125", 
    "claude": "claude-3-haiku-20240307",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "o1m": "o1-mini",
    "o3m": "o3-mini-2025-01-31-high",
    "deepseek_r1": "deepseek-reasoner",
    "deepseek_v3": "deepseek-chat",
    "llama8b": "meta-llama/Llama-3-8b-chat-hf",
    "llama": "meta-llama/Llama-3-70b-chat-hf",
    "qwen3_32b": "qwen3-32b",
    "qwen2.5_32b": "qwen2.5-32b-instruct",
    "qwq_32b": "qwq-32b",
    "gemini": "gemini-1.5-pro",
    "gemma": "gemma-7b-it",
}

correct_algorithm_dict = {
    "Triangle": ["brute force enumeration (check all possible triangles)"],
    "MCP": ["branch and bound", "brute force enumeration (check all possible vertex sequences)"],
    "MaxDegree": ["brute force enumeration (check degree of all vertices)"],
    "MST": ["Kruskal's algorithm", "Prim's algorithm", "brute force enumeration (check all possible spanning trees)"],
    "MKC": ["Peeling algorithm", "brute force enumeration (check all possible k-cores)"],
    "DistanceThreshold": ["Dijikstra's algorithm", "Floyd-Warshall algorithm", "brute force enumeration (check all possible paths)"],
    "PathSum": ["DFS", "BFS", "brute force enumeration (check all possible vertex sequences)"],
    "DistanceK": ["DFS", "BFS", "brute force enumeration (check all possible vertex sequences)"],
    "Diameter": ["DFS", "BFS", "brute force enumeration (check all possible vertex sequences)"]
}

efficient_algorithm_dict = {
    "Triangle": ["brute force enumeration (check all possible triangles)"],
    "MCP": ["branch and bound"],
    "MaxDegree": ["brute force enumeration (check degree of all vertices)"],
    "MST": ["Kruskal's algorithm", "Prim's algorithm"],
    "MKC": ["Peeling algorithm"],
    "DistanceThreshold": ["Dijikstra's algorithm", "Floyd-Warshall algorithm"],
    "PathSum": ["DFS", "BFS"],
    "DistanceK": ["DFS", "BFS"],
    "Diameter": ["DFS", "BFS"]
}


def process_data(i, data, llm, client, llm_to_api, response_dict, args, save_dir, all_results):
    global error_knt
    system_prompt = f"""
You are an intelligent AI assistant. Given a graph problem and an LLM's response to that problem, analyze the LLM’s response to identify any errors it contains. Use the following refined error categories and definitions for your analysis:

# Output Quality
* redundancy: Unnecessary repetition or superfluous information in the solution.

# Algorithm Selection
* incorrect algorithm: The chosen algorithm cannot produce the correct solution for the problem. In this task, the correct algorithms are: {correct_algorithm_dict[args.task]}.
* suboptimal algorithm: The algorithm used is inefficient or is implemented in a less efficient way. In this task, the efficient algorithms are: {efficient_algorithm_dict[args.task]}.

# Information Errors
* graph memorization error: Misunderstanding or incorrect memory of the graph structure, such as extra or missing nodes/edges.

# Algorithm Execution
* state update error: During algorithm execution, state variables or data structures are updated incorrectly, causing subsequent steps to operate on erroneous states.
* omission: Missing important elements during execution, e.g., failure to traverse all nodes or edges.
* condition misjudgment: Mistakes in condition checks, such as incorrect if-statement or loop-condition evaluations.

Instructions: The LLM’s response will be segmented into sections labeled as "[Section 1], content...", "[Section 2], content...", etc.

Your response format should be a list of error annotations as:
"[section index, error type, detailed error analysis]"

If multiple errors exist in one section, list them separately. 
Do not output anything other than these error annotations.
    """

    question = data["problem"]
    llm_response = data["reformated_response"]

    user_prompt = f"""
Here is the problem and the LLM's response:

Graph Problem: {question}

LLM's Response: {llm_response}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": system_prompt + user_prompt}],
            model=llm_to_api[llm],
            seed=42,
            temperature=0.1,
            max_tokens=4096,
        )
        response_dict["predict"] = chat_completion.choices[0].message.content
    except Exception as e:
        print('Call API failed! ', e)
        time.sleep(1)
        response_dict["predict"] = 'Error!'
        error_knt += 1

    all_results.append(response_dict)


def main(args, datas, llm, client, llm_to_api, response_dict, save_dir, all_results):
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for i in range(args.st, args.ed):
            data = datas[i]
            response_dict = {
                "id": data['id'],
                'problem_text': data['problem'],
                'gt': data['answer'],
                'response': data['reformated_response'],
                "predict": ""
            }
            if args.resume:
                done_flag = False
                for result in all_results:
                    if response_dict["id"] == result["id"] and result["predict"] != "Error!":
                        done_flag = True
                        break
                    elif response_dict["id"] == result["id"] and result["predict"] == "Error!":
                        all_results.remove(result)
                        break
                if done_flag:
                    continue
            futures.append(executor.submit(process_data, i, data, llm, client, llm_to_api, response_dict, args, save_dir, all_results))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()

    time.sleep(args.sleep)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='o3m', help='llm model name')
    parser.add_argument('--task', type=str, default='MKC', help='task name')
    parser.add_argument('--difficulty', type=str, default='easy', help='problem difficulty')
    parser.add_argument('--resume', type=bool, default=False, help='resume from last checkpoint')
    parser.add_argument('--sleep', type=int, default=5, help='sleep seconds between API calls')
    parser.add_argument('--st', type=int, default=0, help='start index')
    parser.add_argument('--ed', type=int, default=5, help='end index')
    parser.add_argument('--num_workers', type=int, default=64, help='Number of workers in the thread pool')
    parser.add_argument("--response_generated_from_what_model", type=str, default="Qwen3-32B", help="response generated from what model")

    args = parser.parse_args()
    error_knt = 0
    
    response_dict = defaultdict(dict)
    
    for llm in args.llm.split('-'):
        if 'gpt' in llm or 'mini' in llm or 'o1m' in llm or 'o3m' in llm:
            client = OpenAI(base_url="https://api.openai.com/v1", api_key="YOUR_API_KEY")
        elif llm == 'deepseek_v3' or llm == 'deepseek_r1':
            client = OpenAI(base_url="https://api.deepseek.com", api_key="YOUR_API_KEY")
        elif 'llama' in llm or 'mixtral' in llm:
            client = OpenAI(base_url="https://api.aimlapi.com/", api_key="YOUR_API_KEY")
        elif 'qwen' in llm:
            client = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key="YOUR_API_KEY")
        else:
            client = OpenAI(base_url="https://aigcbest.top/v1", api_key="YOUR_API_KEY")

        date = datetime.now().strftime("%Y-%m-%d")
        save_dir = f"/path/to/GrAlgoBench/error_analysis/results_{date}/{args.response_generated_from_what_model}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        datas = []
        for difficulty in ['easy', 'medium', 'hard', 'challenge']:
            data_path = f'/path/to/GrAlgoBench/error_analysis/reformatted/{args.response_generated_from_what_model}/{args.task}-{difficulty}.json'
            with open(data_path, 'r') as f:
                data = json.load(f)

        print(len(datas))
        args.ed = len(datas)
        all_results = []

        if args.resume and os.path.exists(f"{save_dir}/{args.task}.json"):
            with jsonlines.open(f"{save_dir}/{args.task}.json", mode='r') as reader:
                all_results = list(reader)

        main(args, datas, llm, client, llm_to_api, response_dict, save_dir, all_results)
        
        print('error_knt:', error_knt)

        all_results.sort(key=lambda x: x['id'])
        with jsonlines.open(f"{save_dir}/{args.task}.json", mode='w') as writer:
            for item in all_results:
                writer.write(item)