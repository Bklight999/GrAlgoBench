"""
Unified overthinking test module
Integrates all overthinking-related test functionalities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import jsonlines
from pathlib import Path
from tqdm import tqdm
from common_utils import (
    split_response_by_wait, batch_data, setup_model_and_tokenizer, 
    create_sampling_params, load_json_data, load_pickle_data, load_math_dataset,
    load_math_competition_dataset, extract_math_problem_info, extract_graph_problem_info,
    extract_math_competition_problem_info, setup_output_directory,
    get_common_argument_parser, PATHS, GRAPH_TASKS, MATH_TASKS, MATH_COMPETITION_TASKS, DIFFICULTIES, MODELS
)


def generate_overthinking_template(args, question, gt, answer, tokenizer, task_type="graph"):
    """
    Generate template for overthinking judgment
    
    Args:
        args: Command line arguments
        question: Question text
        gt: Ground truth answer
        answer: Model answer
        tokenizer: Tokenizer
        task_type: Task type ("graph" or "math")
    
    Returns:
        Generated template text
    """
    domain = "graph problem" if task_type == "graph" else "mathematics problem"
    assistant_type = "graph theory problems" if task_type == "graph" else "mathematics problems"
    
    prompt = f"""
Given the following {domain}, the ground truth of the problem, and an answer containing both analysis and the final answer, please determine whether the answer matches the ground truth in meaning (e.g., "1" and "one" are equivalent). Only output YES if the ground truth is explicitly presented and clearly identified as the final answer or conclusion in the answer (e.g., by phrases like "the answer is," "thus," "therefore," "in conclusion," etc.), not just mentioned as part of the reasoning process. If the ground truth is not explicitly and unambiguously identified as the final answer, output NO. Do not provide any explanation.

Problem: {question}

Ground truth: {gt}

Answer: {answer}

"""
    
    system_prompt = f"""
You are a precise grading assistant for {assistant_type}.

Your task is to strictly judge whether the provided answer contains an explicitly stated final answer that matches the given ground truth in meaning (e.g., "1" and "one" are equivalent).

Only output YES if the answer clearly and unambiguously identifies the ground truth value as the final answer or conclusion (e.g., by phrases like "the answer is," "thus," "therefore," "in conclusion," etc.), not just mentioning it as part of the reasoning process.

If the ground truth is not explicitly and unambiguously stated as the final answer, or is only mentioned as an intermediate value, output NO.

Provide the judge without any explanation.
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


def process_graph_overthinking(args, llm, tokenizer, sampling_params):
    """Process graph-related overthinking tests"""
    print("Processing graph overthinking...")
    
    # 设置路径
    data_path = f'{PATHS["graph_inference"]}/{args.response_generated_from_what_model}/{args.task_name}-{args.difficulty}.json'
    dataset_path = f"{PATHS['graph_dataset']}/{args.task_name}_{args.difficulty}.pkl"
    
    # 加载数据
    datas = load_json_data(data_path)
    problem_and_gt = load_pickle_data(dataset_path)
    
    return process_overthinking_data(args, datas, problem_and_gt, llm, tokenizer, sampling_params, "graph")


def process_math_overthinking(args, llm, tokenizer, sampling_params):
    """Process math-related overthinking tests"""
    print("Processing math overthinking...")
    
    # 设置路径
    data_path = f'{PATHS["math_inference"]}/{args.response_generated_from_what_model}/{args.task_name}-{args.difficulty}.json'
    
    # 加载数据
    datas = load_json_data(data_path)
    problem_and_gt = load_math_dataset(args.task_name)
    
    return process_overthinking_data(args, datas, problem_and_gt, llm, tokenizer, sampling_params, "math")


def process_overthinking_data(args, datas, problem_and_gt, llm, tokenizer, sampling_params, task_type):
    """
    Core logic for processing overthinking data
    
    Args:
        args: Command line arguments
        datas: Inference data
        problem_and_gt: Problem and ground truth data
        llm: Language model
        tokenizer: Tokenizer
        sampling_params: Sampling parameters
        task_type: Task type ("graph" or "math")
    
    Returns:
        Processing results
    """
    print(f'Loading {len(datas)} problems for {task_type} overthinking')

    all_evaluations = []  # 存储所有评估结果
    input_texts = []

    for i, (data, qa) in enumerate(zip(datas, problem_and_gt)):
        # Extract question and answer information
        if task_type == "graph":
            question, gt = extract_graph_problem_info(qa, args.task_name)
        elif task_type == "math":
            question, gt = extract_math_problem_info(qa, args.task_name)
        else:  # math_competition
            question, gt = extract_math_competition_problem_info(qa, args.task_name)

        # Process 8 prediction results
        for j in range(8):
            k = j + 1
            answer = data[f"predict{k}"]
            answer = answer.split('</think>')[0]
            
            # Split response by wait
            response_parts = split_response_by_wait(answer)
            print(f"Problem {i}, Predict {k}: {len(response_parts)} parts")
            
            # Create evaluation for each part
            data_evaluations = []
            for t, part in enumerate(response_parts):
                template_text = generate_overthinking_template(args, question, gt, part, tokenizer, task_type)
                input_texts.append((template_text, part, 8*i+j, t))
                data_evaluations.append({"response": part, "label": None})
            
            all_evaluations.append(data_evaluations)

    print(f'Total input texts: {len(input_texts)}')

    # Batch processing
    batch_inputs = batch_data(input_texts, batch_size=args.batch_size)
    
    for batch_input in tqdm(batch_inputs):
        # Extract template texts for model inference
        template_texts = [item[0] for item in batch_input]
        original_parts = [item[1] for item in batch_input]
        data_indices = [item[2] for item in batch_input]
        part_indices = [item[3] for item in batch_input]
        
        completions = llm.generate(template_texts, sampling_params)
        
        for template, original_part, data_idx, part_idx, output in zip(
            template_texts, original_parts, data_indices, part_indices, completions):
            predict = output.outputs[0].text
            all_evaluations[data_idx][part_idx]["label"] = predict

    return all_evaluations


def save_overthinking_results(args, all_evaluations, task_type):
    """Save overthinking results"""
    # Setup output directory
    if task_type == "graph":
        save_dir = PATHS["overthinking_graph"]
    else:
        save_dir = PATHS["overthinking_math"]
    
    output_dir = setup_output_directory(save_dir, args.response_generated_from_what_model)
    output_file = output_dir / f"{args.task_name}-{args.difficulty}.json"
    
    # Write results
    with jsonlines.open(output_file, mode='w') as wf:
        for i, evaluations in enumerate(all_evaluations):
            wf.write({f"evaluations_{i//8}_predict_{i%8+1}": evaluations})
    
    print(f"Results saved to: {output_file}")
    return output_file


def main():
    parser = get_common_argument_parser()
    parser.add_argument("--task_type", type=str, choices=["graph", "math", "math_competition", "both"], 
                       default="graph", help="Type of task to process")
    parser.add_argument("--task_name", type=str, help="Task name")
    parser.add_argument("--difficulty", type=str, choices=DIFFICULTIES, 
                       default="easy", help="Difficulty level")
    parser.add_argument("--response_generated_from_what_model", type=str, 
                       choices=MODELS, default="QWQ-32B", 
                       help="Model that generated the responses")
    
    args = parser.parse_args()
    
    # Validate task name
    if args.task_type == "graph" and args.task_name not in GRAPH_TASKS:
        raise ValueError(f"Graph task {args.task_name} not supported. Choose from: {GRAPH_TASKS}")
    elif args.task_type == "math" and args.task_name not in MATH_TASKS:
        raise ValueError(f"Math task {args.task_name} not supported. Choose from: {MATH_TASKS}")
    elif args.task_type == "math_competition" and args.task_name not in MATH_COMPETITION_TASKS:
        raise ValueError(f"Math competition task {args.task_name} not supported. Choose from: {MATH_COMPETITION_TASKS}")
    
    # Initialize model
    sampling_params = create_sampling_params(args.temperature, args.top_p, args.max_tokens)
    llm, tokenizer = setup_model_and_tokenizer(
        args.model_path, 
        args.tensor_parallel_size, 
        args.gpu_memory_utilization, 
        args.batch_size
    )
    
    # Process different types of tasks
    if args.task_type == "graph":
        all_evaluations = process_graph_overthinking(args, llm, tokenizer, sampling_params)
        save_overthinking_results(args, all_evaluations, "graph")
    elif args.task_type == "math":
        all_evaluations = process_math_overthinking(args, llm, tokenizer, sampling_params)
        save_overthinking_results(args, all_evaluations, "math")
    elif args.task_type == "math_competition":
        # Process math competition data
        data_path = f'{PATHS["math_competition_inference"]}/{args.response_generated_from_what_model}/{args.task_name}-competition.json'
        datas = load_json_data(data_path)
        problem_and_gt = load_math_competition_dataset(args.task_name)
        all_evaluations = process_overthinking_data(args, datas, problem_and_gt, llm, tokenizer, sampling_params, "math_competition")
        
        # Setup output directory for math competition
        save_dir = PATHS["overthinking_math_competition"]
        output_dir = setup_output_directory(save_dir, args.response_generated_from_what_model)
        output_file = output_dir / f"{args.task_name}-competition.json"
        
        # Write results
        with jsonlines.open(output_file, mode='w') as wf:
            for i, evaluations in enumerate(all_evaluations):
                wf.write({f"evaluations_{i//8}_predict_{i%8+1}": evaluations})
        print(f"Results saved to: {output_file}")
    elif args.task_type == "both":
        # Process graph tasks
        if args.task_name in GRAPH_TASKS:
            all_evaluations_graph = process_graph_overthinking(args, llm, tokenizer, sampling_params)
            save_overthinking_results(args, all_evaluations_graph, "graph")
        
        # Process math tasks  
        if args.task_name in MATH_TASKS:
            all_evaluations_math = process_math_overthinking(args, llm, tokenizer, sampling_params)
            save_overthinking_results(args, all_evaluations_math, "math")


if __name__ == '__main__':
    main()

