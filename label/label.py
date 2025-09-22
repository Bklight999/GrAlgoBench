"""
Unified label module
Integrates all labeling functionalities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from common_utils import (
    split_response_by_keywords, setup_model_and_tokenizer, create_sampling_params,
    load_json_data, load_pickle_data, load_math_dataset, load_math_competition_dataset,
    extract_math_problem_info, extract_graph_problem_info, extract_math_competition_problem_info,
    setup_output_directory, save_statistics, print_statistics, get_common_argument_parser,
    PATHS, GRAPH_TASKS, MATH_TASKS, MATH_COMPETITION_TASKS, DIFFICULTIES, MODELS
)


def generate_label_template(args, segment_a, segment_b, tokenizer, task_type="graph"):
    """
    Generate template for segment relationship analysis
    
    Args:
        args: Command line arguments
        segment_a: First text segment
        segment_b: Second text segment
        tokenizer: Tokenizer
        task_type: Task type ("graph" or "math")
    
    Returns:
        Generated template text
    """
    if task_type == "graph":
        domain_desc = "graph theory problem-solving process"
        expert_desc = "text relationships and functions"
    elif task_type == "math":
        domain_desc = "mathematical problem-solving process"
        expert_desc = "mathematical problem-solving text relationships and functions"
    else:  # math_competition
        domain_desc = "mathematical competition problem-solving process"
        expert_desc = "mathematical competition text relationships and functions"
    
    prompt = f"""
Given two text segments from a {domain_desc}, A and B, determine the function of B in relation to A. There are three possible options:

(1) Self-reflection: B serves to evaluate or verify the correctness of A.

(2) Strategy shift: B serves to alter or adjust the strategy presented in A.

(3) None: B has no clear functional relationship to A, or serves a different purpose not covered by the above categories.

Output only one of the following options without any explanation: Self-reflection, Strategy shift, or None.

Segment A: {segment_a}

Segment B: {segment_b}

"""
    
    system_prompt = f"""
You are an expert in analyzing {expert_desc}. Your task is to strictly classify the function of Segment B in relation to Segment A, according to the provided definitions. Always output only the corresponding option without any additional explanation or commentary.
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


def process_label_data(args, datas, problem_and_gt, llm, tokenizer, sampling_params, task_type):
    """
    Core logic for processing labeling data
    
    Args:
        args: Command line arguments
        datas: Inference data
        problem_and_gt: Problem and ground truth data
        llm: Language model
        tokenizer: Tokenizer
        sampling_params: Sampling parameters
        task_type: Task type ("graph" or "math")
    
    Returns:
        Processing results and statistics
    """
    print(f'Loading {len(datas)} problems for {task_type} labeling')

    # Process data and build segment pairs
    all_prompts = []
    all_seg_pairs = []
    all_id_predict = []
    all_indices = []
    all_problem_info = []

    for i, (data, qa) in enumerate(zip(datas, problem_and_gt)):
        # Extract question and answer information
        if task_type == "graph":
            question, gt = extract_graph_problem_info(qa, args.task_name)
        elif task_type == "math":
            question, gt = extract_math_problem_info(qa, args.task_name)
        else:  # math_competition
            question, gt = extract_math_competition_problem_info(qa, args.task_name)
        
        cur_id = data['id']
        
        # Process each prediction result
        for j in range(8):
            k = j + 1
            predict_key = f"predict{k}"
            if predict_key not in data:
                continue
                
            answer = data[predict_key]
            answer = answer.split('</think>')[0]  # Remove thinking tags
            answer = answer.lower()  # Convert to lowercase
            
            # Split by wait and build cumulative segment pairs
            parts = re.split(r'(wait)', answer, flags=re.IGNORECASE)
            segments = []
            segment_a = parts[0].strip()
            
            # Build segment pairs
            for part_idx in range(1, len(parts)-1, 2):
                if part_idx + 1 < len(parts):
                    segment_b = parts[part_idx] + parts[part_idx+1]
                    segments.append((segment_a, segment_b.strip()))
                    segment_a += parts[part_idx] + parts[part_idx+1]  # Accumulate previous content
            
            # Generate labeling task for each segment pair
            for seg_idx in range(len(segments)):
                segment_a, segment_b = segments[seg_idx]
                if segment_a.strip() and segment_b.strip():  # Ensure segments are not empty
                    template_text = generate_label_template(args, segment_a, segment_b, tokenizer, task_type)
                    all_prompts.append(template_text)
                    all_seg_pairs.append((segment_a, segment_b))
                    all_id_predict.append((cur_id, predict_key))
                    all_indices.append(seg_idx)
                    all_problem_info.append({
                        'question': question,
                        'ground_truth': gt,
                        'problem_idx': i,
                        'predict_idx': j
                    })

    print(f'Total segment pairs to process: {len(all_prompts)}')

    # Batch inference
    batch_size = args.batch_size
    results = []
    
    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i:i+batch_size]
        completions = llm.generate(batch_prompts, sampling_params)
        batch_results = [output.outputs[0].text.strip() for output in completions]
        results.extend(batch_results)
        
        # Show progress
        if (i // batch_size + 1) % 10 == 0:
            print(f'Processed {i + len(batch_prompts)} / {len(all_prompts)} segment pairs')

    # Organize output results
    output_dict = defaultdict(lambda: defaultdict(list))
    
    for idx, ((cur_id, predict_key), (segment_a, segment_b), label, seg_idx, problem_info) in enumerate(
        zip(all_id_predict, all_seg_pairs, results, all_indices, all_problem_info)):
        
        output_dict[(cur_id, predict_key)][seg_idx] = {
            "segment_a": segment_a, 
            "segment_b": segment_b, 
            "label": label
        }

    # Generate statistics
    stats = {
        'total_problems': len(datas),
        'total_predictions': len(datas) * 8,
        'total_segment_pairs': len(all_prompts),
        'self_reflection_count': sum(1 for r in results if 'self-reflection' in r.lower()),
        'strategy_shift_count': sum(1 for r in results if 'strategy shift' in r.lower()),
        'other_count': sum(1 for r in results if 'self-reflection' not in r.lower() and 'strategy shift' not in r.lower())
    }
    
    return output_dict, stats, results


def save_label_results(args, output_dict, stats, results, task_type):
    """Save labeling results"""
    # Setup output directory
    if task_type == "graph":
        save_dir = PATHS["label_graph"]
    elif task_type == "math":
        save_dir = PATHS["label_math"]
    else:  # math_competition
        save_dir = PATHS["label_math_competition"]
    
    output_dir = setup_output_directory(save_dir, args.response_generated_from_what_model)
    
    # Save main results
    output_path = output_dir / f"{args.task_name}-{args.difficulty}.jsonl"
    with open(output_path, 'w', encoding='utf-8') as wf:
        for (cur_id, predict_key), segs in output_dict.items():
            # Ensure order
            evaluations = [segs[i] for i in sorted(segs.keys())]
            out_obj = {
                "id": cur_id, 
                "predict_id": predict_key, 
                "evaluations": evaluations
            }
            wf.write(json.dumps(out_obj, ensure_ascii=False) + '\n')

    print(f'Results saved to: {output_path}')
    
    # Save statistics
    stats_path = output_dir / f"{args.task_name}-{args.difficulty}-stats.json"
    save_statistics(stats, str(stats_path))
    
    # Print statistics
    print_statistics(stats, len(results))
    print(f"Statistics saved to: {stats_path}")
    
    return output_path, stats_path


def process_graph_labeling(args, llm, tokenizer, sampling_params):
    """Process graph-related labeling tasks"""
    print("Processing graph labeling...")
    
    # Setup paths
    data_path = f'{PATHS["graph_inference"]}/{args.response_generated_from_what_model}/{args.task_name}-{args.difficulty}.json'
    dataset_path = f"{PATHS['graph_dataset']}/{args.task_name}_{args.difficulty}.pkl"
    
    # Load data
    datas = load_json_data(data_path)
    problem_and_gt = load_pickle_data(dataset_path)
    
    # Process data
    output_dict, stats, results = process_label_data(args, datas, problem_and_gt, llm, tokenizer, sampling_params, "graph")
    
    # Save results
    return save_label_results(args, output_dict, stats, results, "graph")


def process_math_labeling(args, llm, tokenizer, sampling_params):
    """Process math-related labeling tasks"""
    print("Processing math labeling...")
    
    # Setup paths
    data_path = f'{PATHS["math_inference"]}/{args.response_generated_from_what_model}/{args.task_name}-{args.difficulty}.json'
    
    # Load data
    datas = load_json_data(data_path)
    problem_and_gt = load_math_dataset(args.task_name)
    
    # Process data
    output_dict, stats, results = process_label_data(args, datas, problem_and_gt, llm, tokenizer, sampling_params, "math")
    
    # Save results
    return save_label_results(args, output_dict, stats, results, "math")


def process_math_competition_labeling(args, llm, tokenizer, sampling_params):
    """Process math competition labeling tasks"""
    print("Processing math competition labeling...")
    
    # Setup paths
    data_path = f'{PATHS["math_competition_inference"]}/{args.response_generated_from_what_model}/{args.task_name}-competition.json'
    
    # Load data
    datas = load_json_data(data_path)
    problem_and_gt = load_math_competition_dataset(args.task_name)
    
    # Process data
    output_dict, stats, results = process_labeling_data(args, datas, problem_and_gt, llm, tokenizer, sampling_params, "math_competition")
    
    # Save results
    return save_label_results(args, output_dict, stats, results, "math_competition")


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
        process_graph_labeling(args, llm, tokenizer, sampling_params)
    elif args.task_type == "math":
        process_math_labeling(args, llm, tokenizer, sampling_params)
    elif args.task_type == "math_competition":
        process_math_competition_labeling(args, llm, tokenizer, sampling_params)
    elif args.task_type == "both":
        # Process graph tasks
        if args.task_name in GRAPH_TASKS:
            process_graph_labeling(args, llm, tokenizer, sampling_params)
        
        # Process math tasks  
        if args.task_name in MATH_TASKS:
            process_math_labeling(args, llm, tokenizer, sampling_params)


if __name__ == '__main__':
    main()

