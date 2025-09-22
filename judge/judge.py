"""
Unified judge module
Integrates all judgment logic functionalities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import jsonlines
from pathlib import Path
from tqdm import tqdm
from common_utils import (
    batch_data, setup_model_and_tokenizer, create_sampling_params,
    generate_judgment_prompt, parse_model_response, setup_output_directory,
    get_common_argument_parser, PATHS, GRAPH_TASKS, MATH_TASKS, MATH_COMPETITION_TASKS, DIFFICULTIES, MODELS
)


def judge_segment_effectiveness(batch_items, llm, tokenizer, sampling_params):
    """
    Use pre-initialized model and tokenizer to judge effectiveness
    
    Args:
        batch_items: Batch data items
        llm: Language model
        tokenizer: Tokenizer
        sampling_params: Sampling parameters
    
    Returns:
        List of judgment results
    """
    # Batch generate prompts
    prompts = [generate_judgment_prompt(item["segment_a"], item["segment_b"], item["label"]) 
              for item in batch_items]
    
    # Batch call model
    completions = llm.generate(prompts, sampling_params)
    responses = [completion.outputs[0].text for completion in completions]
    
    # Batch parse results
    results = [parse_model_response(response) for response in responses]
    return results


def process_judge_data(args, llm, tokenizer, sampling_params, task_type):
    """
    Core logic for processing judgment data
    
    Args:
        args: Command line arguments
        llm: Language model
        tokenizer: Tokenizer
        sampling_params: Sampling parameters
        task_type: Task type ("graph" or "math")
    
    Returns:
        Processing results
    """
    # Setup input/output paths
    if task_type == "graph":
        dataset_dir = PATHS["label_graph"]
        results_dir = PATHS["judge_graph"]
    elif task_type == "math":
        dataset_dir = PATHS["label_math"]
        results_dir = PATHS["judge_math"]
    else:  # math_competition
        dataset_dir = PATHS["label_math_competition"]
        results_dir = PATHS["judge_math_competition"]
    
    input_file = f"{dataset_dir}/{args.response_generated_from_what_model}/{args.task_name}-{args.difficulty}.jsonl"
    
    # Use specified output model name or extract from model path
    output_model_name = args.output_model_name if args.output_model_name else args.model_path.split('/')[-1]
    output_dir = Path(f"{results_dir}/{args.response_generated_from_what_model}/{output_model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.task_name}-{args.difficulty}.jsonl"

    if not os.path.exists(input_file):
        print(f"Warning: Input file not found: {input_file}, skipping...")
        return None

    # Read evaluation data
    all_evaluations = []
    with jsonlines.open(input_file, mode='r') as rf:
        for line in rf:
            evaluations = line["evaluations"]
            for eval_item in evaluations:
                if eval_item.get("label"):  # Only process data with labels
                    all_evaluations.append(eval_item)

    print(f'Total evaluations to process: {len(all_evaluations)}')

    if not all_evaluations:
        print("No data to process, exiting.")
        # Create an empty output file to indicate task "completion"
        with jsonlines.open(output_file, mode='w') as wf:
            pass 
        return output_file

    # Batch process data
    batch_inputs = batch_data(all_evaluations, args.batch_size)
    results = []

    # Process each batch
    for batch in tqdm(batch_inputs):
        # Pass pre-initialized objects instead of reloading each time
        batch_results = judge_segment_effectiveness(batch, llm, tokenizer, sampling_params)
        
        # Merge original data and judgment results
        for item, result in zip(batch, batch_results):
            evaluation = {
                "segment_a": item["segment_a"],
                "segment_b": item["segment_b"],
                "label": item["label"],
                "is_effective": result["is_effective"],
                "confidence": result["confidence"]
            }
            results.append(evaluation)

    # Save results
    with jsonlines.open(output_file, mode='w') as wf:
        for result in results:
            wf.write(result)

    print(f"Results saved to {output_file}")
    return output_file


def process_graph_judge(args, llm, tokenizer, sampling_params):
    """Process graph-related judgment tasks"""
    print("Processing graph judge...")
    return process_judge_data(args, llm, tokenizer, sampling_params, "graph")


def process_math_judge(args, llm, tokenizer, sampling_params):
    """Process math-related judgment tasks"""
    print("Processing math judge...")
    return process_judge_data(args, llm, tokenizer, sampling_params, "math")


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
    parser.add_argument("--output_model_name", type=str, default=None,
                      help="Name of the model used for judging (for output path). If not specified, will use the model path's last component.")
    
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
        process_graph_judge(args, llm, tokenizer, sampling_params)
    elif args.task_type == "math":
        process_math_judge(args, llm, tokenizer, sampling_params)
    elif args.task_type == "math_competition":
        process_judge_data(args, llm, tokenizer, sampling_params, "math_competition")
    elif args.task_type == "both":
        # Process graph tasks
        if args.task_name in GRAPH_TASKS:
            process_graph_judge(args, llm, tokenizer, sampling_params)
        
        # Process math tasks  
        if args.task_name in MATH_TASKS:
            process_math_judge(args, llm, tokenizer, sampling_params)


if __name__ == "__main__":
    main()

