"""
GraAlgoBench Common Utilities Module
Contains all shared utility functions and configurations
"""

import json
import os
import re
import math
import datetime
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import pickle

try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import jsonlines
    import numpy as np
except ImportError:
    # These imports are optional and will be available in the runtime environment
    pass


def split_response_by_wait(response: str) -> List[str]:
    """
    Split response text by "wait" keyword
    
    Args:
        response: Input response text
        
    Returns:
        List of split text segments
    """
    response = response.lower()
    parts = response.split("wait")
    result = []
    for i in range(len(parts)):
        result.append(parts[i])
    return result


def split_response_by_keywords(response: str) -> List[str]:
    """
    Split response text by multiple keywords (wait, but, so)
    
    Args:
        response: Input response text
        
    Returns:
        List of split text segments
    """
    response = response.lower()
    # Use regex to split by multiple keywords while preserving delimiters
    parts = re.split(r'(\bwait\b|\bbut\b|\bso\b)', response)
    # Filter empty strings and strip whitespace
    result = [part.strip() for part in parts if part.strip()]
    return result


def batch_data(data_list: List[Any], batch_size: int = 1) -> List[List[Any]]:
    """
    Batch data list for processing
    
    Args:
        data_list: Data list to be batched
        batch_size: Batch processing size
        
    Returns:
        List of batched data
    """
    n = math.ceil(len(data_list) / batch_size)
    batch_data = []
    
    for i in range(n-1):
        start = i * batch_size
        end = (i+1) * batch_size
        batch_data.append(data_list[start:end])

    # Handle the last batch
    last_start = (n-1) * batch_size
    last_end = len(data_list)
    if last_start < len(data_list):
        batch_data.append(data_list[last_start:last_end])
    
    return batch_data


def setup_model_and_tokenizer(model_path: str, tensor_parallel_size: int = 2, 
                             gpu_memory_utilization: float = 0.9, 
                             max_num_seqs: int = 256):
    """
    Initialize model and tokenizer
    
    Args:
        model_path: Path to the model
        tensor_parallel_size: Tensor parallel size
        gpu_memory_utilization: GPU memory utilization ratio
        max_num_seqs: Maximum number of sequences
        
    Returns:
        Initialized LLM and tokenizer
    """
    print(f"Initializing model: {model_path}")
    llm = LLM(
        model=model_path, 
        tensor_parallel_size=tensor_parallel_size, 
        gpu_memory_utilization=gpu_memory_utilization, 
        max_num_seqs=max_num_seqs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model initialization completed")
    return llm, tokenizer


def create_sampling_params(temperature: float = 0.0, top_p: float = 1.0, 
                          max_tokens: int = 8000):
    """
    Create sampling parameters
    
    Args:
        temperature: Temperature parameter
        top_p: Top-p parameter
        max_tokens: Maximum number of tokens
        
    Returns:
        SamplingParams object
    """
    return SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)


def load_json_data(file_path: str) -> List[Dict]:
    """
    Load JSON format data file
    
    Args:
        file_path: File path
        
    Returns:
        List of data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    datas = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            datas.append(json.loads(line))
    return datas


def load_pickle_data(file_path: str) -> Any:
    """
    Load pickle format data file
    
    Args:
        file_path: File path
        
    Returns:
        Loaded data
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_math_dataset(task_name: str, dataset_type: str = "test") -> List[Dict]:
    """
    Load math dataset
    
    Args:
        task_name: Task name
        dataset_type: Dataset type (test/train etc.)
        
    Returns:
        Dataset list
    """
    if task_name == 'OlympiadBench':
        file_path = f"math_dataset/OlympiadBench/data/OE_TO_maths_en_COMP.json"
    else:
        file_path = f"math_dataset/{task_name}/data/{dataset_type}.json"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_math_competition_dataset(task_name: str) -> List[Dict]:
    """
    Load math competition dataset
    
    Args:
        task_name: Competition task name (USAMO, AIME, etc.)
        
    Returns:
        Dataset list
    """
    file_path = f"math_dataset/{task_name}/data/competition.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_math_problem_info(qa: Dict, task_name: str) -> Tuple[str, str]:
    """
    Extract question and answer from math problem data
    
    Args:
        qa: Question-answer data
        task_name: Task name
        
    Returns:
        (question text, answer)
    """
    if task_name == 'ASDiv':
        return qa["text"], qa["label"]
    elif task_name == 'GSM-Hard':
        return qa["input"], qa["target"]
    elif task_name == 'SVAMP':
        return qa["Body"] + "\n" + qa["Question"], qa["Answer"]
    elif task_name == 'Math500':
        return qa["problem"], qa["answer"]
    elif task_name == 'GSM8k':
        return qa["question"], qa["answer"]
    elif task_name == 'SAT':
        return qa["query"], qa["gold"]
    elif task_name == 'STEM':
        return qa["question"], qa["answer"]
    elif task_name == 'OlympiadBench':
        return qa["question"], qa["final_answer"]
    else:
        raise ValueError(f"Task name {task_name} not supported")


def extract_math_competition_problem_info(qa: Dict, task_name: str) -> Tuple[str, str]:
    """
    Extract question and answer from math competition data
    
    Args:
        qa: Question-answer data
        task_name: Competition task name
        
    Returns:
        (question text, answer)
    """
    # Math competition datasets typically use consistent structure
    if "problem" in qa and "answer" in qa:
        return qa["problem"], qa["answer"]
    elif "question" in qa and "answer" in qa:
        return qa["question"], qa["answer"]
    elif "text" in qa and "label" in qa:
        return qa["text"], qa["label"]
    else:
        # Fallback - try to find reasonable fields
        question_fields = ["problem", "question", "text", "statement"]
        answer_fields = ["answer", "solution", "label", "target"]
        
        question = None
        answer = None
        
        for field in question_fields:
            if field in qa:
                question = qa[field]
                break
                
        for field in answer_fields:
            if field in qa:
                answer = qa[field]
                break
                
        if question and answer:
            return question, answer
        else:
            raise ValueError(f"Could not extract question/answer from math competition data for task {task_name}")


def extract_graph_problem_info(qa: Dict, task_name: str) -> Tuple[str, str]:
    """
    Extract question and answer from graph problem data
    
    Args:
        qa: Question-answer data
        task_name: Task name
        
    Returns:
        (question text, answer)
    """
    question = qa["problem_text"]
    if task_name == 'LCA':
        id = qa["exact_answer"]
        g = qa["graph"]
        gt = g.nodes[id]['name']
    else:  
        gt = qa["exact_answer"]
    return question, gt


def generate_judgment_prompt(segment_a: str, segment_b: str, label: str) -> str:
    """
    Generate prompt for judging segment effectiveness
    
    Args:
        segment_a: First text segment
        segment_b: Second text segment
        label: Label
        
    Returns:
        Generated prompt
    """
    return f"""
Given two text segments, A and B, where B is labeled as "{label}", determine if B is an effective addition to A. 

For a segment to be effective:
- If labeled as "Self-reflection": B should effectively evaluate or verify the correctness of A
- If labeled as "Strategy shift": B should effectively alter or adjust the strategy presented in A
- If labeled as "None": B has no clear functional relationship to A, or serves a different purpose not covered by the above categories.

Please analyze the relationship between A and B and determine:
1. Is B an effective addition to A given its label? (Yes/No)
2. What is your confidence in this judgment? (High/Medium/Low)

Output your judgment in the following format:
Effectiveness: [Yes/No]
Confidence: [High/Medium/Low]

Segment A: {segment_a}

Segment B: {segment_b}
"""


def parse_model_response(response: str) -> Dict[str, Any]:
    """
    Parse model output and extract judgment results
    
    Args:
        response: Model response
        
    Returns:
        Parsed result dictionary
    """
    response = response.strip().lower()
    result = {
        "is_effective": False,
        "confidence": None
    }
    
    # Parse effectiveness
    if "effectiveness: yes" in response:
        result["is_effective"] = True
    
    # Parse confidence
    if "confidence: high" in response:
        result["confidence"] = "high"
    elif "confidence: medium" in response:
        result["confidence"] = "medium"
    elif "confidence: low" in response:
        result["confidence"] = "low"
    
    return result


def setup_output_directory(base_dir: str, model_name: str) -> Path:
    """
    Setup output directory
    
    Args:
        base_dir: Base directory
        model_name: Model name
        
    Returns:
        Output directory path
    """
    output_dir = Path(f"{base_dir}/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_statistics(stats: Dict, file_path: str):
    """
    Save statistics to file
    
    Args:
        stats: Statistics dictionary
        file_path: Save path
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def print_statistics(stats: Dict, total_results: int):
    """
    Print statistics information
    
    Args:
        stats: Statistics dictionary
        total_results: Total number of results
    """
    print("\n=== Processing Statistics ===")
    print(f"Total problems: {stats['total_problems']}")
    print(f"Total segment pairs: {stats['total_segment_pairs']}")
    if total_results > 0:
        print(f"Self-reflection: {stats['self_reflection_count']} ({stats['self_reflection_count']/total_results*100:.1f}%)")
        print(f"Strategy shift: {stats['strategy_shift_count']} ({stats['strategy_shift_count']/total_results*100:.1f}%)")
        print(f"Other: {stats['other_count']} ({stats['other_count']/total_results*100:.1f}%)")


def get_common_argument_parser() -> argparse.ArgumentParser:
    """
    Get common argument parser
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, 
                       default="models/Qwen/Qwen2___5-32B-Instruct", 
                       help="loading paths of LLM")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    parser.add_argument("--max_tokens", type=int, default=8000, help="Max tokens for inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for sampling")
    return parser


# Dataset path configurations
PATHS = {
    "graph_inference": "Inference/final_results_32B",
    "math_inference": "Inference/final_results_math",
    "math_competition_inference": "Inference/final_results_math_competition",
    "graph_dataset": "data_generation/dataset",
    "math_dataset": "math_dataset",
    "overthinking_graph": "overthinking/final_results_32B",
    "overthinking_math": "overthinking/final_results_math",
    "overthinking_math_competition": "overthinking/final_results_math_competition",
    "label_graph": "label/results",
    "label_math": "label/results_math",
    "label_math_competition": "label/results_math_competition",
    "judge_graph": "judge/results_segment_judge",
    "judge_math": "judge/results_segment_judge_math",
    "judge_math_competition": "judge/results_segment_judge_math_competition",
    "entropy_logits_output": "entropy_analysis/logits_output",
    "entropy_token_analysis": "entropy_analysis/token_analysis",
    "entropy_logs": "entropy_analysis/logs"
}

# Supported tasks and model configurations
GRAPH_TASKS = ["LCA", "DistanceThreshold", "MaximumFlow", "Triangle", "MCP", 
               "DistanceK", "MKC", "MST", "Diameter", "MaxDegree", "PathSum"]

MATH_TASKS = ["ASDiv", "GSM-Hard", "Math500", "GSM8k", "SAT", "STEM", "SVAMP", "OlympiadBench"]

MATH_COMPETITION_TASKS = ["AIME2025-I", "AIME2025-II", "AIMO_AMC", "AIMO_AIME"]

DIFFICULTIES = ["easy", "medium", "hard", "challenge"]

MODELS = ["QWQ-32B", "Qwen3-32B", "Distill_Qwen_32B"]

# Entropy analysis specific tasks
ENTROPY_TASKS = ["MaxDegree", "Diameter"]

def get_entropy_argument_parser():
    """Get argument parser for entropy analysis module"""
    parser = argparse.ArgumentParser(description="Entropy Analysis Tool")
    parser.add_argument('--mode', type=str, choices=['infer', 'analyze', 'wordcloud'], 
                       required=True, help='Operation mode')
    parser.add_argument('--LLM', type=str, help='Model name (required for infer mode)')
    parser.add_argument('--task', type=str, help='Task name (required for infer mode)')
    parser.add_argument('--difficulty', type=str, help='Difficulty level (required for infer mode)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_tokens', type=int, default=32768, help='Maximum tokens')
    parser.add_argument('--gpu_num', type=int, default=2, help='Number of GPUs')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling')
    parser.add_argument('--n', type=int, default=1, help='Number of completions')
    parser.add_argument('--min_p', type=float, default=0.0, help='Min-p sampling')
    parser.add_argument('--enable_thinking', type=str, default='false', help='Enable thinking mode')
    parser.add_argument('--min_freq', type=int, default=40000, help='Minimum frequency threshold')
    parser.add_argument('--top_k', type=int, default=100, help='Top-K tokens to analyze')
    return parser

def setup_entropy_logging(log_type="entropy_analysis"):
    """Setup logging for entropy analysis module"""
    log_dir = PATHS["entropy_logs"]
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'{log_type}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger, log_file, timestamp

def calculate_distribution_entropy(logprobs_dict):
    """
    Calculate entropy from vLLM sparse logprobs dictionary.
    This approximates the true full vocabulary entropy.
    """
    import numpy as np
    
    if not logprobs_dict or len(logprobs_dict) < 2:
        return 0.0
    
    log_probs = [data['log_prob'] for data in logprobs_dict.values()]
    log_probs_array = np.array(log_probs, dtype=np.float64)
    probs = np.exp(log_probs_array)
    
    # Normalize Top-K probability distribution to sum to 1
    observed_mass = probs.sum()
    if observed_mass > 0:
        probs = probs / observed_mass
    else:
        return 0.0
    
    # Calculate Shannon entropy (in bits)
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return entropy

def find_latest_results_file(results_dir, pattern_prefix="entropy_analysis_results"):
    """Find the latest analysis results file"""
    import glob
    pattern = os.path.join(results_dir, f'{pattern_prefix}_*.json')
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Sort by modification time, return latest
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def load_entropy_analysis_results(file_path):
    """Load entropy analysis results data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Failed to load analysis results: {e}")
        return None

def generate_entropy_text_template(problem_text, answer, task):
    """Generate text template for entropy analysis"""
    template = f"""Please solve the following {task} problem step by step:

Problem: {problem_text}

Please provide a detailed solution with clear reasoning steps.

Answer: {answer}"""
    return template

