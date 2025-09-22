#!/usr/bin/env python3
"""
Unified Entropy Analysis Tool
Combines logits inference, entropy analysis, and wordcloud generation functionality.
"""

import os
import sys
import json
import glob
import pickle
from collections import defaultdict, Counter
import math
from datetime import datetime

# Optional imports for specific modes
try:
    import numpy as np
    from tqdm import tqdm
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
except ImportError:
    # These will be available in the runtime environment
    pass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_utils import (
    get_entropy_argument_parser, setup_entropy_logging, calculate_distribution_entropy,
    find_latest_results_file, load_entropy_analysis_results, generate_entropy_text_template,
    setup_model_and_tokenizer, create_sampling_params, batch_data, PATHS, ENTROPY_TASKS
)

def infer_logits_mode(args, logger):
    """Execute logits inference mode"""
    logger.info("Starting logits inference mode")
    
    # Validate required arguments for infer mode
    if not all([args.LLM, args.task, args.difficulty]):
        logger.error("LLM, task, and difficulty are required for infer mode")
        return False
    
    # Setup model and tokenizer
    logger.info(f"Loading model: {args.LLM}")
    llm, tokenizer = setup_model_and_tokenizer(args.LLM, args.gpu_num)
    
    # Create sampling parameters
    sampling_params = create_sampling_params(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n,
        min_p=args.min_p,
        logprobs=32,  # Enable logprobs for entropy analysis
        prompt_logprobs=0
    )
    
    # Load dataset
    data_file = os.path.join(PATHS["graph_dataset"], f"{args.task}_{args.difficulty}.pkl")
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return False
    
    logger.info(f"Loading dataset: {data_file}")
    with open(data_file, 'rb') as f:
        dataset = pickle.load(f)
    
    # Prepare output directory
    output_dir = PATHS["entropy_logits_output"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if output file already exists
    output_file = os.path.join(output_dir, f"{args.LLM}_{args.task}_{args.difficulty}.pkl")
    if os.path.exists(output_file):
        logger.warning(f"Output file already exists: {output_file}")
        return True
    
    # Process dataset
    logger.info(f"Processing {len(dataset)} samples")
    results = []
    
    # Create text templates for inference
    texts = []
    for item in dataset:
        problem_text = item.get('problem', '')
        answer = item.get('answer', '')
        template = generate_entropy_text_template(problem_text, answer, args.task)
        texts.append(template)
    
    # Batch processing
    batched_texts = batch_data(texts, args.batch_size)
    
    for batch_idx, batch in enumerate(tqdm(batched_texts, desc="Processing batches")):
        try:
            # Generate responses with logprobs
            outputs = llm.generate(batch, sampling_params)
            
            for i, output in enumerate(outputs):
                result = {
                    'input_text': batch[i],
                    'generated_text': output.outputs[0].text,
                    'predict_tokens': [token.token for token in output.outputs[0].logprobs],
                    'predict_token_logprobs': [
                        {str(k): {'log_prob': v.logprob, 'rank': v.rank} 
                         for k, v in token_logprobs.items()}
                        for token_logprobs in output.outputs[0].logprobs
                    ]
                }
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            continue
    
    # Save results
    logger.info(f"Saving results to: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Also save as JSONL for compatibility
    jsonl_file = output_file.replace('.pkl', '_logits.json')
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"Logits inference completed. Processed {len(results)} samples")
    return True

def analyze_entropy_mode(args, logger):
    """Execute entropy analysis mode"""
    logger.info("Starting entropy analysis mode")
    
    # Read vLLM data and calculate entropy
    data_path = PATHS["entropy_logits_output"]
    min_freq = args.min_freq
    top_k = args.top_k
    
    # Use defaultdict to simplify code
    token_stats = defaultdict(lambda: {'count': 0, 'sum_entropy': 0.0})
    
    logger.info("Reading vLLM data and calculating entropy...")
    json_files = glob.glob(os.path.join(data_path, '**', '*_logits.json'), recursive=True)
    logger.info(f"Found {len(json_files)} logits files")
    
    if not json_files:
        logger.error(f"No logits files found in {data_path}")
        return False
    
    for file_path in tqdm(json_files, desc="Processing Logits files"):
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    tokens_key = "predict_tokens"
                    logprobs_key = "predict_token_logprobs"

                    if tokens_key in obj and logprobs_key in obj and obj[tokens_key]:
                        tokens = obj[tokens_key]
                        all_logprobs = obj[logprobs_key]

                        if len(tokens) != len(all_logprobs):
                            continue

                        # Iterate through each generated token and its corresponding probability distribution
                        for actual_token, token_logprobs_dict in zip(tokens, all_logprobs):
                            if not actual_token or actual_token.isspace():
                                continue
                            
                            token_key = actual_token.strip().lower()
                            
                            if not token_key:
                                continue
                            
                            # 1. Calculate distribution entropy H_t at that moment
                            entropy_of_the_moment = calculate_distribution_entropy(token_logprobs_dict)
                            
                            # 2. Assign this H_t value to the processed token_key (spaces removed)
                            token_stats[token_key]['count'] += 1
                            token_stats[token_key]['sum_entropy'] += entropy_of_the_moment

                except (json.JSONDecodeError, KeyError):
                    continue

    # Filter and average
    logger.info("Filtering and calculating final average entropy...")
    if not token_stats:
        logger.error("Error: No tokens successfully parsed from data.")
        return False

    token_avg_entropy = {
        token: stats['sum_entropy'] / stats['count']
        for token, stats in token_stats.items()
        if stats['count'] >= min_freq
    }

    if not token_avg_entropy:
        logger.error(f"Error: No tokens with frequency above {min_freq}.")
        return False

    # Sort and filter
    logger.info("Sorting and filtering...")
    sorted_high = sorted(token_avg_entropy.items(), key=lambda x: -x[1])[:top_k]
    sorted_low = sorted(token_avg_entropy.items(), key=lambda x: x[1])[:top_k]

    # Debug information section (fully preserved)
    logger.info("\n=== Debug Information ===")
    key_words = ['wait','but','because','and','if']
    logger.info("Key word entropy analysis:")
    for word in key_words:
        if word in token_avg_entropy:
            entropy_val = token_avg_entropy[word]
            count = token_stats[word]['count']
            logger.info(f"  '{word}': entropy={entropy_val:.4f}, frequency={count}")
        else:
            logger.info(f"  '{word}': not found or frequency too low")

    logger.info(f"\nTotal tokens meeting frequency criteria: {len(token_avg_entropy)}")
    logger.info(f"High entropy top 100: {sorted_high[:100]}")
    logger.info("High entropy top 100 frequencies:")
    for i, (token, entropy) in enumerate(sorted_high[:100]):
        count = token_stats[token]['count']
        logger.info(f"  {i+1}. '{token}': entropy={entropy:.4f}, frequency={count}")
    logger.info(f"Low entropy top 100: {sorted_low[:100]}")
    logger.info("Low entropy top 100 frequencies:")
    for i, (token, entropy) in enumerate(sorted_low[:100]):
        count = token_stats[token]['count']
        logger.info(f"  {i+1}. '{token}': entropy={entropy:.4f}, frequency={count}")
    logger.info("==================\n")

    # Save calculation results
    output_dir = PATHS["entropy_token_analysis"]
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Saving calculation results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save complete token statistics
    full_results = {
        'metadata': {
            'timestamp': timestamp,
            'min_freq': min_freq,
            'top_k': top_k,
            'total_tokens_processed': len(token_stats),
            'tokens_meeting_freq_threshold': len(token_avg_entropy)
        },
        'token_stats': {
            token: {
                'count': stats['count'],
                'avg_entropy': token_avg_entropy.get(token, 0.0)
            }
            for token, stats in token_stats.items()
            if token in token_avg_entropy
        },
        'sorted_results': {
            'high_entropy_top_k': sorted_high,
            'low_entropy_top_k': sorted_low
        }
    }

    # Save to JSON file
    results_file = os.path.join(output_dir, f'entropy_analysis_results_{timestamp}.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Calculation completed! Output files:")
    logger.info(f"  Analysis results: {results_file}")
    logger.info("To generate word clouds, please run with --mode wordcloud")
    return True

def wordcloud_mode(args, logger):
    """Execute wordcloud generation mode"""
    logger.info("Starting wordcloud generation mode")
    
    # Find latest analysis results file
    results_dir = PATHS["entropy_token_analysis"]
    results_file = find_latest_results_file(results_dir)
    if not results_file:
        logger.error(f"No analysis results files found. Please run analyze mode first.")
        logger.error(f"Search path: {os.path.join(results_dir, 'entropy_analysis_results_*.json')}")
        return False
    
    # Load data
    logger.info(f"Found latest analysis results file: {results_file}")
    data = load_entropy_analysis_results(results_file)
    if not data:
        return False
    
    logger.info("Successfully loaded analysis results data")
    logger.info(f"  Timestamp: {data['metadata']['timestamp']}")
    logger.info(f"  Min frequency threshold: {data['metadata']['min_freq']}")
    logger.info(f"  Top-K: {data['metadata']['top_k']}")
    logger.info(f"  Tokens meeting frequency threshold: {data['metadata']['tokens_meeting_freq_threshold']}")
    
    # Ensure output directory exists
    output_dir = PATHS["entropy_token_analysis"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sorted results
    sorted_results = data['sorted_results']
    high_entropy_tokens = sorted_results['high_entropy_top_k']
    low_entropy_tokens = sorted_results['low_entropy_top_k']
    
    # Generate filenames (with timestamp)
    analysis_timestamp = data['metadata']['timestamp']
    high_entropy_path = os.path.join(output_dir, f'high_entropy_wordcloud_{analysis_timestamp}.png')
    low_entropy_path = os.path.join(output_dir, f'low_entropy_wordcloud_{analysis_timestamp}.png')
    
    # Generate word clouds
    success_count = 0
    
    if draw_wordcloud(high_entropy_tokens, 'High Entropy Tokens (Distribution)', high_entropy_path, logger):
        success_count += 1
    
    if draw_wordcloud(low_entropy_tokens, 'Low Entropy Tokens (Distribution)', low_entropy_path, logger):
        success_count += 1
    
    # Output summary
    logger.info(f"Word cloud generation completed!")
    logger.info(f"  Successfully generated {success_count}/2 word clouds")
    logger.info(f"  High entropy wordcloud: {high_entropy_path}")
    logger.info(f"  Low entropy wordcloud: {low_entropy_path}")
    return True

def draw_wordcloud(token_entropy_list, title, out_path, logger):
    """Draw word cloud"""
    if not token_entropy_list:
        logger.warning(f"Warning: Cannot generate wordcloud for '{title}' because filtered token list is empty.")
        return False
        
    logger.info(f"Generating wordcloud: {title}")
    logger.info(f"  Number of tokens included: {len(token_entropy_list)}")
    
    # Convert to frequency dictionary
    freq_dict = {token: entropy for token, entropy in token_entropy_list}
    
    # Create wordcloud object
    wc = WordCloud(
        width=1200, 
        height=600, 
        background_color='white', 
        colormap='viridis',
        max_words=100,
        relative_scaling=0.8,
        min_font_size=10
    )
    wc.generate_from_frequencies(freq_dict)
    
    # Draw and save
    plt.figure(figsize=(15, 7.5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=24, pad=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Wordcloud saved: {out_path}")
    return True

def main():
    """Main function"""
    parser = get_entropy_argument_parser()
    args = parser.parse_args()
    
    # Setup logging based on mode
    logger, log_file, timestamp = setup_entropy_logging(f"entropy_{args.mode}")
    
    logger.info(f"Starting entropy analysis tool in {args.mode} mode")
    logger.info(f"Arguments: {vars(args)}")
    
    success = False
    
    try:
        if args.mode == 'infer':
            success = infer_logits_mode(args, logger)
        elif args.mode == 'analyze':
            success = analyze_entropy_mode(args, logger)
        elif args.mode == 'wordcloud':
            success = wordcloud_mode(args, logger)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            
    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    if success:
        logger.info(f"Entropy analysis {args.mode} mode completed successfully")
    else:
        logger.error(f"Entropy analysis {args.mode} mode failed")
        
    logger.info(f"Log file: {log_file}")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
