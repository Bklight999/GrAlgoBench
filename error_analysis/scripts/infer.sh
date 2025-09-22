#!/bin/bash

# Set up logging
LOG_FILE="/hpc2hdd/home/mpeng885/qifanzhang/GLC-Benchmark/error_analysis/logs/reformated_response_$(date '+%Y%m%d_%H%M%S').log"
echo "===============================================" | tee -a "$LOG_FILE"
echo "Starting test runs at $(date)" | tee -a "$LOG_FILE"
echo "===============================================" | tee -a "$LOG_FILE"

# Define arrays for all possible values
task_names=("Diameter")
difficulties=("easy" "medium" "hard" "challenge")

# "Qwen3-32B" "Qwen3-32B-no-thinking" "Distill_Qwen_32B" "QWQ-32B" "Light-R1-32B" 
models=("Qwen3-8B" "Qwen3-8B-no-thinking")

# Calculate total combinations
total_combinations=$((${#task_names[@]} * ${#difficulties[@]} * ${#models[@]}))
echo "Total combinations to run: $total_combinations" | tee -a "$LOG_FILE"
echo "Using GPUs: 0,1,2,3" | tee -a "$LOG_FILE"
echo "===============================================" | tee -a "$LOG_FILE"

# Counter for successful runs
successful_runs=0
current_run=0

# Run all combinations
for task in "${task_names[@]}"; do
    for difficulty in "${difficulties[@]}"; do
        for model in "${models[@]}"; do
            current_run=$((current_run + 1))
            start_time=$(date +%s)
            
            echo "===============================================" | tee -a "$LOG_FILE"
            echo "Starting run $current_run/$total_combinations at $(date)" | tee -a "$LOG_FILE"
            echo "Parameters:" | tee -a "$LOG_FILE"
            echo "  Task: $task" | tee -a "$LOG_FILE"
            echo "  Difficulty: $difficulty" | tee -a "$LOG_FILE"
            echo "  Model: $model" | tee -a "$LOG_FILE"
            echo "  GPUs: 0,1,2,3" | tee -a "$LOG_FILE"
            
            # Run the test with CUDA_VISIBLE_DEVICES set to 0,1
            if CUDA_VISIBLE_DEVICES=0,1,4,5 python /hpc2hdd/home/mpeng885/qifanzhang/GLC-Benchmark/error_analysis/reformat.py \
                --task_name "$task" \
                --difficulty "$difficulty" \
                --response_generated_from_what_model "$model"; then
                end_time=$(date +%s)
                duration=$((end_time - start_time))
                echo "Test completed successfully for $task-$difficulty-$model" | tee -a "$LOG_FILE"
                echo "Duration: ${duration} seconds" | tee -a "$LOG_FILE"
                successful_runs=$((successful_runs + 1))
            else
                end_time=$(date +%s)
                duration=$((end_time - start_time))
                echo "Error running test for $task-$difficulty-$model" | tee -a "$LOG_FILE"
                echo "Duration: ${duration} seconds" | tee -a "$LOG_FILE"
            fi
            
            echo "===============================================" | tee -a "$LOG_FILE"
            
            # Add a small delay between runs
            sleep 2
        done
    done
done

# Log final statistics
echo "===============================================" | tee -a "$LOG_FILE"
echo "Test runs completed at $(date)" | tee -a "$LOG_FILE"
echo "Final Statistics:" | tee -a "$LOG_FILE"
echo "  Total combinations run: $total_combinations" | tee -a "$LOG_FILE"
echo "  Successful runs: $successful_runs" | tee -a "$LOG_FILE"
echo "  Failed runs: $((total_combinations - successful_runs))" | tee -a "$LOG_FILE"
echo "  Success rate: $(( (successful_runs * 100) / total_combinations ))%" | tee -a "$LOG_FILE"
echo "===============================================" | tee -a "$LOG_FILE" 