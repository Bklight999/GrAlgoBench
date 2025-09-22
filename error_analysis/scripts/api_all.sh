#!/bin/bash

# Set up logging
LOG_FILE="/path/to/error_analysis/logs/error_analysis_$(date '+%Y%m%d_%H%M%S').log"
echo "===============================================" | tee -a "$LOG_FILE"
echo "Starting test runs at $(date)" | tee -a "$LOG_FILE"
echo "===============================================" | tee -a "$LOG_FILE"

# Define arrays for all possible values
task_names=("Diameter" "MST") #"MKC" "DistanceThreshold" "PathSum" "DistanceK" "Diameter" "MaxDegree" "MCP" "Triangle"
llm="o1m"

# "Qwen3-32B" "Qwen3-32B-no-thinking" "Distill_Qwen_32B" "QWQ-32B" "Light-R1-32B" 
models=("Qwen3-8B" "Qwen3-8B-no-thinking")

# Calculate total combinations
total_combinations=$((${#task_names[@]} * ${#models[@]}))
echo "Total combinations to run: $total_combinations" | tee -a "$LOG_FILE"
echo "===============================================" | tee -a "$LOG_FILE"

# Counter for successful runs
successful_runs=0
current_run=0

# Run all combinations
for task in "${task_names[@]}"; do
    for model in "${models[@]}"; do
        current_run=$((current_run + 1))
        start_time=$(date +%s)
        
        echo "===============================================" | tee -a "$LOG_FILE"
        echo "Starting run $current_run/$total_combinations at $(date)" | tee -a "$LOG_FILE"
        echo "Parameters:" | tee -a "$LOG_FILE"
        echo "  Task: $task" | tee -a "$LOG_FILE"
        echo "  Model: $model" | tee -a "$LOG_FILE"
        
        # Run the test with CUDA_VISIBLE_DEVICES set to 0,1
        python /path/to/error_analysis/error_analysis.py \
            --task "$task" \
            --llm "$llm" \
            --response_generated_from_what_model "$model" \
            --resume 0 \
            --st 0 \
            --num_workers 64
        
        # Check if the first run was successful
        if [ $? -eq 0 ]; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            echo "Test completed successfully for $task-$model" | tee -a "$LOG_FILE"
            echo "Duration: ${duration} seconds" | tee -a "$LOG_FILE"
            successful_runs=$((successful_runs + 1))
        else
            echo "First run failed for $task-$model, attempting resume..." | tee -a "$LOG_FILE"
            
            # Initialize retry counter
            retry_count=0
            max_retries=3
            error_knt=1
            
            # Retry with resume flag until error_knt becomes 0
            while [ $error_knt -ne 0 ] && [ $retry_count -lt $max_retries ]; do
                retry_count=$((retry_count + 1))
                echo "Retry attempt $retry_count for $task-$model" | tee -a "$LOG_FILE"
                
                # Capture the output to get error_knt value
                output=$(python /path/to/error_analysis/error_analysis.py \
                    --task "$task" \
                    --llm "$llm" \
                    --response_generated_from_what_model "$model" \
                    --resume 1 \
                    --st 0 \
                    --num_workers 64 2>&1)
                
                exit_code=$?
                
                # Extract error_knt from the output (assuming it's printed in the format "error_knt: X")
                error_knt=$(echo "$output" | grep -o "error_knt:[[:space:]]*[0-9]*" | grep -o "[0-9]*" | tail -1)
                
                # If error_knt is not found in output, set it to 0 to exit the loop
                if [ -z "$error_knt" ]; then
                    error_knt=0
                fi
                
                if [ $exit_code -eq 0 ] && [ $error_knt -eq 0 ]; then
                    end_time=$(date +%s)
                    duration=$((end_time - start_time))
                    echo "Test completed successfully on retry for $task-$model" | tee -a "$LOG_FILE"
                    echo "Duration: ${duration} seconds" | tee -a "$LOG_FILE"
                    successful_runs=$((successful_runs + 1))
                    break
                else
                    echo "Retry $retry_count failed for $task-$model (error_knt: $error_knt)" | tee -a "$LOG_FILE"
                fi
            done
            
            if [ $error_knt -ne 0 ]; then
                echo "All retry attempts failed for $task-$model" | tee -a "$LOG_FILE"
            fi
        fi
        
        echo "===============================================" | tee -a "$LOG_FILE"
        
        # Add a small delay between runs
        sleep 10
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