#!/bin/bash

# Set log file path
LOG_DIR="/path/to/Inference/logs/$(date '+%Y%m%d')"
# LOG_FILE will be set per-task

# Create log directory
mkdir -p "$LOG_DIR"

# Log function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_entry="[$timestamp] [$level] $message"
    
    # Output to console
    echo "$log_entry"
    
    # Save to log file
    echo "$log_entry" >> "$LOG_FILE"
}

# Record script start
# log "INFO" "Log file path: $LOG_FILE"

# LLM list
LLMS=("Qwen2.5-32B" "Llama-3.3-70B" "gpt-oss-120b")
# Task list
TASKS=("MKC" "PathSum" "MCP" "Triangle" "DistanceK" "MaxDegree" "DistanceThreshold" "MST" "Diameter")
# Difficulty levels
DIFFICULTIES=("easy" "medium")

# Record start time
log "INFO" "Starting inference tasks..."
log "INFO" "Start time: $(date)"
log "INFO" "=================================="

# Counter
total_runs=0
completed_runs=0

# Calculate total number of runs
for llm in "${LLMS[@]}"; do
    for task in "${TASKS[@]}"; do
        for difficulty in "${DIFFICULTIES[@]}"; do
            ((total_runs++))
        done
    done
done

log "INFO" "Total tasks to run: $total_runs"
log "INFO" "=================================="

# Run in order: LLM -> task -> difficulty
for llm in "${LLMS[@]}"; do
    log "INFO" "Starting LLM: $llm"
    log "INFO" "----------------------------------"
    
    for task in "${TASKS[@]}"; do
        log "INFO" "  Starting Task: $task"
        log "INFO" "  ----------------------------------"
        
        for difficulty in "${DIFFICULTIES[@]}"; do
            LOG_FILE="${LOG_DIR}/infer_open_${llm}_${task}_gpu4567.log"
            ((completed_runs++))
            log "INFO" "    Log file path: $LOG_FILE"
            log "INFO" "    Running $completed_runs/$total_runs: LLM=$llm, Task=$task, Difficulty=$difficulty"
            
            # Check if data file exists
            data_file="/path/to/data_generation/dataset_0830/${task}_${difficulty}.pkl"
            if [ ! -f "$data_file" ]; then
                log "WARN" "    Warning: Data file $data_file does not exist, skipping this task"
                continue
            fi
            
            # Run inference
            log "INFO" "    Starting infer_open.py..."
            start_time=$(date +%s)
            
            cd ..
            CUDA_VISIBLE_DEVICES=0,1,2,3 python /path/to/Inference/infer_open.py \
                --LLM "$llm" \
                --task "$task" \
                --difficulty "$difficulty" \
                --batch_size 50 \
                --max_tokens 32768 \
                --gpu_num 4 \
                --end_index 50
            
            exit_code=$?
            cd ..
            
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            
            if [ $exit_code -eq 0 ]; then
                log "INFO" "    ✓ Completed: LLM=$llm, Task=$task, Difficulty=$difficulty (Duration: ${duration}s)"
            else
                log "ERROR" "    ✗ Failed: LLM=$llm, Task=$task, Difficulty=$difficulty (Exit code: $exit_code)"
            fi
            log "INFO" ""
        done
        log "INFO" "  Completed Task: $task"
        log "INFO" "  ----------------------------------"
    done
    log "INFO" "Completed LLM: $llm"
    log "INFO" "----------------------------------"
done

# Record end time
log "INFO" "=================================="
log "INFO" "All tasks completed!"
log "INFO" "End time: $(date)"
log "INFO" "Total tasks run: $completed_runs"