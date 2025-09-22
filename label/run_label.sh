#!/bin/bash

# Unified label script
# Usage: ./run_unified_label.sh [task_type] [task_name] [difficulty] [model_name] [label_model_path] [batch_size] [gpu_ids]

set -e

# Default parameters
TASK_TYPE=${1:-"graph"}
TASK_NAME=${2:-"MST"}  
DIFFICULTY=${3:-"easy"}
MODEL_NAME=${4:-"QWQ-32B"}
LABEL_MODEL_PATH=${5:-"models/Qwen/Qwen2___5-32B-Instruct"}
BATCH_SIZE=${6:-256}
GPU_IDS=${7:-"0,1"}

# Logging setup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/unified_label_${TASK_TYPE}_${TASK_NAME}_${DIFFICULTY}_${TIMESTAMP}.log"

echo "========================================" | tee -a $LOG_FILE
echo "Starting unified label test" | tee -a $LOG_FILE
echo "Task type: $TASK_TYPE" | tee -a $LOG_FILE
echo "Task name: $TASK_NAME" | tee -a $LOG_FILE
echo "Difficulty: $DIFFICULTY" | tee -a $LOG_FILE
echo "Source model name: $MODEL_NAME" | tee -a $LOG_FILE
echo "Label model path: $LABEL_MODEL_PATH" | tee -a $LOG_FILE
echo "Batch size: $BATCH_SIZE" | tee -a $LOG_FILE
echo "GPU IDs: $GPU_IDS" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "Start time: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

# Set GPU environment
export CUDA_VISIBLE_DEVICES=$GPU_IDS
echo "Setting CUDA_VISIBLE_DEVICES=$GPU_IDS" | tee -a $LOG_FILE

# Change to project root directory
cd "$(dirname "$(dirname "$0")")"

# Run label test
python label/label.py \
    --task_type $TASK_TYPE \
    --task_name $TASK_NAME \
    --difficulty $DIFFICULTY \
    --response_generated_from_what_model $MODEL_NAME \
    --model_path $LABEL_MODEL_PATH \
    --batch_size $BATCH_SIZE \
    --max_tokens 8000 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --temperature 0.0 \
    --top_p 1.0 \
    2>&1 | tee -a $LOG_FILE

echo "========================================" | tee -a $LOG_FILE
echo "Label test completed" | tee -a $LOG_FILE
echo "End time: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

# Run result analysis (if needed)
echo "Starting result analysis..." | tee -a $LOG_FILE
# Labeling result analysis script calls can be added here
# python label_analysis_${TASK_TYPE}.py --task_name $TASK_NAME --difficulty $DIFFICULTY --model_name $MODEL_NAME

echo "All tasks completed!" | tee -a $LOG_FILE

