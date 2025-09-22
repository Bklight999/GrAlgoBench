#!/bin/bash

MODEL_LIST=("QWQ-32B")
TASK_LIST=("LCA" ) #"Triangle" "LCA" "DistanceK" "MKC"
DIFFICULTY_LIST=("easy" "medium" "hard" "challenge")
LOG_DIR="/path/to/logs"
mkdir -p $LOG_DIR


for model in "${MODEL_LIST[@]}"; do
  for task in "${TASK_LIST[@]}"; do 
    for diff in "${DIFFICULTY_LIST[@]}"; do
      echo "Running for model: $model, task: $task, difficulty: $diff"
      log_file="$LOG_DIR/evaluate_ratio_${model}_${task}_${diff}.log"
    echo "Logging to: $log_file"
    CUDA_VISIBLE_DEVICES=4,5,6,7 python /path/to/evaluate_algorithm_ratio.py \
      --response_generated_from_what_model "$model" \
      --difficulty "$diff" \
      --gpu_num 4 \
      --task_name "$task" \
      > "$log_file" 2>&1
    done  
  done
done

