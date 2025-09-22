#!/bin/bash
trap "kill 0" SIGINT

# Multiple LLM names can be placed here
llms=("gemini2.5_thinking")

tasks=("DistanceThreshold") # Task list
difficulties=("easy" "medium")  

for llm in "${llms[@]}"; do
    for task in "${tasks[@]}"; do
        for difficulty in "${difficulties[@]}"; do
            logfile="logs_${llm}/${task}_${difficulty}.log"
            mkdir -p "logs_${llm}"
            echo "Start running: LLM=$llm, Task=$task, Difficulty=$difficulty"
            python /path/to/Inference/infer_close.py \
                --llm "$llm" --difficulty "$difficulty" --task "$task" > "$logfile" 2>&1
            echo "Finished: LLM=$llm, Task=$task, Difficulty=$difficulty"
        done
    done
done

echo "All tasks completed"