#!/bin/bash

# Entropy Analysis Script
# Supports three modes: infer, analyze, wordcloud
# Usage: ./run_unified_entropy_analysis.sh <mode> [GPU_IDS] [additional_args...]

set -e

# Default configuration
DEFAULT_GPU_IDS="6,7"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$SCRIPT_DIR/entropy_analysis.py"

# Logging setup
LOG_DIR="entropy_analysis/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/entropy_analysis_$(date '+%Y%m%d_%H%M%S').log"

# Function to print usage
print_usage() {
    echo "===============================================" | tee -a "$LOG_FILE"
    echo "Entropy Analysis Tool" | tee -a "$LOG_FILE"
    echo "===============================================" | tee -a "$LOG_FILE"
    echo "Usage: $0 <mode> [GPU_IDS] [additional_args...]" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Modes:" | tee -a "$LOG_FILE"
    echo "  infer     - Generate logits output from model inference" | tee -a "$LOG_FILE"
    echo "  analyze   - Analyze token entropy from logits files" | tee -a "$LOG_FILE"
    echo "  wordcloud - Generate word clouds from analysis results" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Parameters:" | tee -a "$LOG_FILE"
    echo "  GPU_IDS   - GPU device IDs (default: $DEFAULT_GPU_IDS)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Examples:" | tee -a "$LOG_FILE"
    echo "  # Infer logits with specific model and task" | tee -a "$LOG_FILE"
    echo "  $0 infer 0,1 --LLM Qwen3-32B --task MaxDegree --difficulty easy" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "  # Analyze entropy with custom parameters" | tee -a "$LOG_FILE"
    echo "  $0 analyze 0,1 --min_freq 50000 --top_k 200" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "  # Generate wordclouds" | tee -a "$LOG_FILE"
    echo "  $0 wordcloud 0,1" | tee -a "$LOG_FILE"
    echo "===============================================" | tee -a "$LOG_FILE"
}

# Function to log with timestamp
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Check if mode is provided
if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

MODE="$1"
shift

# Check if mode is valid
case "$MODE" in
    infer|analyze|wordcloud)
        ;;
    *)
        log_message "ERROR: Invalid mode '$MODE'"
        print_usage
        exit 1
        ;;
esac

# Parse GPU_IDS (optional second parameter)
if [[ $# -gt 0 && "$1" =~ ^[0-9,]+$ ]]; then
    GPU_IDS="$1"
    shift
else
    GPU_IDS="$DEFAULT_GPU_IDS"
fi

# Remaining arguments are passed to Python script
ADDITIONAL_ARGS="$@"

log_message "==============================================="
log_message "Starting Entropy Analysis"
log_message "==============================================="
log_message "Mode: $MODE"
log_message "GPU IDs: $GPU_IDS"
log_message "Additional arguments: $ADDITIONAL_ARGS"
log_message "Project root: $PROJECT_ROOT"
log_message "Python script: $PYTHON_SCRIPT"
log_message "Log file: $LOG_FILE"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    log_message "ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Set GPU environment variable
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
log_message "Set CUDA_VISIBLE_DEVICES=$GPU_IDS"

# Change to project root directory
cd "$PROJECT_ROOT"
log_message "Changed to project directory: $(pwd)"

# Execute Python script
log_message "Executing: python $PYTHON_SCRIPT --mode $MODE $ADDITIONAL_ARGS"
log_message "==============================================="

START_TIME=$(date +%s)

if python "$PYTHON_SCRIPT" --mode "$MODE" $ADDITIONAL_ARGS; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    log_message "==============================================="
    log_message "SUCCESS: Entropy analysis $MODE mode completed"
    log_message "Duration: ${DURATION} seconds"
    log_message "Log file: $LOG_FILE"
    log_message "==============================================="
    exit 0
else
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    log_message "==============================================="
    log_message "ERROR: Entropy analysis $MODE mode failed"
    log_message "Duration: ${DURATION} seconds"
    log_message "Log file: $LOG_FILE"
    log_message "==============================================="
    exit 1
fi
