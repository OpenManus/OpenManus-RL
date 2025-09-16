#!/bin/bash

# Script to run AlfWorld rollout evaluation
# This script runs the evaluation directly without Docker

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs_${TIMESTAMP}"
RESULTS_DIR="results_${TIMESTAMP}"

echo "Starting AlfWorld rollout evaluation"
echo "Timestamp: ${TIMESTAMP}"
echo "Results will be saved to: ${RESULTS_DIR}"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"

# Activate virtual environment if it exists
if [ -f "/opt/openmanus-venv/bin/activate" ]; then
    source /opt/openmanus-venv/bin/activate
fi

# Function to run a model
run_model() {
    local model_name=$1
    local base_url=$2
    local display_name=$3
    local safe_name=$(echo "$display_name" | tr ' /' '_')
    
    echo "Starting ${display_name}..."
    
    # Build command
    local cmd="python scripts/rollout/openmanus_rollout.py \
        --env alfworld \
        --unique_envs \
        --batch_size 10 \
        --concurrency 10 \
        --total_envs 200 \
        --history_length 30 \
        --model '${model_name}' \
        --chat_root '${RESULTS_DIR}/${safe_name}' \
        --dump_path '${RESULTS_DIR}/${safe_name}/trajectory.jsonl'"
    
    if [ ! -z "$base_url" ]; then
        cmd="${cmd} --base_url '${base_url}/v1'"
    fi
    
    # Run with logging
    echo "Command: ${cmd}" | tee -a "${LOG_DIR}/main.log"
    eval "${cmd}" 2>&1 | tee "${LOG_DIR}/${safe_name}.log"
    
    echo "Completed ${display_name}" | tee -a "${LOG_DIR}/main.log"
}

# Main execution
echo "Launching models..." | tee -a "${LOG_DIR}/main.log"

# OpenAI models (uncomment to use)
# run_model "gpt-4o" "" "GPT-4o"
# run_model "gpt-4o-mini" "" "GPT-4o-mini"

# vLLM models - run sequentially
run_model "qwen3-8b" "http://129.212.187.116:8001" "Qwen3-8B"
run_model "llama-3.1-8b-instruct" "http://129.212.176.75:8001" "Llama3.1-8B"
run_model "qwen2.5-7b-instruct" "http://134.199.196.219:8001" "Qwen2.5-7B"
run_model "qwen2.5-72b-instruct" "http://134.199.196.239:8001" "Qwen2.5-72B"
run_model "llama-3.3-70b-instruct" "http://129.212.178.4:8001" "Llama3.3-70B"

echo ""
echo "========================================="
echo "All models completed!"
echo "========================================="
echo ""
echo "Logs are in: ${LOG_DIR}/"
echo "Results are in: ${RESULTS_DIR}/"
echo ""
echo "To check results:"
echo "  ls -la ${RESULTS_DIR}/"
echo "To view logs:"
echo "  tail -f ${LOG_DIR}/*.log"