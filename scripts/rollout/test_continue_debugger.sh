#!/bin/bash
# Test script for continue debugger functionality

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="experiments/continue_debug_test_${TIMESTAMP}"

# Model server endpoints
QWEN3_8B_URL="${QWEN3_8B_URL:-http://129.212.187.116:8001}"

# Export empty API key for local vLLM servers
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Configuration
ROLLOUT_MODEL="${ROLLOUT_MODEL:-qwen3-8b}"
ROLLOUT_URL="${ROLLOUT_URL:-${QWEN3_8B_URL}/v1}"
DEBUGGER_MODEL="${DEBUGGER_MODEL:-gpt-4.1}"

echo "=== Configuration ==="
echo "Rollout: model=${ROLLOUT_MODEL}, url=${ROLLOUT_URL}"
echo "Debugger: model=${DEBUGGER_MODEL}, url=${DEBUGGER_URL}"
echo ""

# Test: AlfWorld with continue debugger
echo "=== Test: AlfWorld with Continue Debugger ==="
RUN_DIR="${BASE_DIR}/alfworld"
echo "Run directory: ${RUN_DIR}"
python scripts/rollout/openmanus_rollout_debugger.py \
    --env alfworld \
    --total_envs 3 \
    --test_times 1 \
    --max_steps 30 \
    --history_length 40 \
    --model "${ROLLOUT_MODEL}" \
    --base_url "${ROLLOUT_URL}" \
    --temperature 0.0 \
    --enable_debugger \
    --max_try 3 \
    --debugger_model "${DEBUGGER_MODEL}" \
    --debugger_type continue \
    --debugger_temperature 0.0 \
    --experiment_dir "${RUN_DIR}" \
    --save_all_attempts \
    --save_per_task_trajectories \
    --unique_envs \
    --concurrency 2

echo ""
echo "=== Test completed! ==="
echo "Results saved to: ${RUN_DIR}"
echo "Check the debug analysis files to see the continue debugger in action:"
echo "  - Individual attempt trajectories"
echo "  - Debug analysis with follow-up instructions"
echo "  - Task summaries with cumulative guidance"
