#!/bin/bash
# Test script for unified rollout with debugger functionality

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="experiments/unified_debug_${TIMESTAMP}"

# Model server endpoints
QWEN3_8B_URL="${QWEN3_8B_URL:-http://129.212.187.116:8001}"
QWEN72B_URL="${QWEN72B_URL:-http://129.212.188.142:8001}"

# Export empty API key for local vLLM servers
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Configuration
#ROLLOUT_MODEL="${ROLLOUT_MODEL:-qwen3-8b}"
ROLLOUT_URL="${ROLLOUT_URL:-${QWEN3_8B_URL}/v1}"
# DEBUGGER_MODEL="${DEBUGGER_MODEL:-qwen2.5-72b-instruct}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-gpt-4o-mini}"
DEBUGGER_MODEL="${DEBUGGER_MODEL:-gpt-4.1}"
# DEBUGGER_URL="${DEBUGGER_URL:-${QWEN72B_URL}/v1}"

echo "=== Configuration ==="
echo "Rollout: model=${ROLLOUT_MODEL}, url=${ROLLOUT_URL}"
echo "Debugger: model=${DEBUGGER_MODEL}, url=${DEBUGGER_URL}"
echo ""

# Test 1: AlfWorld with debugger
echo "=== Test 1: AlfWorld with Debugger ==="
RUN_DIR="${BASE_DIR}/alfworld"
echo "Run directory: ${RUN_DIR}"
# python scripts/rollout/openmanus_rollout_debugger.py \
#     --env alfworld \
#     --total_envs 10 \
#     --test_times 1 \
#     --max_steps 30 \
#     --history_length 40 \
#     --model "${ROLLOUT_MODEL}" \
#     --temperature 0.0 \
#     --enable_debugger \
#     --max_try 5 \
#     --debugger_model "${DEBUGGER_MODEL}" \
#     --debugger_type continue \
#     --debugger_temperature 0.0 \
#     --experiment_dir "${RUN_DIR}" \
#     --save_all_attempts \
#     --save_per_task_trajectories \
#     --unique_envs \
#     --concurrency 10 \
#     --llm_concurrency 20
# # --base_url "${ROLLOUT_URL}" \
# #    --debugger_base_url "${DEBUGGER_URL}" \

python scripts/rollout/openmanus_rollout_debugger.py \
    --env alfworld \
    --total_envs 10 \
    --test_times 1 \
    --max_steps 5 \
    --history_length 40 \
    --model "${ROLLOUT_MODEL}" \
    --temperature 1.2 \
    --strategy tot \
    --max_try 3 \
    --beam_size 3 \
    --value_threshold 0.2 \
    --experiment_dir "${RUN_DIR}" \
    --save_all_attempts \
    --save_per_task_trajectories \
    --unique_envs \
    --concurrency 10 \
    --llm_concurrency 20
# --base_url "${ROLLOUT_URL}" \
#    --debugger_base_url "${DEBUGGER_URL}" \




# # Test 2: GAIA with debugger
# echo "Testing GAIA with debugger..."
# python scripts/rollout/openmanus_rollout_debugger.py \
#     --env gaia \
#     --batch_size 2 \
#     --total_envs 4 \
#     --test_times 1 \
#     --max_steps 20 \
#     --model gpt-4o-mini \
#     --gaia_data_path data/gaia/val.json \
#     --gaia_tools google_search wikipedia_knowledge_searcher python_code_generator \
#     --enable_debugger \
#     --max_debug_retry 3 \
#     --debugger_model gpt-4o \
#     --debug_output_dir logs/gaia/debug_analysis \
#     --save_all_attempts \
#     --dump_path logs/gaia/trajectories_debug.jsonl

# # Test 3: WebShop with debugger
# echo "Testing WebShop with debugger..."
# python scripts/rollout/openmanus_rollout_debugger.py \
#     --env webshop \
#     --batch_size 2 \
#     --total_envs 4 \
#     --test_times 1 \
#     --max_steps 25 \
#     --model gpt-4o-mini \
#     --webshop_train \
#     --enable_debugger \
#     --max_debug_retry 4 \
#     --debugger_model gpt-4o \
#     --debug_output_dir logs/webshop/debug_analysis \
#     --save_all_attempts \
#     --dump_path logs/webshop/trajectories_debug.jsonl

# # Test 4: Without debugger (baseline)
# echo "Testing without debugger (baseline)..."
# python scripts/rollout/openmanus_rollout_debugger.py \
#     --env alfworld \
#     --batch_size 2 \
#     --total_envs 4 \
#     --test_times 1 \
#     --max_steps 30 \
#     --model gpt-4o-mini \
#     --dump_path logs/alfworld/trajectories_baseline.jsonl

echo "All tests completed. Check experiments/ directory for results."
