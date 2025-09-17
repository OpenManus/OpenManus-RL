#!/bin/bash
# Test script for unified rollout with debugger functionality

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="experiments/unified_debug_${TIMESTAMP}"

# Check if API keys are set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set. Make sure to set it before running."
fi

# Test 1: AlfWorld with debugger (small scale for testing)
echo "=== Test 1: AlfWorld with Debugger ==="
RUN_DIR="${BASE_DIR}/alfworld"
echo "Run directory: ${RUN_DIR}"
python scripts/rollout/openmanus_rollout_debugger.py \
    --env alfworld \
    --total_envs 10 \
    --test_times 1 \
    --max_steps 30 \
    --history_length 40 \
    --model gpt-4o-mini \
    --temperature 0.0 \
    --enable_debugger \
    --max_try 5 \
    --debugger_model gpt-4.1 \
    --debugger_type advanced \
    --debugger_temperature 0.0 \
    --experiment_dir ${RUN_DIR} \
    --save_all_attempts \
    --save_per_task_trajectories \
    --unique_envs \
    --concurrency 10 \
    --llm_concurrency 20 


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
