#!/bin/bash
# Test script for unified rollout with debugger functionality

# Test 1: AlfWorld with debugger
echo "Testing AlfWorld with debugger..."
python scripts/rollout/openmanus_rollout_debugger.py \
    --env alfworld \
    --batch_size 2 \
    --total_envs 4 \
    --test_times 1 \
    --max_steps 30 \
    --model gpt-4o-mini \
    --temperature 0.4 \
    --enable_debugger \
    --max_debug_retry 3 \
    --debugger_model gpt-4o \
    --debugger_temperature 0.3 \
    --debug_output_dir logs/alfworld/debug_analysis \
    --save_all_attempts \
    --dump_path logs/alfworld/trajectories_debug.jsonl \
    --chat_root logs/alfworld/chats

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

# Test 4: Without debugger (baseline)
echo "Testing without debugger (baseline)..."
python scripts/rollout/openmanus_rollout_debugger.py \
    --env alfworld \
    --batch_size 2 \
    --total_envs 4 \
    --test_times 1 \
    --max_steps 30 \
    --model gpt-4o-mini \
    --dump_path logs/alfworld/trajectories_baseline.jsonl

echo "All tests completed. Check logs/ directory for results."
