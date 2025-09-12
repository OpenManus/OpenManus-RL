#!/bin/bash
# Test script for AlfWorld debugger with retry logic

# Basic test without debugger (original behavior)
echo "Running basic test without debugger..."
python scripts/rollout/alfworld_debugger.py \
    --env_name alfworld \
    --batch_size 4 \
    --total_envs 4 \
    --test_times 1 \
    --max_steps 30 \
    --model gpt-4o-mini \
    --temperature 0.0 \
    --dump_path logs/test_no_debugger.jsonl

# Test with debugger ebled
echo "Running test with debugger enabled..."
python scripts/rollout/alfworld_debugger.py \
    --env_name alfworld \
    --batch_size 2 \
    --total_envs 2 \
    --test_times 1 \
    --max_steps 30 \
    --model gpt-4o-mini \
    --temperature 0.4 \
    --enable_debugger \
    --max_retries 3 \
    --debugger_model gpt-4o \
    --debugger_temperature 0.3 \
    --debug_output_dir logs/debug_analysis \
    --dump_path logs/test_with_debugger.jsonl \
    --chat_root logs/chat_histories

# # Test with vLLM backend and debugger
# echo "Running test with vLLM backend and debugger..."
# python scripts/rollout/alfworld_debugger.py \
#     --env_name alfworld \
#     --batch_size 2 \
#     --total_envs 2 \
#     --test_times 1 \
#     --max_steps 30 \
#     --model gpt-4o-mini \
#     --temperature 0.4 \
#     --base_url http://127.0.0.1:8000/v1 \
#     --enable_debugger \
#     --max_retries 5 \
#     --debugger_model gpt-4o \
#     --debug_output_dir logs/debug_vllm \
#     --dump_path logs/test_vllm_debugger.jsonl

# echo "Tests completed. Check logs/ directory for results."
