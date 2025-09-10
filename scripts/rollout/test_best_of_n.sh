#!/bin/bash
# Test script for best-of-N sampling functionality

# Test 1: AlfWorld with best-of-3
echo "Testing AlfWorld with best-of-3 sampling..."
python scripts/rollout/openmanus_rollout.py \
    --env alfworld \
    --batch_size 2 \
    --total_envs 4 \
    --test_times 1 \
    --max_steps 30 \
    --model gpt-4o-mini \
    --best_of_n 3 \
    --best_of_n_temp 1.2 \
    --dump_path logs/alfworld/trajectories_best_of_3.jsonl \
    --chat_root logs/alfworld/chats_best_of_n

# Test 2: GAIA with best-of-5
echo "Testing GAIA with best-of-5 sampling..."
python scripts/rollout/openmanus_rollout.py \
    --env gaia \
    --batch_size 2 \
    --total_envs 4 \
    --test_times 1 \
    --max_steps 20 \
    --model gpt-4o-mini \
    --gaia_data_path data/gaia/val.json \
    --gaia_tools google_search wikipedia_knowledge_searcher python_code_generator \
    --best_of_n 5 \
    --best_of_n_temp 1.2 \
    --dump_path logs/gaia/trajectories_best_of_5.jsonl

# Test 3: WebShop with best-of-4
echo "Testing WebShop with best-of-4 sampling..."
python scripts/rollout/openmanus_rollout.py \
    --env webshop \
    --batch_size 2 \
    --total_envs 4 \
    --test_times 1 \
    --max_steps 25 \
    --model gpt-4o-mini \
    --webshop_train \
    --best_of_n 4 \
    --best_of_n_temp 1.2 \
    --dump_path logs/webshop/trajectories_best_of_4.jsonl

# Test 4: Compare with baseline (no best-of-N)
echo "Testing baseline (no best-of-N) for comparison..."
python scripts/rollout/openmanus_rollout.py \
    --env alfworld \
    --batch_size 2 \
    --total_envs 4 \
    --test_times 1 \
    --max_steps 30 \
    --model gpt-4o-mini \
    --temperature 0.4 \
    --dump_path logs/alfworld/trajectories_baseline_comparison.jsonl

# Test 5: Default temperature vs best-of-N temperature
echo "Testing default temperature (0.4) vs best-of-N temperature (1.2)..."
python scripts/rollout/openmanus_rollout.py \
    --env alfworld \
    --batch_size 2 \
    --total_envs 4 \
    --test_times 1 \
    --max_steps 30 \
    --model gpt-4o-mini \
    --temperature 1.2 \
    --best_of_n 1 \
    --dump_path logs/alfworld/trajectories_high_temp_single.jsonl

echo "All best-of-N tests completed. Check logs/ directory for results."
echo ""
echo "Expected improvements:"
echo "- Best-of-N should show higher success rates than baseline"
echo "- Higher temperature (1.2) should provide more diversity"
echo "- Early stopping should reduce computation time when all tasks succeed"
