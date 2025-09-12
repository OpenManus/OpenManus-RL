#!/bin/bash

# Unified debugger baseline testing script
# Contains running commands for multiple strategies

echo "Starting unified debugger baseline tests..."

# Debugger strategy
# echo "Running Debugger strategy..."
# python scripts/rollout/openmanus_rollout_debugger.py \
#   --env gaia \
#   --strategy debugger \
#   --enable_debugger \
#   --max_debug_retry 3 \
#   --gaia_data_path data/gaia/val.json \
#   --model gpt-4o-mini \
#   --temperature 0.4 \
#   --total_envs 1 \
#   --max_steps 3 \
#   --experiment_dir experiments/debugger_smoke

# # Best-of-N strategy
# echo "Running Best-of-N strategy..."
# python scripts/rollout/openmanus_rollout_debugger.py \
#   --env gaia \
#   --strategy bon \
#   --bon_n 5 \
#   --gaia_data_path data/gaia/val.json \
#   --model gpt-4o-mini \
#   --temperature 0.7 \
#   --total_envs 1 \
#   --max_steps 3 \
#   --experiment_dir experiments/bon_smoke

# # ToT-DFS strategy
# echo "Running ToT-DFS strategy..."
# python scripts/rollout/openmanus_rollout_debugger.py \
#   --env webshop \
#   --strategy tot \
#   --beam_size 4 \
#   --value_threshold 0.2 \
#   --max_nodes 100 \
#   --model gpt-4o-mini \
#   --temperature 0.4 \
#   --total_envs 1 \
#   --max_steps 10 \
#   --experiment_dir experiments/tot_smoke

# DFSDT strategy
echo "Running DFSDT strategy..."
python scripts/rollout/openmanus_rollout_debugger.py \
  --env alfworld \
  --strategy dfsdt \
  --diversity_back_steps 2 \
  --diversity_back_steps_alt 3 \
  --propose_k 4 \
  --beam_size 3 \
  --max_nodes 120 \
  --model gpt-4o-mini \
  --temperature 0.4 \
  --total_envs 1 \
  --max_steps 20 \
  --experiment_dir experiments/dfsdt_smoke


 python scripts/rollout/openmanus_rollout_debugger.py --env alfworld --strategy bon --bon_n 3 --model gpt-4o-mini --temperature 0.7 --alf_env_type alfworld/AlfredTWEnv --total_envs 10 --max_steps 3 --concurrency 10 --experiment_dir experiments/alfworld_bon
  

echo "All baseline tests completed!"