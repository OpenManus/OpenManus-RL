#!/bin/bash

# Unified debugger baseline testing script
# Contains running commands for multiple strategies

echo "Starting unified debugger baseline tests..."

# Debugger strategy (smoke test with trajectory saving)
echo "Running Debugger strategy smoke test with trajectory saving..."
python scripts/rollout/openmanus_rollout_debugger.py \
  --env alfworld \
  --strategy debugger \
  --enable_debugger \
  --max_try 3 \
  --model gpt-4o-mini \
  --temperature 0.4 \
  --total_envs 3 \
  --max_steps 10 \
  --concurrency 3 \
  --unique_envs \
  --save_per_task_trajectories \
  --experiment_dir experiments/debugger_smoke_fixed

# Best-of-N strategy (smoke test with trajectory saving)
echo "Running Best-of-N strategy smoke test with trajectory saving..."
python scripts/rollout/openmanus_rollout_debugger.py \
  --env alfworld \
  --strategy bon \
  --bon_n 3 \
  --model gpt-4o-mini \
  --temperature 0.7 \
  --total_envs 3 \
  --max_steps 10 \
  --concurrency 3 \
  --unique_envs \
  --save_per_task_trajectories \
  --experiment_dir experiments/bon_smoke_fixed

# ToT-DFS strategy (smoke test with trajectory saving)
echo "Running ToT-DFS strategy smoke test with trajectory saving..."
python scripts/rollout/openmanus_rollout_debugger.py \
  --env alfworld \
  --strategy tot \
  --beam_size 4 \
  --value_threshold 0.2 \
  --max_try 5 \
  --model gpt-4o-mini \
  --temperature 0.4 \
  --total_envs 3 \
  --max_steps 10 \
  --concurrency 3 \
  --unique_envs \
  --save_per_task_trajectories \
  --experiment_dir experiments/tot_smoke_fixed

# DFSDT strategy (smoke test with trajectory saving)
echo "Running DFSDT strategy smoke test with trajectory saving..."
python scripts/rollout/openmanus_rollout_debugger.py \
  --env alfworld \
  --strategy dfsdt \
  --diversity_back_steps 2 \
  --diversity_back_steps_alt 3 \
  --propose_k 4 \
  --beam_size 3 \
  --max_try 5 \
  --model gpt-4o-mini \
  --temperature 0 \
  --total_envs 3 \
  --max_steps 3 \
  --concurrency 3 \
  --unique_envs \
  --save_per_task_trajectories \
  --experiment_dir experiments/dfsdt_smoke_fixed


# Best-of-N strategy with per-task storage
echo "Running Best-of-N strategy with per-task trajectory storage..."
python scripts/rollout/openmanus_rollout_debugger.py \
  --env alfworld \
  --strategy bon \
  --bon_n 3 \
  --model gpt-4o-mini \
  --temperature 0.7 \
  --alf_env_type alfworld/AlfredTWEnv \
  --total_envs 10 \
  --max_steps 3 \
  --concurrency 10 \
  --unique_envs \
  --save_per_task_trajectories \
  --experiment_dir experiments/alfworld_bon_fixed

# ToT-DFS strategy with per-task storage
echo "Running ToT-DFS strategy with per-task trajectory storage..."
python scripts/rollout/openmanus_rollout_debugger.py \
  --env alfworld \
  --strategy tot \
  --beam_size 4 \
  --value_threshold 0.2 \
  --max_try 5 \
  --model gpt-4o-mini \
  --temperature 0 \
  --total_envs 5 \
  --max_steps 3 \
  --concurrency 5 \
  --unique_envs \
  --save_per_task_trajectories \
  --experiment_dir experiments/alfworld_tot

# DFSDT strategy with per-task storage  
echo "Running DFSDT strategy with per-task trajectory storage..."
python scripts/rollout/openmanus_rollout_debugger.py \
  --env alfworld \
  --strategy dfsdt \
  --diversity_back_steps 2 \
  --diversity_back_steps_alt 3 \
  --propose_k 4 \
  --beam_size 3 \
  --max_try 5 \
  --model gpt-4o-mini \
  --temperature 0.4 \
  --total_envs 5 \
  --max_steps 20 \
  --concurrency 5 \
  --unique_envs \
  --save_per_task_trajectories \
  --experiment_dir experiments/alfworld_dfsdt
  

echo "All baseline tests completed!"