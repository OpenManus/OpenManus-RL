#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="experiments/debugbench_example_alfworld"
TEMP=0.0
ENV_PARALLEL="${ENV_PARALLEL:-3}"

# AlfWorld test split
# Model selection (override via environment variables if needed)
PRIMARY_MODEL="${PRIMARY_MODEL:-${ROLLOUT_MODEL:-qwen3-8b}}"
PRIMARY_HOST="${PRIMARY_HOST:-${ROLLOUT_HOST:-${QWEN3_8B_URL:-http://129.212.187.116:8001}}}"
SECONDARY_MODEL="${SECONDARY_MODEL:-gpt-4o-mini}"
SECONDARY_HOST="${SECONDARY_HOST:-${OPENAI_BASE_URL:-}}"
TERTIARY_MODEL="${TERTIARY_MODEL:-${AUX_MODEL:-qwen3-32b}}"
TERTIARY_HOST="${TERTIARY_HOST:-${AUX_HOST:-${QWEN3_32B_URL:-http://134.199.197.179:8001}}}"

python scripts/debugbench/generate_debugbench.py \
  --environment alfworld \
  --output_dir "${OUTPUT_ROOT}" \
  --target_failures 100 \
  --history_length 40 \
  --max_steps 30 \
  --start_index 0 \
  --batch_size 10 \
  --env_parallel "${ENV_PARALLEL}" \
  --temperature "${TEMP}" \
  --model_primary "${PRIMARY_MODEL}" \
  --model_primary_base_url "${PRIMARY_HOST}" \
  --model_secondary "${SECONDARY_MODEL}" \
  --model_secondary_base_url "${SECONDARY_HOST}" \
  --model_tertiary "${TERTIARY_MODEL}" \
  --model_tertiary_base_url "${TERTIARY_HOST}"

# # GAIA validation split (loaded from data/gaia/val.json by default)
# python scripts/debugbench/generate_debugbench.py \
#   --environment gaia \
#   --output_dir "${OUTPUT_ROOT}" \
#   --target_failures 5 \
#   --history_length 40 \
#   --max_steps 30 \
#   --start_index 0 \
#   --batch_size 5 \
#   --temperature "${TEMP}" \
#   --model_primary "${PRIMARY_MODEL}" \
#   --model_primary_base_url "${PRIMARY_HOST}" \
#   --model_secondary "${SECONDARY_MODEL}" \
#   --model_secondary_base_url "${SECONDARY_HOST}" \
#   --model_tertiary "${TERTIARY_MODEL}" \
#   --model_tertiary_base_url "${TERTIARY_HOST}"

# # WebShop test set
# python scripts/debugbench/generate_debugbench.py \
#   --environment webshop \
#   --output_dir "${OUTPUT_ROOT}" \
#   --target_failures 5 \
#   --history_length 40 \
#   --max_steps 30 \
#   --start_index 0 \
#   --batch_size 5 \
#   --temperature "${TEMP}" \
#   --model_primary "${PRIMARY_MODEL}" \
#   --model_primary_base_url "${PRIMARY_HOST}" \
#   --model_secondary "${SECONDARY_MODEL}" \
#   --model_secondary_base_url "${SECONDARY_HOST}" \
#   --model_tertiary "${TERTIARY_MODEL}" \
#   --model_tertiary_base_url "${TERTIARY_HOST}"
