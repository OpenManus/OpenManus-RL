#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="experiments/debugbench_example_alfworld"
TEMP="${TEMP:-0.0}"
ENV_PARALLEL="${ENV_PARALLEL:-20}"
START_INDEX_VALUE="${START_INDEX:-11}"

if [[ -n "${START_ID:-}" ]]; then
  if ! [[ "${START_ID}" =~ ^[0-9]+$ ]]; then
    echo "START_ID must be a non-negative integer" >&2
    exit 1
  fi
  START_INDEX_VALUE=$((START_ID - 1))
fi

if [[ -z "${START_INDEX_VALUE}" ]]; then
  START_INDEX_VALUE=0
fi

if (( START_INDEX_VALUE < 0 )); then
  START_INDEX_VALUE=0
fi

# AlfWorld test split
# Model selection (override via environment variables if needed)
PRIMARY_MODEL="${PRIMARY_MODEL:-${ROLLOUT_MODEL:-kunlunz2/Qwen/Qwen3-8B-9f9838eb}}"
PRIMARY_HOST="${PRIMARY_HOST:-${ROLLOUT_HOST:-${TOGETHER_API_BASE_URL:-https://api.together.xyz}}}"
SECONDARY_MODEL="${SECONDARY_MODEL:-gpt-4o-mini}"
SECONDARY_HOST="${SECONDARY_HOST:-${OPENAI_BASE_URL:-}}"
TERTIARY_MODEL="${TERTIARY_MODEL:-${AUX_MODEL:-Qwen/Qwen2.5-72B-Instruct}}"
TERTIARY_HOST="${TERTIARY_HOST:-${AUX_HOST:-${TOGETHER_API_BASE_URL:-https://api.together.xyz}}}"

python scripts/debugbench/generate_debugbench.py \
  --environment alfworld \
  --output_dir "${OUTPUT_ROOT}" \
  --target_failures 100 \
  --history_length 40 \
  --max_steps 30 \
  --start_index "${START_INDEX_VALUE}" \
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
