#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

RUN_NAME="gaia_gpt-4o-mini_bon_maxtry5"
BASE_DIR="experiments/table2"
RUN_DIR="${BASE_DIR}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

MODEL_NAME="gpt-4o-mini"
DEBUGGER_MODEL="gpt-4o-mini"
TOGETHER_ARG=""

TOTAL_ENVS=50
TEST_TIMES=1
START_ID=1
MAX_STEPS=30
HISTORY_LENGTH=30
TEMPERATURE=0.0
MAX_TRY=5
CONCURRENCY=10
LLM_CONCURRENCY=80
PARALLEL_PHASE1=5
BON_N=3
BEAM_SIZE=4
VALUE_THRESHOLD=0.2
SPLIT="test"

cmd=(
  python scripts/rollout/openmanus_rollout_debugger.py
  --env gaia
  --strategy bon
  --model "${MODEL_NAME}"
  --total_envs ${TOTAL_ENVS}
  --test_times ${TEST_TIMES}
  --start_id ${START_ID}
  --max_steps ${MAX_STEPS}
  --history_length ${HISTORY_LENGTH}
  --split "${SPLIT}"
  --temperature ${TEMPERATURE}
  --max_try ${MAX_TRY}
  --experiment_dir "${RUN_DIR}"
  --save_all_attempts
  --save_per_task_trajectories
  --unique_envs
  --debug
  --concurrency ${CONCURRENCY}
  --llm_concurrency ${LLM_CONCURRENCY}
  --bon_n ${BON_N}
)

if [[ -n "${TOGETHER_ARG}" ]]; then
  # shellcheck disable=SC2206
  together_tokens=(${TOGETHER_ARG})
  cmd+=("${together_tokens[@]}")
fi

"${cmd[@]}"
