#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if git_root="$(cd "${SCRIPT_DIR}" >/dev/null 2>&1 && git rev-parse --show-toplevel 2>/dev/null)"; then
  REPO_ROOT="${git_root}"
else
  CANDIDATE="${SCRIPT_DIR}"
  REPO_ROOT=""
  while [[ "${CANDIDATE}" != "/" ]]; do
    if [[ -d "${CANDIDATE}/scripts" && -f "${CANDIDATE}/pyproject.toml" ]]; then
      REPO_ROOT="${CANDIDATE}"
      break
    fi
    CANDIDATE="$(dirname "${CANDIDATE}")"
  done
  if [[ -z "${REPO_ROOT}" ]]; then
    echo "Failed to locate repository root" >&2
    exit 1
  fi
fi

cd "${REPO_ROOT}"

# Generate timestamp for unique run identification
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

RUN_NAME="gaia_qwen3-next-80b_vanilla_maxtry5_env50_start1_${TIMESTAMP}"
BASE_DIR="experiments/table2"
RUN_DIR="${BASE_DIR}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

MODEL_NAME="Qwen/Qwen3-Next-80B-A3B-Instruct"
DEBUGGER_MODEL="gpt-4.1"
TOGETHER_ARG="--together rollout"

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
BON_N=5
BEAM_SIZE=4
VALUE_THRESHOLD=0.2
SPLIT="test"

cmd=(
  python scripts/rollout/openmanus_rollout_debugger.py
  --env gaia
  --strategy debugger
  --enable_debugger
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
  --debugger_model "${DEBUGGER_MODEL}"
  --debugger_type vanilla
  --debugger_temperature 0.0
  --parallel_num_phase_1 ${PARALLEL_PHASE1}
)

if [[ -n "${TOGETHER_ARG}" ]]; then
  # shellcheck disable=SC2206
  together_tokens=(${TOGETHER_ARG})
  cmd+=("${together_tokens[@]}")
fi

"${cmd[@]}"
