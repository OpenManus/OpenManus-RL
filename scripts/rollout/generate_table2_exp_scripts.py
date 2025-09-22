#!/usr/bin/env python3
"""Generate Table 2 experiment bash scripts with predefined sweeps."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from textwrap import dedent


ENVIRONMENTS = ["alfworld", "gaia", "webshop"]

MODELS = {
    "qwen3-8b": {
        "model_name": "kunlunz2/Qwen/Qwen3-8B-9f9838eb",
        "use_together": True,
    },
    "qwen3-next-80b": {
        "model_name": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "use_together": True,
    },
    "gpt-4o-mini": {
        "model_name": "gpt-4o-mini",
        "use_together": False,
    },
}

DEBUGGER_TYPES = ["vanilla", "advanced", "self-refine"]
STRATEGIES = ["bon", "tot"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate bash scripts for Table 2 experiments")
    parser.add_argument("--output-dir", default="./table2_exp_scripts", help="Where to write bash scripts")
    parser.add_argument("--base-dir", default="experiments/table2", help="Base directory for experiment outputs")
    parser.add_argument("--total-envs", type=int, default=50, help="Number of environments to test")
    parser.add_argument("--test-times", type=int, default=1)
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--history-length", type=int, default=30)
    parser.add_argument("--split", default="test")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-try", type=int, default=5, help="Number of retries for each environment")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of environments to test in parallel")
    parser.add_argument("--llm-concurrency", type=int, default=80, help="Number of LLM calls to make in parallel")
    parser.add_argument("--parallel-phase1", type=int, default=5, help="Phase-1 workers for debugger variants")
    parser.add_argument("--bon-n", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--value-threshold", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true", help="Print planned scripts without writing files")
    return parser.parse_args()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def render_script(
    env: str,
    model_key: str,
    model_cfg: dict[str, object],
    variant: str,
    variant_type: str,
    args: argparse.Namespace,
) -> str:
    run_name = f"{env}_{model_key}_{variant}_maxtry{args.max_try}"
    together_default = ""
    if bool(model_cfg["use_together"]):
        together_default = "--together both" if variant_type == "debugger" else "--together rollout"

    model_name = model_cfg["model_name"]
    debugger_model = model_name

    header = dedent(
        f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

if git_root="$(cd "${{SCRIPT_DIR}}" >/dev/null 2>&1 && git rev-parse --show-toplevel 2>/dev/null)"; then
  REPO_ROOT="${{git_root}}"
else
  CANDIDATE="${{SCRIPT_DIR}}"
  REPO_ROOT=""
  while [[ "${{CANDIDATE}}" != "/" ]]; do
    if [[ -d "${{CANDIDATE}}/scripts" && -f "${{CANDIDATE}}/pyproject.toml" ]]; then
      REPO_ROOT="${{CANDIDATE}}"
      break
    fi
    CANDIDATE="$(dirname "${{CANDIDATE}}")"
  done
  if [[ -z "${{REPO_ROOT}}" ]]; then
    echo "Failed to locate repository root" >&2
    exit 1
  fi
fi

cd "${{REPO_ROOT}}"

RUN_NAME="{run_name}"
BASE_DIR="{args.base_dir}"
RUN_DIR="${{BASE_DIR}}/${{RUN_NAME}}"
mkdir -p "${{RUN_DIR}}"

MODEL_NAME="{model_name}"
DEBUGGER_MODEL="{debugger_model}"
TOGETHER_ARG="{together_default}"

TOTAL_ENVS={args.total_envs}
TEST_TIMES={args.test_times}
START_ID={args.start_id}
MAX_STEPS={args.max_steps}
HISTORY_LENGTH={args.history_length}
TEMPERATURE={args.temperature}
MAX_TRY={args.max_try}
CONCURRENCY={args.concurrency}
LLM_CONCURRENCY={args.llm_concurrency}
PARALLEL_PHASE1={args.parallel_phase1}
BON_N={args.bon_n}
BEAM_SIZE={args.beam_size}
VALUE_THRESHOLD={args.value_threshold}
SPLIT="{args.split}"
"""
    ).strip()

    common_lines = [
        "cmd=(",
        "  python scripts/rollout/openmanus_rollout_debugger.py",
        f"  --env {env}",
        f"  --model \"${{MODEL_NAME}}\"",
        "  --total_envs ${TOTAL_ENVS}",
        "  --test_times ${TEST_TIMES}",
        "  --start_id ${START_ID}",
        "  --max_steps ${MAX_STEPS}",
        "  --history_length ${HISTORY_LENGTH}",
        "  --split \"${SPLIT}\"",
        "  --temperature ${TEMPERATURE}",
        "  --max_try ${MAX_TRY}",
        "  --experiment_dir \"${RUN_DIR}\"",
        "  --save_all_attempts",
        "  --save_per_task_trajectories",
        "  --unique_envs",
        "  --debug",
        "  --concurrency ${CONCURRENCY}",
        "  --llm_concurrency ${LLM_CONCURRENCY}",
    ]

    if variant_type == "debugger":
        common_lines.insert(3, "  --strategy debugger")
        common_lines.insert(4, "  --enable_debugger")
        common_lines.extend(
            [
                "  --debugger_model \"${DEBUGGER_MODEL}\"",
                f"  --debugger_type {variant}",
                "  --debugger_temperature 0.0",
                "  --parallel_num_phase_1 ${PARALLEL_PHASE1}",
            ]
        )
    else:
        common_lines.insert(3, f"  --strategy {variant}")
        if variant == "bon":
            common_lines.append("  --bon_n ${BON_N}")
        if variant == "tot":
            common_lines.extend(
                [
                    "  --beam_size ${BEAM_SIZE}",
                    "  --value_threshold ${VALUE_THRESHOLD}",
                ]
            )

    common_lines.append(")")

    together_block = dedent(
        """
if [[ -n "${TOGETHER_ARG}" ]]; then
  # shellcheck disable=SC2206
  together_tokens=(${TOGETHER_ARG})
  cmd+=("${together_tokens[@]}")
fi

"${cmd[@]}"
"""
    ).strip()

    script_body = "\n".join(common_lines)
    return f"{header}\n\n{script_body}\n\n{together_block}\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        ensure_directory(output_dir)

    scripts_written = 0
    for env in ENVIRONMENTS:
        for model_key, model_cfg in MODELS.items():
            for dbg_type in DEBUGGER_TYPES:
                content = render_script(env, model_key, model_cfg, dbg_type, "debugger", args)
                script_name = f"{env}_{model_key}_{dbg_type}_maxtry{args.max_try}.sh"
                if args.dry_run:
                    print(f"[DRY-RUN] {output_dir / script_name}")
                else:
                    script_path = output_dir / script_name
                    script_path.write_text(content, encoding="utf-8")
                    os.chmod(script_path, 0o755)
                scripts_written += 1

            for strategy in STRATEGIES:
                content = render_script(env, model_key, model_cfg, strategy, "strategy", args)
                script_name = f"{env}_{model_key}_{strategy}_maxtry{args.max_try}.sh"
                if args.dry_run:
                    print(f"[DRY-RUN] {output_dir / script_name}")
                else:
                    script_path = output_dir / script_name
                    script_path.write_text(content, encoding="utf-8")
                    os.chmod(script_path, 0o755)
                scripts_written += 1

    if args.dry_run:
        print(f"Planned {scripts_written} scripts (dry run)")
    else:
        print(f"Wrote {scripts_written} scripts to {output_dir}")


if __name__ == "__main__":
    main()
