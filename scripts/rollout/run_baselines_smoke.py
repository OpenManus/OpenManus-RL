#!/usr/bin/env python3
"""
Quick smoke runner to test all rollout strategies (debugger, best-of-n, ToT-DFS, DFSDT)
on a tiny setup. This script just shells out to the unified rollout script with
small budgets so you can verify end-to-end wiring.

By default uses GAIA with the included val.json to avoid heavy external deps.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def build_common_args(args, strategy: str):
    out_dir = args.experiment_root or os.path.join(
        "experiments", f"smoke_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    cmd = [
        sys.executable, "-u", os.path.join("scripts", "rollout", "openmanus_rollout_debugger.py"),
        "--env", args.env,
        "--strategy", strategy,
        "--model", args.model,
        "--temperature", str(args.temperature),
        "--max_steps", str(args.max_steps),
        "--total_envs", str(args.total_envs),
        "--test_times", "1",
        "--concurrency", str(args.concurrency),
        "--llm_concurrency", str(max(1, args.concurrency * 2)),
        "--experiment_dir", out_dir,
        "--seed", str(args.seed),
    ]
    if args.base_url:
        cmd += ["--base_url", args.base_url]
    if args.debug:
        cmd.append("--debug")
    if args.dry_run:
        cmd.append("--dry_run")

    # Env-specific defaults for GAIA
    if args.env == "gaia":
        cmd += [
            "--gaia_data_path", args.gaia_data_path,
        ]
        if args.gaia_tools:
            cmd += ["--gaia_tools", *args.gaia_tools]
    elif args.env == "alfworld":
        cmd += ["--alf_env_type", args.alf_env_type]
    elif args.env == "webshop":
        if args.webshop_train:
            cmd.append("--webshop_train")

    # Strategy-specific parameters
    if strategy == "bon":
        cmd += ["--bon_n", str(args.bon_n)]
    elif strategy in ("tot", "dfsdt"):
        cmd += [
            "--beam_size", str(args.beam_size),
            "--value_threshold", str(args.value_threshold),
            "--max_nodes", str(args.max_nodes),
            "--diversity_back_steps", str(args.diversity_back_steps),
            "--diversity_back_steps_alt", str(args.diversity_back_steps_alt),
            "--propose_k", str(args.propose_k),
            "--search_retry", str(args.search_retry),
        ]
    elif strategy == "debugger":
        cmd += [
            "--enable_debugger",
            "--max_debug_retry", str(args.max_debug_retry),
            "--debugger_model", args.debugger_model or args.model,
            "--debugger_temperature", str(args.debugger_temperature),
            "--save_per_task_trajectories",
        ]

    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["gaia", "alfworld", "webshop"], default="gaia")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--total_envs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--experiment_root", default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--dry_run", action="store_true")

    # GAIA
    ap.add_argument("--gaia_data_path", default="data/gaia/val.json")
    ap.add_argument("--gaia_tools", nargs="+", default=["google_search"])  # minimal tool set

    # AlfWorld/WebShop
    ap.add_argument("--alf_env_type", default="alfworld/AlfredTWEnv")
    ap.add_argument("--webshop_train", action="store_true")

    # Debugger params
    ap.add_argument("--enable_debugger", action="store_true")
    ap.add_argument("--max_debug_retry", type=int, default=2)
    ap.add_argument("--debugger_model", default=None)
    ap.add_argument("--debugger_temperature", type=float, default=0.3)

    # Best-of-N
    ap.add_argument("--bon_n", type=int, default=2)

    # ToT/DFSDT
    ap.add_argument("--beam_size", type=int, default=3)
    ap.add_argument("--value_threshold", type=float, default=0.15)
    ap.add_argument("--max_nodes", type=int, default=30)
    ap.add_argument("--diversity_back_steps", type=int, default=2)
    ap.add_argument("--diversity_back_steps_alt", type=int, default=3)
    ap.add_argument("--propose_k", type=int, default=4)
    ap.add_argument("--search_retry", type=int, default=5)

    args = ap.parse_args()

    strategies = ["debugger", "bon", "tot", "dfsdt"]
    for s in strategies:
        cmd = build_common_args(args, s)
        print("\n==> Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Strategy '{s}' failed with code {e.returncode}")
            # Continue to next strategy


if __name__ == "__main__":
    main()
