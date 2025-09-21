#!/usr/bin/env python3
"""Generate debugbench datasets where multiple rollout agents all fail."""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.rollout.openmanus_rollout_debugger import EnvironmentFactory, UnifiedAgent, load_gaia_tasks
from openmanus_rl.environments.env_package.alfworld.envs import load_config_file as load_alf_config
from openmanus_rl.environments.env_package.alfworld.alfworld.agents.environment import get_environment

DEFAULT_GAIA_DATA_PATH = Path("data/gaia/val.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build debugbench failure dataset")
    parser.add_argument("--environment", choices=["alfworld", "gaia", "webshop"], required=True,
                        help="Environment to evaluate (one per run)")
    parser.add_argument("--output_dir", required=True, help="Directory to store generated trajectories")
    parser.add_argument("--target_failures", type=int, default=10,
                        help="Number of failing tasks to collect")
    parser.add_argument("--history_length", type=int, default=2, help="Prompt history length")
    parser.add_argument("--seed", type=int, default=2025, help="Base random seed")
    parser.add_argument("--start_index", type=int, default=0, help="Global task start index")
    parser.add_argument("--max_steps", type=int, default=30, help="Maximum rollout steps for any environment")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of task indices to evaluate per iteration")
    parser.add_argument("--env_parallel", type=int, default=1,
                        help="Maximum number of tasks evaluated concurrently")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature shared across all models")
    parser.add_argument("--model_primary", default=os.getenv("PRIMARY_MODEL", os.getenv("ROLLOUT_MODEL", "qwen3-8b")),
                        help="Primary rollout model identifier")
    parser.add_argument("--model_primary_base_url", default=os.getenv("PRIMARY_URL", os.getenv("ROLLOUT_URL", os.getenv("QWEN3_8B_URL", "http://129.212.187.116:8001"))),
                        help="Primary rollout model base URL (without /v1)")
    parser.add_argument("--model_secondary", default=os.getenv("SECONDARY_MODEL", "gpt-4o-mini"),
                        help="Secondary rollout model identifier")
    parser.add_argument("--model_secondary_base_url", default=os.getenv("SECONDARY_URL", os.getenv("OPENAI_BASE_URL")),
                        help="Secondary rollout model base URL (without /v1)")
    parser.add_argument("--model_tertiary", default=os.getenv("TERTIARY_MODEL", os.getenv("QWEN32B_MODEL", "qwen3-32b")),
                        help="Tertiary rollout model identifier")
    parser.add_argument("--model_tertiary_base_url", default=os.getenv("TERTIARY_URL", os.getenv("QWEN3_32B_URL", "http://134.199.197.179:8001")),
                        help="Tertiary rollout model base URL (without /v1)")

    parser.add_argument("--alfworld_config", default=str(Path(__file__).resolve().parent.parent / ".." /
                                                           "openmanus_rl" / "environments" /
                                                           "env_package" / "alfworld" / "configs" /
                                                           "config_tw.yaml"),
                        help="Path to AlfWorld config file")
    parser.add_argument("--alfworld_eval_split", default="eval_in_distribution",
                        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
                        help="AlfWorld split to sample from")
    parser.add_argument("--gaia_tools", nargs='+',
                        default=['google_search', 'wikipedia_knowledge_searcher', 'python_code_generator'])

    return parser.parse_args()


def normalize_base_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    trimmed = url.rstrip('/')
    if trimmed.endswith('/v1'):
        return trimmed
    return f"{trimmed}/v1"


def build_model_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    return [
        {
            "id": "primary",
            "model": args.model_primary,
            "base_url": normalize_base_url(args.model_primary_base_url),
        },
        {
            "id": "secondary",
            "model": args.model_secondary,
            "base_url": normalize_base_url(args.model_secondary_base_url),
        },
        {
            "id": "tertiary",
            "model": args.model_tertiary,
            "base_url": normalize_base_url(args.model_tertiary_base_url),
        },
    ]


def ensure_args(args: argparse.Namespace) -> None:
    if args.max_steps <= 0:
        raise ValueError("--max_steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")


def json_safe(data: Any) -> Any:
    try:
        return json.loads(json.dumps(data, default=str))
    except Exception:
        return str(data)


def collect_alfworld_game_files(config_path: str, split: str) -> List[str]:
    cfg = load_alf_config(config_path)
    env_type = cfg['env']['type']
    BaseEnvCls = get_environment(env_type)
    tmp_env = BaseEnvCls(cfg, train_eval=split)
    tmp_env.collect_game_files()
    files = list(dict.fromkeys(tmp_env.game_files))
    return files


def build_agent(spec: Dict[str, Any], env_type: str, temperature: float) -> UnifiedAgent:
    agent = UnifiedAgent(
        model_name=spec["model"],
        temperature=temperature,
        base_url=spec.get("base_url"),
        env_type=env_type,
        use_together=False,
    )
    return agent


def run_episode(env_type: str,
                env_manager,
                agent: UnifiedAgent,
                max_steps: int,
                reset_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
    agent.env_type = env_type
    reset_kwargs = reset_kwargs or {}
    obs_dict, infos = env_manager.reset(**reset_kwargs)
    obs_text = obs_dict['text'][0]
    trajectory: List[Dict[str, Any]] = []
    success = False

    for step_idx in range(max_steps):
        action = agent.get_action_from_llm(obs_text, log_timing=False)
        next_obs, rewards, dones, step_infos = env_manager.step([action])
        info = step_infos[0]
        reward = float(rewards[0]) if hasattr(rewards, '__len__') else float(rewards)
        done = bool(dones[0]) if hasattr(dones, '__len__') else bool(dones)
        success = bool(info.get('won', False))

        trajectory.append({
            "step": step_idx,
            "observation": obs_text,
            "action": action,
            "reward": reward,
            "done": done,
            "info": json_safe(info),
        })

        if done:
            break
        obs_text = next_obs['text'][0]

    final_metadata = {
        "steps": len(trajectory),
        "terminated": bool(trajectory and trajectory[-1]['done']),
        "success": success,
    }
    return success, trajectory, final_metadata


def save_trajectory(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def build_gaia_manager(task: Dict[str, Any], args: argparse.Namespace):
    return EnvironmentFactory.build_env(
        "gaia",
        with_debugger=False,
        tasks_data=[task],
        available_tools=args.gaia_tools,
        env_num=1,
        seed=args.seed,
        history_length=args.history_length,
        max_steps=args.max_steps,
    )


def build_webshop_manager(args: argparse.Namespace):
    return EnvironmentFactory.build_env(
        "webshop",
        with_debugger=False,
        env_num=1,
        seed=args.seed,
        history_length=args.history_length,
        use_train_set=False,
        summary_api_key=None,
        summary_endpoint=None,
        max_steps=args.max_steps,
        is_train=False,
    )


def build_alfworld_manager(game_file: str, args: argparse.Namespace):
    return EnvironmentFactory.build_env(
        "alfworld",
        with_debugger=False,
        env_num=1,
        seed=args.seed,
        history_length=args.history_length,
        alf_env_type="alfworld/AlfredTWEnv",
        game_files=[game_file],
        is_train=False,
    )


def process_environment(env_name: str,
                        args: argparse.Namespace,
                        model_specs: List[Dict[str, Any]],
                        agents: Dict[str, UnifiedAgent],
                        resources: Dict[str, Any],
                        summary: Dict[str, List[int]]) -> None:
    target_failures = args.target_failures
    total_available = resources['total']
    start_index = max(0, int(args.start_index))
    batch_size = max(1, int(args.batch_size))
    env_parallel = max(1, int(args.env_parallel))

    logging.info("Processing %s: target=%d start=%d available=%d batch=%d env_parallel=%d",
                 env_name, target_failures, start_index, total_available, batch_size, env_parallel)

    root_path = Path(args.output_dir)
    failures: List[int] = []
    current_index = start_index

    while len(failures) < target_failures and current_index < total_available:
        batch_indices = list(range(current_index, min(total_available, current_index + batch_size)))

        def evaluate_task(task_index: int):
            logging.debug("Evaluating %s task index %d", env_name, task_index)
            task_dir = root_path / env_name / f"task_{task_index:05d}"
            task_dir.mkdir(parents=True, exist_ok=True)

            per_model_payload: Dict[str, Tuple[bool, List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]] = {}
            all_failed = True

            for spec in model_specs:
                agent = agents[spec['id']]
                success, trajectory, meta, task_info = rollout_task_for_model(
                    env_name,
                    task_index,
                    agent,
                    spec,
                    args,
                    resources,
                )
                if trajectory is None:
                    logging.debug("Model %s failed to evaluate task %d due to exception", spec['id'], task_index)
                    return None

                per_model_payload[spec['id']] = (success, trajectory, meta, task_info)

                save_trajectory(
                    task_dir / f"{spec['id']}.json",
                    {
                        "environment": env_name,
                        "task_index": task_index,
                        "task_info": json_safe(task_info),
                        "trajectory": trajectory,
                        "metadata": meta,
                    },
                )

                all_failed = all_failed and (not success)

                if success:
                    logging.info(
                        "%s index %d succeeded on model %s; skipping remaining models",
                        env_name,
                        task_index,
                        spec['id'],
                    )
                    break

            if all_failed and len(per_model_payload) == len(model_specs):
                return task_index, per_model_payload
            return None

        with ThreadPoolExecutor(max_workers=env_parallel) as executor:
            future_map = {executor.submit(evaluate_task, task_index): task_index for task_index in batch_indices}

            for future in as_completed(future_map):
                if len(failures) >= target_failures:
                    break
                result = future.result()
                if not result:
                    continue
                task_index, per_model_payload = result
                if len(per_model_payload) == len(model_specs):
                    logging.info("%s index %d failed across all models", env_name, task_index)
                    failures.append(task_index)
                    write_outputs_for_failure(env_name, task_index, per_model_payload, args.output_dir)

        current_index += batch_size

    if len(failures) < target_failures:
        logging.warning(
            "Environment %s exhausted before reaching target failures (%d/%d)",
            env_name,
            len(failures),
            target_failures,
        )

    summary[env_name] = failures


def rollout_task_for_model(env_name: str,
                           index: int,
                           agent: UnifiedAgent,
                           spec: Dict[str, Any],
                           args: argparse.Namespace,
                           resources: Dict[str, Any]) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    env_manager = None
    task_info: Dict[str, Any] = {}
    reset_kwargs: Dict[str, Any] = {}

    try:
        if env_name == "gaia":
            task = resources['tasks'][index]
            env_manager = build_gaia_manager(task, args)
            task_info = {"pid": task.get('pid'), "question": task.get('question'), "answer": task.get('answer'), "index": index}
        elif env_name == "webshop":
            env_manager = build_webshop_manager(args)
            reset_kwargs = {"session_indices": [resources['indices'][index]]}
            task_info = {"session_index": resources['indices'][index]}
        elif env_name == "alfworld":
            game_file = resources['files'][index]
            env_manager = build_alfworld_manager(game_file, args)
            task_info = {"game_file": game_file, "index": index}
        else:
            raise ValueError(f"Unsupported environment {env_name}")

        success, trajectory, meta = run_episode(env_name, env_manager, agent, args.max_steps, reset_kwargs)
        task_info.update(meta)
        return success, trajectory, meta, task_info
    except Exception as exc:  # noqa: BLE001
        logging.error("%s model %s encountered error on index %d: %s", env_name, spec['id'], index, exc)
        return False, None, None, None
    finally:
        if env_manager is not None:
            try:
                env_manager.close()
            except Exception:
                pass


def write_outputs_for_failure(env_name: str,
                              index: int,
                              payloads: Dict[str, Tuple[bool, List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]],
                              output_root: str) -> None:
    root = Path(output_root)
    task_dir = root / env_name / f"task_{index:05d}"
    artifacts: Dict[str, str] = {
        model_id: str(task_dir / f"{model_id}.json") for model_id in payloads
    }
    task_infos = {model_id: json_safe(details[3]) for model_id, details in payloads.items()}

    summary_path = root / f"{env_name}_failures.jsonl"
    summary_entry = {
        "environment": env_name,
        "task_index": index,
        "task_directory": str(task_dir),
        "artifacts": artifacts,
        "task_info": task_infos,
    }
    with SUMMARY_WRITE_LOCK:
        with summary_path.open('a', encoding='utf-8') as fp:
            fp.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")


def prepare_resources(args: argparse.Namespace, env_name: str) -> Dict[str, Any]:
    if env_name == "alfworld":
        files = collect_alfworld_game_files(args.alfworld_config, args.alfworld_eval_split)
        return {
            "files": files,
            "total": len(files),
        }
    if env_name == "gaia":
        gaia_tasks = load_gaia_tasks(str(DEFAULT_GAIA_DATA_PATH))
        return {
            "tasks": gaia_tasks,
            "total": len(gaia_tasks),
        }
    if env_name == "webshop":
        webshop_indices = list(range(500))
        return {
            "indices": webshop_indices,
            "total": len(webshop_indices),
        }
    raise ValueError(f"Unsupported environment {env_name}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    model_specs = build_model_specs(args)
    ensure_args(args)

    env_name = args.environment
    resources = prepare_resources(args, env_name)

    agents: Dict[str, UnifiedAgent] = {
        spec['id']: build_agent(spec, env_name, args.temperature)
        for spec in model_specs
    }

    summary: Dict[str, List[int]] = {}

    start_time = time.time()
    process_environment(env_name, args, model_specs, agents, resources, summary)
    logging.info("Finished %s in %.2f seconds", env_name, time.time() - start_time)

    summary_path = Path(args.output_dir) / f"failure_summary_{env_name}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open('w', encoding='utf-8') as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    logging.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logging.error("Generation failed: %s", exc)
        sys.exit(1)
