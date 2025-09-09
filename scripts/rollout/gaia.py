import os
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass
import time
import json
import logging
import argparse
from types import SimpleNamespace
from datetime import datetime
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
import random
import sys
from openmanus_rl.environments.env_manager import *
from openai import OpenAI
from together import Together


def build_env(env_name, tasks_data, available_tools, env_num=1, seed=1, history_length=2, max_steps=30):
    group_n = 1
    if env_name == "gaia":
        # Build GAIA/Tool Use environments
        from openmanus_rl.environments.env_package.tool_use.projection import tool_use_projection
        from openmanus_rl.environments.env_package.tool_use.envs import build_tool_use_envs
        from openmanus_rl.environments.env_package.tool_use.manager import ToolUseEnvironmentManager
        
        envs = build_tool_use_envs(
            tasks_data=tasks_data, 
            available_tools=available_tools,
            seed=seed, 
            env_num=env_num, 
            group_n=group_n, 
            is_train=True
        )
        
        # Minimal config object with required fields
        cfg = SimpleNamespace(
            env=SimpleNamespace(
                env_name="tool_use",
                history_length=history_length,
                max_steps=max_steps  # Controlled by CLI
            )
        )
        env_manager = ToolUseEnvironmentManager(envs, tool_use_projection, cfg)
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")

    return env_manager


class Agent:
    def __init__(self, model_name="gpt-4o", temperature: float = 0.4, base_url: str | None = None):
        self.model_name = model_name
        self.temperature = temperature
        
        # Check if model is a Together model (contains "/" and no base_url provided)
        self.is_together = "/" in model_name and base_url is None
        
        if self.is_together:
            self.client = Together(
                api_key=os.environ.get('TOGETHER_API_KEY', ''),
            )
        elif base_url:
            self.client = OpenAI(
                api_key=os.getenv('OPENAI_API_KEY', 'EMPTY'),
                base_url=base_url,
            )
        else:
            self.client = OpenAI(
                api_key=os.environ['OPENAI_API_KEY'],
            )
        
    def get_action_from_gpt(self, obs):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user", 
                    "content": obs
                }
            ],
            temperature=self.temperature,
            n=1,
        )
        action = response.choices[0].message.content.strip()
        return action

    def get_actions_batch(self, prompts: List[str], concurrency: int = 4, retries: int = 3, backoff: float = 0.5) -> List[str]:
        actions = [None] * len(prompts)

        def _one(idx_prompt):
            idx, prompt = idx_prompt
            delay = backoff
            for attempt in range(retries):
                try:
                    act = self.get_action_from_gpt(prompt)
                    return idx, act
                except Exception as e:
                    if attempt == retries - 1:
                        return idx, "None"
                    time.sleep(delay)
                    delay *= 2

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futures = [ex.submit(_one, (i, p)) for i, p in enumerate(prompts)]
            for fut in as_completed(futures):
                i, act = fut.result()
                actions[i] = act

        return actions


def load_gaia_tasks(data_path: str, max_tasks: int = None) -> List[dict]:
    """Load GAIA tasks from JSON file"""
    with open(data_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    if max_tasks:
        tasks = tasks[:max_tasks]
    
    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="gaia")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of envs to process per batch")
    parser.add_argument("--total_envs", type=int, default=100, help="Total number of environments to rollout")
    parser.add_argument("--test_times", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=30, help="Maximum steps per task")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--history_length", type=int, default=2)
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name (OpenAI: gpt-4o, gpt-4o-mini; Together: Qwen/Qwen2.5-7B-Instruct-Turbo, etc.)")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent OpenAI requests per step")
    parser.add_argument("--retries", type=int, default=3, help="Retries per request on failure")
    parser.add_argument("--dump_path", default=None, help="If set, write JSONL trajectory to this file")
    parser.add_argument("--base_url", default=None, help="OpenAI-compatible base URL (e.g., vLLM http://127.0.0.1:8000/v1)")
    parser.add_argument("--chat_root", default=None, help="If set, save per-episode chat histories under this root")
    parser.add_argument("--data_path", default="data/gaia/val.json", help="Path to GAIA dataset")
    parser.add_argument("--tools", nargs='+', default=['google_search', 'wikipedia_knowledge_searcher', 'python_code_generator'], 
                       help="List of available tools")
    parser.add_argument("--dry_run", action="store_true", help="仅打印任务分配，不创建环境、不调用模型")
    args = parser.parse_args()

    # -------- logging ----------
    os.makedirs("logs/gaia", exist_ok=True)
    log_fp = os.path.join(
        "logs/gaia", f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )

    # -------- Parameters ----------
    max_steps = args.max_steps
    batch_size = args.batch_size
    total_envs = args.total_envs
    test_times = args.test_times
    env_name = args.env_name
    
    # Load GAIA tasks
    logging.info(f"Loading GAIA tasks from {args.data_path}")
    all_tasks = load_gaia_tasks(args.data_path)
    logging.info(f"Loaded {len(all_tasks)} total tasks from {args.data_path}")
    
    # Ensure we have enough tasks for requested envs
    if len(all_tasks) < total_envs:
        logging.warning(f"Only {len(all_tasks)} tasks available, adjusting total_envs from {total_envs} to {len(all_tasks)}")
        total_envs = len(all_tasks)
    
    # Calculate number of batches needed
    num_batches = (total_envs + batch_size - 1) // batch_size
    logging.info(f"Running {total_envs} envs in {num_batches} batches of {batch_size}") 

    # -------- Agent setup ----------
    agent = Agent(model_name=args.model, temperature=args.temperature, base_url=args.base_url)

    # Prepare trajectory dump file if requested
    dump_fp = None
    if args.dump_path:
        os.makedirs(os.path.dirname(args.dump_path) or ".", exist_ok=True)
        dump_fp = open(args.dump_path, "a", encoding="utf-8")

    # Prepare chat history directories if requested
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    chat_ts_root = None
    chat_base_dir = None
    if args.chat_root:
        chat_ts_root = os.path.join(args.chat_root, 'trajectories', run_ts)
        chat_base_dir = os.path.join(chat_ts_root, args.env_name, args.model)
        os.makedirs(chat_base_dir, exist_ok=True)

    def _sanitize(s: str) -> str:
        return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '-' for c in s)[:200]

    # Accumulated statistics across all batches
    all_overall_success_rates = []
    all_task_success_history = defaultdict(list)
    global_env_counter = 0

    # Shuffle tasks for random sampling
    rng = random.Random(args.seed)
    rng.shuffle(all_tasks)

    # Dry-run mode
    if args.dry_run:
        logging.info(f"[Dry-Run] total_envs={total_envs}, batch_size={batch_size}, num_batches={num_batches}")
        for b in range(num_batches):
            start = b * batch_size
            end = start + min(batch_size, total_envs - start)
            batch_tasks = all_tasks[start:end]
            pids = [t.get('pid', f'task_{i}') for i, t in enumerate(batch_tasks)]
            logging.info(f"[Dry-Run] Batch {b+1:02d}: {len(batch_tasks)} tasks; PIDs: {', '.join(pids[:3])}...")
        sys.exit(0)

    # ======================= Main Batch Loop =======================
    for batch_idx in range(num_batches):
        # Calculate actual batch size for this batch (last batch might be smaller)
        current_batch_size = min(batch_size, total_envs - batch_idx * batch_size)
        logging.info(f"\n========== Starting Batch {batch_idx + 1}/{num_batches} with {current_batch_size} envs ==========")
        
        # Select tasks for this batch
        start = batch_idx * batch_size
        end = start + current_batch_size
        batch_tasks = all_tasks[start:end]

        # Create environment for this batch
        env_manager = build_env(
            env_name,
            tasks_data=batch_tasks,
            available_tools=args.tools,
            env_num=current_batch_size,
            seed=args.seed + batch_idx,
            history_length=args.history_length,
            max_steps=args.max_steps,
        )
        
        # Batch-level statistics
        batch_overall_success_rates = []
        batch_task_success_history = defaultdict(list)
        
        try:
            # ======================= Test Loop for this Batch =======================
            for test_idx in range(test_times):
                logging.info(f"\n========== Start Batch {batch_idx + 1} Test {test_idx} ==========")
                start_time = time.time()

                obs, infos = env_manager.reset()
                env_dones = [False] * current_batch_size
                # Persist PIDs from reset infos for later logging (step infos omit pid)
                pids = [infos[i].get("pid", f"task_{i}") for i in range(len(infos))]

                # per-env chat buffers
                chats = [[] for _ in range(current_batch_size)]
                # track which envs already dumped to disk
                saved_flags = [False] * current_batch_size
                # keep last infos for fallback dump (failure/timeout)
                last_infos = infos

                # Statistics for single round
                overall_success_this_round = np.zeros(current_batch_size, dtype=bool)
                task_success_cnt = defaultdict(int)
                task_total_cnt = defaultdict(int)

                for step_idx in range(max_steps):
                    logging.info(f"Batch {batch_idx + 1} Step {step_idx}; Dones ({np.array(env_dones).sum().item()}/{current_batch_size}); SR {overall_success_this_round.mean().item()}")

                    # --- Assemble actions ---
                    prompts = []
                    idx_map = []  # map from prompts index back to env index
                    for i in range(current_batch_size):
                        if not env_dones[i]:
                            prompts.append(obs["text"][i])
                            idx_map.append(i)

                    batch_actions = agent.get_actions_batch(prompts, concurrency=args.concurrency, retries=args.retries)
                    actions = ["None"] * current_batch_size
                    for k, i in enumerate(idx_map):
                        actions[i] = batch_actions[k]

                    # --- Environment stepping ---
                    prev_prompts = obs["text"]  # keep for logging & chat history
                    # Preserve the model's raw outputs for logging/chat before any projection mutates them
                    raw_actions = actions.copy()
                    # Pass a copy into the env manager so in-place projection does not alter our raw copy
                    obs, rewards, dones, infos = env_manager.step(actions.copy())
                    last_infos = infos

                    # --- Determine endings and successes ---
                    for i in range(current_batch_size):
                        if env_dones[i]:
                            continue

                        # Append chat turns for acted envs
                        if prev_prompts and i < len(prev_prompts):
                            chats[i].append({"role": "user", "content": prev_prompts[i]})
                        # Save the model's full raw reply (not the post-projection/action-only string)
                        chats[i].append({"role": "assistant", "content": raw_actions[i]})

                        # Dump trajectory row (only for envs that acted this step, including final step)
                        if args.dump_path and (i in idx_map):
                            try:
                                row = {
                                    "batch_idx": batch_idx,
                                    "test_idx": test_idx,
                                    "step": step_idx,
                                    "env_id": global_env_counter + i,  # Global env ID across all batches
                                    "prompt": prev_prompts[i],
                                    # Save the full raw model output for this step
                                    "action": raw_actions[i],
                                    # Also save the executed (post-projection) action for debugging
                                    "action_exec": actions[i],
                                    "reward": float(rewards[i]) if i < len(rewards) else None,
                                    "done": bool(dones[i]) if i < len(dones) else None,
                                    "won": bool(infos[i].get("won", False)),
                                    "pid": pids[i] if i < len(pids) else "unknown",
                                    "is_action_valid": bool(infos[i].get("is_action_valid", False)),
                                }
                                dump_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                            except Exception:
                                pass

                        if dones[i]:
                            env_dones[i] = True
                            won = bool(infos[i].get("won", False))
                            overall_success_this_round[i] = won

                            # Track success for this task
                            pid = pids[i] if i < len(pids) else f"task_{i}"
                            task_total_cnt[pid] = 1
                            if won:
                                task_success_cnt[pid] = 1

                            # If this env just finished, dump chat history if requested
                            if chat_base_dir and not saved_flags[i]:
                                try:
                                    pid = pids[i] if i < len(pids) else f"task_{i}"
                                    task_dir = os.path.join(chat_base_dir, _sanitize(pid))
                                    os.makedirs(task_dir, exist_ok=True)
                                    unique_id = f"b{batch_idx:03d}_t{test_idx:02d}_e{i:02d}"
                                    base = f"chat_{_sanitize(pid)}-{unique_id}"
                                    out_path = os.path.join(task_dir, base + ".json")
                                    meta = {
                                        "batch_idx": batch_idx,
                                        "env_id": global_env_counter + i,
                                        "test_idx": test_idx,
                                        "model": args.model,
                                        "pid": pid,
                                        "steps": step_idx + 1,
                                        "won": bool(infos[i].get("won", False)),
                                        "timestamp": run_ts,
                                    }
                                    with open(out_path, "w", encoding="utf-8") as f:
                                        json.dump({"messages": chats[i], "metadata": meta}, f, ensure_ascii=False, indent=2)
                                    saved_flags[i] = True
                                except Exception:
                                    pass

                    if all(env_dones):
                        logging.info("All environments finished early!")
                        break

                # After loop: dump any unfinished envs (failures/timeouts)
                if chat_base_dir:
                    for i in range(current_batch_size):
                        if not saved_flags[i]:
                            try:
                                pid = pids[i] if i < len(pids) else f"task_{i}"
                                task_dir = os.path.join(chat_base_dir, _sanitize(pid))
                                os.makedirs(task_dir, exist_ok=True)
                                unique_id = f"b{batch_idx:03d}_t{test_idx:02d}_e{i:02d}"
                                base = f"chat_{_sanitize(pid)}-{unique_id}"
                                out_path = os.path.join(task_dir, base + ".json")
                                steps_taken = max(0, len(chats[i]) // 2)
                                meta = {
                                    "batch_idx": batch_idx,
                                    "env_id": global_env_counter + i,
                                    "test_idx": test_idx,
                                    "model": args.model,
                                    "pid": pid,
                                    "steps": steps_taken,
                                    "won": bool(last_infos[i].get("won", False)) if isinstance(last_infos, list) and i < len(last_infos) else False,
                                    "timestamp": run_ts,
                                }
                                with open(out_path, "w", encoding="utf-8") as f:
                                    json.dump({"messages": chats[i], "metadata": meta}, f, ensure_ascii=False, indent=2)
                                saved_flags[i] = True
                            except Exception:
                                pass

                # -------- Single round results --------
                round_success_rate = overall_success_this_round.mean()
                batch_overall_success_rates.append(round_success_rate)

                logging.info(f"Batch {batch_idx + 1} Test {test_idx} overall success: {round_success_rate:.4f}")

                # Aggregate per-task results for this round
                for pid, total in task_total_cnt.items():
                    if total > 0:
                        rate = task_success_cnt.get(pid, 0) / total
                        batch_task_success_history[pid].append(rate)

                # Log individual task results (top few)
                success_pids = [pid for pid, cnt in task_success_cnt.items() if cnt > 0]
                if success_pids:
                    logging.info(f"    Successful tasks: {', '.join(success_pids[:5])}{'...' if len(success_pids) > 5 else ''}")

                logging.info(
                    f"Batch {batch_idx + 1} Test {test_idx} time elapsed: {time.time() - start_time:.2f}s\n"
                )

        finally:
            # Accumulate batch results to global results
            all_overall_success_rates.extend(batch_overall_success_rates)
            for task, rates in batch_task_success_history.items():
                all_task_success_history[task].extend(rates)

            # Update global env counter
            global_env_counter += current_batch_size

            # Clean up resources for this batch
            try:
                env_manager.envs.close()
                logging.info(f"Released resources for Batch {batch_idx + 1}")
            except Exception as e:
                logging.warning(f"Failed to release resources for Batch {batch_idx + 1}: {e}")

            logging.info(f"========== Finished Batch {batch_idx + 1}/{num_batches}, processed {global_env_counter}/{total_envs} envs ==========\n")

    # ======================= Final Summary =======================
    logging.info("=============== Final Summary ===============")
    logging.info(
        f"Total batches: {num_batches} | Batch size: {batch_size} | Total envs processed: {global_env_counter}"
    )
    logging.info(
        f"Overall success avg ± std: "
        f"{np.mean(all_overall_success_rates):.4f} ± {np.std(all_overall_success_rates):.4f}"
    )

    # Summary of task-level success
    successful_tasks = sum(1 for rates in all_task_success_history.values() if any(r > 0 for r in rates))
    logging.info(f"Successfully completed {successful_tasks} out of {len(all_task_success_history)} unique tasks")

    if dump_fp is not None:
        dump_fp.flush()
        dump_fp.close()
