import os
import time
import json
import logging
import argparse
from types import SimpleNamespace
from datetime import datetime
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import hashlib
import sys

from openmanus_rl.environments.env_manager import WebshopEnvironmentManager
from openmanus_rl.environments.env_package.webshop import build_webshop_envs, webshop_projection
from openai import OpenAI
try:
    import dotenv
    dotenv.load_dotenv()
except Exception:
    pass

def build_env(env_name: str, env_num: int = 1, seed: int = 1, history_length: int = 2, use_train_set: bool = False) -> WebshopEnvironmentManager:
    if env_name != "webshop":
        raise ValueError(f"Unsupported environment name: {env_name}")

    env_kwargs = {"observation_mode": "text"}

    envs = build_webshop_envs(
        seed=seed,
        env_num=env_num,
        group_n=1,
        is_train=use_train_set,
        env_kwargs=env_kwargs,
    )

    cfg = SimpleNamespace(env=SimpleNamespace(env_name="webshop/WebAgentTextEnv", history_length=history_length))
    return WebshopEnvironmentManager(envs, webshop_projection, cfg)


class Agent:
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.4, base_url: str | None = None):
        self.model_name = model_name
        self.temperature = temperature
        # vLLM/OpenAI-compatible: when base_url provided, point to e.g. http://127.0.0.1:8000/v1
        if base_url:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "EMPTY"), base_url=base_url)
        else:
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # will raise if missing

        self.system_prompt = (
            "You are an expert web shopping agent. Respond strictly as \n"
            "<think>...</think><action>...</action>. The <action> must be a single \n"
            "admissible action exactly from the provided list, or a search[query]."
        )

    def get_action(self, obs_text: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": obs_text},
            ],
            temperature=self.temperature,
            n=1,
        )
        return resp.choices[0].message.content.strip()

    def get_actions_batch(self, prompts: List[str], concurrency: int = 4, retries: int = 3, backoff: float = 0.5) -> List[str]:
        actions = [None] * len(prompts)

        def _one(idx_prompt):
            idx, prompt = idx_prompt
            delay = backoff
            for attempt in range(retries):
                try:
                    act = self.get_action(prompt)
                    return idx, act
                except Exception:
                    if attempt == retries - 1:
                        return idx, "<think>error</think><action>search[product]</action>"
                    time.sleep(delay)
                    delay *= 2

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futures = [ex.submit(_one, (i, p)) for i, p in enumerate(prompts)]
            for fut in as_completed(futures):
                i, act = fut.result()
                actions[i] = act
        return actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="webshop")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--total_envs", type=int, default=8)
    parser.add_argument("--test_times", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--history_length", type=int, default=2)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dump_path", default=None, help="Write JSONL trajectory to this file if set")
    parser.add_argument("--base_url", default=None, help="OpenAI-compatible base URL (e.g., vLLM http://127.0.0.1:8000/v1)")
    parser.add_argument("--chat_root", default=None, help="Optional chat history root dir")
    parser.add_argument("--use_train_set", action="store_true", help="Use training set goals instead of test set")
    parser.add_argument("--unique_envs", action="store_true", help="Ensure unique goal indices across total_envs (no repeats)")
    parser.add_argument("--dry_run", action="store_true", help="Only print planned batches when --unique_envs is set, then exit")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # logging
    os.makedirs("logs/webshop", exist_ok=True)
    log_fp = os.path.join("logs/webshop", f"run2_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )

    batch_size = args.batch_size
    total_envs = args.total_envs
    num_batches = (total_envs + batch_size - 1) // batch_size
    logging.info(f"Running {total_envs} envs in {num_batches} batches of {batch_size}")
    logging.info(f"Model={args.model}, base_url={args.base_url}, temp={args.temperature}")

    agent = Agent(model_name=args.model, temperature=args.temperature, base_url=args.base_url)

    dump_fp = None
    if args.dump_path:
        os.makedirs(os.path.dirname(args.dump_path) or ".", exist_ok=True)
        dump_fp = open(args.dump_path, "a", encoding="utf-8")
        logging.info(f"Dumping trajectories to: {args.dump_path}")

    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    chat_base_dir = None
    if args.chat_root:
        chat_ts_root = os.path.join(args.chat_root, 'trajectories', run_ts)
        chat_base_dir = os.path.join(chat_ts_root, args.env_name, args.model.replace('/', '_'))
        os.makedirs(chat_base_dir, exist_ok=True)
        logging.info(f"Saving chats to: {chat_base_dir}")

    def _sanitize(s: str) -> str:
        return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '-' for c in s)[:200]

    all_succ = []
    all_reward = []
    global_env_counter = 0

    # Pre-assign unique goal indices across all envs when requested
    preassigned_goal_indices = None
    if args.unique_envs:
        # Build a temporary env to fetch the available goal index pool (train or test)
        try:
            tmp_env = build_env(args.env_name, env_num=1, seed=args.seed, history_length=args.history_length, use_train_set=args.use_train_set)
            pool = list(tmp_env.envs.goal_idxs)
            try:
                tmp_env.envs.close()
            except Exception:
                pass
        except Exception as e:
            logging.error(f"Failed to probe goal pool: {e}")
            raise

        if len(pool) < args.total_envs:
            raise ValueError(f"Not enough unique goals in the {'train' if args.use_train_set else 'test'} set: need {args.total_envs}, available {len(pool)}")

        rng = np.random.RandomState(args.seed)
        preassigned_goal_indices = rng.choice(pool, size=args.total_envs, replace=False).tolist()
        logging.info(f"Unique envs enabled: sampled {len(preassigned_goal_indices)} unique goal indices from pool size {len(pool)}")

        if args.dry_run:
            logging.info(f"[Dry-Run] total_envs={args.total_envs}, batch_size={batch_size}, num_batches={num_batches}")
            for b in range(num_batches):
                start = b * batch_size
                end = min(start + min(batch_size, args.total_envs - start), args.total_envs)
                batch_slice = preassigned_goal_indices[start:end]
                logging.info(f"[Dry-Run] Batch {b+1:02d}: {len(batch_slice)} goals; examples: {batch_slice[:5]}")
            sys.exit(0)

    try:
        for b in range(num_batches):
            cur_n = min(batch_size, total_envs - b * batch_size)
            logging.info(f"\n== Batch {b+1}/{num_batches} with {cur_n} envs ==")
            env = build_env(args.env_name, env_num=cur_n, seed=args.seed + b, history_length=args.history_length, use_train_set=args.use_train_set)

            # If unique envs requested, force this batch to use the pre-assigned, non-overlapping goal indices
            if preassigned_goal_indices is not None:
                start = b * batch_size
                end = start + cur_n
                batch_slice = preassigned_goal_indices[start:end]
                env.envs.goal_idxs = list(batch_slice)
                logging.info(f"Batch {b+1}: using preassigned goals (len={len(batch_slice)}) from {'train' if args.use_train_set else 'test'} set")

            for t in range(args.test_times):
                obs, infos = env.reset()
                dones = [False] * cur_n
                chats = [[] for _ in range(cur_n)]
                saved = [False] * cur_n
                last_infos = infos
                succ = np.zeros(cur_n, dtype=bool)
                rew = np.zeros(cur_n, dtype=float)

                for step in range(args.max_steps):
                    logging.info(f"Batch {b+1} Step {step}; dones {np.sum(dones)}/{cur_n}")

                    prompts = []
                    idx_map = []
                    for i in range(cur_n):
                        if not dones[i]:
                            prompts.append(obs["text"][i])
                            idx_map.append(i)

                    actions = ["None"] * cur_n
                    raw = [None] * cur_n
                    if prompts:
                        batch_actions = agent.get_actions_batch(prompts, concurrency=args.concurrency, retries=args.retries)
                        for k, i in enumerate(idx_map):
                            actions[i] = batch_actions[k]
                            raw[i] = batch_actions[k]

                    prev_prompts = obs["text"]
                    obs, rewards, dones_vec, infos = env.step(actions)
                    last_infos = infos

                    for i in range(cur_n):
                        if not dones[i]:
                            chats[i].append({"role": "user", "content": prev_prompts[i]})
                            chats[i].append({"role": "assistant", "content": raw[i]})

                        rew[i] += rewards[i]
                        if dones_vec[i] and not dones[i]:
                            dones[i] = True
                            won = bool(infos[i].get("won", False))
                            succ[i] = won
                            logging.info(f"Env {i} finished @step {step}: won={won}, task_score={infos[i].get('task_score', 0):.3f}, total_reward={rew[i]:.3f}")

                            # Save per-episode chat history when an env finishes (short-hash filename)
                            if chat_base_dir and not saved[i]:
                                try:
                                    # Task text -> short hash for compact naming
                                    try:
                                        task_text = env.tasks[i]
                                    except Exception:
                                        task_text = "unknown"
                                    task_hash = hashlib.sha1(task_text.encode("utf-8")).hexdigest()[:8]

                                    unique_id = f"g{global_env_counter + i}-b{b:03d}_t{t:02d}_e{i:02d}-{task_hash}"
                                    out_path = os.path.join(chat_base_dir, f"chat_{unique_id}.json")

                                    meta = {
                                        "batch_idx": b,
                                        "test_idx": t,
                                        "env_id": global_env_counter + i,
                                        "model": args.model,
                                        "task": task_text,
                                        "steps": max(0, len(chats[i]) // 2),
                                        "won": bool(infos[i].get("won", False)),
                                        "task_score": float(infos[i].get("task_score", 0)),
                                        "timestamp": run_ts,
                                        "task_hash": task_hash,
                                    }
                                    with open(out_path, "w", encoding="utf-8") as f:
                                        json.dump({"messages": chats[i], "metadata": meta}, f, ensure_ascii=False, indent=2)
                                    saved[i] = True
                                    logging.info(f"Saved chat: {out_path}")
                                except Exception as e:
                                    logging.warning(f"Failed to save chat for env {i}: {e}")

                        # dump JSONL per acted env
                        if args.dump_path and i in idx_map:
                            try:
                                exec_act, valid = webshop_projection([raw[i]])
                                row = {
                                    "batch_idx": b,
                                    "test_idx": t,
                                    "step": step,
                                    "env_id": global_env_counter + i,
                                    "prompt": prev_prompts[i],
                                    "action": raw[i],
                                    "action_exec": exec_act[0] if exec_act else None,
                                    "is_action_valid": bool(valid[0]) if valid else None,
                                    "reward": float(rewards[i]),
                                    "done": bool(dones_vec[i]),
                                    "won": bool(infos[i].get("won", False)),
                                    "task_score": float(infos[i].get("task_score", 0)),
                                    "available_actions": infos[i].get("available_actions"),
                                }
                                dump_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                            except Exception as e:
                                logging.warning(f"Dump error: {e}")

                    if np.all(dones):
                        break

                all_succ.append(succ.mean())
                all_reward.append(rew.mean())
                logging.info(f"Batch {b+1} Test {t}: SR={succ.mean():.4f}, Reward={rew.mean():.4f}")

                # Save unfinished env chats (timeouts/failures) using short-hash filename
                if chat_base_dir:
                    for i in range(cur_n):
                        if not saved[i]:
                            try:
                                try:
                                    task_text = env.tasks[i]
                                except Exception:
                                    task_text = "unknown"
                                task_hash = hashlib.sha1(task_text.encode("utf-8")).hexdigest()[:8]

                                unique_id = f"g{global_env_counter + i}-b{b:03d}_t{t:02d}_e{i:02d}-{task_hash}"
                                out_path = os.path.join(chat_base_dir, f"chat_{unique_id}.json")

                                meta = {
                                    "batch_idx": b,
                                    "test_idx": t,
                                    "env_id": global_env_counter + i,
                                    "model": args.model,
                                    "task": task_text,
                                    "steps": max(0, len(chats[i]) // 2),
                                    "won": bool(last_infos[i].get("won", False)) if isinstance(last_infos, list) and i < len(last_infos) else False,
                                    "task_score": float(last_infos[i].get("task_score", 0)) if isinstance(last_infos, list) and i < len(last_infos) else 0.0,
                                    "timestamp": run_ts,
                                    "task_hash": task_hash,
                                }
                                with open(out_path, "w", encoding="utf-8") as f:
                                    json.dump({"messages": chats[i], "metadata": meta}, f, ensure_ascii=False, indent=2)
                                saved[i] = True
                                logging.info(f"Saved unfinished chat: {out_path}")
                            except Exception as e:
                                logging.warning(f"Failed to save unfinished chat for env {i}: {e}")

            global_env_counter += cur_n
            try:
                env.envs.close()
            except Exception:
                pass
            logging.info(f"== Finished Batch {b+1}/{num_batches}, processed {global_env_counter}/{total_envs} envs ==\n")

    finally:
        if dump_fp is not None:
            dump_fp.flush()
            dump_fp.close()
            logging.info(f"Trajectories saved to: {args.dump_path}")

    if all_succ:
        logging.info(f"Overall SR: {np.mean(all_succ):.4f} ± {np.std(all_succ):.4f}")
    if all_reward:
        logging.info(f"Overall Reward: {np.mean(all_reward):.4f} ± {np.std(all_reward):.4f}")
