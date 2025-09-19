#!/usr/bin/env python3
"""
Unified rollout script for AlfWorld, GAIA, and WebShop environments.
Provides a single interface for running rollouts across all three environments.
"""

import os
import time
import json
import logging
import argparse
from types import SimpleNamespace
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
import random
import hashlib
import sys
from openmanus_rl.environments.env_manager import *
from openai import OpenAI
from together import Together

try:
    import dotenv
    dotenv.load_dotenv()
except Exception:
    pass


class OpenManusAgent:
    """OpenManus agent that can work with all environments"""
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.7, 
                 base_url: str | None = None, env_type: str = "alfworld"):
        self.model_name = model_name
        self.temperature = temperature
        self.env_type = env_type
        
        # Determine which client to use based on model name and base_url
        # Use Together client only for models that explicitly look like Together models
        # (e.g., meta-llama/Llama-2-7b-chat-hf, Qwen/Qwen2.5-7B-Instruct-Turbo)
        together_providers = ['meta-llama/', 'Qwen/', 'mistralai/', 'NousResearch/', 'teknium/']
        self.is_together = any(model_name.startswith(provider) for provider in together_providers) and base_url is None
        
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
                api_key=os.environ.get('OPENAI_API_KEY'),
            )
        
        # Set environment-specific system prompts
        self.system_prompts = {
            "webshop": (
                "You are an expert web shopping agent. Respond strictly as "
                "<think>...</think><action>...</action>. The <action> must be a single "
                "admissible action exactly from the provided list, or a search[query]."
            ),
            "gaia": None,  # GAIA uses prompt templates in the environment
            "alfworld": None,  # AlfWorld uses prompt templates in the environment
        }
        
    def get_action_from_llm(self, obs: str) -> str:
        """Get action from LLM for a single observation"""
        messages = []
        
        # Add system prompt if available for this environment
        system_prompt = self.system_prompts.get(self.env_type)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": obs})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            n=1,
        )
        return response.choices[0].message.content.strip()
    
    def get_actions_batch(self, prompts: List[str], concurrency: int = 4, 
                         retries: int = 3, backoff: float = 0.5) -> List[str]:
        """Get actions for multiple observations in parallel"""
        actions = [None] * len(prompts)
        
        def _one(idx_prompt):
            idx, prompt = idx_prompt
            delay = backoff
            for attempt in range(retries):
                try:
                    act = self.get_action_from_llm(prompt)
                    return idx, act
                except Exception as e:
                    if attempt == retries - 1:
                        # Return a default action based on environment
                        default_actions = {
                            "webshop": "<think>error</think><action>search[product]</action>",
                            "gaia": "None",
                            "alfworld": "None"
                        }
                        return idx, default_actions.get(self.env_type, "None")
                    time.sleep(delay)
                    delay *= 2
        
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futures = [ex.submit(_one, (i, p)) for i, p in enumerate(prompts)]
            for fut in as_completed(futures):
                i, act = fut.result()
                actions[i] = act
        
        return actions


class EnvironmentFactory:
    """Factory for creating different environment types"""
    
    @staticmethod
    def build_env(env_type: str, **kwargs) -> Any:
        """Build environment based on type"""
        
        if env_type == "alfworld":
            return EnvironmentFactory._build_alfworld(**kwargs)
        elif env_type == "gaia":
            return EnvironmentFactory._build_gaia(**kwargs)
        elif env_type == "webshop":
            return EnvironmentFactory._build_webshop(**kwargs)
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")
    
    @staticmethod
    def _build_alfworld(env_num: int = 1, seed: int = 1, history_length: int = 2,
                       alf_env_type: str = "alfworld/AlfredTWEnv",
                       game_files: Optional[List[str]] = None,
                       use_summary: bool = False,
                       summary_api_key: Optional[str] = None,
                       summary_endpoint: Optional[str] = None,
                       summary_model: Optional[str] = "gpt-4o",
                       summary_concurrency: int = 8,
                       **kwargs):
        """Build AlfWorld environment"""
        from openmanus_rl.environments.env_package.alfworld import alfworld_projection
        from openmanus_rl.environments.env_package.alfworld import build_alfworld_envs
        
        alf_config_path = os.path.join(
            os.path.dirname(__file__), 
            '../../openmanus_rl/environments/env_package/alfworld/configs/config_tw.yaml'
        )
        
        envs = build_alfworld_envs(
            alf_config_path, 
            seed=seed, 
            env_num=env_num, 
            group_n=1, 
            is_train=True, 
            env_kwargs={}, 
            game_files=game_files
        )
        
        cfg = SimpleNamespace(
            env=SimpleNamespace(
                env_name=alf_env_type,
                history_length=history_length,
                use_summary=use_summary,
                summary_api_key=summary_api_key,
                summary_endpoint=summary_endpoint,
                summary_model=summary_model,
                summary_concurrency=summary_concurrency,
            )
        )
        
        return AlfWorldEnvironmentManager(envs, alfworld_projection, cfg)
    
    @staticmethod
    def _build_gaia(tasks_data: List[Dict], available_tools: List[str], 
                   env_num: int = 1, seed: int = 1, history_length: int = 2,
                   max_steps: int = 30, **kwargs):
        """Build GAIA/Tool Use environment"""

        from openmanus_rl.environments.env_package.tool_use.projection import tool_use_projection
        from openmanus_rl.environments.env_package.tool_use.envs import build_tool_use_envs
        from openmanus_rl.environments.env_package.tool_use.manager import ToolUseEnvironmentManager
        
        envs = build_tool_use_envs(
            tasks_data=tasks_data,
            available_tools=available_tools,
            seed=seed,
            env_num=env_num,
            group_n=1,
            is_train=True,
            model_string=kwargs.get('model', 'gpt-4o')
        )
        
        cfg = SimpleNamespace(
            env=SimpleNamespace(
                env_name="tool_use",
                history_length=history_length,
                max_steps=max_steps
            )
        )
        
        return ToolUseEnvironmentManager(envs, tool_use_projection, cfg)
    
    @staticmethod
    def _build_webshop(env_num: int = 1, seed: int = 1, history_length: int = 2,
                       use_train_set: bool = False,
                       use_summary: bool = False,
                       summary_api_key: Optional[str] = None,
                       summary_endpoint: Optional[str] = None,
                       summary_model: Optional[str] = "gpt-4o",
                       summary_concurrency: int = 8,
                       **kwargs):
        """Build WebShop environment"""
        from openmanus_rl.environments.env_package.webshop import build_webshop_envs, webshop_projection
        
        env_kwargs = {"observation_mode": "text"}
        
        envs = build_webshop_envs(
            seed=seed,
            env_num=env_num,
            group_n=1,
            is_train=use_train_set,
            env_kwargs=env_kwargs,
        )
        
        cfg = SimpleNamespace(
            env=SimpleNamespace(
                env_name="webshop/WebAgentTextEnv",
                history_length=history_length,
                use_summary=use_summary,
                summary_api_key=summary_api_key,
                summary_endpoint=summary_endpoint,
                summary_model=summary_model,
                summary_concurrency=summary_concurrency,
            )
        )
        
        return WebshopEnvironmentManager(envs, webshop_projection, cfg)


def load_gaia_tasks(data_path: str, max_tasks: Optional[int] = None) -> List[Dict]:
    """Load GAIA tasks from JSON file"""
    with open(data_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    if max_tasks:
        tasks = tasks[:max_tasks]
    
    return tasks


def prepare_alfworld_game_files(env_type: str, total_envs: int, seed: int) -> Optional[List[str]]:
    """Prepare unique game files for AlfWorld if requested"""
    if env_type != "alfworld":
        return None
        
    from openmanus_rl.environments.env_package.alfworld.envs import load_config_file
    from openmanus_rl.environments.env_package.alfworld.alfworld.agents.environment import get_environment
    
    alf_config_path = os.path.join(
        os.path.dirname(__file__),
        '../../openmanus_rl/environments/env_package/alfworld/configs/config_tw.yaml'
    )
    
    try:
        cfg = load_config_file(alf_config_path)
        env_type = cfg['env']['type']
        BaseEnvCls = get_environment(env_type)
        tmp_env = BaseEnvCls(cfg, train_eval='train')
        tmp_env.collect_game_files()
        all_game_files = list(getattr(tmp_env, 'game_files', []))
        
        if len(all_game_files) < total_envs:
            logging.error(f"Not enough game files: need {total_envs}, have {len(all_game_files)}")
            return None
            
        rng = random.Random(seed)
        rng.shuffle(all_game_files)
        return all_game_files[:total_envs]
        
    except Exception as e:
        logging.error(f"Failed to collect game files: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Unified rollout script for multiple environments")
    
    # Environment selection
    parser.add_argument("--env", choices=["alfworld", "gaia", "webshop"], required=True,
                       help="Environment to run")
    
    # Common parameters
    parser.add_argument("--batch_size", type=int, default=10, 
                       help="Number of envs to process per batch")
    parser.add_argument("--total_envs", type=int, default=100, 
                       help="Total number of environments to rollout")
    parser.add_argument("--test_times", type=int, default=1,
                       help="Number of test runs per batch")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Maximum steps per episode (default: 30)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--history_length", type=int, default=3)
    
    # Sharding parameters (workload splitting across processes)
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help=(
            "Total number of shards to split the workload into. "
            "Use with --shard_id to run disjoint subsets in parallel."
        ),
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help=(
            "Zero-based shard index for this process; must satisfy 0 <= shard_id < num_shards."
        ),
    )
    
    # Model parameters
    parser.add_argument("--model", default="gpt-4o",
                       help="Model name (OpenAI: gpt-4o, gpt-4o-mini; Together: Qwen/Qwen2.5-7B-Instruct-Turbo, etc.)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--base_url", default=None,
                       help="OpenAI-compatible base URL (e.g., vLLM http://127.0.0.1:8000/v1)")
    
    # Execution parameters
    parser.add_argument("--concurrency", type=int, default=4,
                       help="Max concurrent LLM requests per step")
    parser.add_argument("--retries", type=int, default=3,
                       help="Retries per request on failure")
    
    # Output parameters
    parser.add_argument("--dump_path", default=None,
                       help="If set, write JSONL trajectory to this file")
    parser.add_argument(
        "--chat_root",
        default=os.getcwd(),
        help=(
            "Root directory to save per-episode chat histories. "
            "Defaults to the current working directory."
        ),
    )
    
    # Environment-specific parameters
    parser.add_argument("--alf_env_type", default="alfworld/AlfredTWEnv",
                       help="AlfWorld environment type")
    parser.add_argument("--gaia_data_path", default="data/gaia/val.json",
                       help="Path to GAIA dataset")
    parser.add_argument("--gaia_tools", nargs='+', 
                       default=['google_search', 'wikipedia_knowledge_searcher', 'python_code_generator'],
                       help="List of available tools for GAIA")
    parser.add_argument("--webshop_train", action="store_true",
                       help="Use WebShop training set instead of test set")
    
    # Other options
    parser.add_argument("--unique_envs", action="store_true",
                       help="Ensure unique tasks/games across all environments")
    parser.add_argument("--dry_run", action="store_true",
                       help="Only print batch allocation without running")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    # Summary-related options  
    parser.add_argument("--use_summary", action="store_true",
                       help="Enable memory summarization instead of sliding window")
    parser.add_argument("--summary_api_key", default=None,
                       help="API key for summary LLM (defaults to environment variables)")
    parser.add_argument("--summary_endpoint", default=None, 
                       help="API endpoint for summary LLM (defaults to environment variables)")
    parser.add_argument("--summary_model", default="gpt-4o",
                       help="Model name for summarization (OpenAI), default: gpt-4o")
    parser.add_argument("--summary_concurrency", type=int, default=5,
                       help="Max parallel summary requests per step (default: 8)")
    
    args = parser.parse_args()
    
    # Set default max_steps based on environment
    if args.max_steps is None:
        args.max_steps = {
            "alfworld": 30,
            "gaia": 30,
            "webshop": 30
        }[args.env]
    
    # Setup logging
    os.makedirs(f"logs/{args.env}", exist_ok=True)
    log_fp = os.path.join(
        f"logs/{args.env}", 
        f"unified_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )
    
    logging.info(f"Starting unified rollout for {args.env}")
    logging.info(f"Model: {args.model}, Temperature: {args.temperature}")
    logging.info(f"Total envs: {args.total_envs}, Batch size: {args.batch_size}, Max steps: {args.max_steps}")
    
    # Validate sharding
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must be in [0, num_shards)")
    
    # Calculate number of batches (may be updated after task/game-file preparation & sharding)
    num_batches = (args.total_envs + args.batch_size - 1) // args.batch_size
    logging.info(f"Running {args.total_envs} envs in {num_batches} batches (pre-sharding)")
    
    # Prepare output files
    dump_fp = None
    if args.dump_path:
        os.makedirs(os.path.dirname(args.dump_path) or ".", exist_ok=True)
        dump_fp = open(args.dump_path, "a", encoding="utf-8")
        logging.info(f"Dumping trajectories to: {args.dump_path}")
    
    # Prepare chat history directories
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    chat_base_dir = None
    memory_base_dir = None
    if args.chat_root:
        chat_ts_root = os.path.join(args.chat_root, 'trajectories', run_ts)
        chat_base_dir = os.path.join(chat_ts_root, args.env, args.model.replace('/', '_'))
        os.makedirs(chat_base_dir, exist_ok=True)
        logging.info(f"Saving chats to: {chat_base_dir}")
        # Also prepare base directory for memory histories
        mem_ts_root = os.path.join(args.chat_root, 'histories', run_ts)
        memory_base_dir = os.path.join(mem_ts_root, args.env, args.model.replace('/', '_'))
        os.makedirs(memory_base_dir, exist_ok=True)
        logging.info(f"Saving memory histories to: {memory_base_dir}")
    
    def _sanitize(s: str) -> str:
        """Sanitize string for filename"""
        return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '-' for c in str(s))[:200]
    
    # Prepare environment-specific data
    gaia_tasks = None
    alfworld_game_files = None
    
    if args.env == "gaia":
        logging.info(f"Loading GAIA tasks from {args.gaia_data_path}")
        gaia_tasks = load_gaia_tasks(args.gaia_data_path)
        logging.info(f"Loaded {len(gaia_tasks)} tasks")
        
        if len(gaia_tasks) < args.total_envs:
            logging.warning(f"Only {len(gaia_tasks)} tasks available, adjusting total_envs")
            args.total_envs = len(gaia_tasks)
            num_batches = (args.total_envs + args.batch_size - 1) // args.batch_size
        
        # Shuffle tasks for random sampling
        rng = random.Random(args.seed)
        rng.shuffle(gaia_tasks)
        
        # Apply sharding after shuffle to ensure disjoint subsets across processes
        if args.num_shards > 1:
            indices = [i for i in range(len(gaia_tasks)) if i % args.num_shards == args.shard_id]
            gaia_tasks = [gaia_tasks[i] for i in indices]
            args.total_envs = len(gaia_tasks)
            num_batches = (args.total_envs + args.batch_size - 1) // args.batch_size
            logging.info(
                f"Applied sharding to GAIA tasks: shard_id={args.shard_id}/{args.num_shards}, "
                f"effective_total_envs={args.total_envs}, num_batches={num_batches}"
            )
        
    elif args.env == "alfworld" and args.unique_envs:
        alfworld_game_files = prepare_alfworld_game_files(args.env, args.total_envs, args.seed)
        if alfworld_game_files:
            logging.info(f"Prepared {len(alfworld_game_files)} unique game files")
            # Apply sharding to pre-collected game files
            if args.num_shards > 1:
                indices = [i for i in range(len(alfworld_game_files)) if i % args.num_shards == args.shard_id]
                alfworld_game_files = [alfworld_game_files[i] for i in indices]
                args.total_envs = len(alfworld_game_files)
                num_batches = (args.total_envs + args.batch_size - 1) // args.batch_size
                logging.info(
                    f"Applied sharding to AlfWorld files: shard_id={args.shard_id}/{args.num_shards}, "
                    f"effective_total_envs={args.total_envs}, num_batches={num_batches}"
                )
    
    # Dry run mode
    if args.dry_run:
        logging.info(f"[Dry-Run] Environment: {args.env}")
        logging.info(f"[Dry-Run] Total envs: {args.total_envs}, Batches: {num_batches}")
        
        for b in range(num_batches):
            start = b * args.batch_size
            end = min(start + args.batch_size, args.total_envs)
            batch_size = end - start
            
            if args.env == "gaia" and gaia_tasks:
                batch_tasks = gaia_tasks[start:end]
                pids = [t.get('pid', f'task_{i}') for i, t in enumerate(batch_tasks)]
                logging.info(f"[Dry-Run] Batch {b+1:02d}: {batch_size} tasks; PIDs: {', '.join(pids[:3])}...")
            elif args.env == "alfworld" and alfworld_game_files:
                batch_files = alfworld_game_files[start:end]
                examples = [os.path.basename(f) for f in batch_files[:3]]
                logging.info(f"[Dry-Run] Batch {b+1:02d}: {batch_size} files; Examples: {', '.join(examples)}")
            else:
                logging.info(f"[Dry-Run] Batch {b+1:02d}: {batch_size} environments")
        
        sys.exit(0)
    
    # Initialize agent (defer until after potential dry-run exit to avoid requiring API keys)
    agent = OpenManusAgent(
        model_name=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
        env_type=args.env
    )

    # Statistics tracking
    all_overall_success_rates = []
    all_task_success_history = defaultdict(list)
    global_env_counter = 0
    
    # Main rollout loop
    try:
        for batch_idx in range(num_batches):
            # Calculate actual batch size
            current_batch_size = min(args.batch_size, args.total_envs - batch_idx * args.batch_size)
            logging.info(f"\n========== Starting Batch {batch_idx + 1}/{num_batches} with {current_batch_size} envs ==========")
            
            # Prepare environment-specific kwargs
            env_kwargs = {
                "env_num": current_batch_size,
                "seed": args.seed + batch_idx,
                "history_length": args.history_length,
                # Summary configuration
                "use_summary": args.use_summary,
                "summary_api_key": args.summary_api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OAI_KEY"),
                "summary_endpoint": args.summary_endpoint or os.getenv("OPENAI_ENDPOINT") or os.getenv("OAI_ENDPOINT"),
                "summary_model": args.summary_model,
                "summary_concurrency": args.summary_concurrency,
            }
            
            if args.env == "gaia":
                start = batch_idx * args.batch_size
                end = start + current_batch_size
                env_kwargs["tasks_data"] = gaia_tasks[start:end]
                env_kwargs["available_tools"] = args.gaia_tools
                env_kwargs["max_steps"] = args.max_steps
                env_kwargs["model"] = args.model
                
            elif args.env == "alfworld":
                env_kwargs["alf_env_type"] = args.alf_env_type
                if alfworld_game_files:
                    start = batch_idx * args.batch_size
                    end = start + current_batch_size
                    env_kwargs["game_files"] = alfworld_game_files[start:end]
                    
            elif args.env == "webshop":
                env_kwargs["use_train_set"] = args.webshop_train
            
            # Create environment
            env_manager = EnvironmentFactory.build_env(args.env, **env_kwargs)
            
            # Batch-level statistics
            batch_overall_success_rates = []
            batch_task_success_history = defaultdict(list)
            
            try:
                # Test loop for this batch
                for test_idx in range(args.test_times):
                    logging.info(f"\n========== Start Batch {batch_idx + 1} Test {test_idx} ==========")
                    start_time = time.time()
                    
                    obs, infos = env_manager.reset()
                    env_dones = [False] * current_batch_size
                    
                    # Per-env chat buffers
                    chats = [[] for _ in range(current_batch_size)]
                    saved_flags = [False] * current_batch_size
                    last_infos = infos
                    
                    # Statistics for single round
                    overall_success_this_round = np.zeros(current_batch_size, dtype=bool)
                    task_success_cnt = defaultdict(int)
                    task_total_cnt = defaultdict(int)
                    
                    for step_idx in range(args.max_steps):
                        # Log both overall SR (successes over all envs) and SR among finished envs.
                        dones_cnt = int(np.array(env_dones).sum())
                        succ_cnt = int(overall_success_this_round.sum())
                        sr_overall = float(succ_cnt / current_batch_size)
                        sr_done = float(succ_cnt / dones_cnt) if dones_cnt > 0 else 0.0
                        logging.info(
                            f"Batch {batch_idx + 1} Step {step_idx}; "
                            f"Dones ({dones_cnt}/{current_batch_size}); "
                            f"SR_overall {sr_overall:.3f}; SR_done {sr_done:.3f}"
                        )
                        
                        # Assemble actions
                        prompts = []
                        idx_map = []
                        for i in range(current_batch_size):
                            if not env_dones[i]:
                                prompts.append(obs["text"][i])
                                idx_map.append(i)
                        
                        if not prompts:
                            break
                        
                        batch_actions = agent.get_actions_batch(
                            prompts, 
                            concurrency=args.concurrency, 
                            retries=args.retries
                        )
                        
                        actions = ["None"] * current_batch_size
                        for k, i in enumerate(idx_map):
                            actions[i] = batch_actions[k]
                        
                        # Environment stepping
                        prev_prompts = obs["text"]
                        raw_actions = actions.copy()
                        obs, rewards, dones, infos = env_manager.step(actions.copy())
                        last_infos = infos
                        
                        # Process results
                        for i in range(current_batch_size):
                            if env_dones[i]:
                                continue
                            
                            # Append chat history
                            if prev_prompts and i < len(prev_prompts):
                                chats[i].append({"role": "user", "content": prev_prompts[i]})
                            chats[i].append({"role": "assistant", "content": raw_actions[i]})
                            
                            # Dump trajectory
                            if args.dump_path and (i in idx_map):
                                try:
                                    row = {
                                        "batch_idx": batch_idx,
                                        "test_idx": test_idx,
                                        "step": step_idx,
                                        "env_id": global_env_counter + i,
                                        "prompt": prev_prompts[i],
                                        "action": raw_actions[i],
                                        "reward": float(rewards[i]) if i < len(rewards) else None,
                                        "done": bool(dones[i]) if i < len(dones) else None,
                                        "won": bool(infos[i].get("won", False)),
                                        "is_action_valid": bool(infos[i].get("is_action_valid", False)),
                                    }
                                    
                                    # Add environment-specific fields
                                    if args.env == "gaia":
                                        row["pid"] = infos[i].get("pid", "unknown")
                                    elif args.env == "alfworld":
                                        row["gamefile"] = infos[i].get("extra.gamefile", "")
                                    elif args.env == "webshop":
                                        row["task_score"] = float(infos[i].get("task_score", 0))
                                    
                                    dump_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                                except Exception as e:
                                    logging.debug(f"Dump error: {e}")
                            
                            # Check if done
                            if dones[i]:
                                env_dones[i] = True
                                won = bool(infos[i].get("won", False))
                                overall_success_this_round[i] = won
                                
                                # Track task success
                                if args.env == "gaia":
                                    task_id = infos[i].get("pid", f"task_{i}")
                                elif args.env == "alfworld":
                                    gamefile = infos[i].get("extra.gamefile", "")
                                    # Extract task type from gamefile
                                    task_types = ["pick_and_place", "pick_two_obj_and_place", 
                                                 "look_at_obj_in_light", "pick_heat_then_place_in_recep",
                                                 "pick_cool_then_place_in_recep", "pick_clean_then_place_in_recep"]
                                    task_id = "other"
                                    for t in task_types:
                                        if t in gamefile:
                                            task_id = t
                                            break
                                else:  # webshop
                                    task_id = f"task_{i}"
                                
                                task_total_cnt[task_id] = 1
                                if won:
                                    task_success_cnt[task_id] = 1
                                
                                # Save chat history
                                if chat_base_dir and not saved_flags[i]:
                                    try:
                                        task_hash = hashlib.sha1(str(task_id).encode()).hexdigest()[:8]
                                        unique_id = f"b{batch_idx:03d}_t{test_idx:02d}_e{i:02d}-{task_hash}"
                                        out_path = os.path.join(chat_base_dir, f"chat_{unique_id}.json")
                                        
                                        meta = {
                                            "batch_idx": batch_idx,
                                            "env_id": global_env_counter + i,
                                            "test_idx": test_idx,
                                            "model": args.model,
                                            "steps": step_idx + 1,
                                            "won": won,
                                            "timestamp": run_ts,
                                            "environment": args.env,
                                        }
                                        
                                        # Add environment-specific task identifiers
                                        if args.env == "alfworld":
                                            meta["gamefile"] = infos[i].get("extra.gamefile", "")
                                        elif args.env == "gaia":
                                            meta["pid"] = infos[i].get("pid", "unknown")
                                        
                                        with open(out_path, "w", encoding="utf-8") as f:
                                            json.dump({"messages": chats[i], "metadata": meta}, f, ensure_ascii=False, indent=2)
                                        saved_flags[i] = True
                                    except Exception as e:
                                        logging.debug(f"Failed to save chat: {e}")

                                # Save memory history aligned with this episode
                                if memory_base_dir:
                                    try:
                                        task_hash = hashlib.sha1(str(task_id).encode()).hexdigest()[:8]
                                        unique_id = f"b{batch_idx:03d}_t{test_idx:02d}_e{i:02d}-{task_hash}"
                                        mem_out_path = os.path.join(memory_base_dir, f"memory_{unique_id}.json")

                                        # Extract raw memory records from the environment manager.
                                        records = []
                                        try:
                                            env_memory = getattr(env_manager, 'memory', None)
                                            env_mem_data = getattr(env_memory, '_data', None)
                                            if env_mem_data is not None and i < len(env_mem_data):
                                                for step_no, rec in enumerate(env_mem_data[i], start=1):
                                                    step_rec = {
                                                        'step': step_no,
                                                        'text_obs': rec.get('text_obs'),
                                                        'action': rec.get('action'),
                                                    }
                                                    if 'reflection' in rec:
                                                        step_rec['reflection'] = rec.get('reflection')
                                                    records.append(step_rec)
                                        except Exception as mem_e:
                                            logging.debug(f"Memory extraction failed: {mem_e}")

                                        # Include cached summary if SummarizedMemory is used
                                        summary_text = None
                                        try:
                                            if env_memory is not None:
                                                summaries = getattr(env_memory, 'summaries', None)
                                                if summaries is not None and i < len(summaries):
                                                    summary_text = summaries[i]
                                        except Exception:
                                            pass

                                        mem_meta = {
                                            "batch_idx": batch_idx,
                                            "env_id": global_env_counter + i,
                                            "test_idx": test_idx,
                                            "model": args.model,
                                            "steps": len(records),
                                            "won": won,
                                            "timestamp": run_ts,
                                            "environment": args.env,
                                        }
                                        if args.env == "alfworld":
                                            mem_meta["gamefile"] = infos[i].get("extra.gamefile", "")
                                        elif args.env == "gaia":
                                            mem_meta["pid"] = infos[i].get("pid", "unknown")

                                        with open(mem_out_path, "w", encoding="utf-8") as f:
                                            json.dump({"memory": records, "summary": summary_text, "metadata": mem_meta}, f, ensure_ascii=False, indent=2)
                                    except Exception as e:
                                        logging.debug(f"Failed to save memory: {e}")
                        
                        if all(env_dones):
                            logging.info("All environments finished early!")
                            break
                    
                    # Save any unfinished chats
                    if chat_base_dir:
                        for i in range(current_batch_size):
                            if not saved_flags[i]:
                                try:
                                    task_hash = hashlib.sha1(f"unfinished_{i}".encode()).hexdigest()[:8]
                                    unique_id = f"b{batch_idx:03d}_t{test_idx:02d}_e{i:02d}-{task_hash}"
                                    out_path = os.path.join(chat_base_dir, f"chat_{unique_id}.json")
                                    
                                    meta = {
                                        "batch_idx": batch_idx,
                                        "env_id": global_env_counter + i,
                                        "test_idx": test_idx,
                                        "model": args.model,
                                        "steps": len(chats[i]) // 2,
                                        "won": False,
                                        "timestamp": run_ts,
                                        "environment": args.env,
                                    }
                                    
                                    # Add environment-specific task identifiers for unfinished tasks
                                    if last_infos and i < len(last_infos):
                                        if args.env == "alfworld":
                                            meta["gamefile"] = last_infos[i].get("extra.gamefile", "")
                                        elif args.env == "gaia":
                                            meta["pid"] = last_infos[i].get("pid", "unknown")
                                    
                                    with open(out_path, "w", encoding="utf-8") as f:
                                        json.dump({"messages": chats[i], "metadata": meta}, f, ensure_ascii=False, indent=2)
                                    saved_flags[i] = True
                                except Exception as e:
                                    logging.debug(f"Failed to save unfinished chat: {e}")

                    # Save unfinished memory for envs that didn't finish
                    if memory_base_dir:
                        for i in range(current_batch_size):
                            if not env_dones[i]:
                                try:
                                    task_hash = hashlib.sha1(f"unfinished_{i}".encode()).hexdigest()[:8]
                                    unique_id = f"b{batch_idx:03d}_t{test_idx:02d}_e{i:02d}-{task_hash}"
                                    mem_out_path = os.path.join(memory_base_dir, f"memory_{unique_id}.json")

                                    records = []
                                    try:
                                        env_memory = getattr(env_manager, 'memory', None)
                                        env_mem_data = getattr(env_memory, '_data', None)
                                        if env_mem_data is not None and i < len(env_mem_data):
                                            for step_no, rec in enumerate(env_mem_data[i], start=1):
                                                step_rec = {
                                                    'step': step_no,
                                                    'text_obs': rec.get('text_obs'),
                                                    'action': rec.get('action'),
                                                }
                                                if 'reflection' in rec:
                                                    step_rec['reflection'] = rec.get('reflection')
                                                records.append(step_rec)
                                    except Exception as mem_e:
                                        logging.debug(f"Memory extraction failed: {mem_e}")

                                    summary_text = None
                                    try:
                                        if env_memory is not None:
                                            summaries = getattr(env_memory, 'summaries', None)
                                            if summaries is not None and i < len(summaries):
                                                summary_text = summaries[i]
                                    except Exception:
                                        pass

                                    mem_meta = {
                                        "batch_idx": batch_idx,
                                        "env_id": global_env_counter + i,
                                        "test_idx": test_idx,
                                        "model": args.model,
                                        "steps": len(records),
                                        "won": False,
                                        "timestamp": run_ts,
                                        "environment": args.env,
                                    }
                                    if last_infos and i < len(last_infos):
                                        if args.env == "alfworld":
                                            mem_meta["gamefile"] = last_infos[i].get("extra.gamefile", "")
                                        elif args.env == "gaia":
                                            mem_meta["pid"] = last_infos[i].get("pid", "unknown")

                                    with open(mem_out_path, "w", encoding="utf-8") as f:
                                        json.dump({"memory": records, "summary": summary_text, "metadata": mem_meta}, f, ensure_ascii=False, indent=2)
                                except Exception as e:
                                    logging.debug(f"Failed to save unfinished memory: {e}")
                    
                    # Round statistics
                    round_success_rate = overall_success_this_round.mean()
                    batch_overall_success_rates.append(round_success_rate)
                    
                    logging.info(f"Batch {batch_idx + 1} Test {test_idx} overall success: {round_success_rate:.4f}")
                    
                    # Calculate and store per-task success rates for this test
                    for task, total in task_total_cnt.items():
                        if total > 0:
                            rate = task_success_cnt.get(task, 0) / total
                            batch_task_success_history[task].append(rate)
                            
                            # Log task-specific results for alfworld
                            if args.env == "alfworld":
                                logging.info(f"    {task:<35s}: {rate:.4f} ({task_success_cnt.get(task, 0)}/{task_total_cnt[task]})")
                    
                    logging.info(f"Batch {batch_idx + 1} Test {test_idx} time elapsed: {time.time() - start_time:.2f}s\n")
                
            finally:
                # Accumulate batch results
                all_overall_success_rates.extend(batch_overall_success_rates)
                for task, rates in batch_task_success_history.items():
                    all_task_success_history[task].extend(rates)
                
                # Update global counter
                global_env_counter += current_batch_size
                
                # Clean up resources
                try:
                    env_manager.envs.close()
                    logging.info(f"Released resources for Batch {batch_idx + 1}")
                except Exception as e:
                    logging.warning(f"Failed to release resources: {e}")
                
                logging.info(f"========== Finished Batch {batch_idx + 1}/{num_batches}, processed {global_env_counter}/{args.total_envs} envs ==========\n")
        
    finally:
        if dump_fp is not None:
            dump_fp.flush()
            dump_fp.close()
            logging.info(f"Trajectories saved to: {args.dump_path}")
    
    # Final summary
    logging.info("=============== Final Summary ===============")
    logging.info(f"Environment: {args.env}")
    logging.info(f"Total batches: {num_batches} | Batch size: {args.batch_size} | Total envs processed: {global_env_counter}")
    
    # Echo save locations to make it easy to find outputs.
    if args.dump_path:
        logging.info(f"Trajectory file: {args.dump_path}")
    if chat_base_dir:
        logging.info(f"Chats directory: {chat_base_dir}")
    if memory_base_dir:
        logging.info(f"Memory directory: {memory_base_dir}")
    
    if all_overall_success_rates:
        logging.info(
            f"Overall success avg ± std: "
            f"{np.mean(all_overall_success_rates):.4f} ± {np.std(all_overall_success_rates):.4f}"
        )
    
    # Environment-specific summaries
    if args.env == "alfworld":
        task_types = ["pick_and_place", "pick_two_obj_and_place", "look_at_obj_in_light",
                     "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", 
                     "pick_clean_then_place_in_recep", "other"]
        for task in task_types:
            if task in all_task_success_history and all_task_success_history[task]:
                rates = [r for r in all_task_success_history[task] if r is not None]
                if rates:
                    logging.info(f"{task:<35s}: {np.mean(rates):.4f} ± {np.std(rates):.4f}")
    
    elif args.env == "gaia":
        successful_tasks = sum(1 for rates in all_task_success_history.values() if any(r > 0 for r in rates))
        logging.info(f"Successfully completed {successful_tasks} out of {len(all_task_success_history)} unique tasks")


if __name__ == "__main__":
    main()
