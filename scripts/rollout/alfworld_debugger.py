import os
import time
import json
import logging
import argparse
from types import SimpleNamespace
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
import random
import sys
from openmanus_rl.environments.env_manager import *
from openai import OpenAI
from together import Together
from openmanus_rl.environments.env_package.alfworld.envs import load_config_file
from openmanus_rl.environments.env_package.alfworld.alfworld.agents.environment import get_environment

def build_env(env_name, env_num=1, seed=1, history_length=2, alf_env_type="alfworld/AlfredTWEnv", game_files=None):
    group_n = 1
    if env_name == "alfworld":
        # Test AlfWorldEnvironmentManager
        from openmanus_rl.environments.env_package.alfworld import alfworld_projection
        from openmanus_rl.environments.env_package.alfworld import build_alfworld_envs
        alf_config_path = os.path.join(os.path.dirname(__file__), '../../agent_system/environments/env_package/alfworld/configs/config_tw.yaml')
        envs = build_alfworld_envs(alf_config_path, seed=seed, env_num=env_num, group_n=group_n, is_train=True, env_kwargs={}, game_files=game_files)
        # Minimal config object with required fields
        cfg = SimpleNamespace(env=SimpleNamespace(env_name=alf_env_type, history_length=history_length))
        env_manager = ExtendedAlfWorldEnvironmentManager(envs, alfworld_projection, cfg)
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")

    return env_manager


class ExtendedAlfWorldEnvironmentManager(AlfWorldEnvironmentManager):
    """Extended environment manager with single-environment step and reset methods"""
    
    def reset_single(self, env_id: int):
        """Reset a single environment and return its observation"""
        # Store current states
        stored_obs = self.last_obs if hasattr(self, 'last_obs') else None
        stored_infos = self.last_infos if hasattr(self, 'last_infos') else None
        
        # Reset all environments (limitation of current implementation)
        obs, infos = self.reset()
        
        # Store for future use
        self.last_obs = obs
        self.last_infos = infos
        
        return obs, infos
    
    def step_single(self, env_id: int, action: str):
        """Step a single environment with the given action"""
        # Create action list with None for other environments
        actions = ["None"] * self.env_num
        actions[env_id] = action
        
        # Step all environments
        obs, rewards, dones, infos = self.step(actions)
        
        # Store for future use
        self.last_obs = obs
        self.last_infos = infos
        
        return obs, rewards, dones, infos

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


class LLMDebugger:
    """LLM-based debugger that analyzes failed trajectories and provides feedback"""
    
    def __init__(self, model_name="gpt-4o", temperature: float = 0.3, base_url: str | None = None):
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize OpenAI client
        if base_url:
            self.client = OpenAI(
                api_key=os.getenv('OPENAI_API_KEY', 'EMPTY'),
                base_url=base_url,
            )
        else:
            self.client = OpenAI(
                api_key=os.environ.get('OPENAI_API_KEY', ''),
            )
    
    def analyze_trajectory(self, trajectory: List[Dict]) -> Dict:
        """
        Analyze a failed trajectory and identify the failure point and reason
        
        Args:
            trajectory: List of trajectory steps, each containing:
                - step: int
                - observation: str
                - action: str
                - reward: float
                - done: bool
                - won: bool
        
        Returns:
            Analysis dict containing:
                - failure_step: int - the step where the failure occurred
                - failure_type: str - type of failure (e.g., "wrong_object", "invalid_action", etc.)
                - reason: str - detailed reason for failure
                - suggestion: str - suggestion for fixing the issue
                - critical_step: int - the last correct step before failure
        """
        
        # Format trajectory for LLM analysis
        trajectory_text = self._format_trajectory(trajectory)
        
        # Create analysis prompt
        prompt = f"""You are an expert debugger for an AI agent playing text-based games. Analyze the following failed trajectory and identify where and why the agent failed.

TRAJECTORY:
{trajectory_text}

TASK CONTEXT:
The agent is trying to complete household tasks in a text-based environment. Common tasks include:
- pick_and_place: Pick up an object and place it somewhere
- pick_two_obj_and_place: Pick up two objects and place them
- look_at_obj_in_light: Examine an object under a light source
- pick_heat_then_place_in_recep: Heat an object and place it in a receptacle
- pick_cool_then_place_in_recep: Cool an object and place it in a receptacle
- pick_clean_then_place_in_recep: Clean an object and place it in a receptacle

ANALYSIS REQUIRED:
1. Identify the EXACT step where the agent made a critical error
2. Determine the type of failure (e.g., wrong object selection, invalid action sequence, missed step, wrong location)
3. Explain WHY this was an error in the context of the task
4. Suggest a specific correction for that step

Please provide your analysis in the following JSON format:
{{
    "failure_step": <int: the step number where the critical error occurred>,
    "failure_type": "<string: one of 'wrong_object', 'invalid_action', 'wrong_location', 'missed_step', 'wrong_sequence', 'exploration_failure'>",
    "reason": "<string: detailed explanation of why this was an error>",
    "suggestion": "<string: specific action or approach the agent should take instead>",
    "critical_step": <int: the last step that was definitely correct before the error>
}}

IMPORTANT: Be precise about the failure step. Look for actions that:
- Target the wrong object for the task
- Go to wrong locations
- Miss required intermediate steps (like heating/cooling/cleaning)
- Use invalid or nonsensical commands
- Show the agent is lost or confused

Output only the JSON, no additional text."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            logging.error(f"Failed to analyze trajectory: {e}")
            # Return a default analysis on error
            return {
                "failure_step": len(trajectory) - 1,
                "failure_type": "unknown",
                "reason": "Failed to analyze trajectory",
                "suggestion": "Try a different approach",
                "critical_step": 0
            }
    
    def generate_feedback(self, observation: str, analysis: Dict, previous_action: str) -> str:
        """
        Generate feedback to inject into the agent's next action
        
        Args:
            observation: Current observation from the environment
            analysis: Analysis dict from analyze_trajectory
            previous_action: The action that led to failure
        
        Returns:
            Feedback string to prepend to the agent's next prompt
        """
        
        feedback_prompt = f"""Based on a previous failed attempt, generate helpful feedback for the agent.

CURRENT OBSERVATION:
{observation}

PREVIOUS FAILED ACTION:
{previous_action}

FAILURE ANALYSIS:
- Type: {analysis['failure_type']}
- Reason: {analysis['reason']}
- Suggestion: {analysis['suggestion']}

Generate a concise, actionable feedback message (2-3 sentences max) that:
1. Warns the agent about the previous mistake
2. Suggests the correct approach
3. Is formatted as a hint or reminder

Output only the feedback message, nothing else."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": feedback_prompt}],
                temperature=self.temperature
            )
            
            feedback = response.choices[0].message.content.strip()
            return f"\n[DEBUGGER FEEDBACK: {feedback}]\n"
            
        except Exception as e:
            logging.error(f"Failed to generate feedback: {e}")
            return f"\n[DEBUGGER FEEDBACK: Previous attempt failed. {analysis.get('suggestion', 'Try a different approach.')}]\n"
    
    def _format_trajectory(self, trajectory: List[Dict]) -> str:
        """Format trajectory for LLM analysis"""
        lines = []
        for step in trajectory:
            lines.append(f"Step {step['step']}:")
            lines.append(f"  Observation: {step['observation']}")
            lines.append(f"  Action: {step['action']}")
            if step.get('reward') is not None:
                lines.append(f"  Reward: {step['reward']}")
            if step.get('done'):
                lines.append(f"  Done: {step['done']}, Won: {step.get('won', False)}")
            lines.append("")
        return "\n".join(lines)


class TrajectoryManager:
    """Manages trajectory storage and replay"""
    
    def __init__(self):
        self.trajectories = {}  # env_id -> list of trajectory steps
        self.checkpoints = {}   # env_id -> list of environment checkpoints
    
    def reset(self, env_id: int):
        """Reset trajectory for an environment"""
        self.trajectories[env_id] = []
        self.checkpoints[env_id] = []
    
    def add_step(self, env_id: int, step_data: Dict):
        """Add a step to the trajectory"""
        if env_id not in self.trajectories:
            self.trajectories[env_id] = []
        self.trajectories[env_id].append(step_data)
    
    def get_trajectory(self, env_id: int) -> List[Dict]:
        """Get the full trajectory for an environment"""
        return self.trajectories.get(env_id, [])
    
    def get_replay_point(self, env_id: int, target_step: int) -> Tuple[List[Dict], int]:
        """
        Get trajectory up to a certain step for replay
        
        Returns:
            Tuple of (trajectory_up_to_step, actual_step_reached)
        """
        trajectory = self.trajectories.get(env_id, [])
        if target_step >= len(trajectory):
            return trajectory, len(trajectory) - 1
        return trajectory[:target_step + 1], target_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="alfworld")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of envs to process per batch")
    parser.add_argument("--total_envs", type=int, default=1000, help="Total number of environments to rollout")
    parser.add_argument("--test_times", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--history_length", type=int, default=2)
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name (OpenAI: gpt-4o, gpt-4o-mini; Together: Qwen/Qwen2.5-7B-Instruct-Turbo, etc.)")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent OpenAI requests per step")
    parser.add_argument("--retries", type=int, default=3, help="Retries per request on failure")
    parser.add_argument("--dump_path", default=None, help="If set, write JSONL trajectory to this file")
    parser.add_argument("--base_url", default=None, help="OpenAI-compatible base URL (e.g., vLLM http://127.0.0.1:8000/v1)")
    parser.add_argument("--chat_root", default=None, help="If set, save per-episode chat histories under this root: trajectories/react/<model>/<timestamp>/chat_histories")
    parser.add_argument("--alf_env_type", default="alfworld/AlfredTWEnv", help="alfworld/AlfredTWEnv or alfworld/AlfredThorEnv")
    parser.add_argument("--unique_envs", action="store_true", help="确保每个环境使用唯一的游戏文件（无重复采样）")
    parser.add_argument("--dry_run", action="store_true", help="仅打印唯一任务的批次分配，不创建环境、不调用模型")
    
    # Debugging related arguments
    parser.add_argument("--enable_debugger", action="store_true", help="Enable LLM debugger for failed trajectories")
    parser.add_argument("--max_retries", type=int, default=5, help="Maximum number of retry attempts with debugger feedback")
    parser.add_argument("--debugger_model", default="gpt-4o", help="Model to use for trajectory debugging")
    parser.add_argument("--debugger_temperature", type=float, default=0.3, help="Temperature for debugger model")
    parser.add_argument("--debug_output_dir", default=None, help="Directory to save debug analysis results")
    
    args = parser.parse_args()

    # -------- logging ----------
    os.makedirs("logs/alfworld", exist_ok=True)
    log_fp = os.path.join(
        "logs/alfworld", f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    
    # Calculate number of batches needed
    num_batches = (total_envs + batch_size - 1) // batch_size
    logging.info(f"Running {total_envs} envs in {num_batches} batches of {batch_size}") 

    # Keywords for 6 subtasks
    TASKS = [
        "pick_and_place",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_clean_then_place_in_recep",
    ]

    # -------- Agent setup ----------
    agent = Agent(model_name=args.model, temperature=args.temperature, base_url=args.base_url)
    
    # -------- Initialize debugger and trajectory manager if enabled ----------
    debugger = None
    trajectory_manager = None
    if args.enable_debugger:
        debugger = LLMDebugger(
            model_name=args.debugger_model,
            temperature=args.debugger_temperature,
            base_url=args.base_url
        )
        trajectory_manager = TrajectoryManager()
        logging.info(f"Debugger enabled with model {args.debugger_model}, max retries: {args.max_retries}")
        
        # Create debug output directory if specified
        if args.debug_output_dir:
            os.makedirs(args.debug_output_dir, exist_ok=True)

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
        # <chat_root>/trajectories/<timestamp>/<env>/<model>/
        chat_ts_root = os.path.join(args.chat_root, 'trajectories', run_ts)
        chat_base_dir = os.path.join(chat_ts_root, args.env_name, args.model)
        os.makedirs(chat_base_dir, exist_ok=True)

def _sanitize(s: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '-' for c in s)[:200]


def run_environment_with_retry(
    env_id: int,
    env_manager,
    agent: Agent,
    max_steps: int,
    debugger: Optional[LLMDebugger] = None,
    trajectory_manager: Optional[TrajectoryManager] = None,
    max_retries: int = 5,
    dump_fp=None,
    chat_base_dir: str = None,
    batch_idx: int = 0,
    test_idx: int = 0,
    global_env_counter: int = 0,
    run_ts: str = "",
    debug_output_dir: str = None
) -> Dict:
    """
    Run a single environment with retry logic using debugger feedback
    
    Returns:
        Dict containing results and statistics for this environment
    """
    
    best_trajectory = None
    best_reward = -float('inf')
    won = False
    final_info = {}
    
    for retry_idx in range(max_retries):
        logging.info(f"  Env {env_id} - Attempt {retry_idx + 1}/{max_retries}")
        
        # Reset trajectory manager for this attempt
        if trajectory_manager:
            trajectory_manager.reset(env_id)
        
        # Initialize tracking variables
        env_done = False
        chat_history = []
        trajectory_steps = []
        cumulative_reward = 0
        
        # Variables for replay
        replay_to_step = -1
        debugger_feedback = ""
        
        # If this is a retry, we need to replay to the critical step
        if retry_idx > 0 and debugger and best_trajectory:
            # Analyze the failed trajectory
            analysis = debugger.analyze_trajectory(best_trajectory)
            
            # Save debug analysis if output dir specified
            if debug_output_dir:
                debug_file = os.path.join(
                    debug_output_dir,
                    f"debug_b{batch_idx}_t{test_idx}_e{env_id}_r{retry_idx}.json"
                )
                with open(debug_file, "w") as f:
                    json.dump({
                        "retry": retry_idx,
                        "analysis": analysis,
                        "trajectory": best_trajectory
                    }, f, indent=2)
            
            logging.info(f"    Debugger analysis - Failure at step {analysis['failure_step']}: {analysis['failure_type']}")
            logging.info(f"    Suggestion: {analysis['suggestion']}")
            
            replay_to_step = analysis.get('critical_step', 0)
            
        # Get initial observation
        obs_dict, info_dict = env_manager.reset_single(env_id)
        obs = obs_dict["text"][env_id]
        info = info_dict[env_id] if isinstance(info_dict, dict) else info_dict
        
        for step_idx in range(max_steps):
            if env_done:
                break
                
            # Check if we're replaying previous trajectory
            if replay_to_step >= 0 and step_idx <= replay_to_step and best_trajectory:
                # Replay action from previous trajectory
                action = best_trajectory[step_idx]['action'] if step_idx < len(best_trajectory) else "None"
                logging.debug(f"    Replaying step {step_idx}: {action}")
            else:
                # Get new action from agent
                prompt = obs
                
                # Add debugger feedback at the critical point
                if replay_to_step >= 0 and step_idx == replay_to_step + 1 and debugger:
                    # Generate feedback for this specific step
                    prev_action = best_trajectory[step_idx]['action'] if step_idx < len(best_trajectory) else ""
                    debugger_feedback = debugger.generate_feedback(obs, analysis, prev_action)
                    prompt = debugger_feedback + obs
                    logging.info(f"    Injecting debugger feedback at step {step_idx}")
                
                action = agent.get_action_from_gpt(prompt)
            
            # Store raw action for trajectory
            raw_action = action
            
            # Step environment
            obs_dict, reward_dict, done_dict, info_dict = env_manager.step_single(env_id, action)
            
            obs = obs_dict["text"][env_id]
            reward = reward_dict[env_id]
            done = done_dict[env_id]
            info = info_dict[env_id] if isinstance(info_dict, dict) else info_dict
            
            cumulative_reward += reward
            
            # Store trajectory step
            trajectory_step = {
                "step": step_idx,
                "observation": obs,
                "action": raw_action,
                "reward": float(reward),
                "done": bool(done),
                "won": bool(info.get("won", False))
            }
            trajectory_steps.append(trajectory_step)
            
            if trajectory_manager:
                trajectory_manager.add_step(env_id, trajectory_step)
            
            # Update chat history
            chat_history.append({"role": "user", "content": prompt})
            chat_history.append({"role": "assistant", "content": raw_action})
            
            # Write to dump file if specified
            if dump_fp:
                try:
                    row = {
                        "batch_idx": batch_idx,
                        "test_idx": test_idx,
                        "retry_idx": retry_idx,
                        "step": step_idx,
                        "env_id": global_env_counter + env_id,
                        "prompt": prompt,
                        "action": raw_action,
                        "reward": float(reward),
                        "done": bool(done),
                        "won": bool(info.get("won", False)),
                        "gamefile": info.get("extra.gamefile"),
                        "is_action_valid": bool(info.get("is_action_valid", False)),
                    }
                    dump_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                    dump_fp.flush()
                except Exception as e:
                    logging.error(f"Failed to write trajectory: {e}")
            
            if done:
                env_done = True
                won = bool(info.get("won", False))
                final_info = info
                break
        
        # Check if this attempt was successful
        if won:
            logging.info(f"  Env {env_id} - SUCCESS on attempt {retry_idx + 1}")
            best_trajectory = trajectory_steps
            best_reward = cumulative_reward
            break
        
        # Update best trajectory if this one is better
        if cumulative_reward > best_reward:
            best_trajectory = trajectory_steps
            best_reward = cumulative_reward
        
        # If debugger is not enabled, don't retry
        if not debugger:
            break
    
    # Save final chat history if requested
    if chat_base_dir:
        try:
            task = final_info.get("extra.gamefile", "unknown")
            task_dir = os.path.join(chat_base_dir, _sanitize(task))
            os.makedirs(task_dir, exist_ok=True)
            unique_id = f"b{batch_idx:03d}_t{test_idx:02d}_e{env_id:02d}_final"
            base = f"chat_{_sanitize(task)}-{unique_id}"
            out_path = os.path.join(task_dir, base + ".json")
            
            meta = {
                "batch_idx": batch_idx,
                "env_id": global_env_counter + env_id,
                "test_idx": test_idx,
                "model": agent.model_name,
                "task": task,
                "gamefile": final_info.get("extra.gamefile"),
                "steps": len(best_trajectory) if best_trajectory else 0,
                "won": won,
                "retries": retry_idx + 1,
                "timestamp": run_ts,
            }
            
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "messages": chat_history,
                    "metadata": meta,
                    "best_trajectory": best_trajectory
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Failed to save chat history: {e}")
    
    return {
        "env_id": env_id,
        "won": won,
        "reward": best_reward,
        "retries": retry_idx + 1,
        "steps": len(best_trajectory) if best_trajectory else 0,
        "gamefile": final_info.get("extra.gamefile", ""),
        "trajectory": best_trajectory
    }

    # Helper: collect all train game files
    def collect_all_game_files(alf_config_path, is_train=True, eval_dataset='eval_in_distribution'):
        cfg = load_config_file(alf_config_path)
        env_type = cfg['env']['type']
        BaseEnvCls = get_environment(env_type)
        tmp_env = BaseEnvCls(cfg, train_eval='train' if is_train else eval_dataset)
        tmp_env.collect_game_files()
        return list(getattr(tmp_env, 'game_files', []))

    # Pre-assign unique game files when requested
    alf_config_path = os.path.join(os.path.dirname(__file__), '../../agent_system/environments/env_package/alfworld/configs/config_tw.yaml')
    preassigned_game_files = None
    if args.unique_envs:
        try:
            all_game_files = collect_all_game_files(alf_config_path, is_train=True)
        except Exception as e:
            logging.error(f"Failed to collect game files for unique_envs: {e}")
            sys.exit(1)
        rng = random.Random(args.seed)
        rng.shuffle(all_game_files)
        if len(all_game_files) < total_envs:
            logging.error(f"游戏文件不足：需要{total_envs}个，只有{len(all_game_files)}个")
            sys.exit(1)
        preassigned_game_files = all_game_files[:total_envs]
        logging.info(f"Unique envs enabled: using {len(preassigned_game_files)} distinct game files from {len(all_game_files)} available")

        # Dry-run: only print allocation then exit
        if args.dry_run:
            logging.info(f"[Dry-Run] total_envs={total_envs}, batch_size={batch_size}, num_batches={num_batches}")
            for b in range(num_batches):
                start = b * batch_size
                end = start + min(batch_size, total_envs - start)
                batch_slice = preassigned_game_files[start:end]
                examples = ", ".join(os.path.basename(p) for p in batch_slice[:3])
                logging.info(f"[Dry-Run] Batch {b+1:02d}: {len(batch_slice)} files; examples: {examples}")
            sys.exit(0)
    else:
        if args.dry_run:
            logging.warning("--dry_run 需要配合 --unique_envs 使用；当前未启用 unique_envs，直接退出。")
            sys.exit(0)
    
    # Accumulated statistics across all batches
    all_overall_success_rates = []
    all_task_success_history = defaultdict(list)
    global_env_counter = 0

    # ======================= Main Batch Loop =======================
    for batch_idx in range(num_batches):
        # Calculate actual batch size for this batch (last batch might be smaller)
        current_batch_size = min(batch_size, total_envs - batch_idx * batch_size)
        logging.info(f"\n========== Starting Batch {batch_idx + 1}/{num_batches} with {current_batch_size} envs ==========")
        
        # Select per-batch game files if unique_envs is on
        batch_game_files = None
        if preassigned_game_files is not None:
            start = batch_idx * batch_size
            end = start + current_batch_size
            batch_game_files = preassigned_game_files[start:end]

        # Create environment for this batch
        env_manager = build_env(
            env_name,
            env_num=current_batch_size,
            seed=args.seed + batch_idx,
            history_length=args.history_length,
            alf_env_type=args.alf_env_type,
            game_files=batch_game_files,
        )
        
        # Batch-level statistics
        batch_overall_success_rates = []
        batch_task_success_history = defaultdict(list)
        try:
            # ======================= Test Loop for this Batch =======================
            for test_idx in range(test_times):
                logging.info(f"\n========== Start Batch {batch_idx + 1} Test {test_idx} ==========")
                start_time = time.time()
                
                # Run each environment with retry logic if debugger is enabled
                if args.enable_debugger:
                    # Run environments with retry logic
                    env_results = []
                    for env_id in range(current_batch_size):
                        logging.info(f"\nRunning Env {env_id + 1}/{current_batch_size} in Batch {batch_idx + 1}")
                        result = run_environment_with_retry(
                            env_id=env_id,
                            env_manager=env_manager,
                            agent=agent,
                            max_steps=max_steps,
                            debugger=debugger,
                            trajectory_manager=trajectory_manager,
                            max_retries=args.max_retries,
                            dump_fp=dump_fp,
                            chat_base_dir=chat_base_dir,
                            batch_idx=batch_idx,
                            test_idx=test_idx,
                            global_env_counter=global_env_counter,
                            run_ts=run_ts,
                            debug_output_dir=args.debug_output_dir
                        )
                        env_results.append(result)
                    
                    # Collect statistics from results
                    overall_success_this_round = np.array([r['won'] for r in env_results])
                    task_success_cnt = defaultdict(int)
                    task_total_cnt = defaultdict(int)
                    
                    for result in env_results:
                        gamefile = result['gamefile']
                        matched = False
                        for task in TASKS:
                            if task in gamefile:
                                task_total_cnt[task] += 1
                                if result['won']:
                                    task_success_cnt[task] += 1
                                matched = True
                                break
                        if not matched:
                            task_total_cnt["other"] += 1
                            if result['won']:
                                task_success_cnt["other"] += 1
                
                else:
                    # Original batch processing logic (without retry)
                    obs, infos = env_manager.reset()
                    env_dones = [False] * current_batch_size

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
                                        "gamefile": infos[i].get("extra.gamefile"),
                                        "is_action_valid": bool(infos[i].get("is_action_valid", False)),
                                    }
                                    dump_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                                except Exception:
                                    pass

                            if dones[i]:
                                env_dones[i] = True
                                won = bool(infos[i].get("won", False))
                                overall_success_this_round[i] = won

                                # Parse task type
                                gamefile = infos[i].get("extra.gamefile", "")
                                matched = False
                                for task in TASKS:
                                    if task in gamefile:
                                        task_total_cnt[task] += 1
                                        if won:
                                            task_success_cnt[task] += 1
                                        matched = True
                                        break
                                if not matched:
                                    # Unrecognized tasks are also counted in total
                                    task_total_cnt["other"] += 1
                                    if won:
                                        task_success_cnt["other"] += 1

                                # If this env just finished, dump chat history if requested
                                if chat_base_dir and not saved_flags[i]:
                                    try:
                                        task = None
                                        try:
                                            task = env_manager.tasks[i]
                                        except Exception:
                                            task = "unknown"
                                        gamefile = infos[i].get("extra.gamefile", "")
                                        task_dir = os.path.join(chat_base_dir, _sanitize(task))
                                        os.makedirs(task_dir, exist_ok=True)
                                        unique_id = f"b{batch_idx:03d}_t{test_idx:02d}_e{i:02d}"
                                        base = f"chat_{_sanitize(task)}-{_sanitize(gamefile) or f'env{i}'}-{unique_id}"
                                        out_path = os.path.join(task_dir, base + ".json")
                                        meta = {
                                            "batch_idx": batch_idx,
                                            "env_id": global_env_counter + i,
                                            "test_idx": test_idx,
                                            "model": args.model,
                                            "task": task,
                                            "gamefile": gamefile,
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
                                    task = None
                                    try:
                                        task = env_manager.tasks[i]
                                    except Exception:
                                        task = "unknown"
                                    gamefile = last_infos[i].get("extra.gamefile", "") if isinstance(last_infos, list) and i < len(last_infos) else ""
                                    task_dir = os.path.join(chat_base_dir, _sanitize(task))
                                    os.makedirs(task_dir, exist_ok=True)
                                    unique_id = f"b{batch_idx:03d}_t{test_idx:02d}_e{i:02d}"
                                    base = f"chat_{_sanitize(task)}-{_sanitize(gamefile) or f'env{i}'}-{unique_id}"
                                    out_path = os.path.join(task_dir, base + ".json")
                                    steps_taken = max(0, len(chats[i]) // 2)
                                    meta = {
                                        "batch_idx": batch_idx,
                                        "env_id": global_env_counter + i,
                                        "test_idx": test_idx,
                                        "model": args.model,
                                        "task": task,
                                        "gamefile": gamefile,
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

                for task in TASKS + ["other"]:
                    if task_total_cnt.get(task, 0) > 0:
                        rate = task_success_cnt[task] / task_total_cnt[task]
                        batch_task_success_history[task].append(rate)
                        logging.info(
                            f"    {task:<35s}: {rate:.4f} "
                            f"({task_success_cnt[task]}/{task_total_cnt[task]})"
                        )

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

            # Clean up Ray actors for this batch to free resources
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

    for task in TASKS + ["other"]:
        if all_task_success_history.get(task):
            logging.info(
                f"{task:<35s}: "
                f"{np.mean(all_task_success_history[task]):.4f} ± "
                f"{np.std(all_task_success_history[task]):.4f}"
            )

    if dump_fp is not None:
        dump_fp.flush()
        dump_fp.close()
