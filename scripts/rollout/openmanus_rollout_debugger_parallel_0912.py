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
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
import random
import hashlib
import sys
from openmanus_rl.environments.env_manager import *
from openai import OpenAI
from together import Together
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor as AsyncThreadPoolExecutor

try:
    import dotenv
    dotenv.load_dotenv()
except Exception:
    pass


class UnifiedAgent:
    """Unified agent that can work with all environments"""
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.4, 
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
        
    def get_action_from_llm(self, obs: str, log_timing: bool = True) -> str:
        """Get action from LLM for a single observation"""
        if log_timing:
            llm_start = time.time()
            thread_id = threading.get_ident()
        
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
        
        if log_timing:
            llm_time = time.time() - llm_start
            logging.debug(f"[LLM] Thread {thread_id} LLM call took {llm_time:.3f}s")
        
        return response.choices[0].message.content.strip()
    
    def get_action_from_llm_with_shared_pool(self, obs: str, shared_executor, log_timing: bool = True):
        """Get action from LLM using a shared thread pool executor for better global concurrency"""
        def _call_llm():
            return self.get_action_from_llm(obs, log_timing)
        
        # Submit to shared executor and return future
        return shared_executor.submit(_call_llm)
    
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
    
    def analyze_trajectory(self, trajectory: List[Dict], env_type: str) -> Dict:
        """
        Analyze a failed trajectory and identify the failure point and reason
        
        Args:
            trajectory: List of trajectory steps
            env_type: Type of environment (alfworld, gaia, webshop)
        
        Returns:
            Analysis dict containing failure info and suggestions
        """
        
        # Format trajectory for LLM analysis
        trajectory_text = self._format_trajectory(trajectory)
        
        # Create environment-specific analysis prompt
        env_context = {
            "alfworld": """The agent is trying to complete household tasks in a text-based environment. Common tasks include:
- pick_and_place: Pick up an object and place it somewhere
- pick_two_obj_and_place: Pick up two objects and place them
- look_at_obj_in_light: Examine an object under a light source
- pick_heat_then_place_in_recep: Heat an object and place it in a receptacle
- pick_cool_then_place_in_recep: Cool an object and place it in a receptacle
- pick_clean_then_place_in_recep: Clean an object and place it in a receptacle""",
            
            "gaia": """The agent is solving complex reasoning tasks using various tools. Available tools include:
- google_search: Search for information online
- wikipedia_knowledge_searcher: Search Wikipedia for knowledge
- python_code_generator: Generate and execute Python code
- Other specialized tools for information gathering and processing""",
            
            "webshop": """The agent is shopping for products on a web interface. The agent needs to:
- Search for products matching specific requirements
- Navigate through product listings
- Select products that meet the given criteria
- Complete the purchase process"""
        }.get(env_type, "The agent is solving a task in an interactive environment.")
        
        prompt = f"""You are an expert debugger for an AI agent. Analyze the following failed trajectory and identify where and why the agent failed.

ENVIRONMENT TYPE: {env_type}

TASK CONTEXT:
{env_context}

TRAJECTORY:
{trajectory_text}

ANALYSIS REQUIRED:
1. Identify the EXACT step where the agent made a critical error
2. Determine the type of failure
3. Explain WHY this was an error in the context of the task
4. Suggest a specific correction for that step

Please provide your analysis in the following JSON format:
{{
    "failure_step": <int: the step number where the critical error occurred>,
    "failure_type": "<string: one of 'wrong_action', 'invalid_syntax', 'wrong_selection', 'missed_requirement', 'wrong_sequence', 'exploration_failure', 'reasoning_error'>",
    "reason": "<string: detailed explanation of why this was an error>",
    "suggestion": "<string: specific action or approach the agent should take instead>",
    "critical_step": <int: the last step that was definitely correct before the error>
}}

IMPORTANT: Be precise about the failure step. Look for actions that:
- Use incorrect syntax or invalid commands
- Select wrong items or options
- Miss key requirements from the task
- Show logical errors or misunderstanding
- Fail to explore properly

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
    
    def generate_feedback(self, observation: str, analysis: Dict, previous_action: str, env_type: str) -> str:
        """
        Generate feedback to inject into the agent's next action
        
        Args:
            observation: Current observation from the environment
            analysis: Analysis dict from analyze_trajectory
            previous_action: The action that led to failure
            env_type: Type of environment
        
        Returns:
            Feedback string to prepend to the agent's next prompt
        """
        
        feedback_prompt = f"""Based on a previous failed attempt, generate helpful feedback for the agent.

ENVIRONMENT: {env_type}

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
            lines.append(f"  Observation: {step.get('observation', 'N/A')}")
            lines.append(f"  Action: {step.get('action', 'N/A')}")
            if step.get('reward') is not None:
                lines.append(f"  Reward: {step['reward']}")
            if step.get('done'):
                lines.append(f"  Done: {step['done']}, Won: {step.get('won', False)}")
            lines.append("")
        return "\n".join(lines)


class TrajectoryManager:
    """Manages trajectory storage and replay for multiple environments"""
    
    def __init__(self):
        self.trajectories = {}  # env_id -> list of trajectory steps
        self.attempts = {}  # env_id -> list of all attempt trajectories
    
    def reset(self, env_id: int):
        """Reset trajectory for an environment"""
        self.trajectories[env_id] = []
        if env_id not in self.attempts:
            self.attempts[env_id] = []
    
    def add_step(self, env_id: int, step_data: Dict):
        """Add a step to the current trajectory"""
        if env_id not in self.trajectories:
            self.trajectories[env_id] = []
        self.trajectories[env_id].append(step_data)
    
    def save_attempt(self, env_id: int):
        """Save current trajectory as an attempt"""
        if env_id in self.trajectories:
            self.attempts[env_id].append(self.trajectories[env_id].copy())
    
    def get_trajectory(self, env_id: int) -> List[Dict]:
        """Get the current trajectory for an environment"""
        return self.trajectories.get(env_id, [])
    
    def get_all_attempts(self, env_id: int) -> List[List[Dict]]:
        """Get all attempt trajectories for an environment"""
        return self.attempts.get(env_id, [])
    
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


class ExtendedEnvironmentManager:
    """Single‑env helpers on top of an existing manager.

    Important:
    - These helpers are only correct when the underlying manager controls
      exactly one environment instance (env_num == 1).
    - This script uses one‑env managers for debugger rollouts so we can
      freely reset during a rollout without affecting others.
    """

    def __init__(self, base_manager):
        # Keep a handle to the original manager; delegate attribute access.
        self.base_manager = base_manager
        self.__dict__.update(base_manager.__dict__)

    def reset_single(self, env_id: int):
        """Reset the underlying single environment.

        Note: env_id is ignored by design since this wrapper is only used
        when env_num == 1. We keep the signature for compatibility.
        """
        obs, infos = self.reset()
        return obs, infos

    def step_single(self, env_id: int, action: str):
        """Step the underlying single environment with the given action."""
        # For a single environment we just forward a singleton action list.
        obs, rewards, dones, infos = self.step([action])
        return obs, rewards, dones, infos

    def __getattr__(self, name):
        # Delegate unknown attributes to the base manager instance.
        return getattr(self.base_manager, name)


class EnvironmentFactory:
    """Factory for creating different environment types"""
    
    @staticmethod
    def build_env(env_type: str, with_debugger: bool = False, **kwargs) -> Any:
        """Build environment based on type"""
        
        if env_type == "alfworld":
            env = EnvironmentFactory._build_alfworld(**kwargs)
        elif env_type == "gaia":
            env = EnvironmentFactory._build_gaia(**kwargs)
        elif env_type == "webshop":
            env = EnvironmentFactory._build_webshop(**kwargs)
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")
        
        # Wrap with extended manager if debugger is enabled
        if with_debugger:
            return ExtendedEnvironmentManager(env)
        return env
    
    @staticmethod
    def _build_alfworld(env_num: int = 1, seed: int = 1, history_length: int = 2,
                       alf_env_type: str = "alfworld/AlfredTWEnv", 
                       game_files: Optional[List[str]] = None, **kwargs):
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
                history_length=history_length
            )
        )
        
        return AlfWorldEnvironmentManager(envs, alfworld_projection, cfg)
    
    @staticmethod
    def _build_gaia(tasks_data: List[Dict], available_tools: List[str], 
                   env_num: int = 1, seed: int = 1, history_length: int = 2,
                   max_steps: int = 30, **kwargs):
        """Build GAIA/Tool Use environment"""
        from openmanus_rl.environments.env_package.tool_use import tool_use_projection
        from openmanus_rl.environments.env_package.tool_use import build_tool_use_envs
        from openmanus_rl.environments.env_package.tool_use.manager import ToolUseEnvironmentManager
        
        envs = build_tool_use_envs(
            tasks_data=tasks_data,
            available_tools=available_tools,
            seed=seed,
            env_num=env_num,
            group_n=1,
            is_train=True
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
                       use_train_set: bool = False, **kwargs):
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
                history_length=history_length
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


def get_task_id(env_type: str, env_id: int, info: Dict, batch_idx: int = 0) -> str:
    """
    Get a unique task identifier for organizing outputs
    
    Args:
        env_type: Type of environment
        env_id: Environment ID within batch
        info: Info dictionary from environment
        batch_idx: Batch index
        
    Returns:
        Unique task identifier string
    """
    if env_type == "alfworld":
        # Try to extract from gamefile
        gamefile = info.get("extra.gamefile", "")
        if gamefile:
            # Extract just the filename without path
            task_name = os.path.basename(gamefile).replace(".json", "")
            return f"alfworld_b{batch_idx:03d}_e{env_id:03d}_{task_name[:50]}"
        else:
            return f"alfworld_b{batch_idx:03d}_e{env_id:03d}_unknown"
    elif env_type == "gaia":
        pid = info.get("pid", f"unknown_{env_id}")
        return f"gaia_b{batch_idx:03d}_e{env_id:03d}_{pid}"
    elif env_type == "webshop":
        # Try to extract task ID from info
        task_id = info.get("task_id", f"task_{env_id}")
        return f"webshop_b{batch_idx:03d}_e{env_id:03d}_{task_id}"
    else:
        return f"{env_type}_b{batch_idx:03d}_e{env_id:03d}"


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


def run_environment_with_retry(
    env_id: int,
    env_manager,
    agent: UnifiedAgent,
    max_steps: int,
    env_type: str,
    debugger: Optional[LLMDebugger] = None,
    trajectory_manager: Optional[TrajectoryManager] = None,
    max_retries: int = 5,
    dump_fp=None,
    dump_lock=None,
    chat_base_dir: str = None,
    batch_idx: int = 0,
    test_idx: int = 0,
    global_env_counter: int = 0,
    run_ts: str = "",
    debug_output_dir: str = None,
    save_all_attempts: bool = False,
    task_dir: str = None,
    shared_llm_executor=None
) -> Dict:
    """
    Run a single environment with retry logic using debugger feedback
    
    Returns:
        Dict containing results and statistics for this environment
    """
    
    last_trajectory = None  # Track the last trajectory for debugging
    won = False
    final_info = {}
    all_attempt_trajectories = []
    final_reward = 0
    first_attempt_success = False  # Track if first attempt was successful
    
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
        analysis = None
        
        # If this is a retry, analyze the failed trajectory
        if retry_idx > 0 and debugger and last_trajectory and not won:
            analysis = debugger.analyze_trajectory(last_trajectory, env_type)
            
            # Save debug analysis to task dir if specified
            if task_dir:
                debug_file = os.path.join(
                    task_dir,
                    f"debug_analysis_retry_{retry_idx}.json"
                )
                with open(debug_file, "w") as f:
                    json.dump({
                        "retry": retry_idx,
                        "analysis": analysis,
                        "trajectory": last_trajectory,
                        "env_type": env_type
                    }, f, indent=2)
            
            logging.info(f"    Debugger analysis - Failure at step {analysis['failure_step']}: {analysis['failure_type']}")
            logging.info(f"    Suggestion: {analysis['suggestion']}")
            
            # Handle case where failure happens at step 0 (first action)
            critical_step = analysis.get('critical_step', -1)
            
            # Ensure critical_step doesn't exceed trajectory bounds
            if critical_step >= len(last_trajectory):
                logging.warning(f"    Critical step {critical_step} exceeds trajectory length {len(last_trajectory)}, adjusting to {len(last_trajectory) - 1}")
                critical_step = len(last_trajectory) - 1
            
            replay_to_step = critical_step
            if replay_to_step == -1:
                logging.info(f"    First step failure detected, will inject feedback at initial prompt")
            else:
                logging.info(f"    Will replay up to step {replay_to_step} and inject feedback at step {replay_to_step + 1}")
        
        # Get initial observation
        obs_dict, info_dict = env_manager.reset_single(env_id)
        obs = obs_dict["text"][env_id]
        info = info_dict[env_id] if isinstance(info_dict, dict) else info_dict
        
        for step_idx in range(max_steps):
            if env_done:
                break
            
            # Check if we're replaying previous trajectory
            if replay_to_step >= 0 and step_idx <= replay_to_step and last_trajectory:
                # Replay action from previous trajectory
                # Ensure we don't go beyond the trajectory length
                if step_idx < len(last_trajectory):
                    action = last_trajectory[step_idx]['action']
                    logging.debug(f"    Replaying step {step_idx}: {action}")
                else:
                    # This shouldn't happen if replay_to_step is set correctly
                    logging.warning(f"    Replay step {step_idx} beyond trajectory length {len(last_trajectory)}, using 'None'")
                    action = "None"
            else:
                # Get new action from agent
                prompt = obs
                
                # Add debugger feedback at the critical point
                should_inject_feedback = False
                if debugger and analysis:
                    if replay_to_step == -1 and step_idx == 0:
                        # First step failure - inject feedback at initial prompt
                        should_inject_feedback = True
                        logging.info(f"    Injecting debugger feedback at step {step_idx} (first step failure)")
                    elif replay_to_step >= 0 and step_idx == replay_to_step + 1:
                        # Normal case - inject feedback after critical step
                        should_inject_feedback = True
                        logging.info(f"    Injecting debugger feedback at step {step_idx}")
                
                if should_inject_feedback:
                    # Generate feedback for this specific step
                    prev_action = ""
                    if last_trajectory:
                        # Get the previous action that failed at this step
                        if replay_to_step == -1 and step_idx == 0:
                            # First step failure - use the first action from last_trajectory
                            prev_action = last_trajectory[0]['action'] if len(last_trajectory) > 0 else ""
                        elif step_idx < len(last_trajectory):
                            # Normal case - use the action at this step from last_trajectory
                            prev_action = last_trajectory[step_idx]['action']
                        else:
                            # Step beyond trajectory length - shouldn't happen but handle gracefully
                            logging.warning(f"Step {step_idx} beyond trajectory length {len(last_trajectory)}")
                            prev_action = ""
                    
                    debugger_feedback = debugger.generate_feedback(obs, analysis, prev_action, env_type)
                    prompt = debugger_feedback + obs
                
                # Use shared LLM executor if available for better concurrency
                if shared_llm_executor is not None:
                    llm_future = agent.get_action_from_llm_with_shared_pool(prompt, shared_llm_executor)
                    action = llm_future.result()  # This will block until LLM responds, but allows other tasks to proceed
                else:
                    action = agent.get_action_from_llm(prompt)
            
            # Store raw action for trajectory
            raw_action = action
            
            # Step environment
            obs_dict, reward_dict, done_dict, info_dict = env_manager.step_single(env_id, action)
            
            obs = obs_dict["text"][env_id]
            reward = reward_dict[env_id]
            done = done_dict[env_id]
            # info_dict is a list, get the element at env_id
            info = info_dict[env_id] if isinstance(info_dict, list) else info_dict
            
            # Ensure info is a dictionary
            if not isinstance(info, dict):
                info = {}
            
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
            if dump_fp and (save_all_attempts or retry_idx == 0 or won):
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
                        "is_action_valid": bool(info.get("is_action_valid", False)),
                        "env_type": env_type
                    }
                    
                    # Add environment-specific fields
                    if env_type == "gaia":
                        row["pid"] = info.get("pid", "unknown")
                    elif env_type == "alfworld":
                        row["gamefile"] = info.get("extra.gamefile", "")
                    elif env_type == "webshop":
                        row["task_score"] = float(info.get("task_score", 0))
                    
                    line = json.dumps(row, ensure_ascii=False) + "\n"
                    if dump_lock is not None:
                        # Serialize writes across threads
                        with dump_lock:
                            dump_fp.write(line)
                            dump_fp.flush()
                    else:
                        dump_fp.write(line)
                        dump_fp.flush()
                except Exception as e:
                    logging.error(f"Failed to write trajectory: {e}")
            
            if done:
                env_done = True
                won = bool(info.get("won", False))
                final_info = info
                break
        
        # Save this attempt's trajectory
        if trajectory_manager:
            trajectory_manager.save_attempt(env_id)
        
        attempt_data = {
            "retry_idx": retry_idx,
            "trajectory": trajectory_steps.copy(),
            "won": won,
            "reward": cumulative_reward,
            "steps": len(trajectory_steps)
        }
        all_attempt_trajectories.append(attempt_data)
        
        # Save individual attempt trajectory to task dir
        if task_dir:
            attempt_file = os.path.join(task_dir, f"attempt_{retry_idx + 1}_trajectory.json")
            with open(attempt_file, "w") as f:
                json.dump(attempt_data, f, indent=2)
        
        # Save current trajectory for potential debugging
        last_trajectory = trajectory_steps
        final_reward = cumulative_reward
        
        # Check if this attempt was successful
        if won:
            logging.info(f"  Env {env_id} - SUCCESS on attempt {retry_idx + 1}")
            if retry_idx == 0:
                first_attempt_success = True
            break  # Success! No need to retry
        else:
            logging.info(f"  Env {env_id} - FAILED on attempt {retry_idx + 1}, will retry with debugging" if debugger and retry_idx < max_retries - 1 else f"  Env {env_id} - FAILED on attempt {retry_idx + 1}")
        
        # If debugger is not enabled, don't retry
        if not debugger:
            break
    
    # Save final summary to task dir
    if task_dir:
        try:
            summary_file = os.path.join(task_dir, "task_summary.json")
            
            meta = {
                "batch_idx": batch_idx,
                "env_id": global_env_counter + env_id,
                "test_idx": test_idx,
                "model": agent.model_name,
                "env_type": env_type,
                "total_attempts": retry_idx + 1,
                "won": won,
                "first_attempt_success": first_attempt_success,
                "final_reward": final_reward,
                "timestamp": run_ts,
                "steps_in_final_attempt": len(last_trajectory) if last_trajectory else 0
            }
            
            # Add environment-specific metadata
            if env_type == "gaia":
                meta["pid"] = final_info.get("pid", "unknown")
            elif env_type == "alfworld":
                meta["gamefile"] = final_info.get("extra.gamefile", "")
            elif env_type == "webshop":
                meta["task_score"] = float(final_info.get("task_score", 0))
            
            # Save summary with all attempts info
            save_data = {
                "metadata": meta,
                "all_attempts_summary": [
                    {
                        "attempt": i + 1,
                        "won": att["won"],
                        "reward": att["reward"],
                        "steps": att["steps"]
                    }
                    for i, att in enumerate(all_attempt_trajectories)
                ]
            }
            
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to save task summary: {e}")
    
    return {
        "env_id": env_id,
        "won": won,
        "first_attempt_success": first_attempt_success,
        "reward": final_reward,
        "retries": retry_idx + 1,
        "steps": len(last_trajectory) if last_trajectory else 0,
        "env_type": env_type,
        "trajectory": last_trajectory,
        "all_attempts": all_attempt_trajectories if save_all_attempts else None
    }


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
                       help="Maximum steps per episode (default: 50 for alfworld, 30 for gaia/webshop)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--history_length", type=int, default=2)
    
    # Model parameters
    parser.add_argument("--model", default="gpt-4o-mini",
                       help="Model name (OpenAI: gpt-4o, gpt-4o-mini; Together: Qwen/Qwen2.5-7B-Instruct-Turbo, etc.)")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--base_url", default=None,
                       help="OpenAI-compatible base URL (e.g., vLLM http://127.0.0.1:8000/v1)")
    
    # Execution parameters
    parser.add_argument("--concurrency", type=int, default=4,
                       help="Max concurrent task workers")
    parser.add_argument("--llm_concurrency", type=int, default=None,
                       help="Max concurrent LLM requests across all tasks (default: 3x task concurrency)")
    parser.add_argument("--retries", type=int, default=3,
                       help="Retries per request on failure")
    
    # Output parameters
    parser.add_argument("--dump_path", default=None,
                       help="If set, write JSONL trajectory to this file")
    parser.add_argument("--chat_root", default=None,
                       help="If set, save per-episode chat histories under this root")
    parser.add_argument("--experiment_dir", default=None,
                       help="Root directory for all experiment outputs")
    parser.add_argument("--save_per_task_trajectories", action="store_true",
                       help="Save each task's trajectories in a separate folder")
    
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
    
    # Debugger options
    parser.add_argument("--enable_debugger", action="store_true",
                       help="Enable LLM debugger for failed trajectories")
    parser.add_argument("--max_debug_retry", type=int, default=5,
                       help="Maximum number of retry attempts with debugger feedback (default: 5)")
    parser.add_argument("--debugger_model", default="gpt-4o",
                       help="Model to use for trajectory debugging")
    parser.add_argument("--debugger_temperature", type=float, default=0.3,
                       help="Temperature for debugger model")
    parser.add_argument("--debug_output_dir", default=None,
                       help="Directory to save debug analysis results")
    parser.add_argument("--save_all_attempts", action="store_true",
                       help="Save trajectories for all retry attempts")
    
    # Other options
    parser.add_argument("--unique_envs", action="store_true",
                       help="Ensure unique tasks/games across all environments")
    parser.add_argument("--dry_run", action="store_true",
                       help="Only print batch allocation without running")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set default max_steps based on environment
    if args.max_steps is None:
        args.max_steps = {
            "alfworld": 50,
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
    
    # Calculate number of batches (deprecated: we keep a fixed env pool)
    num_batches = 1
    logging.info(f"Using fixed env pool; batch_size is ignored.")
    
    # Prepare experiment directory structure
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.experiment_dir:
        # Use the provided experiment directory
        experiment_root = args.experiment_dir
        os.makedirs(experiment_root, exist_ok=True)
        
        # Create subdirectories
        trajectories_dir = os.path.join(experiment_root, "trajectories")
        summaries_dir = os.path.join(experiment_root, "summaries")
        os.makedirs(trajectories_dir, exist_ok=True)
        os.makedirs(summaries_dir, exist_ok=True)
        
        # Set up dump file path
        if not args.dump_path:
            args.dump_path = os.path.join(summaries_dir, "all_trajectories.jsonl")
        
        logging.info(f"Experiment directory: {experiment_root}")
    else:
        trajectories_dir = None
        summaries_dir = None
    
    # Prepare output files
    dump_fp = None
    if args.dump_path:
        os.makedirs(os.path.dirname(args.dump_path) or ".", exist_ok=True)
        dump_fp = open(args.dump_path, "a", encoding="utf-8")
        logging.info(f"Dumping trajectories to: {args.dump_path}")

    # Pre-initialize Ray (once) for Ray-based envs to avoid thread-race on ray.init
    if args.env in ("alfworld", "webshop"):
        try:
            import os as _os
            _os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
            import ray as _ray
            if not _ray.is_initialized():
                _ray.init(ignore_reinit_error=True, include_dashboard=False)
        except Exception as e:
            logging.warning(f"Ray pre-initialization skipped or failed: {e}")
    
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
        
        # Trim to what will actually be used by a fixed pool of envs
        pool_size = max(1, int(args.total_envs))
        rounds = max(1, int(args.test_times))
        max_needed = min(len(gaia_tasks), pool_size * rounds)
        if len(gaia_tasks) > max_needed:
            logging.info(f"Trimming GAIA tasks to {max_needed} (pool_size={pool_size}, rounds={rounds})")
            gaia_tasks = gaia_tasks[:max_needed]
        elif len(gaia_tasks) < max_needed:
            logging.warning(f"Only {len(gaia_tasks)} GAIA tasks available, reducing rounds to fit availability")
            rounds = max(1, len(gaia_tasks) // pool_size)
            args.test_times = rounds
        
        # Shuffle tasks for random sampling variety
        rng = random.Random(args.seed)
        rng.shuffle(gaia_tasks)
        
    elif args.env == "alfworld" and args.unique_envs:
        alfworld_game_files = prepare_alfworld_game_files(args.env, args.total_envs, args.seed)
        if alfworld_game_files:
            logging.info(f"Prepared {len(alfworld_game_files)} unique game files")
    
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
    agent = UnifiedAgent(
        model_name=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
        env_type=args.env
    )
    
    # Create shared LLM executor pool for better concurrency across all tasks
    # Use more workers than task concurrency to handle LLM I/O wait time
    if args.llm_concurrency is not None:
        llm_pool_size = args.llm_concurrency
    else:
        llm_pool_size = min(50, args.concurrency * 3)  # 3x task concurrency for LLM calls
    
    shared_llm_executor = ThreadPoolExecutor(max_workers=llm_pool_size)
    logging.info(f"Created shared LLM executor pool with {llm_pool_size} workers")
    
    # Initialize debugger and trajectory manager if enabled
    debugger = None
    trajectory_manager = None
    if args.enable_debugger:
        debugger = LLMDebugger(
            model_name=args.debugger_model,
            temperature=args.debugger_temperature,
            base_url=args.base_url
        )
        trajectory_manager = TrajectoryManager()
        logging.info(f"Debugger enabled with model {args.debugger_model}, max retries: {args.max_debug_retry}")
        
        # Create debug output directory if specified
        if args.debug_output_dir:
            os.makedirs(args.debug_output_dir, exist_ok=True)
            logging.info(f"Debug analysis will be saved to: {args.debug_output_dir}")

    # Statistics tracking
    all_overall_success_rates = []
    all_first_attempt_success_rates = []  # Track first attempt success rates
    all_debugger_success_rates = []  # Track success rates after debugger assistance
    all_task_success_history = defaultdict(list)
    global_env_counter = 0
    
    # Track overall statistics
    total_first_attempt_successes = 0
    total_debugger_successes = 0
    total_tasks = 0
    
    # Main rollout loop
    try:
        # Legacy batch loop disabled — we use a fixed env pool below.
        for batch_idx in range(0):
            # Calculate actual batch size
            current_batch_size = min(args.batch_size, args.total_envs - batch_idx * args.batch_size)
            logging.info(f"\n========== Starting Batch {batch_idx + 1}/{num_batches} with {current_batch_size} envs ==========")
            
            # Prepare environment-specific kwargs
            env_kwargs = {
                "env_num": current_batch_size,
                "seed": args.seed + batch_idx,
                "history_length": args.history_length,
            }
            
            if args.env == "gaia":
                start = batch_idx * args.batch_size
                end = start + current_batch_size
                env_kwargs["tasks_data"] = gaia_tasks[start:end]
                env_kwargs["available_tools"] = args.gaia_tools
                env_kwargs["max_steps"] = args.max_steps
                
            elif args.env == "alfworld":
                env_kwargs["alf_env_type"] = args.alf_env_type
                if alfworld_game_files:
                    start = batch_idx * args.batch_size
                    end = start + current_batch_size
                    env_kwargs["game_files"] = alfworld_game_files[start:end]
                    
            elif args.env == "webshop":
                env_kwargs["use_train_set"] = args.webshop_train
            
            # Batch-level statistics
            batch_overall_success_rates = []
            batch_task_success_history = defaultdict(list)
            
            try:
                # Test loop for this batch
                for test_idx in range(args.test_times):
                    logging.info(f"\n========== Start Batch {batch_idx + 1} Test {test_idx} ==========")
                    start_time = time.time()
                    
                    # Run with retry logic if debugger is enabled
                    if args.enable_debugger:
                        # Build per-env managers (env_num == 1) so each rollout can reset independently
                        per_env_build_args = []
                        for i in range(current_batch_size):
                            single_kwargs = dict(env_kwargs)
                            single_kwargs["env_num"] = 1
                            # Slice GAIA tasks to a single task
                            if args.env == "gaia" and "tasks_data" in single_kwargs:
                                # Select the i-th task within this batch
                                start = batch_idx * args.batch_size
                                single_kwargs["tasks_data"] = [gaia_tasks[start + i]]
                            # Pin one gamefile for AlfWorld if provided
                            if args.env == "alfworld" and "game_files" in single_kwargs and single_kwargs["game_files"]:
                                single_kwargs["game_files"] = [single_kwargs["game_files"][i]]
                            per_env_build_args.append(single_kwargs)

                        # Helper to run a single rollout in its own env
                        def _run_one(env_idx: int):
                            task_start_time = time.time()
                            thread_id = threading.get_ident()
                            logging.info(f"[PARALLEL] Task {env_idx + 1} starting on thread {thread_id} at {task_start_time:.3f}")
                            
                            # Create one-env manager for this rollout
                            env_init_start = time.time()
                            local_env = EnvironmentFactory.build_env(
                                args.env,
                                with_debugger=True,
                                **per_env_build_args[env_idx]
                            )
                            env_init_time = time.time() - env_init_start
                            logging.info(f"[PARALLEL] Task {env_idx + 1} env init took {env_init_time:.3f}s")
                            
                            try:
                                # Reset once to compute task id and make task dir
                                reset_start = time.time()
                                init_obs, init_infos = local_env.reset()
                                info0 = init_infos[0] if isinstance(init_infos, list) else init_infos
                                task_id_local = get_task_id(args.env, env_idx, info0, batch_idx)
                                task_dir_local = None
                                if args.save_per_task_trajectories and trajectories_dir:
                                    task_dir_local = os.path.join(trajectories_dir, task_id_local)
                                    os.makedirs(task_dir_local, exist_ok=True)
                                reset_time = time.time() - reset_start
                                logging.info(f"[PARALLEL] Task {env_idx + 1} reset took {reset_time:.3f}s")

                                logging.info(f"[PARALLEL] Task {env_idx + 1}/{current_batch_size} in Batch {batch_idx + 1} - {task_id_local}")

                                # Execute rollout with retry/debugging; env_id is 0 for single-env managers
                                rollout_start = time.time()
                                res = run_environment_with_retry(
                                    env_id=0,
                                    env_manager=local_env,
                                    agent=agent,
                                    max_steps=args.max_steps,
                                    env_type=args.env,
                                    debugger=debugger,
                                    trajectory_manager=trajectory_manager,
                                    max_retries=args.max_debug_retry,
                                    dump_fp=dump_fp,
                                    dump_lock=dump_lock,
                                    chat_base_dir=None,
                                    batch_idx=batch_idx,
                                    test_idx=test_idx,
                                    global_env_counter=global_env_counter + env_idx,
                                    run_ts=run_ts,
                                    debug_output_dir=None,
                                    save_all_attempts=args.save_all_attempts,
                                    task_dir=task_dir_local,
                                    shared_llm_executor=shared_llm_executor,
                                )
                                rollout_time = time.time() - rollout_start
                                total_time = time.time() - task_start_time
                                logging.info(f"[PARALLEL] Task {env_idx + 1} rollout took {rollout_time:.3f}s, total time: {total_time:.3f}s")
                                
                                res["task_id"] = task_id_local
                                res["timing"] = {
                                    "env_init_time": env_init_time,
                                    "reset_time": reset_time, 
                                    "rollout_time": rollout_time,
                                    "total_time": total_time,
                                    "thread_id": thread_id
                                }
                                # Preserve original batch env_id in result for consistency
                                res["env_id"] = env_idx
                                return res
                            finally:
                                # Ensure resources for this single env are released
                                cleanup_start = time.time()
                                try:
                                    local_env.envs.close()
                                    cleanup_time = time.time() - cleanup_start
                                    logging.info(f"[PARALLEL] Task {env_idx + 1} cleanup took {cleanup_time:.3f}s")
                                except Exception as e:
                                    logging.warning(f"[PARALLEL] Task {env_idx + 1} cleanup failed: {e}")

                        # Run all envs in parallel using a thread pool (LLM calls are also IO-bound)
                        dump_lock = threading.Lock() if dump_fp is not None else None
                        env_results = []
                        
                        batch_parallel_start = time.time()
                        logging.info(f"[PARALLEL] Starting parallel execution of {current_batch_size} tasks with {args.concurrency} workers")
                        
                        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
                            futures = [ex.submit(_run_one, i) for i in range(current_batch_size)]
                            logging.info(f"[PARALLEL] All {len(futures)} tasks submitted to thread pool")
                            
                            completed_count = 0
                            for fut in as_completed(futures):
                                result = fut.result()
                                env_results.append(result)
                                completed_count += 1
                                logging.info(f"[PARALLEL] Task completed ({completed_count}/{current_batch_size}): Task {result['env_id'] + 1} in {result['timing']['total_time']:.3f}s")
                        
                        batch_parallel_time = time.time() - batch_parallel_start
                        logging.info(f"[PARALLEL] All {current_batch_size} tasks completed in {batch_parallel_time:.3f}s")
                        
                        # Analyze parallel performance
                        if env_results and "timing" in env_results[0]:
                            total_task_times = [r["timing"]["total_time"] for r in env_results]
                            avg_task_time = np.mean(total_task_times)
                            max_task_time = np.max(total_task_times)
                            min_task_time = np.min(total_task_times)
                            theoretical_sequential_time = sum(total_task_times)
                            parallel_efficiency = theoretical_sequential_time / (batch_parallel_time * current_batch_size) if batch_parallel_time > 0 else 0
                            
                            logging.info(f"[PARALLEL] Performance Analysis:")
                            logging.info(f"  Average task time: {avg_task_time:.3f}s")
                            logging.info(f"  Max task time: {max_task_time:.3f}s (bottleneck)")
                            logging.info(f"  Min task time: {min_task_time:.3f}s")
                            logging.info(f"  Theoretical sequential time: {theoretical_sequential_time:.3f}s")
                            logging.info(f"  Actual parallel time: {batch_parallel_time:.3f}s")
                            logging.info(f"  Speedup: {theoretical_sequential_time/batch_parallel_time:.2f}x")
                            logging.info(f"  Parallel efficiency: {parallel_efficiency:.2f} ({parallel_efficiency*100:.1f}%)")
                        
                        # Collect statistics from results
                        overall_success_this_round = np.array([r['won'] for r in env_results])
                        first_attempt_success_this_round = np.array([r['first_attempt_success'] for r in env_results])
                        task_success_cnt = defaultdict(int)
                        task_total_cnt = defaultdict(int)
                        
                        # Update overall statistics
                        total_tasks += len(env_results)
                        total_first_attempt_successes += first_attempt_success_this_round.sum()
                        total_debugger_successes += overall_success_this_round.sum()
                        
                        # Process results for task-specific statistics
                        for result in env_results:
                            task_id = result.get('task_id', f"task_{result['env_id']}")
                            task_total_cnt[task_id] = 1
                            if result['won']:
                                task_success_cnt[task_id] = 1
                        
                        # Calculate success rates
                        round_success_rate = overall_success_this_round.mean()
                        first_attempt_rate = first_attempt_success_this_round.mean()
                        
                        batch_overall_success_rates.append(round_success_rate)
                        all_first_attempt_success_rates.append(first_attempt_rate)
                        all_debugger_success_rates.append(round_success_rate)
                        
                        logging.info(f"Batch {batch_idx + 1} Test {test_idx} Results:")
                        logging.info(f"  First attempt success rate: {first_attempt_rate:.4f}")
                        logging.info(f"  Success rate after debugger: {round_success_rate:.4f}")
                        
                        # Log per-task results if needed
                        for task, total in task_total_cnt.items():
                            if total > 0:
                                rate = task_success_cnt.get(task, 0) / total
                                batch_task_success_history[task].append(rate)
                        
                        logging.info(f"Batch {batch_idx + 1} Test {test_idx} time elapsed: {time.time() - start_time:.2f}s\n")
                        continue  # Skip the normal rollout code below
                    # Normal rollout without debugger (original code)
                    env_manager = EnvironmentFactory.build_env(
                        args.env,
                        with_debugger=False,
                        **env_kwargs
                    )
                    obs, infos = env_manager.reset()
                    env_dones = [False] * current_batch_size
                    
                    # Set chat_base_dir from args
                    chat_base_dir = args.chat_root
                    
                    # Per-env chat buffers
                    chats = [[] for _ in range(current_batch_size)]
                    saved_flags = [False] * current_batch_size
                    last_infos = infos
                    
                    # Statistics for single round
                    overall_success_this_round = np.zeros(current_batch_size, dtype=bool)
                    task_success_cnt = defaultdict(int)
                    task_total_cnt = defaultdict(int)
                    
                    for step_idx in range(args.max_steps):
                        logging.info(f"Batch {batch_idx + 1} Step {step_idx}; Dones ({np.array(env_dones).sum()}/{current_batch_size}); SR {overall_success_this_round.mean():.3f}")
                        
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
                                        
                                        with open(out_path, "w", encoding="utf-8") as f:
                                            json.dump({"messages": chats[i], "metadata": meta}, f, ensure_ascii=False, indent=2)
                                        saved_flags[i] = True
                                    except Exception as e:
                                        logging.debug(f"Failed to save chat: {e}")
                        
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
                                    
                                    with open(out_path, "w", encoding="utf-8") as f:
                                        json.dump({"messages": chats[i], "metadata": meta}, f, ensure_ascii=False, indent=2)
                                    saved_flags[i] = True
                                except Exception as e:
                                    logging.debug(f"Failed to save unfinished chat: {e}")
                    
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
                
                # Clean up resources for non-debugger batch manager (if any)
                try:
                    if 'env_manager' in locals() and hasattr(env_manager, 'envs'):
                        env_manager.envs.close()
                        logging.info(f"Released resources for Batch {batch_idx + 1}")
                except Exception as e:
                    logging.warning(f"Failed to release resources: {e}")
                
                logging.info(f"========== Finished Batch {batch_idx + 1}/{num_batches}, processed {global_env_counter}/{args.total_envs} envs ==========\n")

        # ===== Fixed-pool parallel rollout (preferred path) =====
        pool_size = max(1, int(args.total_envs))
        rounds = max(1, int(args.test_times))

        logging.info(f"\n========== Fixed Env Pool ==========")
        logging.info(f"Parallel envs: {pool_size} | Rounds per env: {rounds}")

        if args.enable_debugger:
            # Build per-env managers (single-env each) once and reuse with reset()
            common_kwargs = {"env_num": 1, "seed": args.seed, "history_length": args.history_length}
            if args.env == "gaia":
                common_kwargs["available_tools"] = args.gaia_tools
                common_kwargs["max_steps"] = args.max_steps
                # Distribute trimmed tasks across envs
                per_env_tasks: List[List[Dict]] = [[] for _ in range(pool_size)]
                for k, task in enumerate(gaia_tasks or []):
                    per_env_tasks[k % pool_size].append(task)
            elif args.env == "alfworld":
                common_kwargs["alf_env_type"] = args.alf_env_type
                per_env_files: List[Optional[str]] = [None] * pool_size
                if args.unique_envs and alfworld_game_files:
                    take = min(len(alfworld_game_files), pool_size)
                    for i in range(take):
                        per_env_files[i] = alfworld_game_files[i]
            elif args.env == "webshop":
                common_kwargs["use_train_set"] = args.webshop_train

            env_pool = []
            for i in range(pool_size):
                kwargs_i = dict(common_kwargs)
                if args.env == "gaia":
                    kwargs_i["tasks_data"] = per_env_tasks[i] if gaia_tasks is not None else []
                if args.env == "alfworld" and (args.unique_envs and alfworld_game_files):
                    kwargs_i["game_files"] = [per_env_files[i]] if per_env_files[i] else None
                mgr = EnvironmentFactory.build_env(args.env, with_debugger=True, **kwargs_i)
                env_pool.append(mgr)

            def _run_one_round(env_idx: int, round_idx: int):
                # Reset to get task id and ensure fresh episode
                init_obs, init_infos = env_pool[env_idx].reset()
                info0 = init_infos[0] if isinstance(init_infos, list) else init_infos
                task_id = get_task_id(args.env, env_idx, info0, round_idx)

                task_dir = None
                if summaries_dir and args.save_per_task_trajectories:
                    task_dir = os.path.join(summaries_dir, _sanitize(task_id))
                    os.makedirs(task_dir, exist_ok=True)

                res = run_environment_with_retry(
                    env_id=0,
                    env_manager=env_pool[env_idx],
                    agent=agent,
                    max_steps=args.max_steps,
                    env_type=args.env,
                    debugger=debugger,
                    trajectory_manager=trajectory_manager,
                    max_retries=args.max_debug_retry,
                    dump_fp=dump_fp,
                    dump_lock=(threading.Lock() if dump_fp is not None else None),
                    chat_base_dir=None,
                    batch_idx=0,
                    test_idx=round_idx,
                    global_env_counter=env_idx,
                    run_ts=run_ts,
                    debug_output_dir=None,
                    save_all_attempts=args.save_all_attempts,
                    task_dir=task_dir,
                    shared_llm_executor=shared_llm_executor,
                )
                res["task_id"] = task_id
                res["env_id"] = env_idx
                return res

            for r in range(rounds):
                logging.info(f"\n========== Round {r + 1}/{rounds} ==========")
                env_results = []
                with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
                    futures = [ex.submit(_run_one_round, i, r) for i in range(pool_size)]
                    for fut in as_completed(futures):
                        env_results.append(fut.result())

                # Update stats
                overall = np.array([rr['won'] for rr in env_results])
                first_attempt = np.array([rr['first_attempt_success'] for rr in env_results])
                total_tasks += len(env_results)
                total_first_attempt_successes += first_attempt.sum()
                total_debugger_successes += overall.sum()
                all_first_attempt_success_rates.append(first_attempt.mean())
                all_debugger_success_rates.append(overall.mean())

            # Close pool
            for mgr in env_pool:
                try:
                    mgr.envs.close()
                except Exception:
                    pass

            global_env_counter = pool_size

        else:
            # Without debugger: build a single multi-env manager once and reuse
            env_kwargs = {
                "env_num": pool_size,
                "seed": args.seed,
                "history_length": args.history_length,
            }
            if args.env == "gaia":
                # Distribute trimmed tasks across envs as a flat list
                env_kwargs["tasks_data"] = gaia_tasks
                env_kwargs["available_tools"] = args.gaia_tools
                env_kwargs["max_steps"] = args.max_steps
            elif args.env == "alfworld":
                env_kwargs["alf_env_type"] = args.alf_env_type
                if args.unique_envs and alfworld_game_files:
                    env_kwargs["game_files"] = alfworld_game_files[:pool_size]
            elif args.env == "webshop":
                env_kwargs["use_train_set"] = args.webshop_train

            env_manager = EnvironmentFactory.build_env(args.env, with_debugger=False, **env_kwargs)

            # Repeat for a number of rounds; each round calls reset() and steps to done
            for test_idx in range(rounds):
                obs, infos = env_manager.reset()
                env_dones = [False] * pool_size
                overall_success_this_round = np.zeros(pool_size, dtype=bool)

                for step_idx in range(args.max_steps):
                    # Collect prompts for active envs
                    prompts, idx_map = [], []
                    for i in range(pool_size):
                        if not env_dones[i]:
                            prompts.append(obs["text"][i])
                            idx_map.append(i)
                    if not prompts:
                        break

                    batch_actions = agent.get_actions_batch(prompts, concurrency=args.concurrency, retries=args.retries)
                    actions = ["None"] * pool_size
                    for k, i in enumerate(idx_map):
                        actions[i] = batch_actions[k]

                    prev_prompts = obs["text"]
                    raw_actions = actions.copy()
                    obs, rewards, dones, infos = env_manager.step(actions.copy())

                    for i in range(pool_size):
                        if env_dones[i]:
                            continue
                        if dones[i]:
                            env_dones[i] = True
                            overall_success_this_round[i] = bool(infos[i].get("won", False))
                    if all(env_dones):
                        break

                # Update simple aggregate for non-debugger path
                all_overall_success_rates.append(overall_success_this_round.mean())

            try:
                env_manager.envs.close()
            except Exception:
                pass
            global_env_counter = pool_size

    finally:
        # Clean up shared LLM executor
        if 'shared_llm_executor' in locals():
            shared_llm_executor.shutdown(wait=True)
            logging.info("Shared LLM executor pool shut down")
        
        if dump_fp is not None:
            dump_fp.flush()
            dump_fp.close()
            logging.info(f"Trajectories saved to: {args.dump_path}")
    
    # Final summary
    logging.info("=============== Final Summary ===============")
    logging.info(f"Environment: {args.env}")
    logging.info(f"Total batches: {num_batches} | Parallel envs: {max(1, int(args.total_envs))} | Total tasks run: {total_tasks}")
    
    # Report both first attempt and debugger-assisted success rates
    if args.enable_debugger and total_tasks > 0:
        first_attempt_success_rate = total_first_attempt_successes / total_tasks
        debugger_success_rate = total_debugger_successes / total_tasks
        improvement = debugger_success_rate - first_attempt_success_rate
        
        logging.info("\n========== Success Rate Analysis ==========")
        logging.info(f"First Attempt Success Rate: {first_attempt_success_rate:.4f} ({total_first_attempt_successes}/{total_tasks})")
        logging.info(f"Success Rate with Debugger: {debugger_success_rate:.4f} ({total_debugger_successes}/{total_tasks})")
        logging.info(f"Improvement from Debugger: +{improvement:.4f} ({improvement*100:.2f}%)")
        
        if all_first_attempt_success_rates:
            logging.info(
                f"First Attempt (avg ± std): "
                f"{np.mean(all_first_attempt_success_rates):.4f} ± {np.std(all_first_attempt_success_rates):.4f}"
            )
        if all_debugger_success_rates:
            logging.info(
                f"With Debugger (avg ± std): "
                f"{np.mean(all_debugger_success_rates):.4f} ± {np.std(all_debugger_success_rates):.4f}"
            )
    elif all_overall_success_rates:
        logging.info(
            f"Overall success avg ± std: "
            f"{np.mean(all_overall_success_rates):.4f} ± {np.std(all_overall_success_rates):.4f}"
        )
    
    # Save final experiment summary to file if experiment_dir is set
    if summaries_dir:
        summary_file = os.path.join(summaries_dir, "experiment_summary.json")
        experiment_summary = {
            "experiment_info": {
                "environment": args.env,
                "model": args.model,
                "temperature": args.temperature,
                "debugger_enabled": args.enable_debugger,
                "debugger_model": args.debugger_model if args.enable_debugger else None,
                "debugger_temperature": args.debugger_temperature if args.enable_debugger else None,
                "max_retries": args.max_debug_retry if args.enable_debugger else 1,
                "total_tasks": total_tasks,
                "total_batches": num_batches,
                "batch_size": args.batch_size,
                "max_steps": args.max_steps,
                "timestamp": run_ts
            },
            "results": {
                "first_attempt_success_rate": float(total_first_attempt_successes) / total_tasks if total_tasks > 0 else 0,
                "debugger_success_rate": float(total_debugger_successes) / total_tasks if total_tasks > 0 else 0,
                "improvement": (float(total_debugger_successes) - float(total_first_attempt_successes)) / total_tasks if total_tasks > 0 else 0,
                "first_attempt_successes": int(total_first_attempt_successes),
                "debugger_successes": int(total_debugger_successes),
                "total_tasks": int(total_tasks)
            },
            "statistics": {
                "first_attempt_mean": float(np.mean(all_first_attempt_success_rates)) if all_first_attempt_success_rates else 0,
                "first_attempt_std": float(np.std(all_first_attempt_success_rates)) if all_first_attempt_success_rates else 0,
                "debugger_mean": float(np.mean(all_debugger_success_rates)) if all_debugger_success_rates else 0,
                "debugger_std": float(np.std(all_debugger_success_rates)) if all_debugger_success_rates else 0
            }
        }
        
        with open(summary_file, "w") as f:
            json.dump(experiment_summary, f, indent=2)
        logging.info(f"\nExperiment summary saved to: {summary_file}")
    
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
