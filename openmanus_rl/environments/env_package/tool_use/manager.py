"""
Tool Use Environment Manager
"""

import json
import re
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np

from openmanus_rl.environments.base import EnvironmentManagerBase, to_numpy
from openmanus_rl.memory import SimpleMemory
from openmanus_rl.environments.prompts import *


class ToolUseEnvironmentManager(EnvironmentManagerBase):
    """Environment manager for tool use tasks"""
    
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
        self.memory = SimpleMemory()
        self.current_tasks = []
        self.ground_truths = []
        self.step_counts = []
        self.task_completed = []
        self.task_success = []
        
    def reset(self):
        """Reset environment and get new tasks"""
        obs, infos = self.envs.reset()
        
        # Extract task information
        self.current_tasks = [info.get('task', '') for info in infos]
        self.ground_truths = [info.get('answer', '') for info in infos]
        self.tool_metadata = infos[0].get('tool_metadata', '') if infos else ''
        
        batch_size = len(self.current_tasks)
        self.step_counts = [0] * batch_size
        self.task_completed = [False] * batch_size
        self.task_success = [False] * batch_size
        
        # Initialize memory
        self.memory.reset(batch_size=batch_size)
        
        # Build initial text observation
        full_text_obs = self.build_text_obs(init=True)
        
        return {'text': full_text_obs, 'image': None, 'anchor': self.current_tasks.copy()}, infos
        
    def step(self, text_actions: List[str]):
        """Execute text actions"""
        actions, valids = self.projection_f(text_actions)
        batch_size = len(text_actions)
        
        # Process actions and execute tools
        observations = []
        rewards = np.zeros(batch_size) 
        dones = np.zeros(batch_size, dtype=bool)
        infos = []
        
        for i, (action, valid) in enumerate(zip(actions, valids)):
            if self.task_completed[i]:
                observations.append("Task completed.")
                infos.append({'is_action_valid': True, 'won': self.task_success[i]})
                continue
                
            self.step_counts[i] += 1
            
            # Process action
            obs, info = self._process_action(action, i)
            observations.append(obs)
            
            # Check completion
            if self._is_completion_action(action):
                is_correct = self._evaluate_answer(action, i)
                self.task_success[i] = is_correct
                self.task_completed[i] = True
                dones[i] = True
                obs_feedback = "\n\nEvaluation: final answer matches the ground truth." if is_correct else "\n\nEvaluation: final answer does not match the ground truth."
                observations[-1] = obs + obs_feedback
            elif self.step_counts[i] >= self.config.env.max_steps:
                obs += "\n\nMaximum steps reached. Please provide your final answer in <answer></answer> tags."
                dones[i] = True
                observations[-1] = obs
                
            info['is_action_valid'] = to_numpy(valid)
            info['won'] = self.task_success[i]
            info['step_count'] = self.step_counts[i]
            infos.append(info)
        
        # After processing all envs, store this step's observations and actions into memory
        try:
            self.memory.store({'text_obs': observations, 'action': text_actions})
        except Exception:
            # Be permissive: if memory storage fails, continue without history
            pass

        # Build text observations
        full_text_obs = self.build_text_obs(observations=observations)
        
        next_observations = {'text': full_text_obs, 'image': None, 'anchor': observations.copy()}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        
        return next_observations, rewards, dones, infos
        
    def _process_action(self, action: str, batch_idx: int) -> tuple:
        """Process a single action"""
        info = {'is_action_valid': True}
        
        try:
            # Try to parse as JSON (from projection)
            action_data = json.loads(action)
            if action_data.get('type') == 'tool_call':
                # Execute tool
                tool_name = action_data['tool']
                params = action_data['parameters']
                result = self.envs.envs[batch_idx].tool_manager.execute_tool(tool_name, params)
                observation = f"Tool '{tool_name}' executed.\nResult: {result}"
            else:
                observation = "Action acknowledged."
        except (json.JSONDecodeError, KeyError):
            # Regular action or final answer
            if action.startswith("FINAL_ANSWER:"):
                observation = "Final answer provided. Task completed."
            else:
                observation = "Action acknowledged. Continue reasoning or use tools to gather information."
        
        return observation, info
        
    def _is_completion_action(self, action: str) -> bool:
        """Check if action indicates task completion"""
        return action.startswith("FINAL_ANSWER:") or "<answer>" in action
        
    def _evaluate_answer(self, action: str, batch_idx: int) -> bool:
        """Compare model answer with ground truth"""
        predicted = self._extract_answer_text(action)
        ground_truth = self.ground_truths[batch_idx]
        return self._normalize_answer(predicted) == self._normalize_answer(ground_truth)

    @staticmethod
    def _extract_answer_text(action: str) -> str:
        """Extract answer text from action string"""
        if action.startswith("FINAL_ANSWER:"):
            return action.split("FINAL_ANSWER:", 1)[1].strip()

        match = re.search(r"<answer>(.*?)</answer>", action, re.DOTALL)
        if match:
            return match.group(1).strip()
        return action.strip()

    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Normalize answer string for comparison"""
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        normalized = normalized.strip(".,!?:;\"")
        return normalized

    def build_text_obs(self, observations: List[str] = None, init: bool = False) -> List[str]:
        """Build text observations for agent"""
        batch_size = len(self.current_tasks)
        postprocess_text_obs = []
        max_steps = getattr(self.config.env, "max_steps", None)
        history_length_cfg = getattr(self.config.env, "history_length", 0)

        if not init and history_length_cfg > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                history_length_cfg,
                obs_key="text_obs",
                action_key="action",
            )
        else:
            memory_contexts = [""] * batch_size
            valid_lens = [0] * batch_size

        for i in range(batch_size):
            current_obs = observations[i] if observations else "Continue with your task."
            should_use_last_step = (
                not init
                and not self.task_completed[i]
                and max_steps is not None
                and self.step_counts[i] >= max_steps - 1
            )

            if init:
                obs = TOOL_USE_TEMPLATE_NO_HIS.format(
                    task_description=self.current_tasks[i],
                    available_tools=self.tool_metadata,
                    current_observation="Start working on the task."
                )
            elif should_use_last_step:
                obs = TOOL_USE_TEMPLATE_LAST_STEP.format(
                    task_description=self.current_tasks[i],
                    step_count=self.step_counts[i],
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=self.step_counts[i] + 1,
                    current_observation=current_obs,
                )
            elif history_length_cfg <= 0:
                obs = TOOL_USE_TEMPLATE_NO_HIS.format(
                    task_description=self.current_tasks[i],
                    available_tools=self.tool_metadata,
                    current_observation=current_obs,
                )
            else:
                obs = TOOL_USE_TEMPLATE.format(
                    task_description=self.current_tasks[i],
                    step_count=self.step_counts[i],
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=self.step_counts[i] + 1,
                    current_observation=current_obs,
                    available_tools=self.tool_metadata
                )

            postprocess_text_obs.append(obs)
            
        return postprocess_text_obs
        
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        """Process batch for success evaluation"""
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return
