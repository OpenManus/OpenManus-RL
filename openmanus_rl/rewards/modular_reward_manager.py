"""
Modular Reward Manager for combining environment and model-based rewards
"""

import numpy as np
import requests
import asyncio
import aiohttp
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModuleContent:
    """Container for module content extracted from trajectory"""
    module_name: str
    content: str
    step_num: int

class ModularRewardManager:
    """
    Manages reward computation by combining environment rewards with module-specific evaluations.
    Supports four cognitive modules: planner, executor, reflection, memory_use
    """
    
    def __init__(
        self,
        reward_model_url: str = "http://localhost:8100",
        env_weight: float = 0.5,
        model_weight: float = 0.5,
        timeout: int = 30
    ):
        """
        Initialize the reward manager.
        
        Args:
            reward_model_url: URL of the reward model server
            env_weight: Weight for environment rewards
            model_weight: Weight for model-based rewards
            timeout: Request timeout in seconds
        """
        self.reward_model_url = reward_model_url
        self.env_weight = env_weight
        self.model_weight = model_weight
        self.timeout = timeout
        self.session = None
        
        # Module extraction patterns
        self.module_patterns = {
            'memory_use': r'<memory_recall>(.*?)</memory_recall>',
            'reflection': r'<reflection>(.*?)</reflection>',
            'planner': r'<think>(.*?)</think>',
            'executor': r'<action>(.*?)</action>'
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def extract_module_content(
        self,
        trajectory: str,
        module_name: str,
        step_num: int
    ) -> Optional[ModuleContent]:
        """
        Extract specific module content from trajectory.
        
        Args:
            trajectory: Raw trajectory text
            module_name: Name of module to extract
            step_num: Current step number
            
        Returns:
            ModuleContent or None if not found
        """
        pattern = self.module_patterns.get(module_name)
        if not pattern:
            logger.warning(f"Unknown module: {module_name}")
            return None
        
        match = re.search(pattern, trajectory, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            if content:
                return ModuleContent(
                    module_name=module_name,
                    content=content,
                    step_num=step_num
                )
        return None
    
    def extract_task_description(self, trajectory_data: Dict) -> str:
        """Extract task description from trajectory data"""
        # Try to extract from chat history
        chat_history = trajectory_data.get('chat_history', [])
        for msg in chat_history:
            if msg.get('role') == 'user' and 'task is to:' in msg.get('content', ''):
                task_match = re.search(r'Your task is to: (.+?)(?:\n|\.)', msg['content'])
                if task_match:
                    return task_match.group(1).strip()
        
        # Fallback to metadata
        return trajectory_data.get('task_description', 'Unknown task')
    
    def analyze_failure_context(
        self,
        trajectory_data: Dict,
        step_num: int
    ) -> Dict[str, Any]:
        """
        Analyze failure patterns up to current step.
        
        Args:
            trajectory_data: Full trajectory data
            step_num: Current step number
            
        Returns:
            Dictionary with failure statistics
        """
        chat_history = trajectory_data.get('chat_history', [])
        
        consecutive_failures = 0
        same_strategy_repeats = 0
        total_failures = 0
        last_action = None
        
        failure_indicators = [
            "nothing happens",
            "failed to",
            "cannot",
            "unable to",
            "error"
        ]
        
        current_step = 0
        for i, msg in enumerate(chat_history):
            if msg['role'] == 'assistant':
                current_step += 1
                if current_step > step_num:
                    break
                
                # Get environment response
                env_response = ""
                if i + 1 < len(chat_history) and chat_history[i + 1]['role'] == 'user':
                    env_response = chat_history[i + 1]['content'].lower()
                
                # Check for failure
                is_failure = any(indicator in env_response for indicator in failure_indicators)
                if is_failure:
                    consecutive_failures += 1
                    total_failures += 1
                else:
                    consecutive_failures = 0
                
                # Extract action for repetition check
                action_match = re.search(r'<action>(.*?)</action>', msg['content'], re.DOTALL)
                if action_match:
                    current_action = action_match.group(1).strip().lower()
                    if current_action == last_action:
                        same_strategy_repeats += 1
                    else:
                        same_strategy_repeats = 0
                    last_action = current_action
        
        return {
            'consecutive_failures': consecutive_failures,
            'same_strategy_repeats': same_strategy_repeats,
            'total_failures': total_failures
        }
    
    async def evaluate_module(
        self,
        module_content: ModuleContent,
        task_description: str,
        failure_context: Dict[str, Any]
    ) -> float:
        """
        Get evaluation score from reward model for a specific module.
        
        Args:
            module_content: Extracted module content
            task_description: Task description
            failure_context: Failure analysis context
            
        Returns:
            Normalized score (0-1)
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            payload = {
                "module_name": module_content.module_name,
                "trajectory": module_content.content,
                "task_description": task_description,
                "step_num": module_content.step_num,
                "failure_context": failure_context
            }
            
            async with self.session.post(
                f"{self.reward_model_url}/evaluate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('score', 0.0)
                else:
                    logger.error(f"Reward model returned status {response.status}")
                    return 0.0
                    
        except Exception as e:
            logger.error(f"Error calling reward model: {e}")
            return 0.0
    
    async def compute_modular_rewards(
        self,
        trajectories: List[Dict],
        env_rewards: np.ndarray,
        current_module: str
    ) -> np.ndarray:
        """
        Compute combined rewards for trajectories focusing on a specific module.
        
        Args:
            trajectories: List of trajectory dictionaries
            env_rewards: Environment rewards array
            current_module: Module to evaluate in this iteration
            
        Returns:
            Combined rewards array
        """
        batch_size = len(trajectories)
        model_rewards = np.zeros(batch_size)
        
        # Process each trajectory
        tasks = []
        for i, traj_data in enumerate(trajectories):
            # Extract task description
            task_desc = self.extract_task_description(traj_data)
            
            # Process each step in trajectory
            chat_history = traj_data.get('chat_history', [])
            step_num = 0
            step_rewards = []
            
            for j, msg in enumerate(chat_history):
                if msg['role'] == 'assistant':
                    step_num += 1
                    
                    # Extract module content
                    module_content = self.extract_module_content(
                        trajectory=msg['content'],
                        module_name=current_module,
                        step_num=step_num
                    )
                    
                    if module_content:
                        # Analyze failure context
                        failure_context = self.analyze_failure_context(traj_data, step_num)
                        
                        # Create evaluation task
                        task = self.evaluate_module(
                            module_content=module_content,
                            task_description=task_desc,
                            failure_context=failure_context
                        )
                        tasks.append((i, task))
                        step_rewards.append(task)
            
            # If we have step rewards, we'll aggregate them later
            if step_rewards:
                traj_data['_step_rewards'] = step_rewards
        
        # Execute all evaluation tasks concurrently
        if tasks:
            results = await asyncio.gather(*[task for _, task in tasks])
            
            # Aggregate rewards per trajectory
            result_idx = 0
            for i, traj_data in enumerate(trajectories):
                if '_step_rewards' in traj_data:
                    step_count = len(traj_data['_step_rewards'])
                    if step_count > 0:
                        # Average the step rewards for this trajectory
                        step_scores = results[result_idx:result_idx + step_count]
                        model_rewards[i] = np.mean(step_scores)
                        result_idx += step_count
                    del traj_data['_step_rewards']
        
        # Combine environment and model rewards
        combined_rewards = (
            self.env_weight * env_rewards +
            self.model_weight * model_rewards
        )
        
        logger.info(f"Module {current_module} - Env rewards: {env_rewards.mean():.3f}, "
                   f"Model rewards: {model_rewards.mean():.3f}, "
                   f"Combined: {combined_rewards.mean():.3f}")
        
        return combined_rewards
    
    def get_module_order(self, epoch: int, seed: int = 42) -> List[str]:
        """
        Get randomized module order for current epoch.
        
        Args:
            epoch: Current epoch number
            seed: Random seed (default 42)
            
        Returns:
            List of module names in order
        """
        modules = ['planner', 'executor', 'reflection', 'memory_use']
        
        # Set seed based on epoch
        np.random.seed(seed + epoch)
        np.random.shuffle(modules)
        
        return modules
