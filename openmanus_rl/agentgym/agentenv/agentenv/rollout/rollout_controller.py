from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from agentenv.controller import Agent, BaseAgentEnvController, BaseTask
from agentenv.controller.task import ExperienceOutput
from transformers import GenerationConfig

from .rollout_strategy import IRolloutStrategy, StandardReActStrategy
from .rollout_db import ITrajectoryStorage, MongoDBTrajectoryStorage


class RolloutController(BaseAgentEnvController):
    """
    Advanced rollout controller for AgentGym that extends BaseAgentEnvController
    and supports multiple rollout strategies and trajectory storage.
    """
    
    def __init__(
        self, 
        agent: Agent, 
        tasks: List[BaseTask], 
        strategy: Optional[IRolloutStrategy] = None,
        storage: Optional[ITrajectoryStorage] = None,
        max_workers: int = 10
    ):
        """
        Initialize rollout controller with agent, tasks, strategy, and storage.
        
        Args:
            agent: Agent instance with model and tokenizer
            tasks: List of BaseTask instances
            strategy: Rollout strategy to use (defaults to StandardReActStrategy)
            storage: Trajectory storage implementation
            max_workers: Maximum number of worker threads for parallel rollout
        """
        super().__init__(agent, tasks)
        self.strategy = strategy or StandardReActStrategy()
        self.storage = storage
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def set_strategy(self, strategy: IRolloutStrategy):
        """Change the rollout strategy"""
        self.strategy = strategy
    
    def set_storage(self, storage: ITrajectoryStorage):
        """Set or change the trajectory storage"""
        self.storage = storage
    
    def get_storage(self) -> Optional[ITrajectoryStorage]:
        """Get the current trajectory storage instance
        
        Returns:
            The current storage instance or None if not set
        """
        return self.storage
    
    def rollout(
        self, 
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        idxs: Optional[List[int]] = None,
        save_to_storage: bool = True,
        parallel: bool = True,
        batch_size: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[ExperienceOutput]:
        """
        Execute rollout using the selected strategy.
        """
        if not save_to_storage or self.storage is None:
            save_to_storage = False
        
        if idxs is None:
            idxs = []
            for task in self.tasks:
                idxs.append(list(range(len(task.clients[0]))))
        elif isinstance(idxs[0], int):
            idxs = [idxs] + [[] for _ in range(len(self.tasks) - 1)]

        task = self.tasks[0]
        task_idxs = idxs[0]
        
        results = []
        
        if parallel:
            # Process in batches
            for i in range(0, len(task_idxs), batch_size):
                batch_idxs = task_idxs[i:i+batch_size]
                
                # Submit tasks to thread pool
                futures = {}
                for idx in batch_idxs:
                    future = self.executor.submit(
                        self._rollout_one, 
                        task=task,
                        idx=idx, 
                        generation_config=generation_config, 
                        max_rounds=max_rounds,
                        save_to_storage=save_to_storage,
                        metadata=metadata
                    )
                    futures[future] = idx
                
                # Collect results
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        exp_outputs = future.result()
                        results.extend(exp_outputs)
                    except Exception as e:
                        print(f"Error in rollout for task {idx}: {e}")
        else:
            # Sequential processing
            for idx in task_idxs:
                try:
                    exp_outputs = self._rollout_one(
                        task=task,
                        idx=idx,
                        generation_config=generation_config,
                        max_rounds=max_rounds,
                        save_to_storage=save_to_storage,
                        metadata=metadata
                    )
                    results.extend(exp_outputs)
                except Exception as e:
                    print(f"Error in rollout for task {idx}: {e}")
        
        return results
    
    def _rollout_one(
        self, 
        task: BaseTask,
        idx: int, 
        generation_config: Optional[GenerationConfig],
        max_rounds: Optional[int],
        save_to_storage: bool,
        metadata: Optional[Dict[str, Any]]
    ) -> List[ExperienceOutput]:
        """
        Execute rollout for a single task.
        
        Args:
            task: Task to run
            idx: Task ID
            generation_config: Generation configuration
            max_rounds: Maximum rounds
            save_to_storage: Whether to save trajectories
            metadata: Additional metadata
            
        Returns:
            List of ExperienceOutput objects
        """
        # Get client
        client = task.clients[0]
        
        # Reset environment
        client.reset(idx)
        
        # Get initial observation
        initial_observation = client.observe()
        
        # Execute strategy
        trajectories = self.strategy.execute(
            model=self.agent.model,
            tokenizer=self.agent.tokenizer,
            client=client,
            initial_observation=initial_observation,
            generation_config=generation_config,
            max_rounds=max_rounds
        )
        
        # Save trajectories if requested
        if save_to_storage and self.storage is not None and trajectories:
            self._save_trajectories(task.env_name, idx, trajectories, metadata)
        
        return trajectories
    
    def _save_trajectories(
        self, 
        env_name: str, 
        task_id: int, 
        trajectories: List[ExperienceOutput],
        metadata: Optional[Dict[str, Any]]
    ) -> List[str]:
        """
        Save trajectories using the storage.
        
        Args:
            env_name: Environment name
            task_id: Task ID
            trajectories: List of ExperienceOutput objects
            metadata: Additional metadata
            
        Returns:
            List of trajectory IDs
        """
        # Convert ExperienceOutput to dict format for storage
        trajectory_dicts = []
        
        for traj in trajectories:
            traj_dict = {
                "reward": float(traj.reward),
                "conversation": [
                    {
                        "from": msg["from"],
                        "value": msg["value"],
                        "loss": msg.get("loss")
                    } for msg in traj.conversation
                ],
                "text": traj.text
            }
            trajectory_dicts.append(traj_dict)
        
        # Save using storage
        trajectory_ids = self.storage.save_trajectories(
            env_name=env_name,
            task_ids=[task_id] * len(trajectories),
            trajectories=trajectory_dicts,
            metadata=metadata
        )
        
        return trajectory_ids