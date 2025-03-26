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
        
        Args:
            generation_config: Generation configuration for the model
            max_rounds: Maximum number of interaction rounds
            idxs: List of task IDs to run (defaults to all tasks)
            save_to_storage: Whether to save trajectories to storage
            parallel: Whether to run rollouts in parallel
            batch_size: Number of tasks to process in each batch
            metadata: Additional metadata to store with trajectories
            
        Returns:
            List of ExperienceOutput from rollout
        """
        if idxs is None:
            idxs = []
            for task in self.tasks:
                idxs.append(list(range(len(task.clients[0]))))
        elif isinstance(idxs[0], int):
            # Single list of indices for the first task
            idxs = [idxs] + [[] for _ in range(len(self.tasks) - 1)]
        
        # Use the first task for simplicity
        # This could be extended to support multiple tasks concurrently
        task = self.tasks[0]
        task_idxs = idxs[0]
        
        results = []
        
        if parallel:
            # Process in batches for memory efficiency
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
                except Exception