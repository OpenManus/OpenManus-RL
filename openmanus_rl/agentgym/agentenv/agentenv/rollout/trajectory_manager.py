from typing import Dict, List, Any, Optional, Union, TypeVar, Generic
import uuid
import json
import os
from datetime import datetime
import pymongo
from tensordict import TensorDict
import torch
import numpy as np
import verl


class TrajectoryManager:
    """
    Manages trajectory storage and retrieval operations, decoupled from rollout logic.
    Supports different storage backends and conversion to Verl DataProto format.
    """
    
    def __init__(self, storage_backend: str = 'mongodb', connection_config: Dict[str, Any] = None):
        """
        Initialize the trajectory manager with the specified storage backend.
        
        Args:
            storage_backend: Type of storage ('mongodb', 'file', 'memory')
            connection_config: Configuration for connecting to the storage
        """
        self.storage_backend = storage_backend
        self.connection_config = connection_config or {}
        
        # Initialize the appropriate storage backend
        if storage_backend == 'mongodb':
            self._init_mongodb()
        elif storage_backend == 'file':
            self._init_file_storage()
        elif storage_backend == 'memory':
            self._init_memory_storage()
        else:
            raise ValueError(f"Unsupported storage backend: {storage_backend}")
    
    def _init_mongodb(self):
        """Initialize MongoDB storage backend"""
        conn_str = self.connection_config.get('connection_string', 'mongodb://localhost:27017/')
        db_name = self.connection_config.get('db_name', 'agentgym')
        collection_name = self.connection_config.get('collection_name', 'trajectories')
        
        self.client = pymongo.MongoClient(conn_str)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        
        # Create indexes for efficient querying
        self.collection.create_index([("env_name", 1)])
        self.collection.create_index([("task_id", 1)])
        self.collection.create_index([("timestamp", -1)])
        self.collection.create_index([("reward", -1)])
    
    def _init_file_storage(self):
        """Initialize file-based storage backend"""
        self.file_path = self.connection_config.get('file_path', 'trajectories.jsonl')
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
        
        # Ensure file exists
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                pass
    
    def _init_memory_storage(self):
        """Initialize in-memory storage backend"""
        self.memory_storage = []
    
    def save_trajectory(self, env_name: str, task_id: int, trajectory: Dict, metadata: Dict = None) -> str:
        """
        Save a single trajectory to storage.
        
        Args:
            env_name: Name of the environment
            task_id: ID of the task
            trajectory: Trajectory data
            metadata: Additional metadata
            
        Returns:
            Unique ID of the saved trajectory
        """
        trajectory_id = str(uuid.uuid4())
        
        # Create document
        doc = {
            "trajectory_id": trajectory_id,
            "env_name": env_name,
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "reward": float(trajectory.get("reward", 0.0)),
            "conversation": trajectory.get("conversation", []),
            "metadata": metadata or {}
        }
        
        # Save according to backend
        if self.storage_backend == 'mongodb':
            self.collection.insert_one(doc)
        elif self.storage_backend == 'file':
            with open(self.file_path, 'a') as f:
                f.write(json.dumps(doc) + '\n')
        elif self.storage_backend == 'memory':
            self.memory_storage.append(doc)
        
        return trajectory_id
    
    def save_trajectories(self, env_name: str, task_ids: List[int], trajectories: List[Dict], metadata: Dict = None) -> List[str]:
        """
        Save multiple trajectories to storage.
        
        Args:
            env_name: Name of the environment
            task_ids: List of task IDs
            trajectories: List of trajectory data
            metadata: Additional metadata
            
        Returns:
            List of unique IDs of the saved trajectories
        """
        if len(task_ids) != len(trajectories):
            raise ValueError("Number of task IDs must match number of trajectories")
        
        trajectory_ids = []
        
        # MongoDB can do bulk insert
        if self.storage_backend == 'mongodb':
            docs = []
            for task_id, trajectory in zip(task_ids, trajectories):
                trajectory_id = str(uuid.uuid4())
                trajectory_ids.append(trajectory_id)
                
                doc = {
                    "trajectory_id": trajectory_id,
                    "env_name": env_name,
                    "task_id": task_id,
                    "timestamp": datetime.now().isoformat(),
                    "reward": float(trajectory.get("reward", 0.0)),
                    "conversation": trajectory.get("conversation", []),
                    "metadata": metadata or {}
                }
                docs.append(doc)
            
            if docs:
                self.collection.insert_many(docs)
        else:
            # File and memory backends handle one at a time
            for task_id, trajectory in zip(task_ids, trajectories):
                trajectory_id = self.save_trajectory(env_name, task_id, trajectory, metadata)
                trajectory_ids.append(trajectory_id)
        
        return trajectory_ids
    
    def get_trajectory(self, trajectory_id: str) -> Optional[Dict]:
        """
        Retrieve a trajectory by ID.
        
        Args:
            trajectory_id: Unique ID of the trajectory
            
        Returns:
            Trajectory data or None if not found
        """
        if self.storage_backend == 'mongodb':
            doc = self.collection.find_one({"trajectory_id": trajectory_id})
            return doc
        elif self.storage_backend == 'file':
            with open(self.file_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    doc = json.loads(line.strip())
                    if doc.get("trajectory_id") == trajectory_id:
                        return doc
            return None
        elif self.storage_backend == 'memory':
            for doc in self.memory_storage:
                if doc.get("trajectory_id") == trajectory_id:
                    return doc
            return None
    
    def _trajectory_to_tensor_dict(self, trajectory: Dict) -> TensorDict:
        """
        Convert a trajectory to a TensorDict format suitable for Verl.
        
        Args:
            trajectory: Trajectory data
            
        Returns:
            TensorDict representation of the trajectory
        """
        # Extract relevant data from trajectory
        conversation = trajectory.get("conversation", [])
        
        # Create tensor representations
        observations = []
        actions = []
        rewards = []
        
        for i, msg in enumerate(conversation):
            if msg.get("from") == "human":
                # This is an observation
                observations.append(msg.get("value", ""))
            elif msg.get("from") == "gpt":
                # This is an action
                actions.append(msg.get("value", ""))
                
                # If not the last message, we can extract reward
                if i < len(conversation) - 1:
                    rewards.append(0)  # Intermediate rewards are 0
        
        # Add final reward
        final_reward = trajectory.get("reward", 0.0)
        if rewards:
            rewards[-1] = final_reward
        
        # Convert to tensors
        dummy_obs = torch.zeros(len(observations), 10)  # Placeholder
        dummy_actions = torch.zeros(len(actions), 5)  # Placeholder
        reward_tensor = torch.tensor(rewards + [final_reward])
        
        # Create TensorDict
        td = TensorDict({
            "observation": dummy_obs,
            "action": dummy_actions,
            "reward": reward_tensor,
            "done": torch.tensor([0] * (len(rewards)) + [1]),  # Last step is done
        }, batch_size=[len(rewards) + 1])
        
        return td
    
    def to_data_proto(self, trajectories: List[Dict]) -> 'verl.DataProto':
        """
        Convert trajectories to Verl's DataProto format.
        
        Args:
            trajectories: List of trajectory data
            
        Returns:
            Verl DataProto containing the trajectories
        """
        if not HAS_VERL:
            raise ImportError("Verl is not installed. Please install it to use this feature.")
        
        # Convert each trajectory to TensorDict
        tensor_dicts = [self._trajectory_to_tensor_dict(traj) for traj in trajectories]
        
        # Combine TensorDicts if there are multiple
        if len(tensor_dicts) > 1:
            # This is a simplified approach - may need adjustment based on your exact needs
            combined_td = torch.cat(tensor_dicts, dim=0)
        elif len(tensor_dicts) == 1:
            combined_td = tensor_dicts[0]
        else:
            # Empty batch
            combined_td = TensorDict({}, batch_size=[0])
        
        # Create meta information
        meta_info = {
            "env_names": [traj.get("env_name") for traj in trajectories],
            "task_ids": [traj.get("task_id") for traj in trajectories],
            "trajectory_ids": [traj.get("trajectory_id") for traj in trajectories],
            "timestamp": datetime.now().isoformat()
        }
        
        # Create DataProto
        data_proto = verl.DataProto(batch=combined_td, meta_info=meta_info)
        
        return data_proto
    
    @classmethod
    def from_data_proto(cls, data_proto: 'verl.DataProto') -> List[Dict]:
        """
        Convert a Verl DataProto back to a list of trajectories.
        
        Args:
            data_proto: Verl DataProto containing trajectories
            
        Returns:
            List of trajectory dictionaries
        """     
        # Extract TensorDict and meta information
        batch = data_proto.batch
        meta_info = data_proto.meta_info
        
        # Extract relevant data
        env_names = meta_info.get("env_names", [])
        task_ids = meta_info.get("task_ids", [])
        trajectory_ids = meta_info.get("trajectory_ids", [])
        
        # Determine number of trajectories
        num_trajectories = len(env_names)
        
        # Convert back to trajectory dictionaries
        trajectories = []
        
        for i in range(num_trajectories):
            trajectory = {
                "trajectory_id": trajectory_ids[i] if i < len(trajectory_ids) else str(uuid.uuid4()),
                "env_name": env_names[i] if i < len(env_names) else "unknown",
                "task_id": task_ids[i] if i < len(task_ids) else -1,
                "reward": 0.0,
                "conversation": []
            }
            trajectories.append(trajectory)
        
        return trajectories
    
    def delete_trajectory(self, trajectory_id: str) -> bool:
        """
        Delete a trajectory by ID.
        
        Args:
            trajectory_id: Unique ID of the trajectory
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if self.storage_backend == 'mongodb':
            result = self.collection.delete_one({"trajectory_id": trajectory_id})
            return result.deleted_count > 0
        
        elif self.storage_backend == 'file':
            # For file backend, we need to read all lines, filter out the one to delete,
            # and rewrite the file
            lines = []
            found = False
            
            with open(self.file_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    doc = json.loads(line.strip())
                    if doc.get("trajectory_id") == trajectory_id:
                        found = True
                    else:
                        lines.append(line)
            
            if found:
                with open(self.file_path, 'w') as f:
                    f.writelines(lines)
            
            return found
        
        elif self.storage_backend == 'memory':
            for i, doc in enumerate(self.memory_storage):
                if doc.get("trajectory_id") == trajectory_id:
                    self.memory_storage.pop(i)
                    return True
            
            return False