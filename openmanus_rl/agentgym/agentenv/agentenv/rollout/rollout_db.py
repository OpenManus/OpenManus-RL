import pymongo
from datetime import datetime
import uuid
from typing import Dict, List, Any, Optional, Union
import json

from agentenv.controller.task import ExperienceOutput, ConversationMessage


class ITrajectoryStorage:
    """Interface for trajectory storage"""
    
    def save_trajectory(self, env_name: str, task_id: int, strategy_name: str, trajectory: ExperienceOutput, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a trajectory to storage and return its ID"""
        pass
    
    def save_trajectories(self, env_name: str, task_ids: List[int], strategy_name: str, trajectories: List[ExperienceOutput], metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Save multiple trajectories to storage and return their IDs"""
        pass
    
    def get_trajectory(self, trajectory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a trajectory by ID"""
        pass
    
    def get_trajectories(self, env_name: Optional[str] = None, task_id: Optional[int] = None, strategy_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve trajectories matching the filters"""
        pass
    
    def get_best_trajectory(self, env_name: str, task_id: int) -> Optional[Dict[str, Any]]:
        """Get the trajectory with the highest reward for a specific task"""
        pass


class MongoDBTrajectoryStorage(ITrajectoryStorage):
    """MongoDB implementation of trajectory storage"""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/", db_name: str = "agentgym"):
        """Initialize MongoDB connection"""
        self.client = pymongo.MongoClient(connection_string)
        self.db = self.client[db_name]
        self.collection = self.db["trajectories"]
        
        # Create indexes for efficient querying
        self.collection.create_index([("env_name", 1)])
        self.collection.create_index([("task_id", 1)])
        self.collection.create_index([("strategy_name", 1)])
        self.collection.create_index([("timestamp", -1)])
        self.collection.create_index([("reward", -1)])
    
    def save_trajectory(self, env_name: str, task_id: int, strategy_name: str, trajectory: ExperienceOutput, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a trajectory to MongoDB"""
        trajectory_id = str(uuid.uuid4())
        
        # Create document
        doc = {
            "trajectory_id": trajectory_id,
            "env_name": env_name,
            "task_id": task_id,
            "strategy_name": strategy_name,
            "timestamp": datetime.now(),
            "reward": float(trajectory.reward),
            "conversation": [
                {
                    "from": msg["from"],
                    "value": msg["value"],
                    "loss": msg.get("loss")
                } for msg in trajectory.conversation
            ],
            "metadata": metadata or {}
        }
        
        # Insert document
        self.collection.insert_one(doc)
        return trajectory_id
    
    def save_trajectories(self, env_name: str, task_ids: List[int], strategy_name: str, trajectories: List[ExperienceOutput], metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Save multiple trajectories to MongoDB"""
        if len(task_ids) != len(trajectories):
            raise ValueError("Number of task IDs must match number of trajectories")
        
        trajectory_ids = []
        docs = []
        
        for i, (task_id, trajectory) in enumerate(zip(task_ids, trajectories)):
            trajectory_id = str(uuid.uuid4())
            trajectory_ids.append(trajectory_id)
            
            # Create document
            doc = {
                "trajectory_id": trajectory_id,
                "env_name": env_name,
                "task_id": task_id,
                "strategy_name": strategy_name,
                "timestamp": datetime.now(),
                "reward": float(trajectory.reward),
                "conversation": [
                    {
                        "from": msg["from"],
                        "value": msg["value"],
                        "loss": msg.get("loss")
                    } for msg in trajectory.conversation
                ],
                "metadata": metadata or {}
            }
            docs.append(doc)
        
        # Insert documents
        if docs:
            self.collection.insert_many(docs)
        
        return trajectory_ids
    
    def get_trajectory(self, trajectory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a trajectory by ID"""
        doc = self.collection.find_one({"trajectory_id": trajectory_id})
        return doc
    
    def get_trajectories(self, env_name: Optional[str] = None, task_id: Optional[int] = None, strategy_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve trajectories matching the filters"""
        # Build query
        query = {}
        if env_name:
            query["env_name"] = env_name
        if task_id is not None:
            query["task_id"] = task_id
        if strategy_name:
            query["strategy_name"] = strategy_name
        
        # Execute query
        cursor = self.collection.find(query).sort("timestamp", -1).limit(limit)
        return list(cursor)
    
    def get_best_trajectory(self, env_name: str, task_id: int) -> Optional[Dict[str, Any]]:
        """Get the trajectory with the highest reward for a specific task"""
        query = {"env_name": env_name, "task_id": task_id}
        doc = self.collection.find_one(query, sort=[("reward", -1)])
        return doc


class FileTrajectoryStorage(ITrajectoryStorage):
    """Simple file-based implementation of trajectory storage"""
    
    def __init__(self, file_path: str = "trajectories.jsonl"):
        """Initialize file storage"""
        self.file_path = file_path
        
        # Ensure file exists
        try:
            with open(self.file_path, 'a') as f:
                pass
        except:
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
            with open(self.file_path, 'w') as f:
                pass
    
    def save_trajectory(self, env_name: str, task_id: int, strategy_name: str, trajectory: ExperienceOutput, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a trajectory to file"""
        trajectory_id = str(uuid.uuid4())
        
        # Create document
        doc = {
            "trajectory_id": trajectory_id,
            "env_name": env_name,
            "task_id": task_id,
            "strategy_name": strategy_name,
            "timestamp": datetime.now().isoformat(),
            "reward": float(trajectory.reward),
            "conversation": [
                {
                    "from": msg["from"],
                    "value": msg["value"],
                    "loss": msg.get("loss")
                } for msg in trajectory.conversation
            ],
            "metadata": metadata or {}
        }
        
        # Append to file
        with open(self.file_path, 'a') as f:
            f.write(json.dumps(doc) + '\n')
        
        return trajectory_id
    
    def save_trajectories(self, env_name: str, task_ids: List[int], strategy_name: str, trajectories: List[ExperienceOutput], metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Save multiple trajectories to file"""
        if len(task_ids) != len(trajectories):
            raise ValueError("Number of task IDs must match number of trajectories")
        
        trajectory_ids = []
        
        for task_id, trajectory in zip(task_ids, trajectories):
            trajectory_id = self.save_trajectory(env_name, task_id, strategy_name, trajectory, metadata)
            trajectory_ids.append(trajectory_id)
        
        return trajectory_ids
    
    def get_trajectory(self, trajectory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a trajectory by ID"""
        with open(self.file_path, 'r') as f:
            for line in f:
                doc = json.loads(line.strip())
                if doc.get("trajectory_id") == trajectory_id:
                    return doc
        return None
    
    def get_trajectories(self, env_name: Optional[str] = None, task_id: Optional[int] = None, strategy_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve trajectories matching the filters"""
        results = []
        
        with open(self.file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line.strip())
                
                # Apply filters
                if env_name and doc.get("env_name") != env_name:
                    continue
                if task_id is not None and doc.get("task_id") != task_id:
                    continue
                if strategy_name and doc.get("strategy_name") != strategy_name:
                    continue
                
                results.append(doc)
                if len(results) >= limit:
                    break
        
        # Sort by timestamp (descending)
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results
    
    def get_best_trajectory(self, env_name: str, task_id: int) -> Optional[Dict[str, Any]]:
        """Get the trajectory with the highest reward for a specific task"""
        matching_trajectories = []
        
        with open(self.file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line.strip())
                
                # Apply filters
                if doc.get("env_name") == env_name and doc.get("task_id") == task_id:
                    matching_trajectories.append(doc)
        
        if not matching_trajectories:
            return None
        
        # Return trajectory with highest reward
        return max(matching_trajectories, key=lambda x: x.get("reward", 0))