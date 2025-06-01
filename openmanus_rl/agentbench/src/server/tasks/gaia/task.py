import time
from typing import List, Dict, Any
from src.server.task import Task, Session
from src.typings import (
    SampleIndex, 
    TaskSampleExecutionResult, 
    SampleStatus,
    AgentOutputStatus
)
from src.typings.output import TaskOutput
from .agentenv_gaia.environment import GaiaEnvServer

class GAIATask(Task):
    def __init__(self, 
                 dataset_type: str = "validation",
                 level: str = "level1", 
                 data_dir: str = "data/",
                 max_rounds: int = 20,
                 **configs):
        super().__init__(**configs)
        
        self.dataset_type = dataset_type
        self.level = level
        self.data_dir = data_dir
        self.max_rounds = max_rounds
        
        # Initialize GAIA environment server (this handles everything!)
        self.gaia_server = GaiaEnvServer()
        
        # Get dataset size for indices (use existing preloaded data if available)
        try:
            if dataset_type == "validation" and hasattr(self.gaia_server, 'validation_data') and self.gaia_server.validation_data is not None:
                self.dataset_size = len(self.gaia_server.validation_data)
            elif dataset_type == "test" and hasattr(self.gaia_server, 'test_data') and self.gaia_server.test_data is not None:
                self.dataset_size = len(self.gaia_server.test_data)
            else:
                # Fallback: create a temp environment to determine size 
                temp_env_id = self.gaia_server.create(id=0, dataset_type=dataset_type)
                self.dataset_size = 165 if dataset_type == "validation" else 300  # GAIA dataset sizes
                # Clean up temp environment
                if temp_env_id in self.gaia_server.env_instances:
                    del self.gaia_server.env_instances[temp_env_id]
                    del self.gaia_server.env_locks[temp_env_id]
        except Exception as e:
            print(f"Warning: Could not determine dataset size: {e}")
            self.dataset_size = 165 if dataset_type == "validation" else 300
        
    def get_indices(self) -> List[SampleIndex]:
        return list(range(self.dataset_size))
    
    async def start_sample(self, index: SampleIndex, session: Session) -> TaskSampleExecutionResult:
        """
        Execute a single GAIA sample - minimal wrapper around existing environment
        """
        start_time = time.time()
        
        try:
            env_id = self.gaia_server.create(id=index, dataset_type=self.dataset_type)
            
            # Get initial observatin
            initial_obs = self.gaia_server.observation(env_id)
            session.inject({"role": "user", "content": initial_obs})
            
            # Multi-turn interaction loop
            final_reward = 0.0
            for round_num in range(self.max_rounds):
                response = await session.action()
                
                if response.status == AgentOutputStatus.AGENT_CONTEXT_LIMIT:
                    final_status = SampleStatus.AGENT_CONTEXT_LIMIT
                    break
                elif response.status != AgentOutputStatus.NORMAL:
                    final_status = SampleStatus.AGENT_VALIDATION_FAILED
                    break
                
                observation, reward, done, info = self.gaia_server.step(env_id, response.content or "")
                final_reward = reward
                if observation:
                    session.inject({"role": "user", "content": observation})
                
                # Check if done
                if done:
                    final_status = SampleStatus.COMPLETED
                    break
            else:
                final_status = SampleStatus.TASK_LIMIT_REACHED
            
            env_data = self.gaia_server.env_instances[env_id]
            dataset_item = env_data["dataset"]
            
            return TaskSampleExecutionResult(
                status=final_status,
                result={
                    "score": final_reward,
                    "question": dataset_item.get("question", ""),
                    "true_answer": dataset_item.get("true_answer", ""),
                    "rounds_used": round_num + 1 if 'round_num' in locals() else 0,
                    "execution_time": time.time() - start_time,
                    "level": self.level,
                    "dataset_type": self.dataset_type,
                    "steps_taken": info.get("steps_taken", 0) if 'info' in locals() else 0,
                    "env_state": env_data["state"] if final_status == SampleStatus.COMPLETED else {}
                }
            )
            
        except Exception as e:
            return TaskSampleExecutionResult(
                status=SampleStatus.TASK_ERROR,
                result={
                    "error": f"Task error: {str(e)}",
                    "score": 0.0,
                    "execution_time": time.time() - start_time
                }
            )
        
        finally:
            # Clean up environment
            try:
                if 'env_id' in locals() and env_id in self.gaia_server.env_instances:
                    del self.gaia_server.env_instances[env_id]
                    del self.gaia_server.env_locks[env_id]
            except:
                pass
    
    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        """
        Calculate aggregate metrics - simple and minimal
        """
        valid_results = [r for r in results if r.result]
        
        if not valid_results:
            return {
                "accuracy": 0.0,
                "total_samples": len(results),
                "error_rate": 1.0
            }
        
        scores = [r.result.get("score", 0.0) for r in valid_results]
        error_count = sum(1 for r in valid_results if "error" in r.result)
        
        return {
            "accuracy": sum(scores) / len(scores) if scores else 0.0,
            "total_samples": len(results),
            "valid_samples": len(valid_results),
            "error_rate": error_count / len(valid_results) if valid_results else 1.0,
            "level": self.level,
            "dataset_type": self.dataset_type
        }