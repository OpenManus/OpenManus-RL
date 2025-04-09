# AgentGym Rollout Controller Design Document

## Overview

This document outlines the design and implementation of the Rollout Controller for the AgentGym framework. The Rollout Controller extends AgentGym's capabilities by adding support for advanced exploration strategies (Tree of Thoughts, Monte Carlo Tree Search, etc.) and trajectory storage, while maintaining compatibility with the existing architecture.

## Motivation

The standard AgentGym implementation uses a straightforward ReAct approach for agent interaction with environments. While this works well for simple scenarios, more complex reasoning and decision-making often benefit from advanced exploration strategies that consider multiple possible action paths. Additionally, storing and analyzing trajectories is crucial for reinforcement learning and model improvement.

## Architecture

The Rollout Controller architecture consists of three main components:

1. **Rollout Strategies**: Implementations of different exploration algorithms
2. **Trajectory Storage**: Systems for persisting and retrieving trajectories
3. **Rollout Controller**: Main controller that integrates strategies and storage with AgentGym

### Integration with AgentGym

The implementation extends the existing AgentGym components rather than replacing them:

- `RolloutController` extends `BaseAgentEnvController` from AgentGym
- All strategies accept and return `ExperienceOutput` objects for compatibility
- The controller uses `BaseTask` and `BaseEnvClient` from AgentGym for environment interaction
```
                    BaseAgentEnvController
                                 ↑
                                 |
                         RolloutController ←→ IRolloutStrategy
                                 |              ↑
                                 |              |
                                 |          BaseRolloutStrategy
                                 |              ↑
                                 |              |
                                 |        ┌─────┴─────────┐
                                 |        |               |
                                 |   StandardReAct      ToT/MCTS/etc.
                                 |
                                 ↓
                         ITrajectoryStorage
                                 ↑
                         ┌───────┴───────┐
                         |               |
                  MongoDBStorage   FileStorage
```
## Components

### Rollout Strategies

All strategies implement the `IRolloutStrategy` interface, ensuring a consistent API:

```python
class IRolloutStrategy(ABC):
    @abstractmethod
    def execute(
        self, 
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase, 
        client: BaseEnvClient, 
        initial_observation: str,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None
    ) -> List[ExperienceOutput]:
        """Execute the strategy and return trajectories"""
        pass
```

#### Implemented Strategies

1. **StandardReActStrategy**: The default strategy used in AgentGym, which follows a linear path of observation → action → observation.

2. **ToTStrategy (Tree of Thoughts)**: Implements a tree exploration approach where:
   - The agent considers multiple possible actions at each step
   - For each action, it explores the resulting states recursively
   - This creates a tree of potential trajectories
   - Parameters control the breadth (number of branches) and depth of exploration

3. **MCTSStrategy (Monte Carlo Tree Search)**: Implements the MCTS algorithm for more efficient exploration of large action spaces:
   - Selection: Choose promising nodes to explore
   - Expansion: Add new child nodes
   - Simulation: Run rollouts to estimate node value
   - Backpropagation: Update node values based on simulation results

### Trajectory Storage

The `ITrajectoryStorage` interface defines methods for saving and retrieving trajectories:

```python
class ITrajectoryStorage:
    def save_trajectory(self, env_name, task_id, strategy_name, trajectory, metadata=None) -> str:
        pass
    
    def save_trajectories(self, env_name, task_ids, strategy_name, trajectories, metadata=None) -> List[str]:
        pass
    
    def get_trajectory(self, trajectory_id) -> Optional[Dict]:
        pass
    
    def get_trajectories(self, env_name=None, task_id=None, strategy_name=None, limit=100) -> List[Dict]:
        pass
    
    def get_best_trajectory(self, env_name, task_id) -> Optional[Dict]:
        pass
```

#### Implementations

1. **MongoDBTrajectoryStorage**: Stores trajectories in MongoDB for scalable, queryable access.
2. **FileTrajectoryStorage**: A simpler implementation that stores trajectories in JSONL files.

### Rollout Controller

The `RolloutController` class orchestrates the rollout process:

```python
class RolloutController(BaseAgentEnvController):
    def __init__(
        self, 
        agent: Agent, 
        tasks: List[BaseTask], 
        strategy: Optional[IRolloutStrategy] = None,
        storage: Optional[ITrajectoryStorage] = None,
        max_workers: int = 10
    ):
        # initialization...
    
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
        # implementation...
```

Key features:
- **Configurable strategy**: Use different exploration strategies for different tasks
- **Parallel execution**: Process multiple environments concurrently
- **Trajectory storage**: Automatically save trajectories for later analysis
- **Batch processing**: Process environments in batches for memory efficiency

## Usage Examples

### Basic Usage with Tree of Thoughts

```python
from agentenv.controller import Agent
from agentenv.envs import WebshopTask
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from rollout_controller import RolloutController
from strategies import ToTStrategy
from database import MongoDBTrajectoryStorage

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("model_path")
tokenizer = AutoTokenizer.from_pretrained("model_path")
agent = Agent(model, tokenizer)

# Create task
task = WebshopTask(
    client_args={"env_server_base": "http://localhost:36001", "data_len": 200},
    n_clients=1
)

# Create storage
storage = MongoDBTrajectoryStorage()

# Create strategy
strategy = ToTStrategy(num_branches=3, depth=2)

# Create controller
controller = RolloutController(
    agent=agent,
    tasks=[task],
    strategy=strategy,
    storage=storage
)

# Run rollout
results = controller.rollout(
    generation_config=GenerationConfig(max_length=4096),
    max_rounds=7,
    idxs=[0, 1, 2],  # Run on first three tasks
    parallel=True
)

# Analyze results
for result in results:
    print(f"Reward: {result.reward}")
```

### Switching Strategies

```python
from strategies import MCTSStrategy

# Switch to MCTS strategy
mcts_strategy = MCTSStrategy(num_simulations=50, exploration_weight=1.0)
controller.set_strategy(mcts_strategy)

# Run rollout with new strategy
results = controller.rollout(idxs=[0, 1, 2])
```

## Implementation Considerations

### Concurrency and Thread Safety

- The controller uses ThreadPoolExecutor for parallel rollouts
- Each rollout uses a separate environment client instance
- Careful consideration of thread safety in strategy implementations

### Memory Management

- Batch processing to avoid excessive memory usage
- Proper cleanup of resources after rollout
- Copy-on-write for environment branching

### Error Handling

- Robust error handling at multiple levels
- Failed rollouts don't interrupt the entire process
- Detailed error reporting

## TODO