# Modular Training for ALFWorld

This document describes the modular training system that evaluates and trains agents on specific cognitive modules.

## Overview

The modular training system enables focused training on four cognitive modules:
- **Planner**: Strategic planning and decision-making
- **Executor**: Action selection and execution
- **Reflection**: Analysis of outcomes and learning from experience  
- **Memory Use**: Effective recall and utilization of past experiences

## Architecture

### Components

1. **Reward Model Server** (`scripts/host_reward_model.py`)
   - Hosts a VLLM-based evaluation model
   - Provides REST API for module-specific scoring
   - Returns normalized scores (0-1) for integration with environment rewards

2. **Modular Reward Manager** (`openmanus_rl/rewards/modular_reward_manager.py`)
   - Extracts module content from trajectories
   - Manages communication with reward model server
   - Combines environment and model-based rewards

3. **Enhanced Trajectory Collector** (`openmanus_rl/multi_turn_rollout/rollout_loop.py`)
   - Extended with modular reward support
   - Tracks current module being evaluated
   - Computes combined rewards asynchronously

## Usage

### Basic Training (Non-Modular)

```bash
# Standard training without modular evaluation
bash scripts/ppo_train/train_alfworld_modular.sh vllm false
```

### Modular Training

```bash
# Enable modular training with reward model evaluation
bash scripts/ppo_train/train_alfworld_modular.sh vllm true

# With custom reward model
bash scripts/ppo_train/train_alfworld_modular.sh vllm true /path/to/reward/model 8100
```

### Configuration Parameters

Key parameters in the training script:

- `TRAIN_MODULAR`: Enable/disable modular training (default: false)
- `REWARD_MODEL_PATH`: Path to the reward model (default: Qwen3-4B)
- `REWARD_MODEL_PORT`: Port for reward model server (default: 8100)
- `env_reward_weight`: Weight for environment rewards (default: 0.5)
- `model_reward_weight`: Weight for model-based rewards (default: 0.5)
- `module_iterations_per_epoch`: Number of module iterations per epoch (default: 4)
- `random_seed`: Seed for module order randomization (default: 42)

## Training Process

### Standard Mode
1. Single iteration per epoch
2. Environment rewards only
3. Standard PPO optimization

### Modular Mode
1. **Epoch Structure**:
   - 4 iterations per epoch (one per module)
   - Random module order each epoch (seeded for reproducibility)
   
2. **Per Iteration**:
   - Focus on specific module
   - Extract module content from trajectories
   - Compute module-specific scores via reward model
   - Combine with environment rewards
   - Update policy using combined rewards

3. **Module Evaluation**:
   - Considers failure patterns and repetition
   - Analyzes causal relationships
   - Provides failure type classification
   - Generates improvement suggestions

## Module Scoring Rubrics

### Planner (Thinking)
- 5: Clear strategic planning with contingencies
- 4: Solid planning with logical progression
- 3: Basic planning toward goals
- 2: Vague or inconsistent planning
- 1: Plans work against objectives
- 0: No planning or incoherent plans

### Executor (Action)
- 5: Optimal action aligned with plan
- 4: Effective action advancing task
- 3: Reasonable progress
- 2: Suboptimal with questionable reasoning
- 1: Actions hinder progress
- 0: Invalid or impossible actions

### Reflection
- 5: Deep causal analysis with actionable insights
- 4: Solid analysis with useful conclusions
- 3: Basic outcome recognition
- 2: Shallow analysis missing key implications
- 1: Misinterprets outcomes
- 0: No reflection when needed

### Memory Use
- 5: Perfect recall with strategic learning
- 4: Good recall with minor gaps
- 3: Basic recall but limited learning
- 2: Repetitive memory without strategic value
- 1: Inappropriate or irrelevant memory usage
- 0: False memory or absence when needed

## Implementation Details

### Reward Computation

```python
# Combined reward formula
final_reward = env_weight * env_reward + model_weight * model_reward

# Where:
# - env_reward: Task completion reward from environment
# - model_reward: Module-specific score (0-1) from reward model
```

### Module Order Randomization

```python
# Each epoch gets a different module order
np.random.seed(seed + epoch)
modules = ['planner', 'executor', 'reflection', 'memory_use']
np.random.shuffle(modules)
```

### Failure Pattern Analysis

The system tracks:
- Consecutive failures
- Strategy repetition
- Total failure count
- Action loops

These metrics influence module scoring to prevent reward hacking.

## Benefits

1. **Focused Improvement**: Target specific cognitive weaknesses
2. **Balanced Development**: Ensure all modules receive attention
3. **Failure-Aware**: Consider failure patterns in scoring
4. **Interpretable**: Clear module-specific feedback
5. **Flexible**: Easy to add/modify modules or rubrics

## Monitoring

During training, the system logs:
- Current module being trained
- Module-specific scores
- Combined reward statistics
- Failure pattern metrics

Example output:
```
Epoch 0 module order: ['reflection', 'planner', 'memory_use', 'executor']
Training on module: reflection (iteration 1/4)
Module reflection - Env rewards: 0.432, Model rewards: 0.621, Combined: 0.527
```

## Future Extensions

- Dynamic module weighting based on performance
- Multi-module joint training
- Module dependency modeling
- Transfer learning between modules
- Online rubric adaptation
