# OpenManus-RL Development Guide

## Project Overview

OpenManus-RL is a reinforcement learning framework designed for training large language models (LLMs) to perform agent tasks. The project combines two main repositories:

1. **AgentGym**: Provides environments, rewards, and evaluation tools for agent tasks
2. **Verl**: Handles RL training, rollout methods, and reward computation

The training process follows a pipeline architecture:
1. Start AgentGym environment services
2. Initialize reward manager and rollout worker group
3. Generate trajectories via OpenManus agent
4. Run PPO or GRPO training to update the LLM
5. Save checkpoints and repeat from step 3

### Key Components

- **Data Representation**: Uses Hugging Face parquet files for input and `DataProto` for internal data representation
- **Training Scripts**: `train_ppo.sh` and `train_grpo.sh` orchestrate the entire training process
- **Base Agent**: Implemented in `openmanus_rl/llm_agent/openmanus.py`, handles environment interaction
- **Reward Calculation**: Managed in `verl/utils/reward_score/agentgym.py`, computes cumulative rewards from AgentGym

## Core Components

### Verl Framework

Verl is the underlying reinforcement learning framework that handles the RL training loop, rollout mechanisms, and reward computation.

#### DataProto

`DataProto` is the core data structure used throughout the framework:

- Encapsulates both tensor-based data (stored in `.batch`) and non-tensor metadata (stored in `.meta_info`)
- Provides methods for batch manipulation (slicing, merging, etc.)
- Handles device placement and data consistency

Example:
```python
data = DataProto.from_dict({
    'input_ids': input_tensor,
    'attention_mask': mask_tensor,
    'position_ids': position_tensor
})
data.meta_info['task_idx'] = task_indices
```

#### Ray Trainer

The Ray-based trainer (`verl/trainer/ppo/ray_trainer.py`) implements distributed PPO training:

- `RayPPOTrainer`: Manages the entire training process, including:
  - Environment initialization
  - Worker group coordination
  - Advantage computation
  - Policy updates
  - Validation

Key methods:
- `init_workers()`: Initializes different worker roles
- `fit()`: Main training loop
- `_validate()`: Runs validation on the current policy
- `_save_checkpoint()`: Saves model checkpoints

#### Rollout Worker Group

Rollout workers generate trajectories from the current policy:

- Implemented as a Ray-based worker group that can be distributed across multiple nodes
- Handles generation, log probability computation, and policy updates
- Uses VLLM for efficient inference

#### Reward Computation

Reward computation is handled by dedicated modules:

- `verl/utils/reward_score/agentgym.py`: Specific to AgentGym environments
- Various reward modules support different types of rewards (EM scores, BLEU, etc.)
- `apply_kl_penalty()`: Adds KL divergence penalties to raw rewards

### OpenManus Agent

The OpenManus agent (`openmanus_rl/llm_agent/openmanus.py`) serves as the interface between the RL framework and environment.

#### Key Classes

- `AgentConfig`: Configuration for the agent
- `OpenManusAgent`: Main agent class that handles environment interaction

#### Critical Methods

1. **run_llm_loop**
   ```python
   def run_llm_loop(self, gen_batch: DataProto, output_dir: str = None, global_steps: int = 0) -> DataProto:
   ```
   
   This method orchestrates the interaction loop for a batch of environments:
   - Takes initial prompts as input
   - Runs parallel rollouts using thread pool
   - Collects trajectories and rewards
   - Formats results into DataProto for training
   - Handles visualization if enabled

2. **_run_single_rollout**
   ```python
   def _run_single_rollout(self, initial_prompt_ids: torch.Tensor, task_idx: int) -> Dict[str, Any]:
   ```
   
   Executes a single environment interaction:
   - Resets environment with task index
   - Runs the interaction loop for multiple turns
   - Generates responses using the LLM
   - Processes responses and executes actions
   - Collects rewards and observations

3. **_convert_rollout_results_to_dataproto**
   ```python
   def _convert_rollout_results_to_dataproto(self, results: List[Dict], original_batch: DataProto) -> DataProto:
   ```
   
   Converts rollout results to trainable format:
   - Aligns rewards with token sequences
   - Creates token-level reward tensors
   - Concatenates and pads conversation segments
   - Preserves metadata from original batch

### Training Scripts

The training scripts (`train_ppo.sh` and `train_grpo.sh`) orchestrate the entire process:

1. Initialize the environment:
   - Parse command line arguments
   - Create dedicated conda environment for specific AgentGym environment
   - Start environment server

2. Set up training:
   - Configure data paths and experiment names
   - Initialize logging
   - Set hyperparameters

3. Run training:
   - Launch Verl trainer with appropriate algorithm (PPO or GRPO)
   - Monitor training progress
   - Save checkpoints

## Development Guide

### Adding New Reward Methods

To add new reward methods (e.g., process reward, outcome reward):

1. **Create a new reward module**:
   ```bash
   # Create a new file in the reward_score directory
   touch /home/kunlunz2/github_repos/OpenManus-RL/verl/utils/reward_score/my_reward.py
   ```

2. **Implement the reward function**:
   ```python
   # my_reward.py
   def compute_score(solution_str, ground_truth, **kwargs):
       # Your reward computation logic
       return reward_tensor
   ```

3. **Register the reward in `__init__.py`**:
   ```python
   # Add to verl/utils/reward_score/__init__.py
   from .my_reward import compute_score as my_reward_compute_score
   
   SUPPORTED_REWARD_SCORE_FNS = {
       # ... existing rewards
       'my_reward': my_reward_compute_score,
   }
   ```

4. **Modify agent to collect appropriate information**:
   - Update `OpenManusAgent._run_single_rollout` to collect required information
   - Modify `_convert_rollout_results_to_dataproto` to format rewards properly

5. **Use the new reward in training script**:
   ```bash
   # In train_ppo.sh or train_grpo.sh, add:
   algorithm.reward_score_fn=my_reward
   ```

For process rewards specifically:
- Modify `_run_single_rollout` to track intermediate steps
- Update reward computation to consider the process (steps taken) rather than just the outcome

### Adding New Environments

To integrate a new environment from AgentGym:

1. **Prepare the environment package**:
   - Create a dedicated directory in `openmanus_rl/agentgym/agentenv-<env_name>/`
   - Include `environment.yml` for conda environment specs
   - Add `setup.sh` for any additional setup steps

2. **Update training scripts**:
   - Add the new environment to the case statement in `train_ppo.sh` and `train_grpo.sh`:
   ```bash
   new_env)
       LAUNCH_CMD="new_env --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
       DEFAULT_PORT=XXXX
       ;;
   ```

3. **Update OpenManus agent**:
   - Add the new environment to `ENV_TO_TASK_CLASS` in `_init_env_client`
   ```python
   ENV_TO_TASK_CLASS = {
       # ... existing environments
       "new_env": "NewEnvTask",
   }
   ```

4. **Prepare training data**:
   - Create parquet files for training/validation in `data/<env_name>/`
   - Define appropriate reward models in the data

5. **Test the integration**:
   ```bash
   ./train_ppo.sh --env_name new_env
   ```

### Extending Rollout Methods

To add new rollout or action template methods:

1. **Modify the OpenManus agent**:
   - Add new parsing logic in `postprocess_predictions`:
   ```python
   def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str], List[str]]:
       # Add new template patterns
       new_pattern = r'<new_action>(.*?)</new_action>'
       # ... process accordingly
   ```

2. **Add new action execution logic**:
   - Update `_run_single_rollout` to handle new action types
   - Modify the action execution logic to process new templates

3. **Update the prompt template**:
   - Modify `create_react_prompt` to include instructions for the new action templates
   ```python
   def create_react_prompt(task_description, tool_manager):
       # Add instructions for new action templates
   ```

4. **Configure the agent**:
   - Update `AgentConfig` if new parameters are needed
   - Modify training scripts to pass appropriate configurations

### Advanced Modifications

For more advanced modifications, such as changing the training algorithm or reward structure:

1. **Modifying the PPO algorithm**:
   - Update `verl/trainer/ppo/core_algos.py` for algorithm changes
   - Modify advantage calculation in `compute_advantage`

2. **Changing the rollout worker**:
   - Create a new worker class in `verl/single_controller/ray/`
   - Register the worker in the appropriate factory methods

3. **Custom data processing**:
   - Modify `_convert_rollout_results_to_dataproto` for custom data formats
   - Update `DataProto` methods if needed

## Conclusion

OpenManus-RL provides a flexible framework for reinforcement learning with LLMs in agent environments. By understanding the core components and following this development guide, you can extend the framework to support new environments, reward structures, and action templates.

For more detailed information on AgentGym integration, refer to the documentation in `/home/kunlunz2/github_repos/OpenManus-RL/openmanus_rl/agentgym/2nd_dev_docs`. 