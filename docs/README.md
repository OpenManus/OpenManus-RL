# OpenManus Model Training Overview

This document provides a detailed explanation of the training logic and core functions in OpenManus, focusing on how agent trajectories are utilized for loss calculation and training.

## Overall Training Architecture

OpenManus uses Proximal Policy Optimization (PPO) algorithm with the following core components:

```
RayPPOTrainer (Main Trainer)
├── OpenManusAgent (Environment Interaction)
├── ActorRolloutRefWorker (Policy Network)
└── CriticWorker (Value Network)
```

## Main Training Loop

The training core is implemented in the `RayPPOTrainer.fit()` method:

```python
# Simplified training loop
for epoch in range(epochs):
    for batch in train_dataloader:
        # 1. Collect trajectories
        trajectories = generation_manager.run_llm_loop(batch)
        
        # 2. Calculate composite rewards
        compute_total_reward(trajectories)
        
        # 3. Compute advantage function
        compute_advantage(batch)
        
        # 4. Update critic network
        critic_wg.update_critic(batch)
        
        # 5. Update actor network
        actor_rollout_wg.update_actor(batch)
```

## Key Processes

### 1. Trajectory Collection

Implemented in `OpenManusAgent.run_llm_loop` and `_run_single_rollout`:

```python
# Key logic in _run_single_rollout
while not done:
    # Get current observation
    observation = client.observe()
    
    # Generate LLM response
    response = actor_model.generate(observation)
    
    # Parse action
    action = parse_action(response)
    
    # Execute environment step
    next_obs, reward, done, info = client.step(action)
    
    # Record trajectory
    trajectory.append({"from": "human", "value": next_obs, "reward": reward, "info": info})
```

### 2. Reward Composition

Multiple reward signals are combined through the `RewardComposer`:

```python
# Called in _convert_rollout_results_to_dataproto
total_score, breakdown = reward_composer.compute_total_reward(
    trajectory=trajectory,
    reward_model_info=reward_model_info,
    env_name=env_name
)
```

Main reward components include:
- `GoalReward`: Primary task success reward
- `LengthPenalty`: Penalty for excessive length
- `FormatReward`: Reward for correct output format

### 3. Reward Allocation

In the `_convert_rollout_results_to_dataproto` method, rewards are allocated to individual tokens:

```python
# Different reward allocation strategies:
if reward_allocation == "last_token":
    # Assign reward only to the last token
    token_level_rewards[0, last_segment_end] = reward_to_distribute
    
elif reward_allocation == "uniform_positive":
    # Distribute positive rewards evenly, negative rewards only to the last token
    if reward_to_distribute > 0:
        reward_per_token = reward_to_distribute / total_agent_tokens
        for start, end in agent_indices_in_padded:
            token_level_rewards[0, start:end+1] = reward_per_token
            
elif reward_allocation == "discounted":
    # Discounted rewards, allocated backward from the last segment
    gamma = config.algorithm_config.get('gamma', 1.0)
    current_reward = reward_to_distribute
    for start, end in reversed(agent_indices_in_padded):
        # Calculate reward within each segment
        token_level_rewards[0, start:end+1] = current_reward / segment_len
        current_reward *= (gamma ** segment_len)
```

### 4. Advantage Computation

In the `compute_advantage` function, Generalized Advantage Estimation (GAE) is used:

```python
if adv_estimator == 'gae':
    advantages, returns = core_algos.compute_gae_advantage_return(
        token_level_rewards=token_level_rewards,
        values=values,
        eos_mask=response_mask,
        gamma=gamma,
        lam=lam
    )
```

### 5. Policy Update

The policy is updated in `update_actor` using the PPO objective function:

```python
def update_policy(self, data):
    old_log_probs = data.batch['old_log_probs']
    advantages = data.batch['advantages']
    
    # Calculate log probabilities of current policy
    current_log_probs = self.compute_log_prob(data)
    
    # Calculate policy ratio
    ratio = torch.exp(current_log_probs - old_log_probs)
    
    # Clip ratio
    ratio_clipped = torch.clamp(ratio, 1-clip_eps, 1+clip_eps)
    
    # PPO objective
    policy_loss = -torch.min(
        advantages * ratio,
        advantages * ratio_clipped
    ).mean()
    
    self.optimizer.zero_grad()
    policy_loss.backward()
    self.optimizer.step()
```

### 6. Value Network Update

The critic network is updated in `update_critic` by minimizing the value loss:

```python
def update_critic(self, data):
    values = self.compute_values(data)
    returns = data.batch['returns']
    
    # Value loss
    value_loss = F.mse_loss(values, returns)
    
    self.optimizer.zero_grad()
    value_loss.backward()
    self.optimizer.step()
```

## Distributed Training Architecture

OpenManus uses Ray and FSDP (Fully Sharded Data Parallel) for distributed training:

- `ActorRolloutRefWorker`: Responsible for policy network inference and training
- `CriticWorker`: Responsible for value network training
- `RayPPOTrainer`: Coordinates communication and synchronization between different workers

FSDP shards model parameters across nodes using `ShardingStrategy.FULL_SHARD`, allowing for training larger models.

## Summary

The OpenManus training process integrates several key technologies:
1. PPO-based reinforcement learning framework
2. Trajectory-based environment interaction and reward collection
3. Composite reward calculation and flexible reward allocation strategies
4. Distributed training architecture supporting large-scale models

The core of the entire process lies in how to collect meaningful trajectories from environment interactions, and optimize the LLM's decision-making capabilities through appropriate reward functions and advantage estimation. 