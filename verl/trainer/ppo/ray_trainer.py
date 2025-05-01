# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import re
import json
from collections import defaultdict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import re
from openmanus_rl.llm_agent.openmanus import OpenManusAgent, AgentConfig
from verl.utils.reward_score import SUPPORTED_REWARD_SCORE_FNS
from verl.utils.reward_score.agentgym import compute_score as agentgym_compute_score
from verl.utils.reward_score.reward_components import RewardComposer, GoalReward, LengthPenalty, FormatReward
from verl.utils.tracking import Tracking

import ray

WorkerType = Type[Worker]

# Define known AgentGym envs centrally here
KNOWN_AGENTGYM_ENVS = [
    "webshop", "webarena", "maze", "wordle", "alfworld", 
    "sciworld", "babyai", "textcraft", "weather", "movie", 
    "academia", "todo", "sheet", "sqlgym"
]

class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores'] # Shape (batch_size, total_length)
    total_length = token_level_scores.size(1)
    # batch_size = data.batch.batch_size[0] # Get scalar batch size from TensorDict property

    # --- FIX: Get batch size from a tensor inside batch ---
    # Using data.batch.batch_size directly might fail if TensorDict is empty or inconsistent during init
    # It's safer to get it from a guaranteed tensor like input_ids or attention_mask if available
    # However, batch_size for kl_ctrl update needs to be scalar sum of batch sizes across ranks
    # Let's rely on the TensorDict property for now, assuming it's consistent by this point.
    # If this causes issues later, we might need to pass effective batch size differently.
    batch_size_scalar = data.batch.batch_size[0] # Get scalar batch size for kl_ctrl.update
    # --- END FIX ---

    # Get the attention mask for the full sequence
    attention_mask = data.batch['attention_mask'] # Shape (batch_size, total_length)
    # Extract the mask corresponding only to the response part
    response_mask = attention_mask[:, -response_length:] # Shape (batch_size, response_length)

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys() and 'old_log_probs' in data.batch.keys():
        # Assuming old_log_probs and ref_log_prob have shape (batch_size, response_length)
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # Shape (batch_size, response_length)
        kld = kld * response_mask # Apply mask, shape remains (batch_size, response_length)
        beta = kl_ctrl.value
    else:
        beta = 0
        # kld should have the same shape as the response part it would be subtracted from
        kld = torch.zeros_like(response_mask, dtype=torch.float32) # Shape (batch_size, response_length)

    # Initialize token_level_rewards as a copy of scores (prompt rewards are scores)
    token_level_rewards = token_level_scores.clone()

    # --- FIX: Apply KL penalty only to the response part ---
    # Extract the scores corresponding to the response tokens
    response_scores = token_level_scores[:, -response_length:] # Shape (batch_size, response_length)
    # Calculate the rewards for the response tokens
    response_rewards = response_scores - beta * kld # Shape (batch_size, response_length)
    # Place the calculated response rewards back into the full rewards tensor
    # Ensure rewards are only applied where the response mask is 1
    token_level_rewards[:, -response_length:][response_mask] = response_rewards[response_mask]
    # --- END FIX ---

    # Calculate current_kl based on the response part
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # Update KL controller
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size_scalar) # Use scalar batch_size

    # Update the DataProto with the final token_level_rewards
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    """
    Compute advantage estimates based on the specified estimator (GAE or GRPO).
    Now with improved error handling and debugging.
    """
    try:
        # prepare response group
        if adv_estimator == 'gae':
            # Check if values field exists, which is required for GAE
            if 'values' not in data.batch:
                print(f"[compute_advantage] WARNING: 'values' not found in batch, required for GAE. Falling back to GRPO estimator.")
                # Fall back to GRPO estimator which doesn't require values
                adv_estimator = 'grpo'
                print(f"[compute_advantage] Switched to estimator: {adv_estimator}")
            else:
                values = data.batch['values'] # Assume shape (batch_size, response_length), e.g., (4, 1000)
                responses = data.batch['responses'] # Shape (batch_size, response_length), e.g., (4, 1000)
                token_level_rewards = data.batch['token_level_rewards'] # Shape (batch_size, total_length), e.g., (4, 4096)
                attention_mask = data.batch['attention_mask'] # Shape (batch_size, total_length), e.g., (4, 4096)

                response_length = responses.size(-1) # e.g., 1000

                # --- FIX: Extract response-only parts for GAE calculation ---
                # Rewards corresponding to the response part
                response_rewards = token_level_rewards[:, -response_length:] # Shape (4, 1000)
                # Values corresponding to the response part (already assumed to be this shape)
                response_values = values # Shape (4, 1000)
                # Mask corresponding to the response part
                response_eos_mask = attention_mask[:, -response_length:] # Shape (4, 1000)
                # --- END FIX ---

                # Call GAE with aligned tensors
                advantages_response, returns_response = core_algos.compute_gae_advantage_return(
                    token_level_rewards=response_rewards,
                    values=response_values,
                    eos_mask=response_eos_mask,
                    gamma=gamma,
                    lam=lam
                )
                # advantages_response/returns_response have shape (batch_size, response_length)

                # --- FIX: Pad advantages and returns back to the full sequence length ---
                total_length = token_level_rewards.size(1) # e.g., 4096
                advantages = torch.zeros_like(token_level_rewards)
                returns = torch.zeros_like(token_level_rewards)

                advantages[:, -response_length:] = advantages_response
                returns[:, -response_length:] = returns_response
                # Apply mask again to ensure padding remains zero
                advantages = advantages * attention_mask
                returns = returns * attention_mask
                # --- END FIX ---

                data.batch['advantages'] = advantages # Shape (4, 4096)
                data.batch['returns'] = returns # Shape (4, 4096)
                # Successfully computed GAE, return here
                return data

        # If we reach here, we're using GRPO or we fell back to GRPO
        if adv_estimator == 'grpo':
            print(f"[compute_advantage] Computing GRPO advantages...")
            if 'token_level_rewards' not in data.batch:
                raise KeyError("Missing 'token_level_rewards' in batch, required for GRPO advantage computation")
            if 'uid' not in data.non_tensor_batch:
                raise KeyError("Missing 'uid' in non_tensor_batch, required for GRPO advantage computation")
            if 'responses' not in data.batch:
                raise KeyError("Missing 'responses' in batch, required for GRPO advantage computation")
            
            token_level_rewards = data.batch['token_level_rewards']
            index = data.non_tensor_batch['uid']
            responses = data.batch['responses']
            response_length = responses.size(-1)
            attention_mask = data.batch['attention_mask']
            response_mask = attention_mask[:, -response_length:]
            
            print(f"[compute_advantage] GRPO inputs - token_level_rewards shape: {token_level_rewards.shape}, " + 
                 f"response_length: {response_length}, response_mask shape: {response_mask.shape}, index length: {len(index)}")
            
            # GRPO computation with proper response rewards
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards[:, -response_length:],
                eos_mask=response_mask,
                index=index
            )
            
            # Verify the computation results
            print(f"[compute_advantage] GRPO outputs - advantages shape: {advantages.shape}, returns shape: {returns.shape}")
            
            # Pad back to full sequence length
            total_length = token_level_rewards.size(1)
            padded_advantages = torch.zeros_like(token_level_rewards)
            padded_returns = torch.zeros_like(token_level_rewards)
            padded_advantages[:, -response_length:] = advantages
            padded_returns[:, -response_length:] = returns
            
            # Apply attention mask and store results
            data.batch['advantages'] = padded_advantages * attention_mask
            data.batch['returns'] = padded_returns * attention_mask
            
            print(f"[compute_advantage] GRPO advantages/returns computed successfully")
        else:
            raise NotImplementedError(f"Unsupported advantage estimator: {adv_estimator}")
            
        # Check if the computed advantages and returns are valid
        if torch.isnan(data.batch['advantages']).any() or torch.isnan(data.batch['returns']).any():
            raise ValueError(f"NaN values detected in computed advantages or returns with {adv_estimator}")
            
        # Return the updated DataProto
        return data
    
    except Exception as e:
        import traceback
        print(f"[compute_advantage][ERROR] Failed to compute advantages with {adv_estimator}: {e}")
        print(traceback.format_exc())
        raise RuntimeError(f"Advantage computation failed for {adv_estimator}: {e}")


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    # Ensure dimensions match between advantages and response_mask
    adv_shape = advantages.shape[-1]
    mask_shape = response_mask.shape[-1]
    
    if adv_shape != mask_shape:
        print(f"Shape mismatch detected: advantages({adv_shape}) vs response_mask({mask_shape})")
        if adv_shape > mask_shape:
            # If advantages is longer, use only the last portion
            valid_adv = torch.masked_select(advantages[:, -mask_shape:], response_mask)
        else:
            # If response_mask is longer, use only the first portion
            valid_adv = torch.masked_select(advantages, response_mask[:, :adv_shape])
    else:
        valid_adv = torch.masked_select(advantages, response_mask)
    
    # Similar handling for returns and values
    if adv_shape != mask_shape:
        if adv_shape > mask_shape:
            valid_returns = torch.masked_select(returns[:, -mask_shape:], response_mask)
        else:
            valid_returns = torch.masked_select(returns, response_mask[:, :adv_shape])
    else:
        valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        # Ensure dimensions match between values and response_mask
        val_shape = values.shape[-1]
        if val_shape != mask_shape:
            if val_shape > mask_shape:
                valid_values = torch.masked_select(values[:, -mask_shape:], response_mask)
            else:
                valid_values = torch.masked_select(values, response_mask[:, :val_shape])
        else:
            valid_values = torch.masked_select(values, response_mask)
            
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),

        # metrics for actions
         'env/number_of_actions/mean':
            float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean()),
        'env/number_of_actions/max':
            float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max()),
        'env/number_of_actions/min':
            float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min()),
        'env/finish_ratio':
            1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean()),
        'env/number_of_valid_action':
            float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean()),
        'env/ratio_of_valid_action':
            float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean()),
        'env/number_of_valid_search':
            float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean()),
    }

    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


def get_safe_device(requested_device='cuda'):
    """
    Safely determine a usable device (CUDA or CPU).
    
    Args:
        requested_device: The preferred device, e.g., 'cuda', 'cuda:0', etc.
        
    Returns:
        A torch device that is guaranteed to work.
    """
    import torch
    # First check if CUDA is available at all
    if 'cuda' in str(requested_device) and not torch.cuda.is_available():
        print(f"WARNING: CUDA requested ({requested_device}) but is not available. Falling back to CPU.")
        return torch.device('cpu')
    
    # If a specific CUDA device is requested, make sure it exists
    if str(requested_device).startswith('cuda:'):
        try:
            device_idx = int(str(requested_device).split(':')[1])
            if device_idx >= torch.cuda.device_count():
                print(f"WARNING: CUDA device {device_idx} requested but only {torch.cuda.device_count()} devices available. Using device 0.")
                return torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')
        except (ValueError, IndexError):
            # If the format is invalid, fall back to default CUDA device
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # If we get here, the requested device should be valid
    try:
        return torch.device(requested_device)
    except:
        print(f"WARNING: Failed to use device {requested_device}, falling back to CPU.")
        return torch.device('cpu')


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None,
                 reward_component_config: dict = None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.reward_component_config = reward_component_config or {}

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        self._init_logger()
        self._init_reward_composer()
    
    def _init_logger(self):
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

    def _init_reward_composer(self):
        """Initializes the RewardComposer based on the configuration."""
        components = []
        cfg = self.reward_component_config
        print(f"[Trainer._init_reward_composer] Initializing with config: {cfg}")

        # --- Build Reward Components List --- 
        # Example: Dynamically add components based on config
        if cfg.get('goal_reward', {}).get('enabled', True):
            components.append(GoalReward(weight=cfg['goal_reward'].get('weight', 1.0)))
            print("  - Added GoalReward")

        if cfg.get('length_penalty', {}).get('enabled', False):
            lp_cfg = cfg['length_penalty']
            components.append(LengthPenalty(
                weight=lp_cfg.get('weight', -0.01),
                max_length=lp_cfg.get('max_length', 500),
                min_length=lp_cfg.get('min_length', 10),
                penalty_type=lp_cfg.get('penalty_type', "linear")
            ))
            print("  - Added LengthPenalty")

        if cfg.get('format_reward', {}).get('enabled', False):
            fmt_cfg = cfg['format_reward']
            # Get patterns specific to the current env or use default
            patterns = fmt_cfg.get('patterns_by_env', {}).get(
                self.config.data.env_name, # Assumes env_name is available in self.config.data
                fmt_cfg.get('patterns_by_env', {}).get('default', [])
            )
            components.append(FormatReward(
                weight=fmt_cfg.get('weight', 0.2),
                required_patterns=patterns
            ))
            print(f"  - Added FormatReward with patterns: {patterns}")

        self.reward_composer = RewardComposer(components=components)
        print(f"[Trainer._init_reward_composer] Composer initialized with {len(components)} components.")

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    @torch.no_grad()
    def _validate(self):
        print(f'[Trainer] Validate start at Global steps: {self.global_steps}')

        if self.config.data.env_name in KNOWN_AGENTGYM_ENVS:
            print(f"[Trainer] Detected AgentGym environment ({self.config.data.env_name}), using OpenManusAgent for validation.")

            # --- Instantiate AgentConfig ---
            # Ensure all required fields for AgentConfig are available in self.config.data
            # and self.config.algorithm
            agent_config = AgentConfig(
                max_turns=self.config.max_turns, # Make sure max_turns is accessible
                max_start_length=self.config.data.max_start_length,
                max_prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
                max_obs_length=self.config.data.max_obs_length,
                num_gpus=self.config.trainer.n_gpus_per_node, # Or however GPUs are configured
                env_name=self.config.data.env_name,
                env_ports=self.config.data.env_ports,
                env_server_base=self.config.data.env_server_base,
                react_format=getattr(self.config.data, 'react_format', True), # Optional, with default
                env_data_len=self.config.data.val_data_num or 200, # Use validation specific if available
                rollout_strategy=getattr(self.config.data, 'rollout_strategy', "StandardReAct"), # Optional
                max_workers=getattr(self.config.data, 'max_workers', 10), # Optional
                # Pass the relevant part of the algorithm config
                algorithm_config=self.config.algorithm,
            )

            # --- Instantiate OpenManusAgent ---
            # Need to define self.log_dir before this point if used in run_llm_loop
            # Assuming self.log_dir is defined elsewhere, e.g., in __init__ or fit
            if not hasattr(self, 'log_dir'):
                 # Define a default or derive from config if necessary
                 self.log_dir = self.config.trainer.get("default_local_dir", "./verl_checkpoints/default_log_dir")
                 print(f"[Trainer._validate] Warning: self.log_dir not found, using default: {self.log_dir}")

            self.validation_agent = OpenManusAgent(
                tokenizer=self.tokenizer,
                actor_rollout_wg=self.actor_rollout_wg,
                config=agent_config,
                is_validation=True,
                logger=self.logger # Pass logger here if needed by agent internally
            )

            # --- Run Validation Loop using OpenManusAgent ---
            all_metrics = defaultdict(list)

            for val_batch in self.val_dataloader:
                # Ensure batch is on the correct device (or handled by agent)
                # val_batch = val_batch.to(self.rank) # May not be needed if agent handles device placement

                # Agent's run_llm_loop returns a DataProto with results including rewards/scores
                processed_batch = self.validation_agent.run_llm_loop(val_batch, self.log_dir, self.global_steps)

                # --- Extract metrics from the agent's output ---
                # The reward/score should ideally be in processed_batch.meta_info
                # Let's assume 'env_score' holds the final task score per item
                if 'env_score' in processed_batch.meta_info:
                    scores = processed_batch.meta_info['env_score']
                    if isinstance(scores, torch.Tensor):
                        scores = scores.cpu().tolist()
                    all_metrics['val_reward_score'].extend(scores) # Use a consistent key
                    all_metrics['env_score'].extend(scores) # Also log as env_score

                # Log other stats if available
                if 'turns_stats' in processed_batch.meta_info:
                     turns = processed_batch.meta_info['turns_stats']
                     if isinstance(turns, torch.Tensor): turns = turns.cpu().tolist()
                     all_metrics['turns_stats'].extend(turns)

                if 'valid_action_stats' in processed_batch.meta_info:
                     valid_actions = processed_batch.meta_info['valid_action_stats']
                     if isinstance(valid_actions, torch.Tensor): valid_actions = valid_actions.cpu().tolist()
                     all_metrics['valid_action_stats'].extend(valid_actions)

                # Add any other relevant metrics from the agent's output meta_info
                # ...

                # --- Optional: Save Trajectories/Visualizations ---
                # Make sure save_trajectory_to_output is imported
                from openmanus_rl.utils.visualization import save_trajectory_to_output

                if self.logger and 'rollout_trajectory' in processed_batch.meta_info:
                    # Assuming save_rollout_data can handle the trajectory format
                    # You might need to adapt this based on the logger's interface
                    try:
                        task_indices = processed_batch.meta_info.get('task_idx', list(range(len(processed_batch))))
                        if isinstance(task_indices, torch.Tensor): task_indices = task_indices.cpu().tolist()

                        for idx, trajectory in enumerate(processed_batch.meta_info['rollout_trajectory']):
                            if idx < 5: # Limit saving to avoid excessive logging
                                original_task_idx = task_indices[idx]
                                save_trajectory_to_output(
                                    trajectory,
                                    output_dir=self.log_dir,
                                    global_step=self.global_steps,
                                    task_idx=original_task_idx,
                                    prefix="val"
                                )
                    except Exception as e:
                         print(f"[Trainer] Warning: Failed to save validation trajectory: {e}")
                         import traceback
                         # traceback.print_exc() # Uncomment for more details

            # --- Aggregate and Log Metrics ---
            final_metrics = {}
            for key, values in all_metrics.items():
                if values:
                    try:
                        final_metrics[f'{key}_mean'] = np.mean(values)
                        final_metrics[f'{key}_std'] = np.std(values)
                        final_metrics[f'{key}_median'] = np.median(values)
                    except Exception as e:
                        print(f"[Trainer] Warning: Could not compute stats for metric '{key}': {e}")
                else:
                    final_metrics[f'{key}_mean'] = 0 # Or NaN, or skip

            # Log aggregated metrics (using self.logger.log for structured data)
            if final_metrics:
                self.logger.log(final_metrics, step=self.global_steps)
                print(f"[Trainer] Validation Metrics @ step {self.global_steps}: {final_metrics}")
            else:
                print("[Trainer] Warning: No validation metrics collected.")

            return final_metrics # Return the computed metrics


        # --- Original Validation Logic (for non-AgentGym envs) ---
        else:
            print("[Trainer] Using standard validation logic (non-AgentGym).")
            # ... (Keep the existing validation logic for other environments) ...
            self.actor_rollout_wg.eval()
            # Check if ref_policy_wg exists before calling eval
            if hasattr(self, 'ref_policy_wg') and self.use_reference_policy:
                self.ref_policy_wg.eval()
            # Check if critic_wg exists before calling eval
            if hasattr(self, 'critic_wg') and self.use_critic:
                self.critic_wg.eval()
            if self.config.reward_model.enable and hasattr(self, 'reward_model_wg'):
                self.reward_model_wg.eval()

            all_metrics = defaultdict(list)

            # Ensure self.rank is defined if needed by val_step
            if not hasattr(self, 'rank'):
                 self.rank = 0 # Assuming rank 0 for validation if not otherwise set
                 print("[Trainer._validate] Warning: self.rank not found, assuming 0.")

            # Ensure val_step exists
            if not hasattr(self, 'val_step'):
                 print("[Trainer._validate] Error: val_step method is missing for standard validation.")
                 return {} # Return empty metrics

            for val_batch in self.val_dataloader:
                # Move batch to device if needed by val_step
                # Assuming val_step handles device placement or self.rank determines device
                # val_batch = val_batch.to(self.rank)
                metrics = self.val_step(val_batch)
                for k, v in metrics.items():
                    all_metrics[k].append(v)

            # Aggregate metrics
            aggregated_metrics = {}
            for k, v in all_metrics.items():
                if v and isinstance(v[0], torch.Tensor):
                    try:
                        aggregated_metrics[k] = torch.mean(torch.stack(v)).item()
                    except Exception as e:
                         print(f"[Trainer] Warning: Could not aggregate metric '{k}': {e}")
                elif v:
                    try:
                        aggregated_metrics[k] = np.mean(v) # Handle non-tensor metrics if any
                    except Exception as e:
                         print(f"[Trainer] Warning: Could not aggregate non-tensor metric '{k}': {e}")


            self.logger.log(aggregated_metrics, step=self.global_steps)
            print(f"[Trainer] Standard Validation Metrics: {aggregated_metrics}")
            return aggregated_metrics

    def verify_worker_cuda_setup(self, worker_name, worker_group):
        """Verify if worker has correctly set up CUDA devices"""
        print(f"\n--- Verifying CUDA for {worker_name} --- ")
        try:
            worker_info = None
            if hasattr(worker_group, 'get_worker_info') and callable(getattr(worker_group, 'get_worker_info')):
                 # Wrap remote call in try-except
                 try:
                     worker_info = ray.get(worker_group.get_worker_info.remote())
                     print(f"[CUDA DEBUG] {worker_name} worker info (from group): {worker_info}")
                 except Exception as e_info:
                     print(f"[CUDA DEBUG][ERROR] Failed to get worker_info for {worker_name}: {e_info}")

            # Remotely check worker's internal CUDA status
            gpu_status = None
            model_device_info = None
            if hasattr(worker_group, 'run_function') and callable(getattr(worker_group, 'run_function')):
                 # Define check functions to run remotely
                 def check_gpu_setup_remote():
                     import torch, os
                     cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
                     is_available = torch.cuda.is_available()
                     count = torch.cuda.device_count() if is_available else 0
                     devices = [torch.cuda.get_device_name(i) for i in range(count)] if count > 0 else []
                     return {
                         'pid': os.getpid(),
                         'host': os.uname()[1],
                         'CUDA_VISIBLE_DEVICES': cuda_visible,
                         'torch.cuda.is_available': is_available,
                         'torch.cuda.device_count': count,
                         'device_names': devices
                     }

                 def check_model_device_remote(worker_instance):
                    # Assuming the model is accessible via an attribute like 'model' or similar
                    # This needs adjustment based on actual worker implementation
                    model_attr_names = ['actor_module_fsdp', 'critic_module', 'ref_module_fsdp', 'reward_module']
                    devices = {}
                    for attr_name in model_attr_names:
                        if hasattr(worker_instance, attr_name):
                            model = getattr(worker_instance, attr_name)
                            if hasattr(model, 'device'):
                                devices[attr_name] = str(model.device)
                            elif hasattr(model, 'module') and hasattr(model.module, 'device'): # Check wrapped module
                                 devices[attr_name] = str(model.module.device)
                            elif hasattr(model, 'parameters'):
                                try:
                                    first_param_device = next(model.parameters()).device
                                    devices[attr_name] = str(first_param_device)
                                except StopIteration:
                                    devices[attr_name] = "No parameters"
                    return devices if devices else "Model or device info not accessible"

                 try:
                     # Use run_function_on_all_workers_sync or similar if available,
                     # otherwise run on rank 0. Adjust based on RayWorkerGroup implementation.
                     # Assuming run_function runs on rank 0 by default if not specified:
                     worker_gpu_check = worker_group.run_function.remote(check_gpu_setup_remote)
                     gpu_status = ray.get(worker_gpu_check)
                     print(f"[CUDA DEBUG] {worker_name} internal GPU status: {gpu_status}")

                     # Pass 'self' to check model device on the worker instance
                     model_device_check = worker_group.run_function.remote(check_model_device_remote, args=[worker_group.workers[0]]) # Check on rank 0
                     model_device_info = ray.get(model_device_check)
                     print(f"[CUDA DEBUG] {worker_name} internal model device info: {model_device_info}")

                 except Exception as e_remote:
                     print(f"[CUDA DEBUG][ERROR] Error running remote check on {worker_name}: {e_remote}")

            else:
                 print(f"[CUDA DEBUG] {worker_name} does not support remote function execution for detailed checks.")

            print(f"--- Verification for {worker_name} complete --- \n")
            return gpu_status, model_device_info

        except Exception as e:
            print(f"[CUDA DEBUG][ERROR] Error checking {worker_name} CUDA setup: {e}")
            import traceback
            traceback.print_exc()
            print(f"--- Verification for {worker_name} failed --- \n")
            return False, None

    def init_workers(self):
        """Init resource pool and worker group - add GPU device checks and pass assignments"""
        # Print driver's view of CUDA before starting workers (main_task context)
        print(f"\n[Trainer.init_workers @ {os.uname()[1]}] Running in PID: {os.getpid()}")
        get_safe_device() # This implicitly prints warnings if CUDA is not available
        driver_device_count = torch.cuda.device_count()
        print(f"[Trainer.init_workers] Driver process (main_task) sees {driver_device_count} CUDA devices.")
        for i in range(driver_device_count):
             try:
                 print(f"  Driver Device {i}: {torch.cuda.get_device_name(i)}")
             except Exception as e:
                 print(f"  Error getting name for driver device {i}: {e}")
             
        # Modify the resource pool specification to ensure each node has the correct number of GPUs
        # First, determine total GPUs needed
        total_gpus_needed = 1  # Default minimum
        
        # Examine the config to see if we should allocate more GPUs
        # For now, just use a simple approach - we'll allocate at least 1 GPU per node
        if hasattr(self.config, 'trainer') and hasattr(self.config.trainer, 'n_gpus_per_node'):
            total_gpus_needed = self.config.trainer.n_gpus_per_node
        
        print(f"[Trainer.init_workers] Configuring resource pools with {total_gpus_needed} GPUs per node")
        
        # Create the resource pool with the specified number of GPUs per node
        self.resource_pool_manager.create_resource_pool()
        print(f"[Trainer.init_workers] Resource pools created: {list(self.resource_pool_manager.resource_pool_dict.keys())}")

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # --- Map Roles to Classes and Resource Pools ---
        # create actor and rollout - WITHOUT specifying ray_options
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            
            # Create without ray_options - use original approach
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role='actor_rollout'
            )
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
            print(f"[Trainer.init_workers] ActorRollout mapped to pool '{resource_pool.name_prefix}'")
        else:
            raise NotImplementedError

        # create critic - WITHOUT specifying ray_options
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], 
                config=self.config.critic
            )
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
            print(f"[Trainer.init_workers] Critic mapped to pool '{resource_pool.name_prefix}'")
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed - WITHOUT specifying ray_options
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            
            ref_policy_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role='ref'
            )
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls
            print(f"[Trainer.init_workers] RefPolicy mapped to pool '{resource_pool.name_prefix}'")

        # create a reward model if reward_fn is None - WITHOUT specifying ray_options
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], 
                config=self.config.reward_model
            )
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls
            print(f"[Trainer.init_workers] RewardModel mapped to pool '{resource_pool.name_prefix}'")

        # ... rest of the method remains unchanged
        # initialize WorkerGroup
        all_wg = {}
        self.wg_dicts = []
        print("\n[Trainer.init_workers] Initializing Worker Groups...")
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            print(f"  Initializing group for resource pool: {resource_pool.name_prefix}")
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            # Pass resource requests (like num_gpus) defined in RayClassWithInitArgs to the group
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            print(f"  Spawning workers for group {resource_pool.name_prefix}...")
            try:
                spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
                all_wg.update(spawn_wg)
                self.wg_dicts.append(wg_dict)
                print(f"  Successfully spawned workers: {list(spawn_wg.keys())}")
                
                # --- Log assigned resources --- 
                # Note: Getting precise GPU IDs assigned by Ray to specific actors 
                # after spawn can be tricky from the outside. 
                # We'll rely on checks *inside* the worker for now. 
                # Logging the group's overall placement gives some clue.
                if hasattr(wg_dict, 'get_placement_group') and callable(getattr(wg_dict, 'get_placement_group')):
                    pg = wg_dict.get_placement_group()
                    if pg:
                         print(f"    Group {resource_pool.name_prefix} placement group details: {pg.bundle_specs}")
                    else:
                         print(f"    Group {resource_pool.name_prefix} does not have a placement group.")
                else:
                    print(f"    Cannot get placement group details for group {resource_pool.name_prefix}.")
                    
            except Exception as e:
                 print(f"[ERROR] Failed to spawn workers for group {resource_pool.name_prefix}: {e}")
                 import traceback
                 traceback.print_exc()
                 raise # Re-raise the exception to stop execution

        # --- Assign worker groups --- 
        # Use .get for safety in case spawning failed for a group
        if self.use_critic:
            self.critic_wg = all_wg.get('critic')
            if self.critic_wg:
                print("[Trainer.init_workers] Initializing Critic model...")
                # TODO: Modify init_model call to pass assigned GPU IDs if known
                self.critic_wg.init_model() 
            else:
                print("[Trainer.init_workers][ERROR] Critic worker group not found after spawn.")
                # Decide how to handle this - maybe raise an error?

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg.get('ref')
            if self.ref_policy_wg:
                print("[Trainer.init_workers] Initializing RefPolicy model...")
                # TODO: Modify init_model call
                self.ref_policy_wg.init_model()
            else:
                print("[Trainer.init_workers][ERROR] RefPolicy worker group not found after spawn.")

        if self.use_rm:
            self.rm_wg = all_wg.get('rm')
            if self.rm_wg:
                print("[Trainer.init_workers] Initializing RewardModel model...")
                # TODO: Modify init_model call
                self.rm_wg.init_model()
            else:
                 print("[Trainer.init_workers][ERROR] RewardModel worker group not found after spawn.")

        # Initialize actor_rollout last
        self.actor_rollout_wg = all_wg.get('actor_rollout')
        if self.actor_rollout_wg:
            print("[Trainer.init_workers] Initializing ActorRollout model...")
            # TODO: Modify init_model call
            self.actor_rollout_wg.init_model()
        else:
            print("[Trainer.init_workers][ERROR] ActorRollout worker group not found after spawn.")
        
        # --- Verify CUDA setup for each initialized worker group --- 
        print("\n[Trainer.init_workers] Verifying CUDA setup for initialized workers...")
        if self.actor_rollout_wg:
            self.verify_worker_cuda_setup("actor_rollout", self.actor_rollout_wg)
        if self.use_critic and self.critic_wg:
            self.verify_worker_cuda_setup("critic", self.critic_wg)
        if self.use_reference_policy and self.ref_policy_wg:
            self.verify_worker_cuda_setup("ref_policy", self.ref_policy_wg)
        if self.use_rm and self.rm_wg:
            self.verify_worker_cuda_setup("reward_model", self.rm_wg)
            
        print("[Trainer.init_workers] Worker initialization and verification complete.")

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO/GRPO, modified to properly handle advantage computation.
        """
        logger = self.logger
        self.global_steps = 0
        # Define log_dir here based on config
        self.log_dir = self.config.trainer.get("default_local_dir", "./verl_checkpoints/default_log_dir")
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[Trainer.fit] Log directory set to: {self.log_dir}")

        # Determine if this is an AgentGym run upfront
        self.is_agentgym_run = self.config.data.env_name in KNOWN_AGENTGYM_ENVS
        print(f"[Trainer.fit] Is AgentGym run: {self.is_agentgym_run}")
        
        # Get advantage estimator strategy
        adv_estimator = self.config.algorithm.adv_estimator
        print(f"[Trainer.fit] Using advantage estimator: {adv_estimator}")
        
        # 如果使用GRPO但仍然设置了use_critic为True，发出警告
        if adv_estimator == 'grpo' and self.use_critic:
            print(f"[Trainer.fit] WARNING: Using GRPO estimator with critic enabled. For pure GRPO, critic is not required.")

        # we start from step 1
        self.global_steps += 1

        # Agent config preparation (Only needed if AgentGym run)
        generation_manager = None
        if self.is_agentgym_run:
            print(f"[Trainer.fit] Initializing OpenManusAgent for AgentGym environment: {self.config.data.env_name}")
            try:
                gen_config = AgentConfig(
                    max_turns=self.config.max_turns,
                    max_start_length=self.config.data.max_start_length,
                    max_prompt_length=self.config.data.max_prompt_length,
                    max_response_length=self.config.data.max_response_length,
                    max_obs_length=self.config.data.max_obs_length,
                    num_gpus=self.config.trainer.n_gpus_per_node,
                    env_name=self.config.data.env_name,
                    env_ports=self.config.data.env_ports,
                    env_server_base=self.config.data.env_server_base,
                    env_data_len=self.config.data.get('env_data_len', 200),
                    max_workers=self.config.actor_rollout_ref.rollout.get('max_workers', 10),
                    algorithm_config=self.config.algorithm,
                )
                print(f"[Trainer.fit] AgentConfig initialized successfully")
                
                agent_logger = self.logger if hasattr(self, 'logger') else None
                print(f"[Trainer.fit] Creating OpenManusAgent...")
                generation_manager = OpenManusAgent(
                    tokenizer=self.tokenizer,
                    actor_rollout_wg=self.actor_rollout_wg,
                    config=gen_config,
                    is_validation = False,
                    logger=agent_logger
                )
                print(f"[Trainer.fit] OpenManusAgent created successfully")
            except Exception as e:
                print(f"[Trainer.fit][ERROR] Failed to initialize OpenManusAgent: {e}")
                import traceback
                traceback.print_exc()
                raise

        # start training loop
        print(f"[Trainer.fit] Starting training loop for {self.config.trainer.total_epochs} epochs")
        for epoch in range(self.config.trainer.total_epochs):
            print(f"[Trainer.fit] Starting epoch {epoch}")
            for batch_idx, batch_dict in enumerate(self.train_dataloader):
                print(f"[Trainer.fit][STEP] === Epoch {epoch}, Step {self.global_steps}, Batch {batch_idx} ===")
                metrics = {}
                timing_raw = {}

                print(f"[Trainer.fit][STEP {self.global_steps}] Creating DataProto from batch dictionary")
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                original_batch_size = batch.batch['input_ids'].shape[0]
                print(f"[Trainer.fit][STEP {self.global_steps}] Original batch size: {original_batch_size}")
                
                # Keep necessary keys for agent/rollout in gen_batch
                gen_batch = batch.pop(batch_keys=[
                    'input_ids', 'attention_mask', 'position_ids'
                ])
                # Add metadata if missing
                if 'idx' not in gen_batch.meta_info: 
                    gen_batch.meta_info['idx'] = torch.arange(original_batch_size)
                if 'reward_model' not in gen_batch.meta_info: 
                    gen_batch.meta_info['reward_model'] = [{} for _ in range(original_batch_size)]

                ####################
                # Rollout / Generation Step
                ####################
                print(f"[Trainer.fit][STEP {self.global_steps}] Starting rollout/generation step")
                final_gen_batch_output = None
                with _timer('step', timing_raw):
                    if self.is_agentgym_run:
                        # --- AgentGym Path ---
                        print(f"[Trainer.fit][STEP {self.global_steps}] Using AgentGym path")
                        with _timer('gen', timing_raw):
                            # Prepare output directory if logging images during training (less common)
                            output_dir = os.path.join(
                                self.log_dir, # Use the defined log_dir
                                f"train_step_{self.global_steps}"
                            )
                            print(f"[Trainer.fit][STEP {self.global_steps}] Calling generation_manager.run_llm_loop...")
                            try:
                                final_gen_batch_output = generation_manager.run_llm_loop(
                                    gen_batch=gen_batch,
                                    output_dir=output_dir,
                                    global_steps=self.global_steps
                                )
                                print(f"[Trainer.fit][STEP {self.global_steps}] Returned from generation_manager.run_llm_loop")
                            except Exception as e:
                                print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] Encountered error in run_llm_loop: {e}")
                                import traceback
                                traceback.print_exc()
                                continue # Skip to next batch if rollout failed

                        if not final_gen_batch_output or final_gen_batch_output.batch is None or final_gen_batch_output.batch.is_empty():
                            print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] AgentGym rollout returned empty batch. Skipping step.")
                            continue # Skip to next training batch

                        # Add log probs (needed for PPO loss calculation later)
                        print(f"[Trainer.fit][STEP {self.global_steps}] Computing log probabilities")
                        with torch.no_grad(), _timer('logp', timing_raw):
                            if 'input_ids' in final_gen_batch_output.batch:
                                logp_mbs = self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size
                                final_gen_batch_output.meta_info['micro_batch_size'] = logp_mbs
                                temperature = self.config.actor_rollout_ref.rollout.temperature
                                final_gen_batch_output.meta_info['temperature'] = temperature
                                use_dyn_bsz = self.config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
                                final_gen_batch_output.meta_info['use_dynamic_bsz'] = use_dyn_bsz
                                print(f"[Trainer.fit][STEP {self.global_steps}] Calling actor_rollout_wg.compute_log_prob...")
                                try:
                                    output_logp = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                                    final_gen_batch_output = final_gen_batch_output.union(output_logp)
                                    print(f"[Trainer.fit][STEP {self.global_steps}] Log probabilities computed successfully")
                                except Exception as e:
                                    print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] Error computing log probabilities: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    continue # Skip to next batch if compute_log_prob failed
                            else:
                                print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] Cannot compute log probabilities, 'input_ids' not found in batch")
                                continue # Skip this batch

                        batch = final_gen_batch_output
                        print(f"[Trainer.fit][STEP {self.global_steps}] Setting up token_level_scores")
                        if 'token_level_rewards' in batch.batch:
                            batch.batch['token_level_scores'] = batch.batch['token_level_rewards'].clone()
                            print(f"[Trainer.fit][STEP {self.global_steps}] Cloned token_level_rewards to token_level_scores")
                        else:
                            print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] 'token_level_rewards' not found in batch. Creating zero scores.")
                            if 'input_ids' in batch.batch:
                                batch.batch['token_level_scores'] = torch.zeros_like(batch.batch['input_ids'], dtype=torch.float)
                            else:
                                print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] Cannot create zero 'token_level_scores' because 'input_ids' is missing.")
                                continue

                        # --- FIX: Convert UID list to NumPy array with dtype=object ---
                        print(f"[Trainer.fit][STEP {self.global_steps}] Setting up UID for batch")
                        if 'idx' in batch.meta_info:
                            # Ensure idx tensor is moved to CPU before converting to list
                            uid_list = batch.meta_info['idx'].cpu().tolist()
                            batch.non_tensor_batch['uid'] = np.array(uid_list, dtype=object) # Explicitly set dtype=object
                            print(f"[Trainer.fit][STEP {self.global_steps}] Used existing idx as UID")
                        else: # Fallback UID
                            uid_list = [str(uuid.uuid4()) for _ in range(batch.batch['input_ids'].shape[0])]
                            batch.non_tensor_batch['uid'] = np.array(uid_list, dtype=object) # Explicitly set dtype=object
                            print(f"[Trainer.fit][STEP {self.global_steps}] Created new UUIDs as UID")
                        # --- END FIX ---

                    else:
                        # --- Original Path (Non-AgentGym) ---
                        print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] Non-AgentGym training path not implemented. Skipping.")
                        continue # Skip processing for now

                # Apply batch repetition if configured (AFTER generation/rollout)
                if self.config.actor_rollout_ref.rollout.n > 1:
                    print(f"[Trainer.fit][STEP {self.global_steps}] Repeating batch {self.config.actor_rollout_ref.rollout.n} times")
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                ####################
                # Post-Rollout Processing (Common for both paths after merging)
                ####################
                print(f"[Trainer.fit][STEP {self.global_steps}] Balancing batch")
                self._balance_batch(batch, metrics=metrics)
                print(f"[Trainer.fit][STEP {self.global_steps}] Batch balanced successfully")

                # --- Calculate Advantages and Returns (for both GRPO and GAE) ---
                print(f"[Trainer.fit][STEP {self.global_steps}] Computing advantages with estimator: {adv_estimator}")
                with _timer('adv', timing_raw):
                    try:
                        # Now explicitly call compute_advantage with the current batch
                        batch = compute_advantage(
                            data=batch, 
                            adv_estimator=adv_estimator,
                            gamma=self.config.algorithm.get('gamma', 1.0),
                            lam=self.config.algorithm.get('lambda', 1.0)
                        )
                        print(f"[Trainer.fit][STEP {self.global_steps}] Advantages computed successfully")
                    except Exception as e:
                        print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] Failed to compute advantages: {e}")
                        import traceback
                        traceback.print_exc()
                        continue # Skip to next batch if advantage computation failed

                # --- Compute Critic Values (if needed for GAE) ---
                if self.use_critic and adv_estimator == 'gae':
                    print(f"[Trainer.fit][STEP {self.global_steps}] Computing critic values for GAE")
                    with _timer('compute_values', timing_raw):
                        try:
                            # Ensure batch is on the correct device for the critic
                            worker_device_info = self.critic_wg.get_worker_info() if hasattr(self.critic_wg, 'get_worker_info') else None
                            requested_device = worker_device_info.get('device', 'cuda') if worker_device_info else 'cuda'
                            target_device = get_safe_device(requested_device)
                            critic_batch = batch.clone() if hasattr(batch, 'clone') else batch
                            
                            # Move tensors to target device if needed
                            current_device = None
                            if 'input_ids' in critic_batch.batch:
                                current_device = critic_batch.batch['input_ids'].device
                            if current_device is not None and str(current_device) != str(target_device):
                                if hasattr(critic_batch, 'to') and callable(getattr(critic_batch, 'to')):
                                    critic_batch = critic_batch.to(target_device)
                                else:
                                    for key in critic_batch.batch:
                                        if isinstance(critic_batch.batch[key], torch.Tensor):
                                            critic_batch.batch[key] = critic_batch.batch[key].to(target_device)
                            
                            # Call the critic to compute values (make sure this function exists in your critic worker)
                            values_output = self.critic_wg.compute_values(critic_batch)
                            
                            # Union the values with the original batch
                            batch = batch.union(values_output)
                            print(f"[Trainer.fit][STEP {self.global_steps}] Critic values computed successfully")
                        except Exception as e:
                            print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] Error computing critic values: {e}")
                            print(f"[Trainer.fit][STEP {self.global_steps}][WARNING] Will fall back to GRPO for advantage estimation")
                            # Fall back to GRPO if critic fails (we'll let compute_advantage handle the fallback)
                            import traceback
                            traceback.print_exc()
                
                # --- Calculate Advantages and Returns (for both GRPO and GAE) ---
                print(f"[Trainer.fit][STEP {self.global_steps}] Computing advantages with estimator: {adv_estimator}")
                with _timer('adv', timing_raw):
                    try:
                        # Now explicitly call compute_advantage with the current batch
                        batch = compute_advantage(
                            data=batch, 
                            adv_estimator=adv_estimator,
                            gamma=self.config.algorithm.get('gamma', 1.0),
                            lam=self.config.algorithm.get('lambda', 1.0)
                        )
                        print(f"[Trainer.fit][STEP {self.global_steps}] Advantages computed successfully")
                    except Exception as e:
                        print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] Failed to compute advantages: {e}")
                        import traceback
                        traceback.print_exc()
                        continue # Skip to next batch if advantage computation failed

                # --- KL Penalty (if using reference policy) ---
                if self.use_reference_policy and 'ref_log_prob' in batch.batch and 'old_log_probs' in batch.batch:
                    print(f"[Trainer.fit][STEP {self.global_steps}] Applying KL penalty")
                    with _timer('kl_penalty', timing_raw):
                        try:
                            batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, kl_penalty=self.config.algorithm.get('kl_penalty', 'kl'))
                            metrics.update(kl_metrics)
                            print(f"[Trainer.fit][STEP {self.global_steps}] KL penalty applied successfully")
                        except Exception as e:
                            print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] Error applying KL penalty: {e}")
                            import traceback
                            traceback.print_exc()
                            # Continue anyway, this isn't critical

                # --- Compute Critic Values ---
                if self.use_critic:
                    print(f"[Trainer.fit][STEP {self.global_steps}] Updating critic model")
                    if 'advantages' not in batch.batch or 'returns' not in batch.batch:
                        print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] Missing 'advantages' or 'returns' in batch, required for critic update. Skipping critic update.")
                        continue  # We change this from a warning to error and skip the batch
                    else:
                        with _timer('update_critic', timing_raw):
                            print(f"[Trainer.fit][STEP {self.global_steps}] Calling critic_wg.update_critic...")
                            try:
                                # REMOVED EXPLICIT DEVICE CHECK AND TRANSFER
                                # Let the critic worker handle device placement internally
                                critic_batch = batch # Pass the original batch

                                # Check devices of advantages and returns (for logging only)
                                adv_device = critic_batch.batch['advantages'].device
                                returns_device = critic_batch.batch['returns'].device
                                print(f"[CUDA DEBUG][STEP {self.global_steps}] Pre-critic tensor devices (data passed to worker): advantages={adv_device}, returns={returns_device}")

                                critic_output = self.critic_wg.update_critic(critic_batch)
                                critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                                metrics.update(critic_output_metrics)
                                print(f"[Trainer.fit][STEP {self.global_steps}] Critic model updated successfully")
                            except Exception as e:
                                print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] Error updating critic: {e}")
                                import traceback
                                traceback.print_exc()
                                continue  # Skip to next batch if critic update failed
                else:
                    print(f"[Trainer.fit][STEP {self.global_steps}] Skipping critic update (not enabled for {adv_estimator})")

                # --- Update Actor ---
                print(f"[Trainer.fit][STEP {self.global_steps}] Updating actor model")
                if self.config.trainer.critic_warmup <= self.global_steps:
                    if 'advantages' not in batch.batch or 'old_log_probs' not in batch.batch:
                        print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] Missing 'advantages' or 'old_log_probs' in batch, required for actor update. Skipping actor update.")
                        continue  # We change this from a warning to error and skip the batch
                    else:
                        with _timer('update_actor', timing_raw):
                            print(f"[Trainer.fit][STEP {self.global_steps}] Calling actor_rollout_wg.update_actor...")
                            try:
                                # First check if state_masking is enabled and create loss mask if needed
                                if self.is_agentgym_run and hasattr(self.config.actor_rollout_ref.actor, 'state_masking') and self.config.actor_rollout_ref.actor.state_masking:
                                    print(f"[Trainer.fit][STEP {self.global_steps}] State masking is enabled, creating loss_mask")
                                    # Create loss mask for state tokens
                                    batch, actor_metrics = self._create_loss_mask(batch, metrics)
                                    metrics.update(actor_metrics)
                                else:
                                    print(f"[Trainer.fit][STEP {self.global_steps}] State masking is not enabled, creating default loss_mask")
                                    # Create a default loss_mask (all 1s) if state_masking is not enabled
                                    response_length = batch.batch['responses'].shape[-1]
                                    batch.batch['loss_mask'] = torch.ones_like(batch.batch['attention_mask'][:, -response_length:])

                                # Check devices before actor update (for logging only)
                                loss_mask_device = batch.batch['loss_mask'].device
                                adv_device = batch.batch['advantages'].device
                                old_log_probs_device = batch.batch['old_log_probs'].device
                                print(f"[CUDA DEBUG][STEP {self.global_steps}] Pre-actor tensor devices (data passed to worker): loss_mask={loss_mask_device}, advantages={adv_device}, old_log_probs={old_log_probs_device}") # Added print

                                # REMOVED EXPLICIT DEVICE CHECK AND TRANSFER
                                # Let the actor worker handle device placement internally
                                actor_batch = batch # Pass the original batch

                                actor_output = self.actor_rollout_wg.update_actor(actor_batch)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                                metrics.update(actor_output_metrics)
                                print(f"[Trainer.fit][STEP {self.global_steps}] Actor model updated successfully")
                            except Exception as e:
                                print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] Error updating actor: {e}")
                                import traceback
                                traceback.print_exc()
                                continue  # Skip to next batch if actor update failed
                else:
                    print(f"[Trainer.fit][STEP {self.global_steps}] Skipping actor update (in critic warmup phase)")

                # --- Save Checkpoint ---
                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                    print(f"[Trainer.fit][STEP {self.global_steps}] Saving checkpoint")
                    with _timer('save_checkpoint', timing_raw):
                        try:
                            self._save_checkpoint()
                            print(f"[Trainer.fit][STEP {self.global_steps}] Checkpoint saved successfully")
                        except Exception as e:
                            print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] Error saving checkpoint: {e}")
                            import traceback
                            traceback.print_exc()

            # --- Collect and Log Metrics ---
            print(f"[Trainer.fit][STEP {self.global_steps}] Collecting and logging metrics")
            try:
                # Check for necessary keys before computing metrics
                required_keys = ['token_level_scores', 'token_level_rewards', 'advantages', 'returns', 'responses', 'attention_mask']
                if self.use_critic: required_keys.append('values')
                # Add meta_info keys needed for env metrics
                required_meta_keys = ['turns_stats', 'active_mask', 'valid_action_stats', 'valid_search_stats']
                
                # Ensure all required meta keys exist (add defaults if missing)
                for meta_key in required_meta_keys:
                    if meta_key not in batch.meta_info:
                        if meta_key == 'active_mask':
                            batch.meta_info[meta_key] = np.ones(batch.batch['input_ids'].shape[0], dtype=np.int16)
                        else:
                            batch.meta_info[meta_key] = np.zeros(batch.batch['input_ids'].shape[0], dtype=np.int16)
                        print(f"[Trainer.fit][STEP {self.global_steps}] Added default value for missing meta key: {meta_key}")

                can_compute_metrics = all(key in batch.batch for key in required_keys) and all(key in batch.meta_info for key in required_meta_keys)
                if can_compute_metrics:
                    print(f"[Trainer.fit][STEP {self.global_steps}] Computing all metrics")
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                else:
                    missing_keys = [k for k in required_keys if k not in batch.batch]
                    missing_meta = [k for k in required_meta_keys if k not in batch.meta_info]
                    print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] Cannot compute metrics due to missing keys: {missing_keys}, {missing_meta}")
                    # Log timing separately if main metrics can't be computed
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            except KeyError as e:
                print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] Metrics calculation failed due to KeyError: {e}")
            except Exception as e: # Catch other potential errors during metric calculation
                print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] Error during metric calculation: {e}")
                import traceback
                traceback.print_exc()

            # Log metrics
            print(f"[Trainer.fit][STEP {self.global_steps}] Logging metrics to tracking system")
            try:
                logger.log(data=metrics, step=self.global_steps)
            except Exception as e:
                print(f"[Trainer.fit][STEP {self.global_steps}][ERROR] Error logging metrics: {e}")
            
            print(f"[Trainer.fit][STEP {self.global_steps}] Completed step {self.global_steps}")
            self.global_steps += 1

            if self.global_steps >= self.total_training_steps:
                print(f"[Trainer.fit] Reached total training steps ({self.total_training_steps}). Exiting.")
                return

        print(f"[Trainer.fit] Completed epoch {epoch}")
    
    print(f"[Trainer.fit] Training complete")

    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics
