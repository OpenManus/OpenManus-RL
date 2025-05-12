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
    responses = data.batch['responses']  # Shape (B, L_resp)
    response_length = responses.size(1)  # L_resp
    token_level_scores = data.batch['token_level_scores']  # Shape (B, L_full)
    
    # Assuming old_log_probs and ref_log_prob are also L_full
    old_log_probs_full = data.batch.get('old_log_probs')
    ref_log_prob_full = data.batch.get('ref_log_prob')

    attention_mask_full = data.batch['attention_mask']  # Shape (B, L_full)
    # This mask is for the response part only
    response_mask = attention_mask_full[:, -response_length:]  # Shape (B, L_resp)

    beta = 0.0
    # Initialize with a tensor of correct shape and type for the case where KL is not computed.
    kld_response_part_masked = torch.zeros_like(response_mask, dtype=token_level_scores.dtype, device=token_level_scores.device) 
    
    actual_kld_for_metric = 0.0

    if ref_log_prob_full is not None and old_log_probs_full is not None:
        # Calculate KLD over the full length first
        kld_full = core_algos.kl_penalty(old_log_probs_full, ref_log_prob_full, kl_penalty=kl_penalty)  # Shape (B, L_full)
        
        # Slice KLD to the response part
        kld_response_part = kld_full[:, -response_length:]  # Shape (B, L_resp)
        
        # Apply response_mask to the sliced KLD part
        kld_response_part_masked = kld_response_part * response_mask  # Element-wise, shapes match
        beta = kl_ctrl.value
        
        # For KL controller update and metric, use unmasked kld_response_part with response_mask
        actual_kld_for_metric = masked_mean(kld_response_part, mask=response_mask, axis=-1) 
        actual_kld_for_metric = torch.mean(actual_kld_for_metric, dim=0).item()


    # Initialize token_level_rewards as a clone of full-length scores
    token_level_rewards_full = token_level_scores.clone()  # Shape (B, L_full)

    # Slice scores to the response part
    scores_response_part = token_level_scores[:, -response_length:]  # Shape (B, L_resp)
    
    # Calculate the rewards for the response tokens by subtracting scaled KLD
    # kld_response_part_masked already incorporates the response_mask for zeroing out padded tokens
    actual_response_rewards = scores_response_part - beta * kld_response_part_masked  # Shape (B, L_resp)
    
    # Place the calculated response rewards back into the correct segment of the full rewards tensor
    # We view the response part of the full tensor and update it using the response_mask.
    token_level_rewards_full_response_part_view = token_level_rewards_full[:, -response_length:]
    token_level_rewards_full_response_part_view[response_mask] = actual_response_rewards[response_mask]
    
    # Update KL controller
    current_batch_size = responses.shape[0] 
    kl_ctrl.update(current_kl=actual_kld_for_metric, n_steps=current_batch_size)

    data.batch['token_level_rewards'] = token_level_rewards_full

    metrics = {'critic/kl': actual_kld_for_metric, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    """
    Compute advantage estimates based on the specified estimator (GAE or GRPO).
    Ensures inputs to core_algos.compute_gae_advantage_return are correctly sliced to response_length.
    """
    if adv_estimator == 'gae':
        values_full = data.batch['values'] # Expected Shape (B, L_full)
        responses = data.batch['responses'] # Shape (B, L_resp)
        response_length = responses.size(-1) # L_resp
        
        attention_mask_full = data.batch['attention_mask'] # Shape (B, L_full)
        # This is the EoS mask for the response part
        response_eos_mask = attention_mask_full[:, -response_length:] # Shape (B, L_resp)
        
        token_level_rewards_full = data.batch['token_level_rewards'] # Shape (B, L_full)

        # Slice values and token_level_rewards to the response part
        values_response_part = values_full[:, -response_length:] # Shape (B, L_resp)
        token_level_rewards_response_part = token_level_rewards_full[:, -response_length:] # Shape (B, L_resp)

        # Now all inputs to compute_gae_advantage_return are response-length
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards_response_part,
            values=values_response_part,
            eos_mask=response_eos_mask, # This is already the response-specific mask
            gamma=gamma,
            lam=lam
        )
        # advantages and returns will have shape (B, L_resp)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards_full = data.batch['token_level_rewards']
        responses = data.batch['responses'] # L_resp
        response_length = responses.size(-1) # L_resp
        attention_mask_full = data.batch['attention_mask'] # L_full
        response_eos_mask = attention_mask_full[:, -response_length:] # Shape (B, L_resp)
        
        token_level_rewards_response_part = token_level_rewards_full[:, -response_length:]

        index = data.non_tensor_batch['uid']
        
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards_response_part,
            eos_mask=response_eos_mask,
            index=index
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data

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


def get_safe_device(requested_device='cuda', allow_cpu_fallback=True):
    """
    Get a torch device, with improved error handling for CUDA availability.
    
    Args:
        requested_device: The preferred device, e.g., 'cuda', 'cuda:0', etc.
        allow_cpu_fallback: If True, will return CPU device if CUDA is not available
        
    Returns:
        A torch device that matches the requested device or CPU if CUDA is not available
        and allow_cpu_fallback is True.
        
    Raises:
        RuntimeError: If CUDA is requested but not available and allow_cpu_fallback=False
    """
    import torch, os
    
    # Check if CUDA is available when requested
    if 'cuda' in str(requested_device) and not torch.cuda.is_available():
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
        if allow_cpu_fallback:
            print(f"[WARNING] CUDA requested ({requested_device}) but not available. "
                  f"CUDA_VISIBLE_DEVICES={cuda_visible}. Falling back to CPU.")
            return torch.device('cpu')
        else:
            raise RuntimeError(f"CUDA requested ({requested_device}) but not available. "
                              f"CUDA_VISIBLE_DEVICES={cuda_visible}")
    
    # If requesting a specific CUDA device, verify it exists
    if str(requested_device).startswith('cuda:') and torch.cuda.is_available():
        try:
            device_idx = int(str(requested_device).split(':')[1])
            if device_idx >= torch.cuda.device_count():
                if allow_cpu_fallback:
                    print(f"[WARNING] CUDA device {device_idx} requested but only "
                          f"{torch.cuda.device_count()} devices available. "
                          f"Using device cuda:0 instead.")
                    return torch.device('cuda:0')
                else:
                    raise RuntimeError(
                        f"CUDA device {device_idx} requested but only {torch.cuda.device_count()} "
                        f"devices available. Please specify a valid device index."
                    )
        except ValueError:
            if allow_cpu_fallback:
                print(f"[WARNING] Invalid CUDA device format: {requested_device}. Using default cuda device.")
                return torch.device('cuda')
            else:
                raise RuntimeError(f"Invalid CUDA device format: {requested_device}. Use 'cuda:n' where n is an integer.")
    
    # For non-CUDA devices or if CUDA is properly available
    try:
        return torch.device(requested_device)
    except Exception as e:
        if allow_cpu_fallback:
            print(f"[WARNING] Error creating device {requested_device}: {e}. Falling back to CPU.")
            return torch.device('cpu')
        else:
            raise RuntimeError(f"Error creating device {requested_device}: {e}")


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        reward_component_config: dict = None):

        # Check CUDA availability but don't fail if not available
        # Instead, log detailed information for diagnostics

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
    
    def _init_logger(self):
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

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

            # comment on the validation loop
            # for val_batch in self.val_dataloader:
            #     # Ensure batch is on the correct device (or handled by agent)
            #     # val_batch = val_batch.to(self.rank) # May not be needed if agent handles device placement

            #     # Agent's run_llm_loop returns a DataProto with results including rewards/scores
            #     processed_batch = self.validation_agent.run_llm_loop(val_batch, self.log_dir, self.global_steps)

            #     # --- Extract metrics from the agent's output ---
            #     # The reward/score should ideally be in processed_batch.meta_info
            #     # Let's assume 'env_score' holds the final task score per item
            #     if 'env_score' in processed_batch.meta_info:
            #         scores = processed_batch.meta_info['env_score']
            #         if isinstance(scores, torch.Tensor):
            #             scores = scores.cpu().tolist()
            #         all_metrics['val_reward_score'].extend(scores) # Use a consistent key
            #         all_metrics['env_score'].extend(scores) # Also log as env_score

            #     # Log other stats if available
            #     if 'turns_stats' in processed_batch.meta_info:
            #          turns = processed_batch.meta_info['turns_stats']
            #          if isinstance(turns, torch.Tensor): turns = turns.cpu().tolist()
            #          all_metrics['turns_stats'].extend(turns)

            #     if 'valid_action_stats' in processed_batch.meta_info:
            #          valid_actions = processed_batch.meta_info['valid_action_stats']
            #          if isinstance(valid_actions, torch.Tensor): valid_actions = valid_actions.cpu().tolist()
            #          all_metrics['valid_action_stats'].extend(valid_actions)

                # Add any other relevant metrics from the agent's output meta_info
                # ...

                # --- Optional: Save Trajectories/Visualizations ---
                # Make sure save_trajectory_to_output is imported
                # from openmanus_rl.utils.visualization import save_trajectory_to_output

                # if self.logger and 'rollout_trajectory' in processed_batch.meta_info:
                #     # Assuming save_rollout_data can handle the trajectory format
                #     # You might need to adapt this based on the logger's interface
                #     try:
                #         task_indices = processed_batch.meta_info.get('task_idx', list(range(len(processed_batch))))
                #         if isinstance(task_indices, torch.Tensor): task_indices = task_indices.cpu().tolist()

                #         for idx, trajectory in enumerate(processed_batch.meta_info['rollout_trajectory']):
                #             if idx < 5: # Limit saving to avoid excessive logging
                #                 original_task_idx = task_indices[idx]
                #                 save_trajectory_to_output(
                #                     trajectory,
                #                     output_dir=self.log_dir,
                #                     global_step=self.global_steps,
                #                     task_idx=original_task_idx,
                #                     prefix="val"
                #                 )
                #     except Exception as e:
                #          print(f"[Trainer] Warning: Failed to save validation trajectory: {e}")
                #          import traceback
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


            if aggregated_metrics:
                self.logger.log(aggregated_metrics, step=self.global_steps)
                print(f"[Trainer] Standard Validation Metrics: {aggregated_metrics}")
            else:
                print(f"[Trainer] Warning: No standard validation metrics were aggregated.")
            return aggregated_metrics

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
            
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

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
        print(f"[Trainer.fit][DEBUG] Log directory set to: {self.log_dir}") # DEBUG

        # Determine if this is an AgentGym run upfront
        self.is_agentgym_run = self.config.data.env_name in KNOWN_AGENTGYM_ENVS
        print(f"[Trainer.fit][DEBUG] Is AgentGym run: {self.is_agentgym_run}") # DEBUG
        
        # Get advantage estimator strategy
        adv_estimator = self.config.algorithm.adv_estimator
        print(f"[Trainer.fit][DEBUG] Using advantage estimator: {adv_estimator}") # DEBUG
        print(f"[Trainer.fit][DEBUG] Value of self.use_critic at start: {self.use_critic}") # DEBUG
        
        # 如果使用GRPO但仍然设置了use_critic为True，发出警告
        if adv_estimator == 'grpo' and self.use_critic:
            print(f"[Trainer.fit] WARNING: Using GRPO estimator with critic enabled. For pure GRPO, critic is not required.")

        # we start from step 1
        self.global_steps += 1

        # Agent config preparation (Only needed if AgentGym run)
        generation_manager = None
        if self.is_agentgym_run:
            print(f"[Trainer.fit][DEBUG] Initializing OpenManusAgent for AgentGym environment: {self.config.data.env_name}") # DEBUG
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
                print(f"[Trainer.fit][DEBUG] AgentConfig initialized successfully") # DEBUG
                
                agent_logger = self.logger if hasattr(self, 'logger') else None
                print(f"[Trainer.fit][DEBUG] Creating OpenManusAgent...") # DEBUG
                generation_manager = OpenManusAgent(
                    tokenizer=self.tokenizer,
                    actor_rollout_wg=self.actor_rollout_wg,
                    config=gen_config,
                    is_validation = False,
                    logger=agent_logger
                )
                print(f"[Trainer.fit][DEBUG] OpenManusAgent created successfully") # DEBUG
            except Exception as e:
                print(f"[Trainer.fit][ERROR] Failed to initialize OpenManusAgent: {e}")
                import traceback
                traceback.print_exc()
                raise

        # start training loop
        print(f"[Trainer.fit][DEBUG] Starting training loop for {self.config.trainer.total_epochs} epochs")
        for epoch in range(self.config.trainer.total_epochs):
            print(f"[Trainer.fit][DEBUG] Starting Epoch {epoch}") # DEBUG
            for batch_idx, batch_dict in enumerate(self.train_dataloader):
                print(f"[Trainer.fit][DEBUG] === Starting Epoch {epoch}, Step {self.global_steps}, Batch {batch_idx} ===") # DEBUG
                metrics = {}
                timing_raw = {}

                print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Creating DataProto from batch dictionary.") # DEBUG
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                original_batch_size = batch.batch['input_ids'].shape[0]
                print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Original batch size: {original_batch_size}") # DEBUG
                
                # --- Debug Print: Initial Batch State --- 
                print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Initial Batch Info:")
                print(f"  Batch Keys & Shapes & Devices:")
                # Check if batch attribute exists, is not None, and is not empty
                print(f"  Meta Info Keys: {list(batch.meta_info.keys()) if hasattr(batch, 'meta_info') else 'N/A'}")
                print(f"  Non-Tensor Batch Keys: {list(batch.non_tensor_batch.keys()) if hasattr(batch, 'non_tensor_batch') else 'N/A'}")
                # --- End Debug Print ---
                
                # Keep necessary keys for agent/rollout in gen_batch
                print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Popping keys for generation batch.") # DEBUG
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
                print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Starting rollout/generation step.") # DEBUG
                final_gen_batch_output = None
                with _timer('step', timing_raw):
                    if self.is_agentgym_run:
                        # --- AgentGym Path --- 
                        print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Using AgentGym path.") # DEBUG
                        with _timer('gen', timing_raw):
                            output_dir = os.path.join(
                                self.log_dir, 
                                f"train_step_{self.global_steps}"
                            )
                            print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Calling generation_manager.run_llm_loop...") # DEBUG
                            # --- Debug Print: Input to run_llm_loop --- 
                            print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Input gen_batch to run_llm_loop:")
                            print(f"  Batch Keys & Shapes & Devices:")
                            # Check if batch attribute exists, is not None, and is not empty
                            print(f"  Meta Info Keys: {list(gen_batch.meta_info.keys()) if hasattr(gen_batch, 'meta_info') else 'N/A'}")
                            # --- End Debug Print ---
                            try:
                                final_gen_batch_output = generation_manager.run_llm_loop(
                                    gen_batch=gen_batch,
                                    output_dir=output_dir,
                                    global_steps=self.global_steps
                                )
                                print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Returned from generation_manager.run_llm_loop.") # DEBUG
                                # --- Debug Print: Output from run_llm_loop --- 
                                print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Output final_gen_batch_output from run_llm_loop:")
                                print(f"  Batch Keys & Shapes & Devices:")
                                # Check if batch attribute exists, is not None, and is not empty
                                print(f"  Meta Info Keys: {list(final_gen_batch_output.meta_info.keys()) if hasattr(final_gen_batch_output, 'meta_info') else 'N/A'}")
                                # --- End Debug Print ---
                            except Exception as e:
                                print(f"[Trainer.fit][ERROR] Step {self.global_steps}: Encountered error in run_llm_loop: {e}") # ERROR
                                import traceback
                                traceback.print_exc()
                                continue # Skip to next batch if rollout failed

                        if not final_gen_batch_output or final_gen_batch_output.batch is None or final_gen_batch_output.batch.is_empty():
                            print(f"[Trainer.fit][ERROR] Step {self.global_steps}: AgentGym rollout returned empty batch. Skipping step.") # ERROR
                            raise RuntimeError(f"AgentGym rollout returned empty batch at step {self.global_steps}")

                        # Add log probs (needed for PPO loss calculation later)
                        print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Computing log probabilities.") # DEBUG
                        with torch.no_grad(), _timer('logp', timing_raw):
                            if 'input_ids' in final_gen_batch_output.batch:
                                actor_rollout_world_size = self.actor_rollout_wg.world_size
                                print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: ActorRollout world size for padding: {actor_rollout_world_size}") # DEBUG

                                padded_batch_for_logp, pad_size_logp = pad_dataproto_to_divisor(
                                    final_gen_batch_output, 
                                    actor_rollout_world_size
                                )
                                if pad_size_logp > 0:
                                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Padded batch for compute_log_prob by {pad_size_logp} samples.") # DEBUG
                                
                                # --- Populate meta_info for actor_rollout_wg.compute_log_prob ---
                                # These parameters are expected by DataParallelActor.compute_log_prob on the worker.
                                # Source them from self.config.actor_rollout_ref.rollout
                                logp_mbs = self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size
                                use_dyn_bsz = self.config.actor_rollout_ref.rollout.get('log_prob_use_dynamic_bsz', False)
                                temperature = self.config.actor_rollout_ref.rollout.temperature
                                print(f"[DEBUG][Trainer.fit] Step {self.global_steps}: Sourced config for actor.compute_log_prob: log_prob_micro_batch_size={logp_mbs}, log_prob_use_dynamic_bsz={use_dyn_bsz}, temperature={temperature}") # DEBUG
                                padded_batch_for_logp.meta_info['micro_batch_size'] = logp_mbs
                                padded_batch_for_logp.meta_info['use_dynamic_bsz'] = use_dyn_bsz
                                padded_batch_for_logp.meta_info['temperature'] = temperature
                                if use_dyn_bsz:
                                    max_token_len_logp = self.config.actor_rollout_ref.rollout.get(
                                        'log_prob_max_token_len_per_gpu', 
                                        self.config.data.max_prompt_length 
                                    )
                                    padded_batch_for_logp.meta_info['max_token_len'] = max_token_len_logp
                                    print(f"[DEBUG][Trainer.fit] Step {self.global_steps}: For dynamic log_prob batching, set max_token_len={max_token_len_logp}") # DEBUG
                                else:
                                    padded_batch_for_logp.meta_info.pop('max_token_len', None)
                                print(f"[DEBUG][Trainer.fit] Step {self.global_steps}: Final padded_batch_for_logp.meta_info for compute_log_prob: {padded_batch_for_logp.meta_info}") # DEBUG
                                # --- End of meta_info population ---
                                
                                # --- Debug Print: Input to compute_log_prob --- 
                                print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Input padded_batch_for_logp to compute_log_prob:")
                                print(f"  Batch Keys & Shapes & Devices:")
                                # Check if batch attribute exists, is not None, and is not empty
                                print(f"  Meta Info: {padded_batch_for_logp.meta_info if hasattr(padded_batch_for_logp, 'meta_info') else 'N/A'}")
                                # --- End Debug Print ---
                                print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Calling actor_rollout_wg.compute_log_prob...") # DEBUG
                                try:
                                    output_logp_padded = self.actor_rollout_wg.compute_log_prob(padded_batch_for_logp)
                                    
                                    output_logp = unpad_dataproto(output_logp_padded, pad_size=pad_size_logp)
                                    if pad_size_logp > 0:
                                        print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Unpadded log_prob output.") # DEBUG

                                    final_gen_batch_output = final_gen_batch_output.union(output_logp)
                                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Log probabilities computed successfully.") # DEBUG
                                except Exception as e:
                                    print(f"[Trainer.fit][ERROR] Step {self.global_steps}: Error computing log probabilities: {e}") # ERROR
                                    traceback.print_exc()
                                    raise # Re-raise to halt execution
                            else:
                                print(f"[Trainer.fit][ERROR] Step {self.global_steps}: Cannot compute log probabilities, 'input_ids' not found in batch.") # ERROR
                                raise RuntimeError("Cannot compute log_prob, input_ids missing") # Halt execution

                        batch = final_gen_batch_output # Update batch with the results
                        print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Setting up token_level_scores.") # DEBUG
                        if 'token_level_rewards' in batch.batch:
                            batch.batch['token_level_scores'] = batch.batch['token_level_rewards'].clone()
                            print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Cloned token_level_rewards to token_level_scores.") # DEBUG
                        else:
                            print(f"[Trainer.fit][ERROR] Step {self.global_steps}: 'token_level_rewards' not found in batch after run_llm_loop. Creating zero scores.") # ERROR
                            if 'input_ids' in batch.batch:
                                batch.batch['token_level_scores'] = torch.zeros_like(batch.batch['input_ids'], dtype=torch.float)
                            else:
                                print(f"[Trainer.fit][ERROR] Step {self.global_steps}: Cannot create zero 'token_level_scores' because 'input_ids' is missing.") # ERROR
                                continue

                        print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Setting up UID for batch.") # DEBUG
                        if 'idx' in batch.meta_info:
                            uid_list = batch.meta_info['idx'].cpu().tolist()
                            batch.non_tensor_batch['uid'] = np.array(uid_list, dtype=object)
                            print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Used existing idx as UID.") # DEBUG
                        else: 
                            uid_list = [str(uuid.uuid4()) for _ in range(batch.batch['input_ids'].shape[0])]
                            batch.non_tensor_batch['uid'] = np.array(uid_list, dtype=object)
                            print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Created new UUIDs as UID.") # DEBUG

                    else:
                        # --- Original Path (Non-AgentGym) --- 
                        print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Using Non-AgentGym generation path.") # DEBUG
                        # Add debug logs for non-agentgym path if needed
                        print(f"[Trainer.fit][ERROR] Step {self.global_steps}: Non-AgentGym training path not fully implemented with debug logs. Skipping.") # ERROR
                        continue # Skip processing for now

                # Apply batch repetition if configured (AFTER generation/rollout)
                if self.config.actor_rollout_ref.rollout.n > 1:
                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Repeating batch {self.config.actor_rollout_ref.rollout.n} times.") # DEBUG
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                ####################
                # Post-Rollout Processing (Common for both paths after merging)
                ####################
                print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Starting post-rollout processing.") # DEBUG
                
                print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Balancing batch...") # DEBUG
                self._balance_batch(batch, metrics=metrics)
                print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Batch balanced successfully.") # DEBUG

                # compute global_valid tokens (Maybe move after all data is ready?)
                if 'attention_mask' in batch.batch:
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                else:
                    print(f"[Trainer.fit][WARN] Step {self.global_steps}: 'attention_mask' not in batch for global_token_num calculation.")

                # --- Compute Reference Log Probs --- 
                if self.use_reference_policy:
                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Computing reference log probs...") # DEBUG
                    with _timer('ref', timing_raw):
                        # --- Debug Print: Input to compute_ref_log_prob --- 
                        print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Input batch to compute_ref_log_prob:")
                        print(f"  Batch Keys & Shapes & Devices:")
                        # Check if batch attribute exists, is not None, and is not empty
                        # --- End Debug Print ---
                        ref_log_prob_output = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob_output)
                        print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Reference log probs computed.") # DEBUG
                else:
                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Skipping reference log prob computation.") # DEBUG

                # --- Compute Critic Values --- 
                if self.use_critic and adv_estimator == 'gae':
                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Computing critic values for GAE...") # DEBUG
                    with _timer('compute_values', timing_raw):
                        # Read config values for critic
                        critic_mbs = self.config.critic.ppo_micro_batch_size
                        critic_use_dyn_bsz = self.config.critic.get('use_dynamic_bsz', False)
                        print(f"[CRITICAL DEBUG][Trainer.fit] Step {self.global_steps}: Value read from self.config.critic.ppo_micro_batch_size = {critic_mbs}")
                        print(f"[CRITICAL DEBUG][Trainer.fit] Step {self.global_steps}: Value read from self.config.critic.get('use_dynamic_bsz') = {critic_use_dyn_bsz}")

                        # --- Create a temporary, isolated DataProto for the compute_values call --- 
                        # 1. Create a dedicated meta_info dictionary
                        critic_meta_info = {}
                        critic_meta_info['micro_batch_size'] = critic_mbs
                        critic_meta_info['use_dynamic_bsz'] = critic_use_dyn_bsz
                        if critic_use_dyn_bsz:
                            critic_meta_info['max_token_len'] = self.config.critic.get('ppo_max_token_len_per_gpu', 2048)
                        # No need to pop max_token_len if false, it just won't be added
                        print(f"[DEBUG][Trainer.fit] Step {self.global_steps}: Prepared TEMPORARY critic_meta_info for compute_values: {critic_meta_info}") # DEBUG
                        
                        # 2. Select only the necessary tensors from the original batch
                        critic_input_tensors = {
                            key: batch.batch[key] 
                            for key in ['responses', 'input_ids', 'attention_mask', 'position_ids'] 
                            if key in batch.batch
                        }
                        if len(critic_input_tensors) != 4:
                            print(f"[WARN][Trainer.fit] Step {self.global_steps}: Missing some required keys for critic compute_values in batch.batch. Found: {list(critic_input_tensors.keys())}")
                        
                        # 3. Create the temporary DataProto object
                        critic_input_proto = DataProto.from_dict(critic_input_tensors)
                        critic_input_proto.meta_info = critic_meta_info # Assign the dedicated meta_info

                        # --- Debug print the temporary object --- 
                        print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: TEMPORARY Input critic_input_proto to compute_values:")
                        print(f"  Batch Keys & Shapes & Devices:")
                        print(f"  Meta Info: {critic_input_proto.meta_info}")
                        # --- End Debug print --- 

                        # 4. Call the worker with the temporary object
                        values_output = self.critic_wg.compute_values(critic_input_proto)
                        # --- End modification for temporary object ---

                        # --- Debug Print: Output from compute_values --- 
                        print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Output from compute_values:")
                        # --- End Debug Print ---
                        batch = batch.union(values_output)
                        print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Critic values computed and merged.") # DEBUG
                elif self.use_critic and adv_estimator != 'gae':
                     print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Critic exists but not used for GAE, skipping compute_values for advantage calculation.") # DEBUG
                else:
                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Critic not used, skipping compute_values.") # DEBUG

                # --- Apply Reward Function / KL Penalty --- 
                print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Computing scores and rewards...") # DEBUG
                with _timer('adv', timing_raw):
                    # Compute scores (potentially using RM)
                    if self.use_rm:
                        print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Computing RM score...") # DEBUG
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)
                        print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: RM score computed and merged.") # DEBUG
                    
                    # Combine with rule-based/external reward function if provided
                    if self.reward_fn is not None:
                         print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Applying external reward_fn...") # DEBUG
                         reward_tensor = self.reward_fn(batch) # Assuming reward_fn returns the tensor directly
                         batch.batch['token_level_scores'] = reward_tensor
                         print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: External reward_fn applied.") # DEBUG
                    elif 'reward_model_scores' in batch.batch: # If only RM was used
                         print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Using RM scores as token_level_scores.") # DEBUG
                         batch.batch['token_level_scores'] = batch.batch['reward_model_scores']
                    elif 'token_level_scores' not in batch.batch:
                         print(f"[Trainer.fit][WARN] Step {self.global_steps}: 'token_level_scores' not found after RM/reward_fn. Ensure one is active or rewards are set elsewhere.") # WARN
                         # If scores are set directly by agentgym run_llm_loop, this might be okay.

                    # Apply KL penalty (modifies token_level_scores -> token_level_rewards)
                    if self.use_reference_policy and not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                        print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Applying KL penalty...") # DEBUG
                        batch, kl_metrics = apply_kl_penalty(batch,
                                                             kl_ctrl=self.kl_ctrl,
                                                             kl_penalty=self.config.algorithm.get('kl_penalty', 'kl')) # Use .get
                        metrics.update(kl_metrics)
                        print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: KL penalty applied.") # DEBUG
                    else:
                        if 'token_level_rewards' not in batch.batch and 'token_level_scores' in batch.batch:
                            print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Setting token_level_rewards = token_level_scores (no KL penalty/loss).") # DEBUG
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores'].clone()
                        elif 'token_level_rewards' not in batch.batch:
                             print(f"[Trainer.fit][WARN] Step {self.global_steps}: 'token_level_rewards' not set and KL penalty not applied.") # WARN
                    
                    # --- Compute Advantages --- 
                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Computing advantages (estimator: {adv_estimator})...") # DEBUG
                    # --- Debug Print: Input to compute_advantage --- 
                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Input batch to compute_advantage:")
                    print(f"  Batch Keys & Shapes & Devices:")
                    # Check if batch attribute exists, is not None, and is not empty
                    batch = compute_advantage(batch,
                                              adv_estimator=adv_estimator,
                                              gamma=self.config.algorithm.get('gamma', 1.0),
                                              lam=self.config.algorithm.get('lambda', 1.0),
                                              num_repeat=self.config.actor_rollout_ref.rollout.get('n', 1)) # Use .get
                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Advantages computed.") # DEBUG

                # --- Update Critic --- 
                if self.use_critic:
                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Updating critic...") # DEBUG
                    with _timer('update_critic', timing_raw):
                         # --- Debug Print: Input to update_critic --- 
                        print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Input batch to update_critic:")
                        print(f"  Batch Keys & Shapes & Devices:")
                        # Check if batch attribute exists, is not None, and is not empty
                        # --- End Debug Print ---
                        critic_output = self.critic_wg.update_critic(batch) # Returns DataProto with metrics
                        if hasattr(critic_output, 'meta_info') and 'metrics' in critic_output.meta_info:
                            critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                            metrics.update(critic_output_metrics)
                        else:
                            print(f"[Trainer.fit][WARN] Step {self.global_steps}: Critic update did not return metrics in meta_info.")
                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Critic updated.") # DEBUG
                else:
                     print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Skipping critic update.") # DEBUG

                # --- Update Actor --- 
                if self.config.trainer.get('critic_warmup', 0) <= self.global_steps: # Use .get
                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Updating actor...") # DEBUG
                    with _timer('update_actor', timing_raw):
                        # state masking is only applicable for search agent
                        if self.is_agentgym_run and hasattr(self.config.actor_rollout_ref.actor, 'state_masking') and self.config.actor_rollout_ref.actor.state_masking:
                            print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Applying state masking...") # DEBUG
                            batch, actor_metrics = self._create_loss_mask(batch, metrics)
                            metrics.update(actor_metrics)
                            print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: State masking applied.") # DEBUG
                        elif 'loss_mask' not in batch.batch:
                             # Ensure loss_mask exists if not using state masking (defaults to response mask)
                             if 'responses' in batch.batch and 'attention_mask' in batch.batch:
                                 response_length = batch.batch['responses'].shape[-1]
                                 batch.batch['loss_mask'] = batch.batch['attention_mask'][:, -response_length:].clone()
                                 print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Created default loss_mask (response mask).") # DEBUG
                             else:
                                 print(f"[Trainer.fit][WARN] Step {self.global_steps}: Cannot create default loss_mask, missing 'responses' or 'attention_mask'.") # WARN

                        # --- Debug Print: Input to update_actor --- 
                        print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Input batch to update_actor:")
                        print(f"  Batch Keys & Shapes & Devices:")
                        # Check if batch attribute exists, is not None, and is not empty
                        # --- End Debug Print ---
                        actor_output = self.actor_rollout_wg.update_actor(batch) # Returns DataProto with metrics
                        if hasattr(actor_output, 'meta_info') and 'metrics' in actor_output.meta_info:
                            actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                            metrics.update(actor_output_metrics)
                        else:
                             print(f"[Trainer.fit][WARN] Step {self.global_steps}: Actor update did not return metrics in meta_info.")
                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Actor updated.") # DEBUG
                else:
                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Skipping actor update (critic warmup phase).") # DEBUG

                # --- Validate --- 
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                    self.global_steps % self.config.trainer.test_freq == 0:
                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Starting validation...") # DEBUG
                    with _timer('testing', timing_raw):
                        val_metrics: dict = self._validate()
                    metrics.update(val_metrics)
                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Validation finished.") # DEBUG

                # --- Save Checkpoint --- 
                if self.config.trainer.save_freq > 0 and \
                        self.global_steps % self.config.trainer.save_freq == 0:
                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Saving checkpoint...") # DEBUG
                    with _timer('save_checkpoint', timing_raw):
                        self._save_checkpoint()
                    print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Checkpoint saved.") # DEBUG

                # --- Collect Metrics --- 
                print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Collecting and computing metrics...") # DEBUG
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Metrics computed: {list(metrics.keys())}") # DEBUG

                # --- Log Metrics --- 
                print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Logging metrics...") # DEBUG
                logger.log(data=metrics, step=self.global_steps)
                print(f"[Trainer.fit][DEBUG] Step {self.global_steps}: Metrics logged.") # DEBUG

                self.global_steps += 1

                if self.config.trainer.total_training_steps is not None and self.global_steps >= self.config.trainer.total_training_steps:
                    print(f"[Trainer.fit][DEBUG] Reached total training steps ({self.config.trainer.total_training_steps}). Exiting training loop.") # DEBUG
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        print(f"[Trainer.fit][DEBUG] Performing final validation...") # DEBUG
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                        print(f"[Trainer.fit][DEBUG] Final validation logged.") # DEBUG
                    return
            print(f"[Trainer.fit][DEBUG] Finished Epoch {epoch}") # DEBUG
        print(f"[Trainer.fit][DEBUG] Training loop finished after {self.config.trainer.total_epochs} epochs.") # DEBUG

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
