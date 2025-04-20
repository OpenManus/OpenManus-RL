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
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
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

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
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
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

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
        from verl.utils.tracking import Tracking
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

    def _validate(self):
        """
        Validation loop.
        """
        import torch
        all_metrics = defaultdict(list) 
        all_calculated_scores = [] # Store scores calculated by score_fn

        # --- Determine Score Function --- 
        score_fn = None
        score_fn_name = self.config.algorithm.get('reward_score_fn')
        if score_fn_name and score_fn_name in SUPPORTED_REWARD_SCORE_FNS:
            score_fn = SUPPORTED_REWARD_SCORE_FNS[score_fn_name]
            print(f"[Trainer._validate] Using reward score function: {score_fn_name}")
        else:
            print(f"[Trainer._validate] No valid reward_score_fn configured ('{score_fn_name}'). Using val_reward_fn if available.")
            score_fn = self.val_reward_fn # Fallback to RewardManager if passed
            if score_fn:
                 print(f"[Trainer._validate] Using val_reward_fn (likely RewardManager).")
            else:
                 print(f"[Trainer._validate] No score_fn or val_reward_fn available.")

        # Determine if this is an AgentGym run
        is_agentgym_run = self.config.data.env_name in KNOWN_AGENTGYM_ENVS

        # Agent config preparation (remains the same)
        gen_config = AgentConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node, 
            env_name=self.config.data.env_name, 
            env_port=self.config.data.env_port,
            env_server_base=self.config.data.env_server_base,
            env_data_len=self.config.data.get('env_data_len', 200),
            max_workers=self.config.actor_rollout_ref.rollout.get('max_workers', 10),
        )
        generation_manager = OpenManusAgent(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            tool_manager=None, 
            is_validation = True,
        )

        # --- Run Validation Loop --- 
        for batch_dict in self.val_dataloader:
            timing_raw = {}
            test_batch: DataProto = DataProto.from_single_dict(batch_dict)
            
            final_batch_output = None # To store results from rollout/generation
            
            # --- Rollout/Generation --- 
            if is_agentgym_run:
                # print("[Trainer._validate] Running AgentGym/do_search path.") # Debug
                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                if 'idx' not in test_gen_batch.meta_info:
                     batch_size = test_gen_batch.batch['input_ids'].shape[0]
                     test_gen_batch.meta_info['idx'] = torch.arange(batch_size)
                if 'reward_model' not in test_gen_batch.meta_info:
                     batch_size = test_gen_batch.batch['input_ids'].shape[0]
                     test_gen_batch.meta_info['reward_model'] = [{} for _ in range(batch_size)] 

                with _timer('step', timing_raw):
                    final_batch_output = generation_manager.run_llm_loop(gen_batch=test_gen_batch)
                    
            else: # Original Path (Not AgentGym)
                # print("[Trainer._validate] Running original/non-AgentGym path.") # Debug
                # Check reward model style if needed (original logic)
                # if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                #    continue 
                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                final_batch_output = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            
            # --- Score Calculation (using results in final_batch_output) --- 
            if final_batch_output and score_fn:
                current_batch_size = final_batch_output.batch['input_ids'].shape[0]
                env_name = self.config.data.env_name
                
                # Prepare data needed by the score function
                trajectories = final_batch_output.meta_info.get('rollout_trajectory', [[]] * current_batch_size)
                reward_models = final_batch_output.meta_info.get('reward_model', [{}] * current_batch_size)
                env_scores_from_rollout = final_batch_output.meta_info.get('env_scores', None) # Direct scores from env
                
                batch_scores = []
                for i in range(current_batch_size):
                    # --- Call the selected score function --- 
                    # Check if score_fn is the agentgym one by name (or reference)
                    if score_fn_name == 'agentgym': 
                        # Pass trajectory and reward_model info
                        score_kwargs = {
                            'trajectory': trajectories[i] if i < len(trajectories) else [],
                            'reward_model_info': reward_models[i] if i < len(reward_models) else {}
                        }
                        try:
                            score = score_fn(env_name=env_name, **score_kwargs)
                            batch_scores.append(score)
                        except Exception as e:
                            print(f"[Trainer._validate] Error calling score function {score_fn_name} for sample {i}: {e}")
                            batch_scores.append(0.0)
                    elif score_fn == self.val_reward_fn: # Check if it's the RewardManager
                        # RewardManager expects the full batch DataProto
                        # Reconstruct a single item DataProto for RewardManager
                        single_item_batch = test_batch[i].union(final_batch_output[i])
                        try:
                             # RewardManager.__call__ returns a tensor, get the score
                             reward_tensor = score_fn(single_item_batch) 
                             # Assume score is sum or last non-zero value
                             score = reward_tensor.sum().item() # Or other logic based on RewardManager output
                             batch_scores.append(score)
                        except Exception as e:
                            print(f"[Trainer._validate] Error calling val_reward_fn (RewardManager) for sample {i}: {e}")
                            batch_scores.append(0.0)
                    else:
                        # Handle other potential score functions if needed
                        print(f"[Trainer._validate] Warning: Handling for score function {score_fn_name} not implemented. Skipping.")
                        batch_scores.append(0.0)
                        
                all_calculated_scores.extend(batch_scores)
                # print(f"[Trainer._validate] Calculated Batch Scores: {batch_scores}") # Debug
            elif not score_fn:
                 print("[Trainer._validate] No score function available to calculate scores.")
                 
            # Collect timing or other common metrics if needed
            # ...

        # --- Aggregate and Log Metrics --- 
        final_metrics = {}
        if all_calculated_scores:
            mean_score = np.mean(all_calculated_scores)
            log_key = f'val/{score_fn_name}/mean' if score_fn_name else 'val/calculated_score/mean'
            final_metrics[log_key] = mean_score
            print(f"[Trainer._validate] Final Mean Score ({log_key}): {mean_score}")
        else:
             print("[Trainer._validate] No validation scores collected to report.")
             # ... (Fallback logging if needed) ...

        # Aggregate other metrics if collected
        # ... 

        return final_metrics

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
        The training loop of PPO.
        """
        logger = self.logger
        self.global_steps = 0
        
        # Determine if this is an AgentGym run upfront
        self.is_agentgym_run = self.config.data.env_name in KNOWN_AGENTGYM_ENVS
        print(f"[Trainer.fit] Is AgentGym run: {self.is_agentgym_run}")

        # perform validation before training
        if self.val_reward_fn is not None or self.config.algorithm.get('reward_score_fn') == 'agentgym': # Check if validation is possible
             if self.config.trainer.get('val_before_train', True):
                val_metrics = self._validate()
                pprint(f'Initial validation metrics: {val_metrics}')
                logger.log(data=val_metrics, step=self.global_steps)
                if self.config.trainer.get('val_only', False):
                    return
        else:
             print("[Trainer.fit] Skipping initial validation as no val_reward_fn or agentgym score fn is configured.")

        # we start from step 1
        self.global_steps += 1

        # Agent config preparation (Only needed if AgentGym run)
        generation_manager = None
        if self.is_agentgym_run:
            gen_config = AgentConfig(
                 # ... (ensure all necessary AgentGym params are passed from self.config.data)
                max_turns=self.config.max_turns,
                max_start_length=self.config.data.max_start_length,
                max_prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
                max_obs_length=self.config.data.max_obs_length,
                num_gpus=self.config.trainer.n_gpus_per_node, 
                env_name=self.config.data.env_name, 
                env_port=self.config.data.env_port,
                env_server_base=self.config.data.env_server_base,
                env_data_len=self.config.data.get('env_data_len', 200),
                max_workers=self.config.actor_rollout_ref.rollout.get('max_workers', 10),
            )
            generation_manager = OpenManusAgent(
                tokenizer=self.tokenizer,
                actor_rollout_wg=self.actor_rollout_wg,
                config=gen_config,
                tool_manager=None, # Tool manager likely not needed
                # is_validation = False # Default
            )

        # start training loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # Do NOT repeat batch here initially, repeat happens after rollout/generation if needed
                # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                # pop those keys for generation / initial prompt
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                if 'idx' not in gen_batch.meta_info: # Add index if missing
                     gen_batch.meta_info['idx'] = torch.arange(gen_batch.batch['input_ids'].shape[0])
                if 'reward_model' not in gen_batch.meta_info: # Add placeholder
                     gen_batch.meta_info['reward_model'] = [{} for _ in range(gen_batch.batch['input_ids'].shape[0])] 

                ####################
                # Rollout / Generation Step
                ####################
                with _timer('step', timing_raw):
                    if self.is_agentgym_run:
                        # --- AgentGym Path --- 
                        with _timer('gen', timing_raw):
                            final_gen_batch_output = generation_manager.run_llm_loop(gen_batch=gen_batch)
                            
                        # Check if final_gen_batch_output is empty (e.g., error during rollout)
                        if not final_gen_batch_output.batch: 
                             print("[Trainer.fit] Warning: AgentGym rollout returned empty batch. Skipping step.")
                             continue # Skip to next training batch
                             
                        # Add log probs (needed for PPO loss)
                        with torch.no_grad():
                            output_logp = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                            final_gen_batch_output = final_gen_batch_output.union(output_logp)
                            
                        # Merge rollout results back with original batch info (like index)
                        batch = batch.union(final_gen_batch_output)
                        # Assign UID (can use index)
                        if 'index' in batch.non_tensor_batch:
                            batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                        else: # Fallback UID
                             batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

                    else:
                        # --- Original Path --- 
                        # Generate sequences
                        with _timer('gen', timing_raw):
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        # Add log probs
                        with torch.no_grad():
                             output_logp = self.actor_rollout_wg.compute_log_prob(gen_batch_output)
                             gen_batch_output = gen_batch_output.union(output_logp)
                             
                        # Assign UID
                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                        # Merge generated results
                        batch = batch.union(gen_batch_output)

                    # Apply batch repetition if configured (AFTER generation/rollout)
                    if self.config.actor_rollout_ref.rollout.n > 1:
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        
                    ####################
                    # Post-Rollout Processing
                    ####################
                    self._balance_batch(batch, metrics=metrics)
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # Ensure correct dtypes (mostly long, except log_probs)
                    for key in batch.batch.keys():
                        if key != 'old_log_probs' and 'log_prob' not in key and 'rewards' not in key and 'scores' not in key: # Keep floats for rewards/scores/logprobs
                            if torch.is_tensor(batch.batch[key]):
                                 batch.batch[key] = batch.batch[key].long()

                    # --- Compute Ref Log Probs --- 
                    if self.use_reference_policy:
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # --- Compute Critic Values --- 
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                    
                    # --- Compute Rewards & Advantages --- 
                    with _timer('adv', timing_raw):
                        # Use RM model if configured (and not AgentGym? Check logic)
                        if self.use_rm and not self.is_agentgym_run: # Only use RM model if NOT agentgym?
                            reward_tensor_rm = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor_rm)
                            if 'token_level_scores' not in batch.batch:
                                batch.batch['token_level_scores'] = reward_tensor_rm.get('rm_scores', torch.zeros_like(batch.batch['input_ids'], dtype=torch.float32))
                        
                        # --- Get Token Level Scores/Rewards --- 
                        if self.is_agentgym_run and 'token_level_rewards' in batch.batch:
                            # Trust rewards from agentgym rollout
                            print("[Trainer.fit] Using token_level_rewards directly from AgentGym rollout.")
                            if 'token_level_scores' not in batch.batch: # Need scores for KL penalty
                                batch.batch['token_level_scores'] = batch.batch['token_level_rewards'].clone()
                            # token_level_rewards is already set 
                            
                        elif not self.is_agentgym_run and self.reward_fn: 
                            # Use RewardManager for non-agentgym runs
                            print("[Trainer.fit] Using self.reward_fn (RewardManager) to compute scores.")
                            reward_tensor = self.reward_fn(batch)
                            batch.batch['token_level_scores'] = reward_tensor
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores'].clone()
                        else:
                            # Fallback: No rewards available
                            print(f"[Trainer.fit] Warning: No reward source found (AgentGym: {self.is_agentgym_run}, reward_fn: {self.reward_fn is not None}). Using zeros.")
                            if 'token_level_scores' not in batch.batch:
                                 batch.batch['token_level_scores'] = torch.zeros_like(batch.batch['input_ids'], dtype=torch.float32)
                            if 'token_level_rewards' not in batch.batch:
                                 batch.batch['token_level_rewards'] = torch.zeros_like(batch.batch['input_ids'], dtype=torch.float32)

                        # Apply KL penalty (modifies token_level_rewards)
                        if not self.config.actor_rollout_ref.actor.use_kl_loss and self.use_reference_policy:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        
                        # Compute advantages using the final token_level_rewards
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            # Apply state masking only for agentgym runs if configured
                            if self.is_agentgym_run and self.config.actor_rollout_ref.actor.state_masking:
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    # Check if validation is possible
                    can_validate = self.config.algorithm.get('reward_score_fn') or self.val_reward_fn is not None
                    if can_validate and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    # ... (save checkpoint) ...
                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # Log metrics
                logger.log(data=metrics, step=self.global_steps)
                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform final validation if possible
                    if can_validate:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
    
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
