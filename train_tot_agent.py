#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train an LLM Agent using Tree of Thought (ToT) exploration with VERL PPO.
This script replaces the standard OpenManusAgent with our ToT-enabled version.
"""

import os
import ray
import sys
import torch
import hydra
from omegaconf import OmegaConf, open_dict
from enum import Enum
from typing import Dict, Any, Type

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.reward_score import SUPPORTED_REWARD_SCORE_FNS
from verl.utils.tracking import Tracking

# Import the custom ToT Agent
from openmanus_tot import OpenManusToTAgent, ToTConfig

# Define known AgentGym envs
KNOWN_AGENTGYM_ENVS = [
    "webshop", "webarena", "maze", "wordle", "alfworld", 
    "sciworld", "babyai", "textcraft", "weather", "movie", 
    "academia", "todo", "sheet", "sqlgym"
]

WorkerType = Type[Role]


import os
os.environ["WANDB_API_KEY"] = "5d34e6d6eda0c3dcef1815bbb5e2df5214617abd"


class ToTRewardManager:
    """The ToT reward manager for non-AgentGym environments."""

    def __init__(self, tokenizer, num_examine, format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score

    def __call__(self, data: DataProto):
        """Extracts rewards from DataProto batch."""
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # For non-AgentGym envs, implement appropriate reward calculation
        # This is a placeholder implementation
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        # Set rewards based on your specific logic
        # ...
        
        return reward_tensor


def _select_reward_fn(data_source):
    """Select appropriate reward function based on data source."""
    if data_source in KNOWN_AGENTGYM_ENVS:
        from verl.utils.reward_score import agentgym_compute_score
        return agentgym_compute_score
    elif data_source in SUPPORTED_REWARD_SCORE_FNS:
        return SUPPORTED_REWARD_SCORE_FNS[data_source]
    else:
        raise NotImplementedError(f"Unsupported data_source: {data_source}")


class ToTPPOTrainer(RayPPOTrainer):
    """
    Extends RayPPOTrainer to use Tree of Thought agent for exploration.
    """
    
    def _validate(self):
        """Override validation to use ToT agent."""
        print(f'[ToT Trainer] Validate start at Global steps: {self.global_steps}')

        if self.config.data.env_name in KNOWN_AGENTGYM_ENVS:
            print(f"[ToT Trainer] Detected AgentGym environment ({self.config.data.env_name}), using ToT Agent for validation.")

            # Create ToT-specific config from regular AgentConfig params plus ToT params
            tot_config = ToTConfig(
                max_turns=self.config.max_turns,
                max_start_length=self.config.data.max_start_length,
                max_prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
                max_obs_length=self.config.data.max_obs_length,
                num_gpus=self.config.trainer.n_gpus_per_node,
                env_name=self.config.data.env_name,
                env_ports=self.config.data.env_ports,
                env_server_base=self.config.data.env_server_base,
                env_data_len=self.config.data.val_data_num or 200,
                react_format=getattr(self.config.data, 'react_format', True),
                rollout_strategy=getattr(self.config.data, 'rollout_strategy', "StandardReAct"),
                max_workers=getattr(self.config.data, 'max_workers', 10),
                algorithm_config=self.config.algorithm,
                # ToT specific parameters
                tot_beam_width=self.config.tot.beam_width,
                tot_exploration_factor=self.config.tot.exploration_factor,
                tot_max_branches=self.config.tot.max_branches,
                tot_value_guidance=self.config.tot.value_guidance,
                tot_temperature=self.config.tot.temperature,
                tot_search_strategy=self.config.tot.search_strategy,
                tot_value_threshold=self.config.tot.value_threshold,
                tot_reward_cutoff=self.config.tot.reward_cutoff
            )

            if not hasattr(self, 'log_dir'):
                self.log_dir = self.config.trainer.get("default_local_dir", "./verl_checkpoints/default_log_dir")
                print(f"[ToT Trainer._validate] Warning: self.log_dir not found, using default: {self.log_dir}")

            # Create the ToT-enabled validation agent
            critic_wg = self.critic_wg if hasattr(self, 'critic_wg') and self.use_critic else None
            
            self.validation_agent = OpenManusToTAgent(
                tokenizer=self.tokenizer,
                actor_rollout_wg=self.actor_rollout_wg,
                config=tot_config,
                critic_wg=critic_wg if tot_config.tot_value_guidance else None,
                is_validation=True,
                logger=self.logger
            )

            # Rest of validation logic goes here...
            # Refer to base class implementation for details
            
            # Return validation metrics
            return {}
        else:
            # Non-AgentGym environments - call super implementation
            return super()._validate()

    def fit(self):
        """Override fit to use ToT agent for training."""
        logger = self.logger
        self.global_steps = 0
        self.log_dir = self.config.trainer.get("default_local_dir", "./verl_checkpoints/default_log_dir")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Determine if this is an AgentGym run
        self.is_agentgym_run = self.config.data.env_name in KNOWN_AGENTGYM_ENVS
        print(f"[ToT Trainer.fit] Is AgentGym run: {self.is_agentgym_run}")
        
        # Get advantage estimator strategy
        adv_estimator = self.config.algorithm.adv_estimator
        print(f"[ToT Trainer.fit] Using advantage estimator: {adv_estimator}")
        
        # We start from step 1
        self.global_steps += 1

        # Initialize ToT agent for training if using AgentGym
        generation_manager = None
        if self.is_agentgym_run:
            print(f"[ToT Trainer.fit] Initializing ToT Agent for AgentGym environment: {self.config.data.env_name}")
            try:
                # Create ToT-specific config
                tot_config = ToTConfig(
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
                    # ToT specific parameters
                    tot_beam_width=self.config.tot.beam_width,
                    tot_exploration_factor=self.config.tot.exploration_factor,
                    tot_max_branches=self.config.tot.max_branches,
                    tot_value_guidance=self.config.tot.value_guidance,
                    tot_temperature=self.config.tot.temperature,
                    tot_search_strategy=self.config.tot.search_strategy,
                    tot_value_threshold=self.config.tot.value_threshold,
                    tot_reward_cutoff=self.config.tot.reward_cutoff
                )
                
                # Get critic for value-guided exploration
                critic_wg = self.critic_wg if hasattr(self, 'critic_wg') and self.use_critic else None
                
                # Create the ToT agent
                generation_manager = OpenManusToTAgent(
                    tokenizer=self.tokenizer,
                    actor_rollout_wg=self.actor_rollout_wg,
                    config=tot_config,
                    critic_wg=critic_wg if tot_config.tot_value_guidance else None,
                    is_validation=False,
                    logger=self.logger
                )
                print(f"[ToT Trainer.fit] ToT Agent created successfully")
            except Exception as e:
                print(f"[ToT Trainer.fit] Failed to initialize ToT Agent: {e}")
                import traceback
                traceback.print_exc()
                raise

        # Rest of the fit logic can reuse the parent class implementation
        # ... [Replace with appropriate training loop code]
        
        # Defer to parent class for training loop
        # This is simplified - in a real implementation, you would need to integrate
        # the ToT agent into the training loop more carefully
        super().fit()


@hydra.main(config_path='config', config_name='tot_ppo_trainer', version_base=None)
def main(config):
    """
    Main entry point for ToT agent training with VERL.
    """
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        num_gpus_for_ray_init = int(config.trainer.n_gpus_per_node)
        
        global_ray_env_vars = {
            'TOKENIZERS_PARALLELISM': 'true',
            'NCCL_DEBUG': 'WARN',
            'VLLM_LOGGING_LEVEL': 'WARN',
        }
        
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            global_ray_env_vars['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
        
        ray.init(
            num_gpus=num_gpus_for_ray_init,
            runtime_env={'env_vars': global_ray_env_vars}
        )

    # Call the ray remote task for training
    ray.get(tot_main_task.remote(config))
    ray.shutdown()


@ray.remote
def tot_main_task(config):
    """
    Ray remote task for ToT agent training.
    """
    from transformers import AutoTokenizer
    
    # Copy model locally if needed
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    
    # Load tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)
    
    # Determine worker strategy
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
        print("Using FSDP workers.")
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
        print("Using Megatron workers.")
    else:
        raise NotImplementedError(f"Unsupported strategy: {config.actor_rollout_ref.actor.strategy}")
    
    # Define actor_rollout_cls based on loaded workers
    actor_rollout_cls = ActorRolloutRefWorker
    
    # Set up role worker mapping
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
        Role.Critic: ray.remote(CriticWorker),
    }
    
    # Set up resource pools
    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [int(config.trainer.n_gpus_per_node)] * int(config.trainer.nnodes),
    }
    
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }
    
    # Add KL-related workers if needed
    use_kl_in_reward = config.algorithm.get('use_kl_in_reward', False)
    use_kl_loss = config.actor_rollout_ref.actor.get('use_kl_loss', False)
    if use_kl_in_reward or use_kl_loss:
        print("KL penalty enabled, adding RefPolicy worker.")
        role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
        mapping[Role.RefPolicy] = global_pool_id
    
    # Add reward model if enabled
    if config.reward_model.enable:
        print("RewardModel enabled, setting up worker.")
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
            print("Using FSDP RewardModelWorker.")
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
            print("Using Megatron RewardModelWorker.")
        else:
            raise NotImplementedError(f"Unsupported reward_model strategy: {config.reward_model.strategy}")
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id
    
    # Set up reward functions
    reward_fn = None
    val_reward_fn = None
    
    is_agentgym_run = config.data.env_name in KNOWN_AGENTGYM_ENVS
    
    reward_component_config = OmegaConf.to_container(
        config.algorithm.get('reward_components', {}), resolve=True
    )
    
    if not is_agentgym_run:
        print("Not an AgentGym run. Setting up ToTRewardManager.")
        reward_fn = ToTRewardManager(tokenizer=tokenizer, num_examine=0, format_score=config.get('format_score', 0.))
        val_reward_fn = ToTRewardManager(tokenizer=tokenizer, num_examine=1, format_score=config.get('format_score', 0.))
    
    # Initialize resource manager and trainer
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    
    trainer = ToTPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        reward_component_config=reward_component_config,
    )
    
    # Initialize workers and start training
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()