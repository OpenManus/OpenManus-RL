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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
# from verl.utils.reward_score import agentgym # Keep for _select_rm_score_fn if used
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role # Import necessary classes
import re
import numpy as np
from omegaconf import OmegaConf, open_dict # Ensure open_dict is imported
import ray # Import ray at the module level
import hydra
import os
import sys # For version printing
import time # For potential use in main_task

def _select_rm_score_fn(data_source):
    # Define known AgentGym environment list
    KNOWN_AGENTGYM_ENVS = [
        "webshop", "webarena", "maze", "wordle", "alfworld", 
        "sciworld", "babyai", "textcraft", "weather", "movie", 
        "academia", "todo", "sheet", "sqlgym"
    ]
    
    # Check if data source is an AgentGym environment
    if data_source in KNOWN_AGENTGYM_ENVS:
        from verl.utils.reward_score import agentgym_compute_score # Specific import
        return agentgym_compute_score
    elif data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError(f"Unsupported data_source for reward score function: {data_source}")


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
  
        return reward_tensor


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "NotSet") # Default if not set
    print(f"[main] Initial CUDA_VISIBLE_DEVICES: {original_cuda_visible}")
    print(f"[main] Python version: {sys.version}")
    print(f"[main] PyTorch version: {torch.__version__}")
    print(f"[main] Ray version: {ray.__version__}") # ray is imported at module level

    if not ray.is_initialized():
        num_gpus_for_ray_init = int(config.trainer.n_gpus_per_node) # GPUs for Ray to manage on this node
        
        # Prepare env_vars for ray.init and potentially for actors
        # This will be the global runtime_env for the Ray job
        global_ray_env_vars = {
            'TOKENIZERS_PARALLELISM': 'true',
            'NCCL_DEBUG': 'WARN',
            'VLLM_LOGGING_LEVEL': 'WARN',
        }
        if original_cuda_visible != "NotSet" and original_cuda_visible != "":
             global_ray_env_vars['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
        
        print(f"[main] Initializing Ray with num_gpus={num_gpus_for_ray_init} for the node/cluster.")
        print(f"[main] Global Ray runtime_env to be set with env_vars: {global_ray_env_vars}")

        ray.init(
            num_gpus=num_gpus_for_ray_init,
            runtime_env={'env_vars': global_ray_env_vars}
        )
        print(f"[main] Ray initialized. Available resources: {ray.available_resources()}")
        current_context_env_vars = ray.get_runtime_context().runtime_env.get('env_vars', {})
        print(f"[main] Ray driver runtime_context effective env_vars: {current_context_env_vars}")
        print(f"[main] Driver's Ray context CUDA_VISIBLE_DEVICES: {current_context_env_vars.get('CUDA_VISIBLE_DEVICES', 'Not in Ray context')}")


    print("[main] Calling main_task.remote...")
    # Explicitly pass runtime_env to the main_task actor to ensure CUDA_VISIBLE_DEVICES propagation
    # This uses the same env_vars prepared for ray.init, ensuring consistency.
    actor_runtime_env = {'env_vars': global_ray_env_vars if 'global_ray_env_vars' in locals() else env_vars_for_actors_fallback}
    # Fallback in case global_ray_env_vars is not in scope (e.g. if ray was already initialized)
    # A cleaner way would be to define env_vars_for_actors outside the if block or pass original_cuda_visible
    # For now, let's define a fallback based on original_cuda_visible directly accessible here.
    env_vars_for_main_task = {
            'TOKENIZERS_PARALLELISM': 'true',
            'NCCL_DEBUG': 'WARN',
            'VLLM_LOGGING_LEVEL': 'WARN',
    }
    if original_cuda_visible != "NotSet" and original_cuda_visible != "":
        env_vars_for_main_task['CUDA_VISIBLE_DEVICES'] = original_cuda_visible

    print(f"[main] Runtime env to be passed to main_task actor: {{'env_vars': {env_vars_for_main_task}}}")
    ray.get(main_task.options(runtime_env={'env_vars': env_vars_for_main_task}).remote(config))
    print("[main] main_task finished.")
    ray.shutdown()
    print("[main] Ray shutdown.")


@ray.remote
def main_task(config):
    # This function now contains the logic previously in TaskRunner.run
    # import torch, os, sys, time # Already imported at module or outer scope

    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer
    from pprint import pprint

    # OmegaConf.resolve(config) # Ensure config is fully resolved, if not already
    # It's good practice to resolve it early if there are interpolations

    print("--- Initial Config (Resolved by Hydra, further resolve if needed) ---")
    # pprint(OmegaConf.to_container(config, resolve=True)) # Already resolved by Hydra usually
    print(OmegaConf.to_yaml(config)) # Print YAML for readability
    print("---------------------------------")

    print(f"Copying model from {config.actor_rollout_ref.model.path}...")
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    print(f"Model copied to local path: {local_path}")

    print(f"Loading tokenizer from {local_path}...")
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)
    print("Tokenizer loaded.")

    print(f"Determining worker strategy: {config.actor_rollout_ref.actor.strategy}")
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
        # actor_rollout_cls = ActorRolloutRefWorker # Defined below for clarity
        print("Using FSDP workers.")
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
        # actor_rollout_cls = ActorRolloutRefWorker # Defined below for clarity
        print("Using Megatron workers.")
    else:
        raise NotImplementedError(f"Unsupported strategy: {config.actor_rollout_ref.actor.strategy}")
    
    # Define actor_rollout_cls based on the loaded workers
    actor_rollout_cls = ActorRolloutRefWorker 

    # Role Worker Mapping: GPU assignment is handled by PlacementGroups via RayWorkerGroup
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
        Role.Critic: ray.remote(CriticWorker),
    }
    print(f"Base role mapping created: {list(role_worker_mapping.keys())}")

    global_pool_id = 'global_pool'
    # This is the number of GPUs per node that the trainer config expects.
    # It should align with how Ray was initialized (num_gpus for the node).
    resource_pool_spec = {
        global_pool_id: [int(config.trainer.n_gpus_per_node)] * int(config.trainer.nnodes),
    }
    
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }

    use_kl_in_reward = config.algorithm.get('use_kl_in_reward', False)
    use_kl_loss = config.actor_rollout_ref.actor.get('use_kl_loss', False)
    if use_kl_in_reward or use_kl_loss:
        print("KL penalty enabled, adding RefPolicy worker.")
        role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
        mapping[Role.RefPolicy] = global_pool_id
    else:
        print("KL penalty not enabled, skipping RefPolicy worker.")

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
    else:
        print("RewardModel not enabled, skipping worker setup.")
    
    print(f"Final role_worker_mapping: {list(role_worker_mapping.keys())}")
    print(f"Resource pool spec: {resource_pool_spec}")
    print(f"Final mapping: {mapping}")
    
    # --- Reward Function/Composer Setup ---
    reward_fn = None
    val_reward_fn = None
    
    KNOWN_AGENTGYM_ENVS = [
        "webshop", "webarena", "maze", "wordle", "alfworld",
        "sciworld", "babyai", "textcraft", "weather", "movie",
        "academia", "todo", "sheet", "sqlgym"
    ]
    is_agentgym_run = config.data.env_name in KNOWN_AGENTGYM_ENVS
    print(f"Environment: {config.data.env_name}, AgentGym run: {is_agentgym_run}")

    reward_component_config = OmegaConf.to_container(
        config.algorithm.get('reward_components', {}), resolve=True
    )
    print(f"Reward component configuration: {reward_component_config}")

    if not is_agentgym_run:
        print("Not an AgentGym run. Setting up RewardManager (if defined globally).")
        # Assuming RewardManager class is defined (as it is in this file)
        reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, format_score=config.get('format_score', 0.))
        val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, format_score=config.get('format_score', 0.))
        print("RewardManager loaded for train and validation.")
    else:
        print("AgentGym run detected. Skipping RewardManager setup (AgentGym internal rewards or RewardComposer will be used).")

    print("Initializing ResourcePoolManager...")
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    print("ResourcePoolManager initialized.")
    
    print("Initializing RayPPOTrainer...")
    # Fallback logic removed for now to simplify, can be added back if robustly tested
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        reward_component_config=reward_component_config,
    )
    print("RayPPOTrainer initialized.")

    # Timeout protection removed for now to simplify, can be added back
    print("Initializing workers (trainer.init_workers())...")
    trainer.init_workers() # This is where RayWorkerGroup creates actors within PlacementGroups
    print("Workers initialized successfully.")
    
    print("Starting training loop (trainer.fit())...")
    trainer.fit()
    print("Training loop (trainer.fit()) finished successfully.")

    # Exception handling for main_task, ensuring Ray knows if it fails
    # The @ray.remote decorator handles propagating exceptions from the remote task.
    # If an unhandled exception occurs here, Ray.get() in main() will raise it.

if __name__ == '__main__':
    main()
