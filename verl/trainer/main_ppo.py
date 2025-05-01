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
from verl.utils.reward_score import agentgym
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
from omegaconf import OmegaConf
import ray
import hydra
import os

def _select_rm_score_fn(data_source):
    # 定义已知的AgentGym环境列表
    KNOWN_AGENTGYM_ENVS = [
        "webshop", "webarena", "maze", "wordle", "alfworld", 
        "sciworld", "babyai", "textcraft", "weather", "movie", 
        "academia", "todo", "sheet", "sqlgym"
    ]
    
    # 检查数据源是否为AgentGym环境
    if data_source in KNOWN_AGENTGYM_ENVS:
        from verl.utils.reward_score import agentgym_compute_score
        return agentgym_compute_score
    elif data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # all_scores = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

            reward_tensor[i, valid_response_length - 1] = score
            # all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        
        # print(f"[DEBUG] all_scores: {all_scores}")
        # print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        # print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        # print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        # print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        # print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

        return reward_tensor


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    # Based on verl/verl/trainer/main_ppo.py
    # Ensure necessary env vars for potential conflicts
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,6,7,8")
    if not ray.is_initialized():
        # Use ray_init config if available, otherwise default
        num_cpus = config.get("ray_init", {}).get("num_cpus", None)
        print(f"Initializing Ray... Requesting num_cpus: {num_cpus}")
        ray.init(
            runtime_env={'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true', 
                'NCCL_DEBUG': 'WARN', 
                'VLLM_LOGGING_LEVEL': 'WARN' # Added based on verl example
            }},
            num_cpus=num_cpus, # Pass num_cpus request
            num_gpus=config.trainer.n_gpus_per_node
        )
        print("Ray initialized.")

    # Create and run the TaskRunner actor
    print("Creating TaskRunner actor...")
    runner = TaskRunner.remote()
    print("Calling TaskRunner.run...")
    ray.get(runner.run.remote(config))
    print("TaskRunner finished.")

# Define the TaskRunner Actor based on verl/verl structure
@ray.remote(num_cpus=1) # Request only 1 CPU for the runner itself
class TaskRunner:
    def run(self, config):
        print(f"TaskRunner.run started in PID: {os.getpid()}, Host: {os.uname()[1]}")
        from verl.utils.fs import copy_local_path_from_hdfs # Keep relevant imports inside
        from transformers import AutoTokenizer
        from pprint import pprint

        # print initial config
        print("--- Initial Config (Resolved) ---")
        pprint(OmegaConf.to_container(config, resolve=True))
        print("---------------------------------")
        # Ensure config is fully resolved for subsequent use
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        print(f"Copying model from {config.actor_rollout_ref.model.path}...")
        local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
        print(f"Model copied to local path: {local_path}")

        # instantiate tokenizer
        print(f"Loading tokenizer from {local_path}...")
        from verl.utils import hf_tokenizer
        tokenizer = hf_tokenizer(local_path)
        print("Tokenizer loaded.")

        # define worker classes based on strategy
        print(f"Determining worker strategy: {config.actor_rollout_ref.actor.strategy}")
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup
            actor_rollout_cls = ActorRolloutRefWorker # Assuming non-async for OpenManus
            print("Using FSDP workers.")

        elif config.actor_rollout_ref.actor.strategy == 'megatron':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            ray_worker_group_cls = NVMegatronRayWorkerGroup
            actor_rollout_cls = ActorRolloutRefWorker
            print("Using Megatron workers.")
        else:
            raise NotImplementedError(f"Unsupported strategy: {config.actor_rollout_ref.actor.strategy}")

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role, RayPPOTrainer # Import RayPPOTrainer here

        # Define base role mapping
        role_worker_mapping = {
            # Use the determined actor_rollout_cls
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }
        print(f"Base role mapping created: {list(role_worker_mapping.keys())}")

        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }
        print(f"Resource pool spec: {resource_pool_spec}")
        print(f"Initial mapping: {mapping}")

        # --- Conditionally Add RefPolicy and RewardModel --- 
        # Use reference model if KL penalty is applied in either way
        # Adjusted logic to match verl/verl structure more closely
        use_kl_in_reward = config.algorithm.get('use_kl_in_reward', False) # Default to False if not present
        use_kl_loss = config.actor_rollout_ref.actor.get('use_kl_loss', False) # Default to False if not present
        if use_kl_in_reward or use_kl_loss:
            print("KL penalty enabled, adding RefPolicy worker.")
            # RefPolicy typically uses the same base class as ActorRollout
            role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
            mapping[Role.RefPolicy] = global_pool_id
        else:
            print("KL penalty not enabled, skipping RefPolicy worker.")

        # Setup RewardModel worker if enabled
        if config.reward_model.enable:
            print("RewardModel enabled, setting up worker.")
            # Determine RewardModelWorker class based on strategy (similar to verl/verl)
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

        # --- Reward Function/Composer Setup --- 
        reward_fn = None
        val_reward_fn = None
        reward_component_config = {}

        # Define known AgentGym environments (mirroring agentgym.py or train_ppo.sh)
        KNOWN_AGENTGYM_ENVS = [
            "webshop", "webarena", "maze", "wordle", "alfworld",
            "sciworld", "babyai", "textcraft", "weather", "movie",
            "academia", "todo", "sheet", "sqlgym"
        ]
        is_agentgym_run = config.data.env_name in KNOWN_AGENTGYM_ENVS

        # Load reward components config
        reward_component_config = OmegaConf.to_container(
            config.algorithm.get('reward_components', {}), resolve=True
        )
        print(f"Reward component configuration: {reward_component_config}")

        # Initialize RewardManager only if NOT an AgentGym run AND if RewardManager class exists
        if not is_agentgym_run:
            print("Not an AgentGym run. Attempting to load RewardManager (if defined)...")
            try:
                 # Assuming RewardManager is defined globally or imported elsewhere
                 from verl.trainer.ppo.ray_trainer import RewardManager # Or adjust import path
                 reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, format_score=config.get('format_score', 0.))
                 val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, format_score=config.get('format_score', 0.))
                 print("RewardManager loaded for train and validation.")
            except (ImportError, NameError):
                 print("RewardManager class not found or import failed. Skipping RewardManager setup.")
                 pass # reward_fn and val_reward_fn remain None
        else:
             print("AgentGym run detected. Skipping RewardManager setup (RewardComposer will be used).")

        # --- Initialize Trainer --- 
        print("Initializing ResourcePoolManager...")
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        print("Initializing RayPPOTrainer...")
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            # processor=processor, # Add processor if/when needed for multimodal
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn, # Pass potentially None
            val_reward_fn=val_reward_fn, # Pass potentially None
            reward_component_config=reward_component_config, # Pass the parsed config
        )
        print("RayPPOTrainer initialized. Initializing workers...")
        trainer.init_workers()
        print("Workers initialized. Starting training loop (trainer.fit())...")
        trainer.fit()
        print("Trainer finished.")

if __name__ == '__main__':
    main()
