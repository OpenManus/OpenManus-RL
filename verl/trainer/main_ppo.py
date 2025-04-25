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


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # --- Conditionally Define Reward Functions ---
    reward_fn = None
    val_reward_fn = None

    # Define known AgentGym environments (mirroring agentgym.py or train_ppo.sh)
    KNOWN_AGENTGYM_ENVS = [
        "webshop", "webarena", "maze", "wordle", "alfworld",
        "sciworld", "babyai", "textcraft", "weather", "movie",
        "academia", "todo", "sheet", "sqlgym"
    ]
    is_agentgym_run = config.data.env_name in KNOWN_AGENTGYM_ENVS

    # --- Get Reward Component Configuration --- 
    # Safely get the reward components config, default to empty dict if not present
    reward_component_config = OmegaConf.to_container(
        config.algorithm.get('reward_components', {}), # Use .get for safety
        resolve=True
    )
    print(f"[main_task] Reward component configuration: {reward_component_config}")

    # --- Initialize RewardManager (if needed, e.g., for non-AgentGym) ---
    # Decide if RewardManager is still needed. With RewardComposer, its role might change
    # or become obsolete if all scoring is handled by components.
    # For now, let's assume it might still be used for specific datasets.
    if not is_agentgym_run:
        print("[main_task] Initializing RewardManager for non-AgentGym run.")
        try:
            # Pass reward_component_config to RewardManager if it needs it
            reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, format_score=config.get('format_score', 0.))
            val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, format_score=config.get('format_score', 0.))
        except NameError:
             print("[main_task] Error: RewardManager class not defined. Skipping.")
             pass # reward_fn and val_reward_fn remain None

    # --- Setup RewardModel worker (if needed) ---
    # This logic remains largely the same, depends on reward_model.enable config
    if config.reward_model.enable:
         print("[main_task] Setting up RewardModel worker.")
         # ... (rest of the RewardModel setup logic) ...
         # if config.reward_model.strategy == 'fsdp':
         #    from verl.workers.fsdp_workers import RewardModelWorker
         # ... etc ...
         # role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
         # mapping[Role.RewardModel] = global_pool_id
    else:
        print(f"[main_task] AgentGym run ({config.data.env_name}) or RewardModel not enabled. Skipping RewardManager/RewardModel worker setup.")

    # --- Initialize Trainer --- 
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn, # Pass potentially None
                            val_reward_fn=val_reward_fn, # Pass potentially None
                            reward_component_config=reward_component_config, # Pass the parsed config
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
