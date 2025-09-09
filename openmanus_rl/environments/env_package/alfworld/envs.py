# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
# Try to import torchvision transforms lazily and robustly.
# On ROCm builds, importing torchvision may fail due to missing compiled ops
# (e.g., torchvision::nms). We only need basic ToTensor for multi-modal Thor env.
try:
    import torchvision.transforms as T  # type: ignore
except Exception:  # Fallback lightweight transforms to avoid hard dependency
    class _ToTensor:
        def __call__(self, img):
            import numpy as _np
            import torch as _torch
            arr = _np.array(img)
            if arr.ndim == 2:  # grayscale -> HxW -> 1xHxW
                arr = arr[:, :, None]
            t = _torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            return t

    class _Compose:
        def __init__(self, ts):
            self.ts = ts
        def __call__(self, x):
            for f in self.ts:
                x = f(x)
            return x

    class _T:
        Compose = _Compose
        ToTensor = _ToTensor

    T = _T()  # minimal drop-in replacement
import ray

from openmanus_rl.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

@ray.remote(num_cpus=0.2)
class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env=None, env_type=None, single_gamefile=None, is_train=True, eval_dataset='eval_in_distribution'):
        if base_env is not None:
            # Legacy path: share a base_env and instantiate sub-env from it
            self.env = base_env.init_env(batch_size=1)
        else:
            # Unique path: each worker binds to exactly one gamefile
            assert env_type is not None, "env_type is required when base_env is None"
            BaseEnvCls = get_environment(env_type)
            game_files_override = [single_gamefile] if single_gamefile is not None else None
            be = BaseEnvCls(config, train_eval='train' if is_train else eval_dataset, game_files=game_files_override)
            self.env = be.init_env(batch_size=1)
        self.env.seed(seed)
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed=0, env_num=1, group_n=1, is_train=True, env_kwargs={}, game_files=None):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        self.workers = []
        if game_files is not None:
            assert len(game_files) == self.num_processes, "game_files length must equal env_num * group_n"
            for i in range(self.num_processes):
                single = game_files[i]
                worker = AlfworldWorker.remote(
                    config,
                    seed + (i // self.group_n),
                    base_env=None,
                    env_type=env_type,
                    single_gamefile=single,
                    is_train=is_train,
                    eval_dataset=eval_dataset,
                )
                self.workers.append(worker)
        else:
            base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
            for i in range(self.num_processes):
                worker = AlfworldWorker.remote(config, seed + (i // self.group_n), base_env=base_env)
                self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Collect all image observations from workers.
        """
        # Send getobs commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        # Collect and stack results
        results = ray.get(futures)
        image_obs_list = torch.cat(results, dim=0)
        return image_obs_list

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """Clean up Ray actors"""
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, is_train=True, env_kwargs={}, game_files=None):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, is_train, env_kwargs, game_files)