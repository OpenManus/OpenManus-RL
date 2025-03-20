import gym
import subprocess
import os
import re
import numpy as np


class BaseEnvironment(gym.Env):
    def __init__(self):
        super().__init__()

    def get_info(self):
        pass

    def get_obs(self):
        pass

    def get_goal(self):
        pass

    def get_history(self):
        pass

    def get_action_space(self):
        pass

    def is_done(self):
        pass

    def update(self, action, obs, reward, done, infos):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

    def save_log(self, log_path):
        pass

    @classmethod
    def from_config(cls, config):
        pass