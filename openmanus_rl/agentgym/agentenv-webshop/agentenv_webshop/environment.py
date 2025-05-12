"""
WebshopEnvServer
"""

from typing import Optional

import gym
from web_agent_site.envs import WebAgentTextEnv


class WebshopEnvServer:
    """
    WebshopEnvServer
    """

    def __init__(self) -> None:
        self._max_id = 0
        self.env = {}
        self.ls = []
        self.sz = 8000
        self.now = -1

    def create(self) -> int:
        env_idx = self._max_id
        import random
        import time

        random.seed(time.time())
        idx = random.randint(0, 48950076)
        print(f"-------Env {idx} created--------")
        if len(self.env) == self.sz:
            self.now = self.now + 1
            if self.now == self.sz:
                self.now = 0
            return self.ls[self.now]

        self.env[idx] = gym.make(
            "WebAgentTextEnv-v0",
            observation_mode="text",
            num_products=1000,
        )
        self.env[idx].reset()
        self._max_id += 1
        self.ls.append(idx)
        return idx

    def step(self, env_idx, action: str):
        return self.env[env_idx].step(action)

    def get_available_actions(self, env_idx):
        """
        Return:
            {'has_search_bar': True, 'clickables': ['search']}
        """
        return self.env[env_idx].get_available_actions()

    def get_image(self, env_idx):
        """
        Return:
            tensor()
        """
        return self.env[env_idx].get_image()

    def get_instruction_text(self, env_idx):
        """
        Return:
            Instruction: Find me slim fit, machine wash women's jumpsuits,
            rompers & overalls with short sleeve, high waist, polyester spandex for
            daily wear with color: green stripe, and size: large, and price lower than
            60.00 dollars
        """
        return self.env[env_idx].get_instruction_text()

    def observation(self, env_idx):
        """
        Return:
            "WebShop [SEP] Instruction: [SEP] Find me slim fit, machine wash women's
            jumpsuits, rompers & overalls with short sleeve, high waist, polyester
            spandex for daily wear with color: green stripe, and size: large, and
            price lower than 60.00 dollars [SEP] Search"
        """
        return self.env[env_idx].observation

    def state(self, env_idx):
        """
        Return
            {
                'url': '',
                'html': '',
                'instruction_text': ""
            }
        """
        return self.env[env_idx].state

    def reset(self, env_idx, session_id: Optional[int]):
        return self.env[env_idx].reset(session=session_id)


webshop_env_server = WebshopEnvServer()
