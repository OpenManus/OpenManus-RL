import os
import sys
from typing import Any, Dict, Optional

import requests
from requests.exceptions import RequestException

from agentenv.controller.env import BaseEnvClient, StepOutput
from agentenv.controller.task import BaseTask, ConversationMessage


class GaiaEnvClient(BaseEnvClient):
    """GAIA环境客户端，用于与GAIA环境服务器交互"""

    conversation_start = ("我是GAIA环境助手，请告诉我你想要做什么？",)

    def __init__(
        self,
        server_url: str,
        data_dir: str = "agentenv_gaia/data/",
        level: str = "level1",
        dataset: str = "validation",
        tool_list: Optional[list] = None,
    ):
        """初始化GAIA环境客户端

        Args:
            server_url: 环境服务器URL
            data_dir: 数据目录
            level: 难度级别
            dataset: 数据集类型
            tool_list: 可用工具列表
        """
        super().__init__()
        self.server_url = server_url
        self.data_dir = data_dir
        self.level = level
        self.dataset = dataset
        self.tool_list = tool_list
        self.env_id = None
        self.current_state = None
        self.session = requests.Session()  # 使用会话提高HTTP连接效率
        self._create_env()

    def __len__(self) -> int:
        """返回环境的总数据集大小"""
        # 目前需要手动指定或从服务器获取，这里暂时返回一个固定值
        # 实际应用中应当从环境获取真实大小
        return 100

    def _create_env(self):
        """创建环境实例"""
        params = {
            "data_dir": self.data_dir,
            "level": self.level,
            "dataset": self.dataset,
            "tool_list": self.tool_list,
        }
        response = self.session.post(f"{self.server_url}/createEnv", json=params)
        response.raise_for_status()
        self.env_id = response.json()["env_id"]

    def observe(self) -> str:
        """获取当前环境状态的观察"""
        if self.env_id is None:
            raise ValueError("环境尚未初始化")

        response = self.session.get(
            f"{self.server_url}/observation", params={"env_id": self.env_id}
        )
        response.raise_for_status()
        self.current_state = response.json()["observation"]
        return self.current_state

    def get_available_actions(self) -> Dict[str, Any]:
        """获取可用动作列表"""
        if self.env_id is None:
            raise ValueError("环境尚未初始化")

        response = self.session.get(
            f"{self.server_url}/available_actions", params={"env_id": self.env_id}
        )
        response.raise_for_status()
        return response.json()

    def step(self, action: str) -> StepOutput:
        """执行动作并返回结果"""
        if self.env_id is None:
            raise ValueError("环境尚未初始化")

        payload = {"env_id": self.env_id, "action": action}
        response = self.session.post(f"{self.server_url}/step", json=payload)
        response.raise_for_status()
        result = response.json()

        return StepOutput(
            observation=result["observation"],
            reward=result.get("reward", 0),
            done=result.get("done", False),
            info=result.get("info", {}),
        )

    def reset(self, idx: int) -> None:
        """重置环境到指定索引的任务"""
        if self.env_id is None:
            self._create_env()

        payload = {"env_id": self.env_id, "idx": idx}
        response = self.session.post(f"{self.server_url}/reset", json=payload)
        response.raise_for_status()
        result = response.json()
        self.current_state = result["observation"]


class GaiaTask(BaseTask):
    """GAIA任务类，连接GAIA环境客户端"""

    env_client_cls = GaiaEnvClient
    env_name = "gaia"

    def __init__(self, client_args, n_clients=1):
        """初始化GAIA任务

        Args:
            client_args: 客户端参数
            n_clients: 客户端数量
        """
        super().__init__(client_args, n_clients)

    def _tokenize_conversation_one(self, conversation):
        """处理GAIA特定的对话格式

        Args:
            conversation: 对话历史

        Returns:
            处理后的对话标记
        """
        # 如果需要特殊处理GAIA对话格式，可以在这里实现
        # 默认使用父类的标记化方法
        return super()._tokenize_conversation_one(conversation)


if __name__ == "__main__":
    client = GaiaEnvClient(server_url="http://localhost:8000")
    print(client.observe())
