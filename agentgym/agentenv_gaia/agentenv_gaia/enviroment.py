import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from gymnasium import Env

from agentenv_gaia.utils.load_data import load_gaia_data
from agentenv_gaia.utils.openmanus_imports import get_tool_collection


@dataclass
class StepOutput:
    observation: str
    reward: float
    done: bool
    info: Dict[str, Any] = None


class GaiaEnv(Env):
    """GAIA环境类，实现GAIA基准测试的环境逻辑"""

    def __init__(
        self,
        data_dir: str = "data/",
        level: str = "level1",
        dataset: str = "validation",
        tool_list: Optional[list] = None,
    ):
        """初始化GAIA环境

        Args:
            data_dir: 数据目录路径
            level: GAIA级别，可选"level1"、"level2"等
            dataset: 数据集类型，可选"validation"、"test"等
            tool_list: 需要使用的工具列表，默认为None
        """
        super().__init__()

        self.data_dir = data_dir
        self.level = level
        self.dataset_type = dataset
        self.tool_list = tool_list

        # 加载数据集
        self.dataset = self._load_dataset()

        # 初始化工具集合
        self.tools = self._init_tools()

        # 当前状态信息
        self.current_idx = None
        self.current_state = None
        self.available_actions = []
        self.history = []

    def _load_dataset(self):
        """加载GAIA数据集"""
        return load_gaia_data(
            data_dir=self.data_dir, level=self.level, dataset=self.dataset_type
        )

    def _init_tools(self):
        """初始化工具集"""
        return get_tool_collection(self.tool_list)

    def get_observation(self):
        """获取当前环境状态的观察"""
        return {
            "observation": self.current_state,
            "available_actions": self.available_actions,
        }

    def get_available_actions(self):
        """获取可用动作列表"""
        return {"available_actions": self.available_actions}

    def step(self, action):
        """执行动作并返回结果

        Args:
            action: 智能体执行的动作

        Returns:
            tuple: (observation, reward, done, truncated, info)，符合gymnasium.Env的接口
        """
        # 处理动作
        if isinstance(action, str):
            # 分析动作内容，确定使用哪个工具
            tool_name, tool_input = self._parse_action(action)

            # 执行工具调用
            if tool_name in self.tools.tool_map:
                tool_result = self.tools.execute(name=tool_name, tool_input=tool_input)
                observation = str(tool_result)
            else:
                observation = f"工具 {tool_name} 不存在或不可用。"
        else:
            observation = "不支持的动作格式。"

        # 更新环境状态
        self.current_state = observation

        # 计算奖励和是否完成
        reward = self._calculate_reward(action)
        done = self._check_done()

        # 记录历史
        self.history.append({"action": action, "observation": observation})

        # 返回结果（符合gymnasium.Env的接口）
        return observation, reward, done, False, {"action_count": len(self.history)}

    def _parse_action(self, action: str) -> Tuple[str, Dict[str, Any]]:
        """解析动作，提取工具名称和参数

        简单实现，实际使用中可能需要更复杂的解析逻辑

        Args:
            action: 动作字符串

        Returns:
            tuple: (工具名称, 工具参数字典)
        """
        # 这里是一个简单的实现，假设动作格式为"工具名: 参数1=值1, 参数2=值2"
        try:
            tools = json.loads(action)
            return tools["tool_name"], tools["tool_input"]
        except Exception:
            return "invalid_tool", {}

    def reset(self, *, seed=None, options=None):
        """重置环境到初始状态

        Args:
            seed: 随机种子
            options: 可选参数，可以包含指定的任务索引 options={'idx': task_idx}

        Returns:
            tuple: (observation, info)，符合gymnasium.Env的接口
        """
        super().reset(seed=seed)

        # 确定任务索引
        idx = 0
        if options and "idx" in options:
            idx = options["idx"]

        self.current_idx = idx

        if idx < len(self.dataset):
            # 获取任务数据
            task_data = self.dataset.iloc[idx]

            # 设置初始状态
            self.current_state = task_data["question"]

            # 设置可用动作
            self.available_actions = self._get_initial_actions(task_data)

            # 清空历史
            self.history = []

            return self.current_state, {"task_data": task_data}
        else:
            raise ValueError(f"索引 {idx} 超出数据集范围")

    def _get_initial_actions(self, task_data):
        """获取初始可用动作列表

        Args:
            task_data: 任务数据

        Returns:
            list: 可用动作列表
        """
        # 根据任务类型返回不同的可用工具列表
        task_type = task_data.get("task", "")

        # 返回所有工具作为初始可用工具
        return [tool.name for tool in self.tools]

    def _calculate_reward(self, action):
        """计算执行动作后的奖励

        Args:
            action: 执行的动作

        Returns:
            float: 奖励值
        """
        # 在这个简单实现中，所有动作的奖励都是0，只有最终状态才有奖励
        return 0.0

    def _check_done(self):
        """检查任务是否完成

        Returns:
            bool: 任务是否完成
        """
        # 简单实现：达到一定步骤数或发现特定关键词表示任务完成
        if len(self.history) >= 10:  # 最多10步
            return True

        # 检查最后一个观察是否包含"final_answer"或"答案"等关键词
        if self.history and isinstance(self.history[-1].get("action", ""), str):
            action = self.history[-1]["action"].lower()
            if "final_answer" in action or "答案" in action:
                return True

        return False

    def render(self):
        """渲染环境（可选实现）"""
        # GAIA环境是纯文本环境，可以简单地打印当前状态
        return self.current_state

    def close(self):
        """关闭环境，释放资源"""
        # 清理任何需要释放的资源
        pass
