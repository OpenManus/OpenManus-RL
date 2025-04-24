import pytest
import os
import sys
from pathlib import Path

# Import target functions/classes
from openmanus_rl.llm_agent.openmanus import OpenManusAgent, create_react_prompt

# 将 OpenManus-RL 路径加入 sys.path，确保能够找到 openmanus_rl 包
PROJECT_ROOT = Path(__file__).resolve().parent.parent / "OpenManus-RL"
if PROJECT_ROOT.exists() and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DummyToolManager:
    """简易 ToolManager，仅返回固定的提示指令。"""

    def get_prompt_instructions(self):
        return "These are tool instructions."


class MinimalAgent(OpenManusAgent):
    """覆盖 __init__ 以绕过复杂依赖，仅用于测试无状态方法。"""

    def __init__(self):  # pylint: disable=super-init-not-called
        pass


# ------------------------------
# postprocess_predictions 测试
# ------------------------------

def test_postprocess_action():
    agent = MinimalAgent()
    predictions = ["<action>CLICK_BUTTON</action>"]
    actions, contents = agent.postprocess_predictions(predictions)
    assert actions == ["action"]
    assert contents == ["CLICK_BUTTON"]


def test_postprocess_response():
    agent = MinimalAgent()
    predictions = ["<response>Hello World</response>"]
    actions, contents = agent.postprocess_predictions(predictions)
    assert actions == ["response"]
    assert contents == ["Hello World"]


def test_postprocess_no_tag():
    agent = MinimalAgent()
    predictions = ["Hello there"]
    actions, contents = agent.postprocess_predictions(predictions)
    assert actions == [None]
    assert contents == [""]


def test_postprocess_invalid_type():
    """非字符串输入应抛出 ValueError。"""
    agent = MinimalAgent()
    with pytest.raises(ValueError):
        agent.postprocess_predictions([123])


def test_postprocess_multiple_tags():
    """确保仅解析第一个 <action> 标签内容。"""
    agent = MinimalAgent()
    predictions = ["<action>foo</action><action>bar</action>"]
    actions, contents = agent.postprocess_predictions(predictions)
    assert actions == ["action"]
    assert contents == ["foo"]


# ------------------------------
# create_react_prompt 测试
# ------------------------------

def test_create_react_prompt_contains_sections():
    task_description = "Navigate to the red door."
    tool_manager = DummyToolManager()
    prompt = create_react_prompt(task_description, tool_manager)

    # Prompt 应包含任务描述、工具指令和固定结尾
    assert task_description in prompt
    assert tool_manager.get_prompt_instructions() in prompt
    assert "Let's solve this step by step." in prompt 