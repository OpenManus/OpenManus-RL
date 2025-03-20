# GAIA环境模块

这个模块实现了GAIA数据集的环境服务器和客户端，用于在AgentGym框架中支持GAIA任务。

## 功能特点

- 支持GAIA数据集的加载和处理
- 提供HTTP API服务端，支持并发操作
- 提供客户端接口，集成到AgentGym框架

## 安装方法

```bash
# 从当前目录安装
pip install -e .
```

## 使用方法

### 启动环境服务器

```python
from agentenv_gaia import launch_server

# 启动GAIA环境服务器
launch_server(host="0.0.0.0", port=8000)
```

### 客户端使用

```python
from agentenv_gaia import GaiaEnvClient, GaiaTask
from agentenv.controller.agent import Agent
from agentenv.controller.utils import Evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer

# 初始化LLM模型
model_path = "your_model_path"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
agent = Agent(model, tokenizer)

# 初始化GAIA任务
client_args = {
    "server_url": "http://localhost:8000",
    "data_dir": "path/to/data",
    "level": "level1",
    "dataset": "validation"
}
task = GaiaTask(client_args)

# 创建评估器
evaluator = Evaluator(agent, [task])

# 评估LLM在GAIA任务上的性能
generation_config = {"max_new_tokens": 512, "do_sample": True, "temperature": 0.7}
results = evaluator.eval(generation_config, max_rounds=10, idxs=[0, 1, 2])
print(results)
```

## 环境参数说明

GAIA环境支持以下参数：

- `data_dir`: GAIA数据集目录路径
- `level`: 难度级别，可选值为"level1"、"level2"等
- `dataset`: 数据集类型，可选值为"train"、"validation"、"test"
- `tool_list`: 可用工具列表，默认为None（使用所有可用工具）
