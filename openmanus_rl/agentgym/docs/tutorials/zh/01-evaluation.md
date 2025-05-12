# 测评

本文介绍如何使用 Agent GYM 平台评测模型在 WebShop 任务上的表现。

## 环境配置

首先，请参考 [项目根目录中的 `README.md`](/README.md) 安装 `agentenv` 包。然后，请按照以下流程启动 WebShop 环境服务器。

```bash
# 切换到 agentenv-webshop 文件夹
cd agentenv-webshop

# WebShop 依赖特殊的 Python、PyTorch、Faiss 和 Java 版本，为避免冲突，请创建一个新的 Conda 环境。
conda create -n webshop -f environment.yml

# 创建环境后，请激活这个环境。
conda activate -n webshop

# 然后，运行 setup.sh 配置 WebShop 的运行环境。
bash ./setup.sh
# 这一脚本将按照正确的顺序安装依赖、下载 webshop 的数据并安装 agentenv-webshop 服务器。
# 该过程涉及从 Google Drive 下载文件，请确保网络通畅。
```

## 启动服务

完成环境配置后，你可以从 `webshop` 的 conda 环境中启动 WebShop 环境服务器了。

```bash
# 请确保处于对应的 Conda 环境中
conda activate webshop

# 在对应的端口上启动服务器，端口号请按需修改
webshop --port 36001
```

## 编写评测脚本

前文创建的 `webshop` Conda 环境仅用于启动 WebShop 环境服务器。而其他代码可以在自己常用的环境中运行，也就是您安装 `agentenv` 包的环境。

请参考 `examples/basic/eval_agentlm_webshop.py`。

```python
# Evaluate AgentLM-7B on WebShop task.
# 200 data pieces, max_length=4096, max_rounds=20

import json
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from agentenv.controller import Agent, Evaluator
from agentenv.envs import WebshopTask

MODEL_PATH = "THUDM/agentlm-7b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True).eval()

evaluator = Evaluator(
    Agent(model, tokenizer),
    [
        WebshopTask(
            client_args={
                "env_server_base": "http://127.0.0.1:36001", # 如果你在前文修改了端口号，请在这里一并修改
                "data_len": 200, # data_len 参数目前没有实际用途，将会在后续开发中重构
                "timeout": 300,
            },
            # n_clients 参数保留用于后续批量生成的实现，现阶段留为 1 即可
            n_clients=1,
        )
    ],
)

exps = evaluator.eval(
    generation_config=GenerationConfig(
        max_length=4096,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
    ),
    max_rounds=7,
    idxs=list(range(200)),
)

print("\n\n==== EVALUATION ====\n")
print(f"Score: {exps.score}")
print(f"Success: {exps.success}")

print("\n\n==== EXPERIENCES ====\n")
for idx, exp in enumerate(exps.experiences[:3]):
    print(f"\n\n==== EXP {idx} ====\n")
    for message in exp.conversation:
        if message["from"] == "gpt":
            print(f"\n### Agent\n{message['value']}")
        else:
            print(f"\n### Env\n{message['value']}")

```

这段代码在 id 为 0~199 的任务上评测了 `THUDM/AgentLM-7b` 模型的性能。

## 多卡测评

`agentenv/utils/distributed_eval_task.py` 给出了单机多卡并行推理的脚本。请参考 `agentenv/examples/distributed_eval_scripts/distributed_eval_alfworld.sh` 中的示例来使用该脚本评测模型性能。

该脚本需要准备一个包含数据 id 的 inference_file。其格式类似：

```json
[
    {
        "item_id": "webshop_0",
        "conversations": []
    },
    {
        "item_id": "webshop_1",
        "conversations": []
    },
    {
        "item_id": "webshop_2",
        "conversations": []
    },
    ...
]
```

该脚本除了评测模型能力，还会将模型的交互轨迹（trajectory）保存到文件中。这份数据可以用于后续的自我进化训练。AgentGYM 提供了各个环境的训练集/验证集 id 划分，你可以从 [Hugging Face](https://huggingface.co/AgentGym) 获取我们的数据集。
