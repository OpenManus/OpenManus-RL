# Evaluation

This article introduces how to evaluate LLM-based agents on WebShop task via Agent GYM.

## Setup the Environment

Firstly, please install `agentenv` package. You can refer to the [`README.md`](/README.md) in the root path of this project. Then, you can launch the WebShop environment server through the following instructions.


```bash
# Change to the agentenv-webshop directory
cd agentenv-webshop

# WebShop depends on specific versions of Python, PyTorch, Faiss and Java.
# To avoid any conflict, please create a new Conda virtual environment.
conda create -n webshop -f environment.yml

# After creating, activate it.
conda activate -n webshop

# Then, run `setup.sh` to setup the environment.
bash ./setup.sh
# The script will resolve the dependencies in correct order, download the dataset and install the agentenv-webshop server.
# This script will download data from Google Drive. Please make sure that the network is available.
```

## Launch the Server

After configuration of the environment, you can launch the WebShop environment server fron the `webshop` Conda Environment.

```bash
# Make sure the webshop Conda environment is activated.
conda activate webshop

# Launch the server on port 36001 or other ports you like.
webshop --port 36001
```

## Write the Evaluation Script

The `webshop` Conda should only be used to launch the WebShop environment server. Other code should be executed in your own Conda environment, where the `agentenv` package have been installed.

Please refer to `examples/basic/eval_agentlm_webshop.py`.

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
                "env_server_base": "http://127.0.0.1:36001", # If you have modified the port, modify it here.
                "data_len": 200, # Currently, the data_len argument is of no use. It will be removed in future versions.
                "timeout": 300,
            },
            # The n_clients argument is reserved for subsequent implementations of batch generation. Please leave it at 1.
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

This script evaluate the `THUDM/AgentLM-7b` model on tasks with id from 0 to 199.

## Evaluation on Multi-GPU

`agentenv/utils/distributed_eval_task.py` evaluates the models on multi-gpu. Please refer to `agentenv/examples/distributed_eval_scripts/distributed_eval_alfworld.sh`.


To use the script, you need to prepare an inference_file with the ids of the data. This is the required format:

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

While evaluating the model, this script also saves the trajectories to a file, which can be used for further training or analysis. AgentGYM gives the split of the train and test set. You can get the dataset from [Hugging Face](https://huggingface.co/AgentGym).
