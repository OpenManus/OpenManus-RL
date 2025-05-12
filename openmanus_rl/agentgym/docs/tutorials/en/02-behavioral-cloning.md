# Behavioral Cloning

To make the models 

In order to adapt the model to Agent-related tasks, this article describes how behavioral cloning of the model can be performed using the AgentGYM framework.

AgentGYM is a platform for general-capable agents. Currently, we support the training of models using the Llama 2 Chat template.

## Prepare the Dataset

AgentGYM gives the split of the train and test set. You can get the dataset from [Hugging Face](https://huggingface.co/AgentGym). This is the required format for the dataset file:

```json
[
    {
        "item_id": "webshop_0",
        "conversations": [
            {
                "from": "human",
                "loss": null,
                "value": "Description of the task"
            },
            {
                "from": "gpt",
                "loss": false,
                "value": "Ok."
            },
            {
                "from": "human",
                "loss": null,
                "value": "Observation from the environment."
            },
            {
                "from": "gpt",
                "loss": true,
                "value": "Thought: ...\n\nAction: ..."
            },
            ...
        ],
    },
    ...
]
```

## Train

This section introduces how to perform behavioral cloning through `agentenv.trainer.bc_trainer.BCTrainer`. Please refer to `agentenv/agentenv/trainer/bc_trainer.py`.

```python
from dataclasses import dataclass, field

import torch
import transformers
from agentenv.controller import Agent
from agentenv.envs import (
    AlfWorldTask,
    SciworldTask,
    TextCraftTask,
    WebarenaTask,
    WebshopTask,
)
from agentenv.trainer.bc_trainer import BCTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TrainingArguments:
    train_file: str = field(metadata={"help": "Training dataset."})
    inference_file: str = field(
        default="./data/train/webshop_train.json", metadata={"help": "Inference dataset."}
    )
    test_file: str = field(default="./data/test/webshop_test.json", metadata={"help": "Test dataset."})
    # model path
    model_train_path: str = field(
        default="/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-chat-hf",
        metadata={"help": "Path of initial train model"},
    )
    model_save_path: str = field(
        default="outputs/model",
        metadata={"help": "Directory to save the trained model."},
    )
    task_name: str = field(
        default="webshop", metadata={"help": "Task name for evaluation"}
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for training."},
    )
    eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size for evaluation."}
    )
    n_epochs: int = field(default=40)
    num_workers: int = field(
        default=8, metadata={"help": "Number of subprocesses to use for data loading."}
    )
    learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate."})
    weight_decay: float = field(
        default=1e-6, metadata={"help": "Weight decay for regularization."}
    )
    warmup_step: int = field(
        default=0,
        metadata={"help": "Number of warmup steps for learning rate scheduling."},
    )
    clip_grad_norm: float = field(
        default=1, metadata={"help": "Gradient clipping threshold."}
    )
    gradient_accumulation_steps: int = field(default=1)
    evaluating_epoch_freq: int = field(default=1)
    logging_epoch_freq: int = field(default=1)
    saving_epoch_freq: int = field(default=1)
    logging_step_freq: int = field(default=None)
    seed: int = field(default=42)
    max_input_length: int = field(default=700)

    # environment
    max_round: int = field(
        default=6,
        metadata={"help": "Interaction rounds between agents and environment"},
    )

    # wandb stuff
    wandb_log: bool = field(default=False)
    wandb_project: str = field(default="AgentGym_behavioral_clone")
    wandb_run_name: str = field(default="behavioral_clone")

    # environment parameters
    env_server_base: str = field(default=None)
    data_len: int = field(default=200)
    timeout: int = field(default=2400)


def main():
    parser = transformers.HfArgumentParser(TrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(args.model_train_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_train_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    )
    model.gradient_checkpointing_enable()

    # task_name - task dict
    task_classes = {
        "webshop": WebshopTask,
        "alfworld": AlfWorldTask,
        "sciworld": SciworldTask,
        "textcraft": TextCraftTask,
        "webarena": WebarenaTask,
    }

    # select task according to the name
    task_class = task_classes.get(args.task_name.lower(), None)
    if task_class is None:
        raise ValueError(f"Unsupported task name: {args.task_name}")

    # set environment parameters
    env_args = {
        "env_server_base": args.env_server_base,
        "data_len": args.data_len,
        "timeout": args.timeout,
    }

    trainer = BCTrainer(
        Agent(model, tokenizer),
        [task_class(client_args=env_args, n_clients=1)],
        args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
```

To launch the training using `accelerate`, please refer to `agentenv/examples/behavioral_cloning/train_behavioral_clone_multi_task.sh`.

```bash
exp_name="behavioral_clone_eval_alfworld_mix_24110"

n_epochs='1'

# accelerator config
num_processes='8'
main_process_port='8897'
config_file=""

# training arguments
train_file='PATH/TO/mix_data_24110.json'
inference_file='PATH/TO/webshop_test.json'
model_train_path="meta-llama/Llama-2-7b-chat-hf"
model_save_path="outputs/${exp_name}/"

batch_size="2"
eval_batch_size="1"
gradient_accumulation_steps="2"
max_input_length="4096"
num_workers="8"
learning_rate="1e-5"
weight_decay="0"
warmup_step="-100"
clip_grad_norm="1"
seed="42"

logging_epoch_freq="1"
evaluating_epoch_freq="100"
saving_epoch_freq="1"
logging_step_freq="5"

# wandb config
wandb_log="True"
wandb_project="agentenv"
wandb_run_name="${exp_name}"

# environment parameters
data_len="200"
timeout="2400"

# eval
task_list=("webshop" "alfworld" "textcraft" "sciworld")
# eval parameters
test_file_list=("PATH/TO/webshop_test.json" "PATH/TO/alfworld_test.json" "PATH/TO/textcraft_test.json" "PATH/TO/sciworld_test_small.json")
do_sample="False"
temperature="1.0"
sample_num="2"
max_round_list=("6" "30" "20" "30")
env_server_base_list=("http://127.0.0.1:36004" "http://127.0.0.1:36002" "http://127.0.0.1:36008" "http://127.0.0.1:36010")

mkdir -p "${model_save_path}"
# step1: train
accelerate launch \
        --config_file "${config_file}" \
        --num_processes=${num_processes} \
        --main_process_port=${main_process_port} \
    train_behavioral_clone.py \
        --train_file "${train_file}" \
        --model_train_path "${model_train_path}" \
        --model_save_path "${model_save_path}" \
        --task_name "${task_list[1]}" \
        --batch_size "${batch_size}" \
        --eval_batch_size "${eval_batch_size}" \
        --n_epochs "${n_epochs}" \
        --num_workers "${num_workers}" \
        --learning_rate "${learning_rate}" \
        --weight_decay "${weight_decay}" \
        --warmup_step "${warmup_step}" \
        --clip_grad_norm "${clip_grad_norm}" \
        --evaluating_epoch_freq "${evaluating_epoch_freq}" \
        --logging_epoch_freq "${logging_epoch_freq}" \
        --saving_epoch_freq "${saving_epoch_freq}" \
        --logging_step_freq "${logging_step_freq}" \
        --seed "${seed}" \
        --max_input_length "${max_input_length}" \
        --max_round "${max_round_list[1]}" \
        --gradient_accumulation_steps "${gradient_accumulation_steps}" \
        --wandb_log "${wandb_log}" \
        --wandb_project "${wandb_project}" \
        --wandb_run_name "${wandb_run_name}" \
        --env_server_base "${env_server_base_list[1]}" \
        --data_len "${data_len}" \
        --timeout "${timeout}"
                
# step2: eval on test dataset
for index in "${!task_list[@]}";
do
    cur_task=${task_list[$index]}
    cur_test_file="${test_file_list[$index]}"
    cur_max_round=${max_round_list[$index]}
    cur_env_server_base=${env_server_base_list[$index]}
    cur_eval_output_file="${model_save_path}/eval_${cur_task}.jsonl"

    accelerate launch \
            --config_file "${config_file}" \
            --num_processes=${num_processes} \
            --main_process_port=${main_process_port} \
        ../../utils/distributed_eval_task.py \
            --model_path "${model_save_path}/train_epoch_${n_epochs}" \
            --output_file "${cur_eval_output_file}" \
            --inference_file "${cur_test_file}" \
            --task_name "${cur_task}" \
            --eval_batch_size "${eval_batch_size}" \
            --num_workers "${num_workers}" \
            --seed "${seed}" \
            --do_sample "${do_sample}" \
            --max_round "${cur_max_round}" \
            --env_server_base "${cur_env_server_base}" \
            --data_len "${data_len}" \
            --timeout "${timeout}"
done
```
