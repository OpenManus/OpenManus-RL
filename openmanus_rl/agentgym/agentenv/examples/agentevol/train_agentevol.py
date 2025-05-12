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
from agentenv.trainer.agentevol_trainer import AgentEvolTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TrainingArguments:
    # data path
    train_file: str = field(metadata={"help": "Training dataset."})
    inference_file: str = field(
        default="./data/train/webshop_train.json", metadata={"help": "Inference dataset."}
    )
    test_file: str = field(default="./data/test/webshop_test.json", metadata={"help": "Test dataset."})
    iter_data_path: str = field(
        default="./iter_data/train_iter_0.json", metadata={"help": "Iter data path (dir)"}
    )
    # model path
    model_train_path: str = field(
        default="meta-llama/Llama-2-7b-chat-hf",
        metadata={"help": "Path of initial train model"},
    )
    model_save_path: str = field(
        default="outputs/model",
        metadata={"help": "Directory to save the trained model."},
    )
    task_name: str = field(default="webshop", metadata={"help": "Task name for evaluation"})
    # train config
    batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for training."},
    )
    eval_batch_size: int = field(default=8, metadata={"help": "Batch size for evaluation."})
    n_epochs: int = field(default=40)
    num_workers: int = field(default=8, metadata={"help": "Number of subprocesses to use for data loading."})
    learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate."})
    weight_decay: float = field(default=1e-6, metadata={"help": "Weight decay for regularization."})
    warmup_step: int = field(default=0, metadata={"help": "Number of warmup steps"})
    clip_grad_norm: float = field(default=1, metadata={"help": "Gradient clipping threshold."})
    gradient_accumulation_steps: int = field(default=1)
    evaluating_epoch_freq: int = field(default=1)
    logging_epoch_freq: int = field(default=1)
    saving_epoch_freq: int = field(default=1)
    evaluating_step_freq: int = field(default=None)
    logging_step_freq: int = field(default=None)
    seed: int = field(default=42)
    max_input_length: int = field(default=700)

    # agent evol
    sample_num: int = field(default=5)
    iter_num: int = field(default=0)

    # environment
    max_round: int = field(
        default=6,
        metadata={"help": "Interaction rounds between agents and environment"},
    )

    # wandb stuff
    wandb_log: bool = field(default=False)
    wandb_project: str = field(default="AgentGym_agent_evol")
    wandb_run_name: str = field(default="agent_evol")

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

    trainer = AgentEvolTrainer(
        Agent(model, tokenizer),
        [task_class(client_args=env_args, n_clients=1)],
        args,
    )
    trainer.evol()


if __name__ == "__main__":
    main()
