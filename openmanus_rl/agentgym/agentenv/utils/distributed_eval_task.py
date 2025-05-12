import time
from dataclasses import asdict, dataclass, field

import torch
import transformers
from agentenv.controller import Agent
from agentenv.envs import (
    AcademiaTask,
    AlfWorldTask,
    BabyAITask,
    MazeTask,
    MovieTask,
    SciworldTask,
    SheetTask,
    SqlGymTask,
    TextCraftTask,
    TodoTask,
    WeatherTask,
    WebarenaTask,
    WebshopTask,
    WordleTask,
)
from agentenv.trainer.distributed_evaluator import DistributedEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class EvalArguments:
    model_path: str
    # required path
    output_file: str
    inference_file: str = field(metadata={"help": "Inference dataset."})
    task_name: str = field(default="webshop", metadata={"help": "Task name for evaluation"})

    # eval config
    eval_batch_size: int = field(default=8, metadata={"help": "Batch size for evaluation."})
    num_workers: int = field(default=8, metadata={"help": "Number of subprocesses to use for data loading."})
    seed: int = field(default=42)
    do_sample: bool = field(default=False, metadata={"help": "Do sampling or not."})
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature."})

    # conversation rounds
    max_round: int = field(
        default=6,
        metadata={"help": "Interaction rounds between agents and environment"},
    )

    weight_decay: float = field(default=1e-6, metadata={"help": "Weight decay for regularization."})

    # environment parameters
    env_server_base: str = field(default=None)
    data_len: int = field(default=200)
    timeout: int = field(default=2400)


def main():
    parser = transformers.HfArgumentParser(EvalArguments)
    (args,) = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    )
    model.gradient_checkpointing_enable()

    # task_name - task dict
    task_classes = {
        "webshop": WebshopTask,
        "alfworld": AlfWorldTask,
        "babyai": BabyAITask,
        "sciworld": SciworldTask,
        "textcraft": TextCraftTask,
        "webarena": WebarenaTask,
        "sqlgym": SqlGymTask,
        "lmrlgym_maze": MazeTask,
        "lmrlgym_wordle": WordleTask,
        "weather": WeatherTask,
        "todo": TodoTask,
        "movie": MovieTask,
        "sheet": SheetTask,
        "academia": AcademiaTask,
        "babyai": BabyAITask,
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

    distributed_evaluator = DistributedEvaluator(
        Agent(model, tokenizer),
        [task_class(client_args=env_args, n_clients=1)],
        args,
    )
    start_time = time.time()
    distributed_evaluator.generate()
    process_time = time.time() - start_time
    print(f"==== {process_time} seconds ====")


if __name__ == "__main__":
    main()
