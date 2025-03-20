import os
import pathlib
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from dpo_trainer import DPOMultiTrainer


@dataclass
class ModelArguments:
    model_name_or_path: str = field()
    ref_model_name_or_path: Optional[str] = field()
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )
    beta: Optional[float] = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_prompt_length: int = field(
        default=512,
        metadata={"help": "Maximum target length."},
    )
    max_target_length: int = field(
        default=2048,
        metadata={"help": "Maximum target length."},
    )
    remove_unused_columns: bool = field(default=False)


def tokenize_conversation(
    conversation,
    tokenizer,
):
    def tokenize_conversation_one(
        message,
        tokenizer,
    ):
        if message["from"] == "human":
            input_ids = tokenizer.encode(
                f"<s>[INST] {message['value']} [/INST]", add_special_tokens=False
            )
        else:
            input_ids = tokenizer.encode(
                f"{message['value']}</s>", add_special_tokens=False
            )
        if message["loss"]:
            labels = list(input_ids)
        else:
            labels = [-100] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    input_ids = []
    labels = []

    for message in conversation:
        message_out = tokenize_conversation_one(message, tokenizer)
        input_ids += message_out["input_ids"]
        labels += message_out["labels"]

    return input_ids, [1] * len(input_ids), labels


def preprocess_multi_turn(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict[str, list[int]]:
    prompt = source["prompt"]
    prompt = [
        {
            "role": "user" if x["from"] == "human" else "assistant",
            "content": x["value"],
        }
        for x in prompt
    ]
    prompt_input_ids = tokenizer.apply_chat_template(prompt)
    prompt_attention_mask = [1] * len(prompt_input_ids)

    chosen_input_ids, chosen_attention_mask, chosen_labels = tokenize_conversation(
        source["chosen"], tokenizer
    )
    chosen_input_ids = list(prompt_input_ids) + chosen_input_ids
    chosen_attention_mask = list(prompt_attention_mask) + chosen_attention_mask
    chosen_labels = [-100] * len(prompt_input_ids) + chosen_labels

    rejected_input_ids, rejected_attention_mask, rejected_labels = (
        tokenize_conversation(
            source["rejected"],
            tokenizer,
        )
    )
    rejected_input_ids = list(prompt_input_ids) + rejected_input_ids
    rejected_attention_mask = list(prompt_attention_mask) + rejected_attention_mask
    rejected_labels = [-100] * len(prompt_input_ids) + rejected_labels

    return dict(
        chosen_input_ids=chosen_input_ids,
        chosen_attention_mask=chosen_attention_mask,
        chosen_labels=chosen_labels,
        rejected_input_ids=rejected_input_ids,
        rejected_attention_mask=rejected_attention_mask,
        rejected_labels=rejected_labels,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    parser.add_argument("--local-rank", type=int, default=0)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    global local_rank  # pylint: disable=W0604:global-at-module-level
    local_rank = training_args.local_rank

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()

    if model_args.ref_model_name_or_path is None:
        model_args.ref_model_name_or_path = model_args.model_name_or_path
    model_ref = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.ref_model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=True,
        trust_remote_code=model_args.trust_remote_code,
    )

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    # Load data
    dataset = load_dataset("json", data_files=data_args.data_path)
    preprocess = partial(
        preprocess_multi_turn,
        tokenizer=tokenizer,
    )
    train_dataset = dataset["train"].map(preprocess)
    trainer = DPOMultiTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        beta=model_args.beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=training_args.model_max_length,
        max_target_length=training_args.max_target_length,
        max_prompt_length=training_args.max_prompt_length,
        generate_during_eval=True,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    global local_rank, world_size  # pylint: disable=W0604:global-at-module-level
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    train()
