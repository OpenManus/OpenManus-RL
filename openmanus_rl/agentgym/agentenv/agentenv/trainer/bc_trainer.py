import json
import os
from collections import defaultdict
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from typing import Sequence

import jsonlines
import numpy as np
import torch
import wandb
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import broadcast, gather_object
from agentenv.controller import Agent
from agentenv.controller.agent import Agent
from agentenv.controller.task import BaseTask
from agentenv.controller.utils import BaseTrainer
from agentenv.trainer.utils import set_seed
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, GenerationConfig, get_linear_schedule_with_warmup


class BCTrainer(BaseTrainer):
    def __init__(self, agent: Agent, tasks: Sequence[BaseTask], args) -> None:
        self.agent = agent
        self.tasks = tasks
        self.args = asdict(args)

        # data & loader
        self.train_dataset = None
        self.train_dataloader = None
        self.test_dataloader = None

        # accelerator
        self.accelerator = None

        # train in parallel
        self.optimizer = None
        self.scheduler = None

        # log dict
        self.best_eval_log_dict = {}
        self.summary_log_dict = {}

        self.create_accelerator()
        self.set_seed()
        self.setup_tokenizer()
        self.get_raw_dataset()
        self.get_train_dataloader()
        self.get_inference_test_dataloader()
        self.setup_wandb()
        self.init_train_stuff()

    def create_accelerator(self):
        """
        Create the accelerator.
        """
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args["gradient_accumulation_steps"],
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))],
        )  # wait for processing upto 5hrs

    def set_seed(self):
        """
        Set the random seed.
        """
        set_seed(self.args["seed"] + self.accelerator.process_index)

    def setup_tokenizer(self):
        """
        Setup the tokenizer.
        """
        self.agent.tokenizer.pad_token_id = 0
        self.agent.tokenizer.eos_token_id = 2
        self.accelerator.print(f"[Vocab size]: {len(self.agent.tokenizer)}")
        self.agent.model.resize_token_embeddings(len(self.agent.tokenizer))

    def get_raw_dataset(self):
        with self.accelerator.main_process_first():
            self.raw_dataset = DatasetDict(
                {
                    "train": Dataset.from_list(
                        json.load(open(self.args["train_file"], "r"))
                    ),
                    "inference": Dataset.from_list(
                        json.load(open(self.args["inference_file"], "r"))
                    ),
                    "test": Dataset.from_list(
                        json.load(open(self.args["test_file"], "r"))
                    ),
                }
            )
            self.accelerator.print("Raw data:", self.raw_dataset)

    def get_train_dataloader(self):
        """
        create train_dataset and train_dataloader
        """

        def tokenize_fn(batch, args, tokenizer):
            # tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content | trim + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content | trim + ' ' + eos_token }}{% endif %}{% endfor %}"
            assert tokenizer.eos_token_id is not None, (
                tokenizer.eos_token_id,
                tokenizer.eos_token,
            )
            new_batch = defaultdict(list)
            all_keys = list(batch.keys())
            for item_values in zip(*(batch[k] for k in all_keys)):
                item = {k: item_values[i] for i, k in enumerate(all_keys)}
                item_id, conversations = (item["item_id"], item["conversations"])

                input_ids = []
                labels = []

                for message in conversations:
                    if message["from"] == "human":
                        text = f"<s>[INST] {message['value']} [/INST]"
                        input_encode = tokenizer.encode(text, add_special_tokens=False)
                        input_ids.extend(input_encode)
                        labels.extend([-100] * len(input_encode))
                    else:
                        # message["from"] == "gpt":
                        # text = f" {message['value']}</s>"
                        text = f" {message['value']}"
                        input_encode = tokenizer.encode(text, add_special_tokens=False)
                        input_encode += [tokenizer.eos_token_id]
                        input_ids.extend(input_encode)
                        labels.extend(input_encode)

                attention_mask = [1] * len(input_ids)

                # Truncation
                input_ids_max_length = len(input_ids)
                # assert input_ids_max_length <= args['max_input_length'], input_ids_max_length
                input_ids = input_ids[: args["max_input_length"]]
                labels = labels[: args["max_input_length"]]
                attention_mask = attention_mask[: args["max_input_length"]]

                ##
                new_batch["input_ids"].append(input_ids)
                new_batch["labels"].append(labels)
                new_batch["attention_mask"].append(attention_mask)
                ##
                new_batch["item_id"].append(item_id)
                new_batch["input_ids_max_length"].append(input_ids_max_length)

            return new_batch

        tokenized_dataset = DatasetDict(
            {
                "train": self.raw_dataset["train"].map(
                    tokenize_fn,
                    fn_kwargs={
                        "args": self.args,
                        "tokenizer": self.agent.tokenizer,
                    },
                    batched=True,
                    remove_columns=self.raw_dataset["train"].column_names,
                    num_proc=8,
                    load_from_cache_file=False,
                )
            }
        )
        self.accelerator.print("Processed data:", tokenized_dataset)
        for mode, dataset in tokenized_dataset.items():
            self.accelerator.print(
                f"\n{mode}_input_ids_max_length",
                max(dataset["input_ids_max_length"]),
            )

        def collate_fn(batch, tokenizer):
            max_input_length = max([len(item["input_ids"]) for item in batch])
            max_target_length = max([len(item["labels"]) for item in batch])
            input_ids = []
            attention_mask = []
            labels = []

            for item in batch:
                input_ids.append(
                    item["input_ids"]
                    + [tokenizer.pad_token_id]
                    * (max_input_length - len(item["input_ids"]))
                )
                attention_mask.append(
                    item["attention_mask"]
                    + [0] * (max_input_length - len(item["attention_mask"]))
                )
                labels.append(
                    item["labels"] + [-100] * (max_target_length - len(item["labels"]))
                )

            forward_kwargs = {
                "input_ids": torch.LongTensor(input_ids),
                "attention_mask": torch.BoolTensor(attention_mask),
                "labels": torch.LongTensor(labels),
            }
            return {"forward_kwargs": forward_kwargs}

        self.train_dataset = tokenized_dataset["train"]
        self.train_dataloader = DataLoader(
            tokenized_dataset["train"],
            shuffle=True,
            batch_size=self.args["batch_size"],
            num_workers=self.args["num_workers"],
            pin_memory=True,
            collate_fn=partial(collate_fn, tokenizer=self.agent.tokenizer),
        )
        self.accelerator.print("Number of train batches:", len(self.train_dataloader))

    def get_inference_test_dataloader(self):
        """
        create inference_dataloader, test_dataloader
        """

        def collate_fn(batch):
            result = {
                "data_idxs": [int(item["item_id"].split("_")[-1]) for item in batch]
            }
            return result

        with self.accelerator.main_process_first():

            self.inference_dataloader = DataLoader(
                self.raw_dataset["inference"],
                batch_size=self.args["eval_batch_size"],
                num_workers=self.args["num_workers"],
                pin_memory=True,
                collate_fn=partial(collate_fn),
            )

            self.test_dataloader = DataLoader(
                self.raw_dataset["test"],
                batch_size=self.args["eval_batch_size"],
                num_workers=self.args["num_workers"],
                pin_memory=True,
                collate_fn=partial(collate_fn),
            )
            self.accelerator.print(
                "Number of inference batches:", len(self.inference_dataloader)
            )
            self.accelerator.print("Number of test batches:", len(self.test_dataloader))

    def setup_wandb(self):
        """
        Set the wandb.
        """
        # os.environ["WANDB_MODE"] = "offline"
        if torch.distributed.get_rank() == 0 and self.args["wandb_log"]:
            wandb.init(
                project=self.args["wandb_project"],
                name=self.args["wandb_run_name"],
            )
            wandb.config.update(self.args)

        if self.accelerator.is_main_process and self.args["wandb_log"]:
            wandb.run.summary.update(
                {
                    "pad_token_id": self.agent.tokenizer.pad_token_id,
                    "eos_token_id": self.agent.tokenizer.eos_token_id,
                    "unk_token_id": self.agent.tokenizer.unk_token_id,
                    "vocab_size": len(self.agent.tokenizer),
                }
            )

    def save_model(self, model, tokenizer, save_path):
        os.makedirs(save_path, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(model),
        )
        tokenizer.save_pretrained(save_path)

    def init_train_stuff(self):
        """
        Initialize the training stuff, including the optimizer, scheduler, etc.
        Prepare the model, optimizer, and dataloader.
        """
        num_training_steps = (
            len(self.train_dataloader)
            // self.accelerator.num_processes
            * self.args["n_epochs"]
        ) // self.args["gradient_accumulation_steps"]
        warmup_step = (
            self.args["warmup_step"]
            if self.args["warmup_step"] is not None and self.args["warmup_step"] >= 0
            else int(0.1 * num_training_steps)
        )
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.agent.model.named_parameters()
                    if not any(nd in n for nd in ["bias", "LayerNorm.weight"])
                ],
                "weight_decay": self.args["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.agent.model.named_parameters()
                    if any(nd in n for nd in ["bias", "LayerNorm.weight"])
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.args["learning_rate"], eps=1e-8
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_step,
            num_training_steps=num_training_steps,
        )

        self.accelerator.print(
            f"***** Running training *****\n"
            f"  Num examples = {len(self.train_dataset)}\n"
            f"  Num Epochs = {self.args['n_epochs']}\n"
            f"  Instantaneous batch size per device = {self.args['batch_size']}\n"
            f"  Total train batch size (w. parallel, distributed & accumulation) = {self.args['batch_size']*self.accelerator.num_processes*self.args['gradient_accumulation_steps']}\n"
            f"  Total optimization steps = {num_training_steps}\n"
            f"  Warm up step: {warmup_step}\n"
            f"  Learning rate: {self.args['learning_rate']}\n"
        )

        (
            self.agent.model,
            self.optimizer,
            self.train_dataloader,
            self.inference_dataloader,
            self.test_dataloader,
        ) = self.accelerator.prepare(
            self.agent.model,
            self.optimizer,
            self.train_dataloader,
            self.inference_dataloader,
            self.test_dataloader,
        )

    def train_one_epoch(self, epoch, global_step):
        clip_grad_norm = self.args.get("clip_grad_norm", None)
        logging_step_freq = self.args.get("logging_step_freq", None)
        self.agent.model.train()
        epoch_result_dict = defaultdict(list)
        with tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            disable=not self.accelerator.is_main_process,
            desc=f"Train Loop | Epoch {epoch}",
        ) as t:
            for idx, batch in t:
                with self.accelerator.accumulate(self.agent.model):
                    output = self.agent.model(**batch["forward_kwargs"])
                    # train_data_idx = batch["item_id"]
                    # # Print train_data_idx
                    # self.accelerator.print("Train data idx:", train_data_idx)
                    # Get some metrics
                    loss = output[0]
                    result_dict, extra = {}, None
                    # Update
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        if clip_grad_norm is not None:
                            self.accelerator.clip_grad_norm_(
                                self.agent.model.parameters(), clip_grad_norm
                            )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.accelerator.sync_gradients:
                        self.scheduler.step()

                if self.accelerator.sync_gradients:
                    global_step += 1
                    # Step update metric
                    epoch_result_dict["loss"].append(loss.item())
                    for k, v in result_dict.items():
                        epoch_result_dict[k].append(v)

                    # Step logging
                    train_log_dict = {}
                    if (
                        logging_step_freq is not None
                        and global_step % logging_step_freq == 0
                    ):
                        train_log_dict = {
                            f"T.{k}": sum(v) / len(v) if isinstance(v, list) else v
                            for k, v in epoch_result_dict.items()
                        }

                    if train_log_dict:
                        log_dict = {
                            "lr": self.scheduler.get_last_lr()[0],
                            **train_log_dict,
                        }
                        if self.accelerator.is_main_process and self.args["wandb_log"]:
                            wandb.log(log_dict, step=global_step)
                            log_dict = {
                                "wandb": self.args["wandb_project"]
                                + "|"
                                + self.args["wandb_run_name"],
                                **log_dict,
                            }
                        log_dict = {
                            k: f"{v:.5g}" if isinstance(v, float) else v
                            for k, v in log_dict.items()
                        }
                        self.accelerator.print(
                            f"[E={epoch}/{self.args['n_epochs']}, S={global_step}] {log_dict}"
                        )

                    # Keep only max_record items
                    for k, v in epoch_result_dict.items():
                        if len(v) > 1:
                            epoch_result_dict[k] = v[-1:]

        # Metric summary:
        epoch_result_dict = {
            k: (sum(v) / len(v) if isinstance(v, list) else v)
            for k, v in epoch_result_dict.items()
        }
        return epoch_result_dict, global_step

    def train(self):
        """
        Train the model.
        """
        global_step = 0
        n_epochs = self.args["n_epochs"]
        logging_epoch_freq = self.args["logging_epoch_freq"]
        evaluating_epoch_freq = self.args["evaluating_epoch_freq"]
        saving_epoch_freq = self.args["saving_epoch_freq"]
        model_save_path = self.args["model_save_path"]
        os.makedirs(model_save_path, exist_ok=True)
        with tqdm(range(1, n_epochs + 1), total=n_epochs, disable=False) as t:
            for epoch in t:
                train_epoch_result_dict, global_step = self.train_one_epoch(
                    epoch, global_step
                )

                eval_log_dict = {}
                is_best = False
                if (
                    evaluating_epoch_freq is not None
                    and epoch % evaluating_epoch_freq == 0
                ):
                    evaluate_result_dict = {
                        f"Eval.Gen.{k}": v
                        for k, v in self.eval_test_dataloader().items()
                    }
                    eval_log_dict.update(evaluate_result_dict)
                    if eval_log_dict["Eval.Gen.success"] > self.best_eval_log_dict.get(
                        "Eval.Gen.success_best", 0
                    ):
                        is_best = True
                        self.best_eval_log_dict["Eval.Gen.success_best"] = (
                            eval_log_dict["Eval.Gen.success"]
                        )
                    if "Eval.Gen.success" not in self.summary_log_dict:
                        self.summary_log_dict["Eval.Gen.success"] = []
                    self.summary_log_dict["Eval.Gen.success"].append(
                        eval_log_dict["Eval.Gen.success"]
                    )

                train_log_dict = {}
                if logging_epoch_freq is not None and epoch % logging_epoch_freq == 0:
                    train_log_dict = {
                        f"T.{k}": sum(v) / len(v) if isinstance(v, list) else v
                        for k, v in train_epoch_result_dict.items()
                    }

                if train_log_dict or eval_log_dict:
                    log_dict = {
                        "lr": self.scheduler.get_last_lr()[0],
                        **train_log_dict,
                        **eval_log_dict,
                        **self.best_eval_log_dict,
                    }
                    if self.accelerator.is_main_process and self.args["wandb_log"]:
                        wandb.log(log_dict, step=global_step)
                        log_dict = {
                            "wandb": self.args["wandb_project"]
                            + "|"
                            + self.args["wandb_run_name"],
                            **log_dict,
                        }
                    log_dict = {
                        k: f"{v:.5g}" if isinstance(v, float) else v
                        for k, v in log_dict.items()
                    }
                    self.accelerator.print(
                        f"[E={epoch}/{self.args['n_epochs']}, S={global_step}] {log_dict}"
                    )

                if saving_epoch_freq is not None and epoch % saving_epoch_freq == 0:
                    # if is_best:
                    save_path = os.path.join(model_save_path, f"train_epoch_{epoch}")
                    self.save_model(self.agent.model, self.agent.tokenizer, save_path)
                    self.agent.model = self.accelerator.unwrap_model(self.agent.model)

    def eval_test_dataloader(
        self,
        dataloader=None,
        do_sample=False,
        temperature=1.0,
        record_to_file=True,
    ):
        # test
        self.agent.model.eval()
        all_rewards = []
        all_success = []
        if dataloader is None:
            dataloader = self.test_dataloader

        for _, batch in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=not self.accelerator.is_main_process,
            desc="Evaluation Gen Loop",
        ):
            data_idxs = batch["data_idxs"]
            self.accelerator.print("==== Batch inference data idxs ====", data_idxs)
            with torch.no_grad():
                exps = self.eval(
                    generation_config=GenerationConfig(
                        max_length=4096,
                        do_sample=do_sample,
                        temperature=temperature,
                        eos_token_id=self.agent.tokenizer.eos_token_id,
                        pad_token_id=(
                            self.agent.tokenizer.pad_token_id
                            if self.agent.tokenizer.pad_token_id is not None
                            else self.agent.tokenizer.unk_token_id
                        ),
                    ),
                    max_rounds=self.args["max_round"],
                    idxs=data_idxs,
                )

                cur_batch_rewards = torch.FloatTensor(
                    [exp.reward for exp in exps.experiences]
                ).to(self.accelerator.device)
                cur_batch_success = torch.FloatTensor(
                    [1 if exp.reward == 1 else 0 for exp in exps.experiences]
                ).to(self.accelerator.device)
                cur_batch_data_idx = torch.tensor(data_idxs).to(self.accelerator.device)
                
                # gather operation
                all_device_batch_rewards = self.accelerator.gather(cur_batch_rewards)
                all_device_batch_success = self.accelerator.gather(cur_batch_success)
                all_device_batch_exp = gather_object(exps.experiences)
                all_device_data_idx = self.accelerator.gather(cur_batch_data_idx)
                all_rewards.extend(all_device_batch_rewards.cpu().numpy().tolist())
                all_success.extend(all_device_batch_success.cpu().numpy().tolist())
                
                # write inference results to file
                if record_to_file and self.accelerator.is_main_process:
                    # write to file
                    inference_file_path = os.path.join(
                        self.args["model_save_path"], "inference.jsonl"
                    )
                    with jsonlines.open(inference_file_path, mode="a") as f:
                        for idx, exp in enumerate(all_device_batch_exp):
                            cur_idx = all_device_data_idx[idx]
                            conversation = exp.conversation
                            cur_reward = exp.reward
                            cur_success = 1 if exp.reward == 1 else 0
                            item_id = f"{self.args['task_name']}_{cur_idx}"
                            f.write(
                                {
                                    "conversations": conversation,
                                    "item_id": item_id,
                                    "reward": cur_reward,
                                    "success": cur_success,
                                }
                            )

        # fix for duplicated data
        all_rewards = all_rewards[: len(dataloader.dataset)]
        all_success = all_success[: len(dataloader.dataset)]

        if self.accelerator.is_main_process and self.accelerator.is_local_main_process:
            mean_reward = torch.FloatTensor([np.mean(all_rewards)]).to(
                self.accelerator.device
            )
            mean_success = torch.FloatTensor([np.mean(all_success)]).to(
                self.accelerator.device
            )
        else:
            mean_reward = torch.FloatTensor([-1.0]).to(self.accelerator.device)
            mean_success = torch.FloatTensor([-1.0]).to(self.accelerator.device)

        mean_reward = broadcast(mean_reward).cpu().numpy().tolist()[0]
        mean_success = broadcast(mean_success).cpu().numpy().tolist()[0]
        self.accelerator.print("\n\n==== Test Evaluation ====\n")
        self.accelerator.print(f"Score: {mean_reward:.5f}")
        self.accelerator.print(f"Success: {mean_success:.5f}")

        return {"score": mean_reward, "success": mean_success}

    def train_and_inference(self):
        self.accelerator.print("[BC Trainer] Start training.")
        self.train()
        self.accelerator.print("[BC Trainer] Start inference.")
        self.eval_test_dataloader(
            dataloader=self.inference_dataloader,
            do_sample=True,
            temperature=1.2,
            record_to_file=True,
        )
