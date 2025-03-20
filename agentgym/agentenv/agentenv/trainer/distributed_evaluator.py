import json
import os
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from typing import Sequence

import jsonlines
import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import broadcast, gather_object
from agentenv.controller import Agent
from agentenv.controller.agent import Agent
from agentenv.controller.task import BaseTask, GenerationConfig
from agentenv.controller.utils import BaseTrainer
from agentenv.trainer.utils import set_seed
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, GenerationConfig


class DistributedEvaluator(BaseTrainer):
    def __init__(self, agent: Agent, tasks: Sequence[BaseTask], args) -> None:
        self.agent = agent
        self.tasks = tasks
        self.args = asdict(args)

        # data & loader
        self.raw_dataset = None

        # accelerator
        self.accelerator = None

        self.create_accelerator()
        self.set_seed()
        self.setup_tokenizer()
        self.get_raw_dataset()
        self.get_inference_dataloader()

    def create_accelerator(self):
        """
        Create the accelerator.
        """
        self.accelerator = Accelerator(
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))],
        )

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
                    "inference": Dataset.from_list(
                        json.load(open(self.args["inference_file"], "r"))
                    ),
                }
            )
            self.accelerator.print("Raw data:", self.raw_dataset)

    def get_inference_dataloader(self):
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

            self.accelerator.print(
                "Number of inference batches:", len(self.inference_dataloader)
            )

    def generate(self, dataloader=None):
        self.optimizer = AdamW(self.agent.model.parameters())
        self.agent.model, self.optimizer, self.inference_dataloader = (
            self.accelerator.prepare(
                self.agent.model, self.optimizer, self.inference_dataloader
            )
        )
        self.agent.model.eval()
        all_rewards = []
        all_success = []
        if dataloader is None:
            dataloader = self.inference_dataloader

        for _, batch in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=not self.accelerator.is_main_process,
            desc="Inference Gen Loop",
        ):
            data_idxs = batch["data_idxs"]
            with torch.no_grad():
                exps = self.eval(
                    generation_config=GenerationConfig(
                        max_length=4096,
                        do_sample=self.args["do_sample"],
                        temperature=self.args["temperature"],
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
                if self.accelerator.is_main_process:
                    with jsonlines.open(self.args["output_file"], mode="a") as f:
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
        self.accelerator.print("\n\n==== Inference Evaluation ====\n")
        self.accelerator.print(f"Score: {mean_reward:.5f}")
        self.accelerator.print(f"Success: {mean_success:.5f}")

