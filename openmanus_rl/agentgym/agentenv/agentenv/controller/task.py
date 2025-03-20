from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, TypedDict

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.utils import GenerateOutput

from agentenv.controller import BaseEnvClient

ConversationMessage = TypedDict(
    "ConversationMessage", {"from": str, "loss": Optional[bool], "value": str}
)


@dataclass
class ExperienceOutput:
    conversation: list[ConversationMessage]
    reward: float
    text: str
    seq_ids: list[int]
    attention_mask: list[int]
    action_mask: list[int]


TokenizedConversationOutput = TypedDict(
    "TokenizedConversationOutput",
    {
        "text": str,
        "input_ids": list[int],
        "action_mask": list[int],
    },
)


class BaseTask:
    env_client_cls: Callable
    env_name: str

    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int = 1,
    ) -> None:
        """
        Initializes the Task object.

        Args:
            client_args (Mapping[str, Any]): A mapping of client arguments.
            n_clients (int, optional): The number of clients. Defaults to 1. Larger than 1 for batch generation. Batch generation is not implemented yet.
        """
        if self.env_client_cls is None or self.env_name is None:
            raise NotImplementedError
        self.clients = [self.env_client_cls(**client_args) for _ in range(n_clients)]
        self.len = len(self.clients[0])

    def _tokenize_conversation_one(
        self,
        message: ConversationMessage,
        tokenizer: PreTrainedTokenizerBase,
    ) -> TokenizedConversationOutput:
        """
        This function applied Llama Chat template on the given vicuna-styled conversation message.
        You can provide your own _tokenize_conversation_one to adapt to your own task.
        """
        if message["from"] == "human":
            text = f"<s>[INST] {message['value']} [/INST]"
            input_ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            text = f"{message['value']}</s>"
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            text = f" {text}"
        if message["loss"]:
            action_mask = [1] * len(input_ids)
        else:
            action_mask = [0] * len(input_ids)

        return TokenizedConversationOutput(
            {
                "text": text,
                "input_ids": input_ids,
                "action_mask": action_mask,
            }
        )

    def _tokenize_conversation(
        self,
        conversation: list[ConversationMessage],
        tokenizer: PreTrainedTokenizerBase,
    ) -> TokenizedConversationOutput:
        text = ""
        input_ids = []
        action_mask = []

        for message in conversation:
            message_out = self._tokenize_conversation_one(message, tokenizer)
            text += message_out["text"]
            input_ids += message_out["input_ids"]
            action_mask += message_out["action_mask"]

        return TokenizedConversationOutput(
            {
                "text": text,
                "input_ids": input_ids,
                "action_mask": action_mask,
            }
        )

    def _generate_experience_one(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        client: BaseEnvClient,
        idx: int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> ExperienceOutput:
        client.reset(idx)
        reward = 0.0
        done = False
        state = client.observe()
        conversation = list(client.conversation_start)
        conversation.append(
            ConversationMessage({"from": "human", "loss": None, "value": state})
        )
        conversation_tokenized = self._tokenize_conversation(conversation, tokenizer)
        rounds = 0

        while not done:
            input_length = len(conversation_tokenized["input_ids"])
            # if input_length exceeds 4096, break
            if input_length > 4096:
                break
            output = model.generate(
                torch.tensor(
                    [conversation_tokenized["input_ids"]], device=model.device
                ),
                generation_config=generation_config,
            )
            if isinstance(output, GenerateOutput):
                output = output.sequences

            generated_tokens = output[0][input_length:].cpu().numpy().tolist()
            if generated_tokens[-1] != tokenizer.eos_token_id:
                generated_tokens += [tokenizer.eos_token_id]

            generated_text = tokenizer.decode(generated_tokens)
            conversation_tokenized["text"] += f" {generated_text}"
            conversation_tokenized["input_ids"] += generated_tokens
            conversation_tokenized["action_mask"] += [1] * len(generated_tokens)

            generated_text = generated_text[
                : -len(tokenizer.eos_token)
            ]  # not endswith eos_token
            conversation.append(
                ConversationMessage(
                    {"from": "gpt", "loss": True, "value": generated_text}
                )
            )

            step_output = client.step(generated_text)
            state, reward, done = (
                step_output.state,
                step_output.reward,
                step_output.done,
            )
            env_message = ConversationMessage(
                {"from": "human", "loss": None, "value": state}
            )
            env_message_tokenized = self._tokenize_conversation_one(
                env_message, tokenizer
            )

            conversation.append(env_message)
            conversation_tokenized["text"] += env_message_tokenized["text"]
            conversation_tokenized["input_ids"] += env_message_tokenized["input_ids"]
            conversation_tokenized["action_mask"] += env_message_tokenized[
                "action_mask"
            ]

            rounds += 1
            if max_rounds is not None and rounds >= max_rounds:
                break

        return ExperienceOutput(
            conversation=conversation,
            reward=reward,
            text=conversation_tokenized["text"],
            seq_ids=conversation_tokenized["input_ids"],
            attention_mask=[1] * len(conversation_tokenized["input_ids"]),
            action_mask=conversation_tokenized["action_mask"],
        )

    def _generate_experience_batch(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        idxs: Sequence[int],
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput]:
        # TODO: "Batch experience generation is not implemented. Generate one by one.",
        client = self.clients[0]
        result = [self._generate_experience_one(
                    model=model,
                    tokenizer=tokenizer,
                    client=client,
                    idx=idx,
                    generation_config=generation_config,
                    max_rounds=max_rounds,
                ) for idx in idxs]
        return result

    def generate_experience(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        idxs: Sequence[int] | int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput]:
        if isinstance(idxs, int):
            idxs = [idxs]

        if isinstance(model, DistributedDataParallel):
            model = model.module

        return self._generate_experience_batch(
            model=model,
            tokenizer=tokenizer,
            idxs=idxs,
            generation_config=generation_config,
            max_rounds=max_rounds,
        )
