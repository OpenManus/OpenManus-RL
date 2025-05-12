from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, TypedDict

import torch
import openai # Added for OpenAI client
from transformers import GenerationConfig, PreTrainedTokenizerBase # Tokenizer still needed for output processing

from agentenv.controller import BaseEnvClient

# ConversationMessage remains the same internally but will be transformed for the API
ConversationMessage = TypedDict(
    "ConversationMessage", {"from": str, "value": str}
)


@dataclass
class ExperienceOutput:
    conversation: list[ConversationMessage]
    reward: float
    text: str # This will be the full conversation text, potentially reconstructed
    seq_ids: list[int] # Reconstructed token IDs
    attention_mask: list[int] # Likely all 1s
    action_mask: list[int] # Reconstructed action mask


# TokenizedConversationOutput is no longer needed in the same way,
# as tokenization for input is handled by vLLM server.
# We will, however, tokenize outputs.

class BaseTask:
    env_client_cls: Callable[[Any], BaseEnvClient] # Added type hint for clarity
    env_name: str

    def __init__(
        self,
        client_args: Mapping[str, Any],
        tokenizer: PreTrainedTokenizerBase, # Tokenizer is now needed at init for output processing
        vllm_base_url: str = "http://localhost:8001/v1",
        vllm_api_key: str = "dummy-key",
        vllm_model_name: str = "agent-llm",
    ) -> None:
        """
        Initializes the Task object to use a vLLM OpenAI-compatible server.

        Args:
            client_args (Mapping[str, Any]): Arguments for the environment client.
            tokenizer (PreTrainedTokenizerBase): Tokenizer for processing model outputs and environment states.
            vllm_base_url (str, optional): Base URL of the vLLM server.
            vllm_api_key (str, optional): API key for the vLLM server.
            vllm_model_name (str, optional): The model name served by vLLM.
        """
        if self.env_client_cls is None or self.env_name is None:
            raise NotImplementedError("env_client_cls and env_name must be defined in subclasses.")

        # Initialize environment client (assuming only one is needed as batching is deferred to vLLM's capability)
        # If multiple distinct environment instances are needed for parallel generation,
        # this part might need adjustment, but _generate_experience_batch handles sequential processing over idxs.
        self.env_clients = [self.env_client_cls(**client_args)] # Keep as a list for potential future use or consistency
                                                               # but primarily self.env_clients[0] will be used in _generate_experience_one
        self.len = len(self.env_clients[0]) # Length based on the first environment client

        self.tokenizer = tokenizer
        self.vllm_model_name = vllm_model_name
        self.openai_client = openai.OpenAI(
            base_url=vllm_base_url,
            api_key=vllm_api_key,
        )

    def _format_conversation_for_openai(
        self, conversation: list[ConversationMessage]
    ) -> list[dict[str, str]]:
        """
        Formats the internal conversation list into the OpenAI API messages format.
        The vLLM server is expected to apply the appropriate chat template.
        """
        messages = []
        for msg in conversation:
            if msg["from"] == "human":
                messages.append({"role": "user", "content": msg["value"]})
            elif msg["from"] == "gpt":
                messages.append({"role": "assistant", "content": msg["value"]})
            # 'system' messages could be handled here if needed
        return messages

    def _reconstruct_tokenized_info(
        self,
        conversation: list[ConversationMessage],
    ) -> tuple[str, list[int], list[int]]:
        """
        Reconstructs text, input_ids, and action_mask from the conversation.
        This is called after the full interaction loop to prepare ExperienceOutput.
        It applies tokenization based on the final conversation state.
        The Llama chat template specific tokens (<s>, [INST], etc.) are added here
        if we want the reconstructed `text` and `seq_ids` to exactly match the
        format the model would have seen *if tokenized with the template manually*.
        Alternatively, if vLLM handles templating entirely and we only care about raw content
        token IDs, this can be simpler. For consistency with original, let's try to replicate.
        """
        full_text = ""
        full_input_ids = []
        full_action_mask = []

        # Add BOS if it's part of the template and not automatically added by tokenizer for first turn.
        # Llama template usually starts with <s> for the first user message.
        # The tokenizer.apply_chat_template might be more robust if available and matching vLLM's.
        # For now, manual construction based on original logic:

        for i, message in enumerate(conversation):
            text_segment = ""
            ids_segment = []
            mask_segment = []

            if message["from"] == "human":
                # Based on original: "<s>[INST] {message['value']} [/INST]"
                # However, the first message in a llama sequence is just "<s>[INST] ... [/INST]"
                # Subsequent user messages don't get another <s> if part of the same turn sequence.
                # The vLLM server handles this templating. For reconstruction, we'll tokenize content.
                # Let's use a simplified approach: tokenize role-based content and apply masks.
                # The full templated string might be better assembled by vLLM's logic.
                # For simplicity, let's tokenize the content directly and then decide on special tokens.
                # This part is tricky because vLLM internally applies the template.
                # What we want for seq_ids and action_mask in ExperienceOutput should reflect what PPO/training expects.

                # Let's assume the vLLM server handles the template perfectly.
                # We need to tokenize our "value" to get IDs.
                # The "text" in ExperienceOutput should be the complete, final textual conversation.

                if i == 0 and self.tokenizer.bos_token: # Prepend BOS for the very first message
                    full_text += self.tokenizer.bos_token
                    full_input_ids += self.tokenizer.encode(self.tokenizer.bos_token, add_special_tokens=False)
                    full_action_mask += [0] * len(self.tokenizer.encode(self.tokenizer.bos_token, add_special_tokens=False))


                # Simplified: construct text as model would see it if templated.
                # This is an approximation.
                templated_user_content = f"[INST] {message['value']} [/INST]"
                if i == 0: # First message is <s>[INST]...
                     # Some tokenizers add <s> by default, others don't.
                     # Assuming Llama-style, where <s> is at the start of the whole sequence.
                     # self.tokenizer.apply_chat_template would be ideal here.
                     # For now, we'll manually add based on the original tokenization logic.
                    prefix = self.tokenizer.bos_token if self.tokenizer.bos_token and i==0 else ""
                    # The original code did: text = f"<s>[INST] {message['value']} [/INST]"
                    # We'll adapt slightly:
                    current_text = f"{prefix}[INST] {message['value']} [/INST]"
                    current_ids = self.tokenizer.encode(current_text, add_special_tokens=False)
                else:
                    current_text = f" [INST] {message['value']} [/INST]" # Space prefixed for subsequent turns
                    current_ids = self.tokenizer.encode(current_text, add_special_tokens=False)


                full_text += current_text
                full_input_ids += current_ids
                # Human messages don't contribute to loss, so action_mask is 0
                full_action_mask += [0] * len(current_ids)

            elif message["from"] == "gpt":
                # Original: text = f"{message['value']}</s>"
                # Add a leading space if it's not the first part of the text
                prefix = " " if full_text else ""
                current_text = f"{prefix}{message['value']}{self.tokenizer.eos_token}"
                current_ids = self.tokenizer.encode(current_text, add_special_tokens=False)

                full_text += current_text
                full_input_ids += current_ids
                # Assistant messages contribute to loss if message["loss"] is True
                mask_value = 1 if message["loss"] else 0
                full_action_mask += [mask_value] * len(current_ids)

        return full_text, full_input_ids, full_action_mask


    def _generate_experience_one(
        self,
        # model: PreTrainedModel, # Removed
        # tokenizer: PreTrainedTokenizerBase, # Now self.tokenizer
        client: BaseEnvClient, # Environment client
        idx: int, # Index for environment reset
        generation_config = None,
        max_rounds: Optional[int] = None,
    ) -> ExperienceOutput:
        client.reset(idx)
        reward = 0.0
        done = False
        state = client.observe()
        # Initial conversation state from the environment
        conversation: list[ConversationMessage] = list(client.conversation_start) # type: ignore
        # Append the first human observation
        conversation.append(
            ConversationMessage({"from": "human", "value": state}) # loss is False for human turns
        )
        rounds = 0

        # Prepare parameters for OpenAI API from GenerationConfig
        temperature = generation_config.temperature if generation_config else 0.7 # Default from your example
        max_tokens = generation_config.max_new_tokens if generation_config and generation_config.max_new_tokens is not None else 256 # Default
        # Add other parameters as needed: top_p, presence_penalty, frequency_penalty, etc.
        # stop_sequences = [self.tokenizer.eos_token] if self.tokenizer.eos_token else []

        while not done:
            formatted_messages = self._format_conversation_for_openai(conversation)

            # Estimate current token length to avoid exceeding model limits (optional, but good practice)
            # This requires tokenizing the input with the server's chat template, which can be complex.
            # For now, we rely on vLLM to handle overly long inputs gracefully or error out.
            # A simple check:
            # temp_text_for_len_check = self.tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
            # if len(self.tokenizer.encode(temp_text_for_len_check)) > 3800: # Leave some headroom for generation (e.g. 4096 - 256)
            #     print(f"Warning: Input length {len(self.tokenizer.encode(temp_text_for_len_check))} might be too long.")
            #     break


            api_response = self.openai_client.chat.completions.create(
                model=self.vllm_model_name,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                # stop=stop_sequences, # vLLM handles EOS automatically based on model
                stream=False,
                extra_body={ # vLLM specific
                    "chat_template_kwargs": {"enable_thinking": False} # As per your example
                }
            )

            generated_text = api_response.choices[0].message.content
            if generated_text is None: # Should not happen with stream=False and valid response
                generated_text = ""

            # Append assistant's response to the conversation
            # Loss is True for assistant's generated actions
            conversation.append(
                ConversationMessage(
                    {"from": "gpt", "value": generated_text}
                )
            )

            # Interact with the environment
            step_output = client.step(generated_text)
            state, reward, done = (
                step_output.state,
                step_output.reward,
                step_output.done,
            )

            # Append environment's response (as a human turn)
            if not done: # Only append if not done, or if we want to record the final state
                conversation.append(
                    ConversationMessage({"from": "human", "value": state}) # loss is False for human turns
                )

            rounds += 1
            if max_rounds is not None and rounds >= max_rounds:
                break
            if len(conversation) > 50: # Safety break for very long conversations
                print("Warning: Max conversation turns reached.")
                break

        return ExperienceOutput(
            conversation=conversation,
            reward=0,
            text=None,
            seq_ids=None,
            attention_mask=None,
            action_mask=None,
        )

    def _generate_experience_batch(
        self,
        # model: PreTrainedModel, # Removed
        # tokenizer: PreTrainedTokenizerBase, # Now self.tokenizer
        idxs: Sequence[int],
        generation_config = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput]:
        # Batching here means iterating through specified environment indices
        # and generating experience for each one sequentially.
        # vLLM handles concurrent requests efficiently on the backend if multiple
        # calls to _generate_experience_one were made in parallel (e.g. using asyncio).
        # For now, this remains sequential at the Python level.
        
        # Use the single (or first) environment client. If different envs per idx are needed,
        # self.env_clients would need to be managed/indexed appropriately.
        # Assuming self.env_clients[0] is the one to use for all these idx resets.
        env_client_to_use = self.env_clients[0]
        
        results = []
        for idx in idxs:
            results.append(self._generate_experience_one(
                client=env_client_to_use,
                idx=idx,
                generation_config=generation_config,
                max_rounds=max_rounds,
            ))
        return results

    def generate_experience(
        self,
        model, # Removed
        tokenizer: PreTrainedTokenizerBase, # Removed, use self.tokenizer
        idxs: Sequence[int] | int,
        generation_config = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput] | ExperienceOutput: # Return type can be single if input is int
        if not self.tokenizer:
            raise ValueError("Tokenizer must be provided during BaseTask initialization to use vLLM.")

        single_request = isinstance(idxs, int)
        if single_request:
            idxs_list = [idxs] # type: ignore
        else:
            idxs_list = list(idxs)


        # No DistributedDataParallel model.module check needed
        results = self._generate_experience_batch(
            # model, tokenizer arguments removed
            idxs=idxs_list,
            generation_config=generation_config,
            max_rounds=max_rounds,
        )
        
        return results[0] if single_request else results


    def __len__(self) -> int:
        """Returns the length of the underlying environment dataset."""
        return self.len