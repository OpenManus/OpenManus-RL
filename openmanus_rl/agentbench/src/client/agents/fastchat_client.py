import json
import time
from typing import List, Dict, Union, Any

import requests
from fastchat.model.model_adapter import get_conversation_template
# import TimeoutException
from requests.exceptions import Timeout, ConnectionError

from ..agent import AgentClient
from ...typings import AgentNetworkException


class Prompter:
    @staticmethod
    def get_prompter(prompter: Union[str, None, Dict[str, Any]]):
        name = None
        args = {}
        if isinstance(prompter, str):
            name = prompter
        elif isinstance(prompter, dict):
            name = prompter["name"]
            args = prompter["args"]
        # check if prompter_name is a method and its variable
        if not name:
            return None
        if hasattr(Prompter, name) and callable(getattr(Prompter, name)):
            return getattr(Prompter, name)(**args)

    @staticmethod
    def claude():
        def _prompter(messages: List[Dict[str, str]]):
            prompt = ""
            role_dict = {
                "user": "Human",
                "agent": "Assistant",
            }
            for item in messages:
                prompt += f"{role_dict[item['role']]}: {item['content']}\n\n"
            prompt += "Assistant:"
            return {"prompt": prompt}

        return _prompter

    @staticmethod
    def openchat_v3_1():
        def _prompter(messages: List[Dict[str, str]]):
            prompt = "Assistant is GPT4<|end_of_turn|>"
            role_dict = {
                "user": "User: {content}<|end_of_turn|>",
                "agent": "Assistant: {content}<|end_of_turn|>",
            }
            for item in messages:
                prompt += role_dict[item["role"]].format(content=item["content"])
            prompt += "Assistant:"
            return {"prompt": prompt}

        return _prompter

    @staticmethod
    def openchat_v3_2():
        def _prompter(messages: List[Dict[str, str]]):
            prompt = ""
            role_dict = {
                "user": "GPT4 User: {content}<|end_of_turn|>\n",
                "agent": "GPT4 Assistant: {content}<|end_of_turn|>\n",
            }
            for item in messages:
                prompt += role_dict[item["role"]].format(content=item["content"])
            prompt += "GPT4 Assistant:"
            return {"prompt": prompt}

        return _prompter

    @staticmethod
    def prompt_string(
        prefix: str = "",
        suffix: str = "AGENT:",
        user_format: str = "USER: {content}\n\n",
        agent_format: str = "AGENT: {content}\n\n",
        prompt_key: str = "prompt",
    ):
        def prompter(messages: List[Dict[str, str]]):
            nonlocal prefix, suffix, user_format, agent_format, prompt_key
            prompt = prefix
            for item in messages:
                if item["role"] == "user":
                    prompt += user_format.format(content=item["content"])
                else:
                    prompt += agent_format.format(content=item["content"])
            prompt += suffix
            return {prompt_key: prompt}

        return prompter


class FastChatAgent(AgentClient):
    """This agent is a test agent, which does nothing. (return empty string for each action)"""

    def __init__(
        self,
        model_name,
        controller_address=None,
        worker_address=None,
        temperature=0,
        max_new_tokens=32,
        top_p=0,
        prompter=None,
        args=None,
        **kwargs,
    ) -> None:
        if controller_address is None and worker_address is None:
            raise ValueError(
                "Either controller_address or worker_address must be specified."
            )
        self.controller_address = controller_address
        self.worker_address = worker_address
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.prompter = Prompter.get_prompter(prompter)
        self.args = args or {}
        print(self.max_new_tokens)
        super().__init__(**kwargs)

    def inference(self, history: List[dict]) -> str:
        if self.worker_address:
            worker_addr = self.worker_address
        else:
            controller_addr = self.controller_address
            worker_addr = controller_addr
        if worker_addr == "":
            raise ValueError
        gen_params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "echo": False,
            "top_p": self.top_p,
            **self.args,
        }
        if self.prompter:
            prompt = self.prompter(history)
            gen_params.update(prompt)
        else:
            conv = get_conversation_template(self.model_name)
            for history_item in history:
                role = history_item["role"]
                content = history_item["content"]
                if role == "user":
                    conv.append_message(conv.roles[0], content)
                elif role == "agent":
                    conv.append_message(conv.roles[1], content)
                else:
                    raise ValueError(f"Unknown role: {role}")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            gen_params.update(
                {
                    "prompt": prompt,
                    "stop": conv.stop_str,
                    "stop_token_ids": conv.stop_token_ids,
                }
            )
        headers = {"User-Agent": "FastChat Client"}
        for _ in range(3):
            try:
                response = requests.post(
                    controller_addr + "/worker_generate_stream",
                    headers=headers,
                    json=gen_params,
                    stream=True,
                    timeout=120,
                )
                text = ""
                for line in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if line:
                        data = json.loads(line)
                        if data["error_code"] != 0:
                            raise AgentNetworkException(data["text"])
                        text = data["text"]
                return text
            # if timeout or connection error, retry
            except Timeout:
                print("Timeout, retrying...")
            except ConnectionError:
                print("Connection error, retrying...")
            time.sleep(5)
        else:
            raise Exception("Timeout after 3 retries.")
