import os
from openai import OpenAI
from typing import List
from copy import deepcopy

from ..agent import AgentClient


class QwqAgent(AgentClient):

    def __init__(self, api_args=None, *args, **config):
        super().__init__(*args, **config)
    

        if api_args and isinstance(api_args, dict):
            # print(">>>>>>>>>api_args: ", api_args)

            # Set the API key
            self.key = os.getenv("DASHSCOPE_API_KEY")

            
            body = api_args.get('body')
            self.model = api_args.get('model')
            self.base_url = api_args.get('url')
        
        # Validate required parameters
        if not self.key:
            raise ValueError("Qwq API KEY is required, please assign api_args.key or set DASHSCOPE_API_KEY "
                            "environment variable.")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.key,
            base_url=self.base_url
        )

    def inference(self, history: List[dict]) -> str:
        try:
            messages = []
            for msg in history:
                role = "assistant" if msg["role"] == "agent" else msg["role"]
                messages.append({"role": role, "content": msg["content"]})
            
            params = deepcopy(self.body) if hasattr(self, 'body') else {}
            params["model"] = self.model
            params["messages"] = messages
            params["stream"] = True
            
            completion = self.client.chat.completions.create(**params)
            
            response = []
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    # print(content, end="", flush=True)
                    response.append(content)
            
            response = "".join(response)
            return response
            
        except Exception as e:
            print(f"ERROR: QwqAgent/os-std agent error {e}")
            raise e