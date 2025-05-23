import openai
import os
from copy import deepcopy
from typing import List, Dict, Optional, Any

from ..agent import AgentClient

class QWQAgent(AgentClient):
    def __init__(self, name: str, api_args: Optional[Dict[str, Any]] = None, *args, **config):
        super().__init__(name=name, *args, **config)
        
        if api_args is None:
            print(f"Warning: api_args not explicitly provided for QWQAgent '{name}'. Trying to infer from config.")
            resolved_api_args = deepcopy(config.get('body', {}))
            if 'model' not in resolved_api_args and 'model' in config:
                resolved_api_args['model'] = config['model']
            if 'max_tokens' not in resolved_api_args and 'max_tokens' in config:
                resolved_api_args['max_tokens'] = config['max_tokens']
            if 'temperature' not in resolved_api_args and 'temperature' in config:
                 resolved_api_args['temperature'] = config['temperature']
        else:
            resolved_api_args = deepcopy(api_args)

        self.model_name = resolved_api_args.pop("model", "qwq-32b")
        self.api_key = resolved_api_args.pop("api_key", None) or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = resolved_api_args.pop("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        self.client_params = resolved_api_args 

        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for QWQAgent. Please set it as an environment variable or in api_args.")

        try:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except ImportError:
            raise ImportError("OpenAI library not installed. Please install it using 'pip install openai'")

    def _format_messages(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        messages = []
        for message in history:
            role = message["role"]
            if role == "agent":
                role = "assistant"
            if role in ["user", "assistant"]:
                messages.append({"role": role, "content": message["content"]})
        return messages

    def inference(self, history: List[Dict[str, str]]) -> str:
        formatted_messages = self._format_messages(history)
        
        if not formatted_messages:
            return "Error: No valid messages to send to the model."

        try:
            completion_params = {
                "model": self.model_name,
                "messages": formatted_messages,
                **self.client_params,
                "stream": True
            }
            
            stream = self.client.chat.completions.create(**completion_params)
            
            full_response_content = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    full_response_content.append(chunk.choices[0].delta.content)
            
            if full_response_content:
                return "".join(full_response_content).strip()
            else:
                return "Error: No response content from QWQ model stream."
        except Exception as e:
            print(f"Error during QWQAgent inference: {e}")
            return f"Error: {str(e)}"

