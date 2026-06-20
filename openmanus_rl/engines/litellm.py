"""LiteLLM chat wrapper.

Provides the same callable interface as ``ChatOpenAI`` but routes through
LiteLLM (https://github.com/BerriAI/litellm), giving access to 100+ LLM
providers (Anthropic, Bedrock, Vertex, Gemini, Cohere, Mistral, etc.)
via a single unified API.
"""

import json
import re
from typing import Any, Dict, List, Optional, Type

try:
    from pydantic import BaseModel  # type: ignore
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore


class ChatLiteLLM:
    """Thin wrapper around LiteLLM's completion API.

    Drop-in replacement for ``ChatOpenAI`` that supports any model string
    LiteLLM understands (e.g. ``anthropic/claude-sonnet-4-6``,
    ``bedrock/claude-3.5-sonnet``, ``gpt-4o``, etc.).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        try:
            import litellm  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "litellm package is not installed. "
                "Install it with: pip install litellm"
            ) from exc

        self._litellm = litellm
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.api_key = api_key

    def __call__(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        system: Optional[str] = None,
        response_format: Optional[Type] = None,
        **_: Any,
    ) -> Any:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})

        if not images:
            messages.append({"role": "user", "content": prompt})
        else:
            content = prompt
            for p in images:
                content += f"\n[Image: {p}]"
            messages.append({"role": "user", "content": content})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "n": 1,
            "drop_params": True,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url

        resp = self._litellm.completion(**kwargs)
        text = (resp.choices[0].message.content or "").strip()

        try:
            if response_format and isinstance(response_format, type) and issubclass(response_format, BaseModel):
                try:
                    data = json.loads(text)
                    if isinstance(data, dict):
                        return response_format(**data)
                    if isinstance(data, list):
                        payload: Dict[str, Any] = {}
                        if hasattr(response_format, "model_fields") and "patch" in response_format.model_fields:
                            payload["patch"] = data
                        elif hasattr(response_format, "__fields__") and "patch" in getattr(response_format, "__fields__"):
                            payload["patch"] = data
                        if payload:
                            return response_format(**payload)
                except Exception:
                    pass

                if getattr(response_format, "__name__", "") == "AnswerVerification":
                    analysis = ""
                    tf = False
                    m = re.search(r"<analysis>\s*(.*?)\s*</analysis>", text, re.DOTALL)
                    if m:
                        analysis = m.group(1).strip()
                    m2 = re.search(r"<true_false>\s*(.*?)\s*</true_false>", text, re.DOTALL)
                    if m2:
                        val = m2.group(1).strip().lower()
                        tf = val in ("true", "1", "yes")
                    if not analysis:
                        analysis = text
                    return response_format(analysis=analysis, true_false=tf)

                payload: Dict[str, Any] = {}
                for field in ("analysis", "text"):
                    if (hasattr(response_format, "model_fields") and field in response_format.model_fields) or (
                        hasattr(response_format, "__fields__") and field in getattr(response_format, "__fields__")
                    ):
                        payload[field] = text
                if payload:
                    return response_format(**payload)
        except Exception:
            pass

        return text
