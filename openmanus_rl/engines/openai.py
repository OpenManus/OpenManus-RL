"""Minimal OpenAI chat wrapper.

Provides a small surface compatible with internal code paths that expect
`ChatOpenAI` with a callable interface. Supports OpenAI-compatible backends
such as vLLM by honoring `OPENAI_BASE_URL`.
"""

from typing import Optional, List, Dict, Any, Type
import json
import re
try:
    from pydantic import BaseModel  # type: ignore
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore
import os

try:
    from openai import OpenAI  # type: ignore
except Exception as exc:  # pragma: no cover
    OpenAI = None  # type: ignore


class ChatOpenAI:
    """Thin wrapper around OpenAI's Chat Completions API.

    The instance is callable and returns plain text. Images are not sent as
    binary by design to remain compatible with OpenAI-compatible servers that
    do not support multimodal content; image paths are appended as text hints.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")

        self.model = model
        self.temperature = temperature
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

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
            # Safe multimodal fallback: append image paths as text hints.
            content = prompt
            for p in images:
                content += f"\n[Image: {p}]"
            messages.append({"role": "user", "content": content})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            n=1,
        )
        text = (resp.choices[0].message.content or "").strip()

        # Best-effort structured parsing when a pydantic model is requested
        try:
            if response_format and isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # Try JSON first
                try:
                    data = json.loads(text)
                    if isinstance(data, dict):
                        return response_format(**data)
                    if isinstance(data, list):
                        # Common pattern: patch list
                        payload: Dict[str, Any] = {}
                        if hasattr(response_format, "model_fields") and "patch" in response_format.model_fields:  # pydantic v2
                            payload["patch"] = data
                        elif hasattr(response_format, "__fields__") and "patch" in getattr(response_format, "__fields__"):
                            payload["patch"] = data
                        if payload:
                            return response_format(**payload)
                except Exception:
                    pass

                # Special-case: AnswerVerification(analysis: str, true_false: bool)
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

                # Fallback: try to populate known common fields
                payload: Dict[str, Any] = {}
                for field in ("analysis", "text"):
                    if (hasattr(response_format, "model_fields") and field in response_format.model_fields) or (
                        hasattr(response_format, "__fields__") and field in getattr(response_format, "__fields__")
                    ):
                        payload[field] = text
                if payload:
                    return response_format(**payload)
        except Exception:
            # Swallow parsing errors and return raw text
            pass

        return text
