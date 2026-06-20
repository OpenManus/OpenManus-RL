"""Engine factory helpers.

Exposes `create_llm_engine` returning a callable that maps prompt -> text using
the minimal `ChatOpenAI` wrapper (or `ChatLiteLLM` for 100+ providers via
LiteLLM). Keep the surface small and stable so tools can depend on it without
heavy coupling.
"""

from typing import Callable, Optional
from .openai import ChatOpenAI


def create_llm_engine(
    model_string: str = "gpt-4o-mini",
    is_multimodal: bool = False,
    base_url: Optional[str] = None,
    engine: str = "openai",
    api_key: Optional[str] = None,
) -> Callable[[str], str]:
    if engine == "litellm":
        from .litellm import ChatLiteLLM
        chat = ChatLiteLLM(model=model_string, base_url=base_url, api_key=api_key)
    else:
        chat = ChatOpenAI(model=model_string, base_url=base_url)

    def _engine(prompt: str) -> str:
        return chat(prompt)

    return _engine

