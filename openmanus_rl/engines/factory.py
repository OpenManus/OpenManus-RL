"""Engine factory helpers.

Exposes `create_llm_engine` returning a callable that maps prompt -> text using
the minimal `ChatOpenAI` wrapper. Keep the surface small and stable so tools
can depend on it without heavy coupling.
"""

from typing import Callable, Optional
from .openai import ChatOpenAI


def create_llm_engine(model_string: str = "gpt-4o-mini", is_multimodal: bool = False, base_url: Optional[str] = None) -> Callable[[str], str]:
    chat = ChatOpenAI(model=model_string, base_url=base_url)

    def _engine(prompt: str) -> str:
        # Tools currently call engine(prompt) for text-only flows.
        # If multimodal is needed later, extend by adding optional image args.
        return chat(prompt)

    return _engine

