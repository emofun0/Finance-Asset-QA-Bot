from __future__ import annotations

from typing import Any

from app.core.config import settings

try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover - optional dependency at runtime
    ChatOllama = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency at runtime
    ChatOpenAI = None


def build_langchain_chat_model(provider: str | None = None, model: str | None = None) -> Any | None:
    selected_provider = (provider or settings.llm_provider).lower()

    if selected_provider == "openai":
        if ChatOpenAI is None or not settings.openai_api_key:
            return None
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=model or settings.openai_model,
            timeout=settings.llm_timeout_seconds,
            model_kwargs={"reasoning": {"effort": settings.openai_reasoning_effort}},
        )

    if selected_provider == "ollama":
        if ChatOllama is None:
            return None
        selected_model = model or settings.ollama_model
        if not selected_model:
            return None
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=selected_model,
            temperature=0,
            num_predict=800,
            reasoning=False,
            keep_alive="10m",
            sync_client_kwargs={"timeout": settings.llm_timeout_seconds},
            async_client_kwargs={"timeout": settings.llm_timeout_seconds},
        )

    return None
