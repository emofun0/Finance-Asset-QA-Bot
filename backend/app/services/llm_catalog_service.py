from __future__ import annotations

from typing import Any

import requests

from app.core.config import settings
from app.schemas.response import LLMModelCatalogResponse, LLMModelItem, LLMProviderCatalog


class LLMCatalogService:
    def list_models(self) -> LLMModelCatalogResponse:
        return LLMModelCatalogResponse(
            default_provider=settings.llm_provider.lower(),
            providers=[
                self._list_ollama_models(),
                self._list_openai_models(),
            ],
        )

    def _list_ollama_models(self) -> LLMProviderCatalog:
        try:
            response = requests.get(
                f"{settings.ollama_base_url.rstrip('/')}/api/tags",
                timeout=settings.llm_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            raw_models = payload.get("models", [])
            model_ids = sorted(
                {
                    str(item.get("name") or "").strip()
                    for item in raw_models
                    if str(item.get("name") or "").strip()
                }
            )
            return LLMProviderCatalog(
                provider="ollama",
                label="Ollama",
                enabled=bool(model_ids),
                default_model=self._resolve_default_model("ollama", model_ids),
                models=[LLMModelItem(id=model_id) for model_id in model_ids],
                error=None if model_ids else "Ollama 未返回可用模型。",
            )
        except Exception as exc:
            return LLMProviderCatalog(
                provider="ollama",
                label="Ollama",
                enabled=False,
                default_model=None,
                models=[],
                error=f"Ollama 模型列表获取失败：{exc}",
            )

    def _list_openai_models(self) -> LLMProviderCatalog:
        if not settings.openai_api_key:
            return LLMProviderCatalog(
                provider="openai",
                label="OpenAI",
                enabled=False,
                default_model=None,
                models=[],
                error="未配置 OPENAI_API_KEY。",
            )

        try:
            response = requests.get(
                f"{settings.openai_base_url.rstrip('/')}/models",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                },
                timeout=settings.llm_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            raw_models = payload.get("data", [])
            model_ids = sorted(
                {
                    str(item.get("id") or "").strip()
                    for item in raw_models
                    if self._is_openai_chat_model(item)
                }
            )
            return LLMProviderCatalog(
                provider="openai",
                label="OpenAI",
                enabled=bool(model_ids),
                default_model=self._resolve_default_model("openai", model_ids),
                models=[LLMModelItem(id=model_id) for model_id in model_ids],
                error=None if model_ids else "OpenAI 未返回可用于问答的模型。",
            )
        except Exception as exc:
            return LLMProviderCatalog(
                provider="openai",
                label="OpenAI",
                enabled=False,
                default_model=None,
                models=[],
                error=f"OpenAI 模型列表获取失败：{exc}",
            )

    def _resolve_default_model(self, provider: str, model_ids: list[str]) -> str | None:
        if not model_ids:
            return None
        configured_model = settings.ollama_model if provider == "ollama" else settings.openai_model
        if configured_model in model_ids:
            return configured_model
        return model_ids[0]

    def _is_openai_chat_model(self, model_item: dict[str, Any]) -> bool:
        model_id = str(model_item.get("id") or "").strip().lower()
        if not model_id:
            return False

        blocked_prefixes = (
            "text-embedding",
            "omni-moderation",
            "whisper",
            "tts",
            "dall-e",
            "gpt-image",
            "babbage",
            "davinci",
        )
        if model_id.startswith(blocked_prefixes):
            return False

        allowed_prefixes = ("gpt-", "o", "chatgpt-", "codex-")
        return model_id.startswith(allowed_prefixes)
