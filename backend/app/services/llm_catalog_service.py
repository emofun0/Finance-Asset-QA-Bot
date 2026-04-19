from __future__ import annotations

import re
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
            model_ids = self._build_openai_model_ids(raw_models)
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
        normalized_configured_model = self._strip_openai_date_suffix(configured_model)
        if provider == "openai" and normalized_configured_model in model_ids:
            return normalized_configured_model
        return model_ids[0]

    def _build_openai_model_ids(self, raw_models: list[dict[str, Any]]) -> list[str]:
        filtered_ids = {
            str(item.get("id") or "").strip()
            for item in raw_models
            if self._is_openai_chat_model(item)
        }
        undated_ids = {
            model_id
            for model_id in filtered_ids
            if model_id and model_id == self._strip_openai_date_suffix(model_id)
        }

        deduped_ids: set[str] = set()
        for model_id in filtered_ids:
            normalized_id = self._strip_openai_date_suffix(model_id)
            if not normalized_id:
                continue
            if normalized_id in undated_ids:
                deduped_ids.add(normalized_id)
                continue
            if model_id == normalized_id:
                deduped_ids.add(model_id)

        return sorted(deduped_ids)

    def _is_openai_chat_model(self, model_item: dict[str, Any]) -> bool:
        model_id = str(model_item.get("id") or "").strip().lower()
        if not model_id:
            return False

        if model_id.endswith("-latest"):
            return False
        if re.search(r"^gpt-\d(?:\.\d)?-\d{4}$", model_id):
            return False

        blocked_exact_names = {
            "chatgpt-latest",
            "gpt-image-1",
            "gpt-image-latest",
        }
        if model_id in blocked_exact_names:
            return False

        blocked_prefixes = (
            "text-embedding",
            "omni-moderation",
            "whisper",
            "tts",
            "dall-e",
            "gpt-image",
            "gpt-3.5",
            "babbage",
            "davinci",
            "codex",
        )
        if model_id.startswith(blocked_prefixes):
            return False

        blocked_keywords = (
            "audio",
            "realtime",
            "transcribe",
            "search-preview",
            "moderation",
            "embed",
            "vision-preview",
            "instruct",
            "codex",
            "tts",
            "image",
        )
        if any(keyword in model_id for keyword in blocked_keywords):
            return False

        allowed_prefixes = ("gpt-", "o", "chatgpt-")
        return model_id.startswith(allowed_prefixes)

    def _strip_openai_date_suffix(self, model_id: str) -> str:
        return re.sub(r"-20\d{2}-\d{2}-\d{2}$", "", model_id.strip())
