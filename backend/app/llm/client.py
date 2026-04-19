from __future__ import annotations

from abc import ABC, abstractmethod
import json
from typing import Iterator, TypeVar

import requests
from pydantic import BaseModel

from app.core.config import settings
from app.llm.output_parser import parse_structured_output

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency at runtime
    OpenAI = None


TModel = TypeVar("TModel", bound=BaseModel)


class BaseLLMClient(ABC):
    @abstractmethod
    def is_enabled(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[TModel],
    ) -> TModel:
        raise NotImplementedError

    @abstractmethod
    def generate_text_stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> Iterator[str]:
        raise NotImplementedError


class NullLLMClient(BaseLLMClient):
    def is_enabled(self) -> bool:
        return False

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[TModel],
    ) -> TModel:
        raise RuntimeError("LLM client is not enabled.")

    def generate_text_stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> Iterator[str]:
        raise RuntimeError("LLM client is not enabled.")


class OllamaLLMClient(BaseLLMClient):
    def __init__(self, base_url: str, model: str, timeout_seconds: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def is_enabled(self) -> bool:
        return bool(self.model)

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[TModel],
    ) -> TModel:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        grounded_user_prompt = (
            f"{user_prompt}\n"
            "请严格按以下 JSON Schema 输出，不要输出额外说明：\n"
            f"{schema_json}"
        )
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "format": schema.model_json_schema(),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": grounded_user_prompt},
                ],
                "options": {"temperature": 0},
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        raw_text = response.json()["message"]["content"]
        return parse_structured_output(raw_text, schema)

    def generate_text_stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> Iterator[str]:
        with requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": True,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {"temperature": 0},
            },
            timeout=self.timeout_seconds,
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                payload = json.loads(line)
                message = payload.get("message") or {}
                content = str(message.get("content") or "")
                if content:
                    yield content


class OpenAILLMClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: int,
        reasoning_effort: str,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self._client = None
        if api_key and OpenAI is not None:
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout_seconds,
            )

    def is_enabled(self) -> bool:
        return self._client is not None and bool(self.model)

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[TModel],
    ) -> TModel:
        if self._client is None:
            raise RuntimeError("OpenAI client is not configured.")

        response = self._client.responses.parse(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
            text_format=schema,
            store=False,
            reasoning={"effort": self.reasoning_effort},
        )
        if response.output_parsed is None:
            raise RuntimeError("OpenAI structured response parsing returned no parsed output.")
        return response.output_parsed

    def generate_text_stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> Iterator[str]:
        if not self.api_key or not self.model:
            raise RuntimeError("OpenAI client is not configured.")

        with requests.post(
            f"{settings.openai_base_url.rstrip('/')}/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "instructions": system_prompt,
                "input": user_prompt,
                "stream": True,
                "store": False,
                "reasoning": {"effort": self.reasoning_effort},
            },
            timeout=self.timeout_seconds,
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                data = raw_line[6:].strip()
                if not data or data == "[DONE]":
                    continue
                payload = json.loads(data)
                event_type = str(payload.get("type") or "")
                if event_type == "response.output_text.delta":
                    delta = str(payload.get("delta") or "")
                    if delta:
                        yield delta
                elif event_type == "response.refusal.delta":
                    delta = str(payload.get("delta") or "")
                    if delta:
                        yield delta


def build_llm_client(provider: str | None = None, model: str | None = None) -> BaseLLMClient:
    selected_provider = (provider or settings.llm_provider).lower()

    if selected_provider == "ollama":
        return OllamaLLMClient(
            base_url=settings.ollama_base_url,
            model=model or settings.ollama_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    if selected_provider == "openai":
        return OpenAILLMClient(
            api_key=settings.openai_api_key,
            model=model or settings.openai_model,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
            reasoning_effort=settings.openai_reasoning_effort,
        )

    return NullLLMClient()
