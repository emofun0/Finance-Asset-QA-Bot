from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import requests

from app.core.config import settings

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency at runtime
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency at runtime
    SentenceTransformer = None


class DenseEmbeddingFunction(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class SentenceTransformerEmbeddingFunction(DenseEmbeddingFunction):
    def __init__(self, model_name: str) -> None:
        if SentenceTransformer is None:
            raise RuntimeError("缺少 sentence-transformers 依赖，无法使用本地 dense embedding。")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=settings.rag_embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OllamaEmbeddingFunction(DenseEmbeddingFunction):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.base_url = settings.ollama_base_url.rstrip("/")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": text},
                timeout=settings.llm_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            embeddings = payload.get("embeddings") or []
            if not embeddings:
                raise RuntimeError("Ollama embedding 响应为空。")
            vectors.append(embeddings[0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbeddingFunction(DenseEmbeddingFunction):
    def __init__(self, model_name: str) -> None:
        if OpenAI is None or not settings.openai_api_key:
            raise RuntimeError("OpenAI embedding 未配置。")
        self.model_name = model_name
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_seconds,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=4)
def get_embedding_function(
    provider: str | None = None,
    model_name: str | None = None,
) -> DenseEmbeddingFunction:
    selected_provider = (provider or settings.rag_embedding_provider).strip().lower()
    selected_model = (model_name or settings.rag_embedding_model).strip()

    if selected_provider == "ollama":
        return OllamaEmbeddingFunction(selected_model)
    if selected_provider == "openai":
        return OpenAIEmbeddingFunction(selected_model)
    return SentenceTransformerEmbeddingFunction(selected_model)


def build_embedding_debug_info() -> dict[str, Any]:
    return {
        "provider": settings.rag_embedding_provider,
        "model": settings.rag_embedding_model,
        "batch_size": settings.rag_embedding_batch_size,
    }
