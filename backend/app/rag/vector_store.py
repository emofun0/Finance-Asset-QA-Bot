from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.rag.embeddings import get_embedding_function

try:
    import chromadb
except ImportError:  # pragma: no cover - optional dependency at runtime
    chromadb = None


@dataclass(frozen=True)
class VectorQueryResult:
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    score: float


class ChromaVectorStore:
    def __init__(self, persist_dir: str | Path | None = None, collection_name: str | None = None) -> None:
        self.persist_dir = Path(persist_dir or settings.rag_vector_db_dir)
        self.collection_name = collection_name or settings.rag_collection_name

    def reset(self) -> None:
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass

    def upsert(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return

        collection = self._get_collection()
        embedding_fn = get_embedding_function()
        texts = [str(record["content"]) for record in records]
        embeddings = embedding_fn.embed_documents(texts)

        collection.upsert(
            ids=[str(record["chunk_id"]) for record in records],
            documents=texts,
            metadatas=[self._sanitize_metadata(record["metadata"]) for record in records],
            embeddings=embeddings,
        )

    def query(
        self,
        *,
        query_text: str,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[VectorQueryResult]:
        if not self.persist_dir.exists():
            return []

        collection = self._get_collection()
        embedding_fn = get_embedding_function()
        query_embedding = embedding_fn.embed_query(query_text)
        response = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where or None,
        )
        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[VectorQueryResult] = []
        for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            score = 1.0 / (1.0 + float(distance))
            results.append(
                VectorQueryResult(
                    chunk_id=str(chunk_id),
                    content=str(content),
                    metadata=dict(metadata or {}),
                    score=score,
                )
            )
        return results

    def _get_client(self):
        if chromadb is None:
            raise RuntimeError("缺少 chromadb 依赖，无法进行向量存储与检索。")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(self.persist_dir))

    def _get_collection(self):
        client = self._get_client()
        return client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized
