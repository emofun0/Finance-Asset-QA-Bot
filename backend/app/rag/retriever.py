from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from app.core.config import settings
from app.rag.vector_store import LocalVectorStore


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    score: float
    content: str
    metadata: dict[str, Any]


class KnowledgeRetriever:
    def __init__(self, knowledge_base_dir: str | Path) -> None:
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.index_dir = self.knowledge_base_dir / "index"
        self._chunks = self._load_chunks()
        self._store = LocalVectorStore(self.index_dir)
        self._index = None
        if self._store.vectorizer_path.exists() and self._store.matrix_path.exists():
            self._index = self._store.load()

    def search(
        self,
        query: str,
        top_k: int = 5,
        company: str | None = None,
        symbol: str | None = None,
        doc_type: str | None = None,
        doc_types: list[str] | None = None,
        language: str | None = None,
    ) -> list[RetrievalResult]:
        if not self._chunks or self._index is None:
            return []

        expanded_query = self._expand_query(query)
        query_vector = self._index.vectorizer.transform([expanded_query])
        similarities = (self._index.matrix @ query_vector.T).toarray().ravel()
        query_terms = self._extract_query_terms(query)
        is_chinese_query = self._contains_chinese(query)
        doc_type_priority = {item: len(doc_types) - index for index, item in enumerate(doc_types or [])}

        candidates: list[RetrievalResult] = []
        for item, score in zip(self._chunks, similarities):
            if score <= 0:
                continue
            metadata = item["metadata"]
            if company and metadata.get("company") != company:
                continue
            if symbol and metadata.get("symbol") != symbol:
                continue
            if doc_type and metadata.get("doc_type") != doc_type:
                continue
            if doc_types and metadata.get("doc_type") not in doc_types:
                continue
            if language and metadata.get("language") != language:
                continue
            boosted_score = float(score)
            if company and metadata.get("company") == company:
                boosted_score += 0.15
            if symbol and metadata.get("symbol") == symbol:
                boosted_score += 0.2
            if is_chinese_query and metadata.get("language") == "zh":
                boosted_score += 0.08
            boosted_score += doc_type_priority.get(metadata.get("doc_type"), 0) * 0.05
            boosted_score += self._keyword_boost(
                content=item["content"],
                title=metadata.get("title", ""),
                query_terms=query_terms,
            )
            candidates.append(
                RetrievalResult(
                    chunk_id=item["chunk_id"],
                    score=boosted_score,
                    content=item["content"],
                    metadata=metadata,
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:top_k]

    def _load_chunks(self) -> list[dict[str, Any]]:
        chunks_path = self.index_dir / "chunks.jsonl"
        if not chunks_path.exists():
            return []
        return [
            json.loads(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _expand_query(self, query: str) -> str:
        aliases = {
            "市盈率": "市盈率 price earnings ratio pe ratio",
            "净利润": "净利润 net income profit",
            "营业收入": "营业收入 revenue net sales",
            "收入": "收入 revenue",
            "财报": "财报 annual report interim report earnings results quarterly report",
            "季度": "季度 quarter quarterly interim",
            "年报": "年报 annual report",
            "半年报": "半年报 interim report",
        }
        expanded = query
        for key, value in aliases.items():
            if key in query:
                expanded += f" {value}"
        return expanded

    def _extract_query_terms(self, query: str) -> list[str]:
        known_terms = [
            "市盈率",
            "每股收益",
            "净利润",
            "归母净利润",
            "营业收入",
            "收入",
            "营收",
            "毛利",
            "现金流",
            "年报",
            "半年报",
            "中报",
            "季报",
            "季度",
            "财报",
        ]
        matched = [term for term in known_terms if term in query]
        english_terms = re.findall(r"[A-Za-z]{3,}", query.lower())
        return matched + english_terms

    def _keyword_boost(self, content: str, title: str, query_terms: list[str]) -> float:
        boost = 0.0
        lowered_content = content.lower()
        lowered_title = title.lower()
        for term in query_terms:
            lowered_term = term.lower()
            if lowered_term in lowered_title:
                boost += 0.12
            elif lowered_term in lowered_content:
                boost += 0.04
        return min(boost, 0.28)

    def _contains_chinese(self, value: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", value))


def get_default_retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever(settings.knowledge_base_dir)
