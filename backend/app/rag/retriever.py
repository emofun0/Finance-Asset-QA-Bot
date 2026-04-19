from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.rag.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    score: float
    content: str
    metadata: dict[str, Any]


class KnowledgeRetriever:
    _report_doc_types = {
        "annual_report",
        "interim_report",
        "quarterly_financial_statements",
        "quarterly_results",
        "earnings_release",
        "quarterly_presentation",
    }

    def __init__(self, knowledge_base_dir: str | Path) -> None:
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.index_dir = self.knowledge_base_dir / "index"
        self.vector_store = ChromaVectorStore(settings.rag_vector_db_dir, settings.rag_collection_name)
        self._chunks = self._load_chunks()

    def search(
        self,
        query: str,
        top_k: int = 5,
        company: str | None = None,
        symbol: str | None = None,
        doc_type: str | None = None,
        doc_types: list[str] | None = None,
        language: str | None = None,
        chunk_kinds: list[str] | None = None,
    ) -> list[RetrievalResult]:
        if not query.strip():
            return []
        results = self._query_with_filters(
            query=query,
            top_k=max(top_k * 3, top_k),
            company=company,
            symbol=symbol,
            doc_type=doc_type,
            doc_types=doc_types,
            language=language,
            chunk_kinds=chunk_kinds,
        )
        reranked = self._rerank_results(query, results, company=company, symbol=symbol, preferred_doc_types=doc_types)
        return reranked[:top_k]

    def search_report_documents(
        self,
        query: str,
        *,
        top_k: int = 6,
        company: str | None = None,
        symbol: str | None = None,
        language: str | None = None,
    ) -> list[RetrievalResult]:
        results = self.search(
            query,
            top_k=max(top_k * 3, top_k),
            company=company,
            symbol=symbol,
            doc_types=list(self._report_doc_types),
            language=language,
            chunk_kinds=["report_metric", "report_table", "report_profile", "report_text"],
        )
        return self._rerank_report_results(query, results)[:top_k]

    def get_document_chunks(
        self,
        doc_id: str,
        *,
        chunk_kinds: list[str] | None = None,
    ) -> list[RetrievalResult]:
        selected: list[RetrievalResult] = []
        for item in self._chunks:
            metadata = item.get("metadata") or {}
            if metadata.get("doc_id") != doc_id:
                continue
            if chunk_kinds and metadata.get("chunk_kind") not in chunk_kinds:
                continue
            selected.append(
                RetrievalResult(
                    chunk_id=str(item.get("chunk_id") or ""),
                    score=0.0,
                    content=str(item.get("content") or ""),
                    metadata=dict(metadata),
                )
            )
        return selected

    def _query_with_filters(
        self,
        *,
        query: str,
        top_k: int,
        company: str | None,
        symbol: str | None,
        doc_type: str | None,
        doc_types: list[str] | None,
        language: str | None,
        chunk_kinds: list[str] | None,
    ) -> list[RetrievalResult]:
        where = self._build_where(
            company=company,
            symbol=symbol,
            doc_type=doc_type,
            doc_types=doc_types,
            language=language,
            chunk_kinds=chunk_kinds,
        )
        try:
            records = self.vector_store.query(query_text=query, top_k=top_k, where=where)
        except RuntimeError:
            return []
        return [
            RetrievalResult(
                chunk_id=record.chunk_id,
                score=record.score,
                content=record.content,
                metadata=record.metadata,
            )
            for record in records
        ]

    def _build_where(
        self,
        *,
        company: str | None,
        symbol: str | None,
        doc_type: str | None,
        doc_types: list[str] | None,
        language: str | None,
        chunk_kinds: list[str] | None,
    ) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if company:
            clauses.append({"company": {"$eq": company}})
        if symbol:
            clauses.append({"symbol": {"$eq": symbol}})
        if doc_type:
            clauses.append({"doc_type": {"$eq": doc_type}})
        elif doc_types:
            clauses.append({"doc_type": {"$in": doc_types}})
        if language:
            clauses.append({"language": {"$eq": language}})
        if chunk_kinds:
            clauses.append({"chunk_kind": {"$in": chunk_kinds}})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _rerank_results(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        company: str | None,
        symbol: str | None,
        preferred_doc_types: list[str] | None,
    ) -> list[RetrievalResult]:
        query_terms = self._extract_query_terms(query)
        doc_type_rank = {item: len(preferred_doc_types or []) - index for index, item in enumerate(preferred_doc_types or [])}

        def score(item: RetrievalResult) -> float:
            boosted = item.score
            metadata = item.metadata
            haystack = f"{metadata.get('title', '')}\n{item.content}".lower()
            if company and metadata.get("company") == company:
                boosted += 0.25
            if symbol and metadata.get("symbol") == symbol:
                boosted += 0.3
            if metadata.get("chunk_kind") == "glossary_term":
                boosted += 0.22
            if metadata.get("chunk_kind") == "report_metric":
                boosted += 0.16
            boosted += doc_type_rank.get(metadata.get("doc_type"), 0) * 0.04
            for term in query_terms:
                if term in haystack:
                    boosted += 0.06
                if metadata.get("chunk_kind") == "glossary_term" and metadata.get("term", "").lower() == term:
                    boosted += 0.28
            return boosted

        unique: list[RetrievalResult] = []
        seen: set[str] = set()
        for item in sorted(results, key=score, reverse=True):
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            unique.append(item)
        return unique

    def _rerank_report_results(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        query_terms = self._extract_query_terms(query)

        def score(item: RetrievalResult) -> float:
            boosted = item.score
            chunk_kind = str(item.metadata.get("chunk_kind") or "")
            title = str(item.metadata.get("title") or "").lower()
            content = item.content.lower()
            published_at = str(item.metadata.get("published_at") or "")
            if chunk_kind == "report_metric":
                boosted += 0.35
            elif chunk_kind == "report_table":
                boosted += 0.3
            elif chunk_kind == "report_profile":
                boosted -= 0.08
            if published_at:
                boosted += min(max(int(published_at[:4]) - 2024, 0) * 0.04, 0.24)
            if any(term in f"{title} {content}" for term in query_terms):
                boosted += 0.14
            if self._contains_numbers(item.content):
                boosted += 0.08
            if any(token in content for token in ["收入", "营收", "净利润", "经营利润", "revenue", "net income", "operating income", "cash flow", "gross margin"]):
                boosted += 0.18
            if any(token in content for token in ["company milestones", "business overview", "contents", "目录", "董事会", "founded", "listed on hong kong"]):
                boosted -= 0.35
            return boosted

        return sorted(results, key=score, reverse=True)

    def _extract_query_terms(self, query: str) -> list[str]:
        terms = re.findall(r"[A-Za-z]{3,}", query.lower())
        chinese_terms = [item for item in re.findall(r"[\u4e00-\u9fff]{2,}", query) if len(item) >= 2]
        return list(dict.fromkeys(terms + chinese_terms))

    def _contains_numbers(self, value: str) -> bool:
        return bool(re.search(r"\d", value))

    def _load_chunks(self) -> list[dict[str, Any]]:
        chunks_path = self.index_dir / "chunks.jsonl"
        if not chunks_path.exists():
            return []
        return [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def get_default_retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever(settings.knowledge_base_dir)
