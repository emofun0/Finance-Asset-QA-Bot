from __future__ import annotations

from dataclasses import dataclass

from app.observability.request_trace import trace_event
from app.rag.retriever import KnowledgeRetriever, RetrievalResult


@dataclass(frozen=True)
class RagSearchRequest:
    query: str
    top_k: int = 6
    company: str | None = None
    symbol: str | None = None
    language: str | None = None
    doc_type: str | None = None
    doc_types: list[str] | None = None


class RagSearchTool:
    def __init__(self, retriever: KnowledgeRetriever) -> None:
        self.retriever = retriever

    def search(self, request: RagSearchRequest) -> list[RetrievalResult]:
        trace_event(
            "tool.rag_search",
            {
                "query": request.query,
                "top_k": request.top_k,
                "company": request.company,
                "symbol": request.symbol,
                "language": request.language,
                "doc_type": request.doc_type,
                "doc_types": request.doc_types,
            },
        )
        return self.retriever.search(
            request.query,
            top_k=request.top_k,
            company=request.company,
            symbol=request.symbol,
            doc_type=request.doc_type,
            doc_types=request.doc_types,
            language=request.language,
        )

    def search_report_documents(self, request: RagSearchRequest) -> list[RetrievalResult]:
        trace_event(
            "tool.rag_report_search",
            {
                "query": request.query,
                "top_k": request.top_k,
                "company": request.company,
                "symbol": request.symbol,
                "language": request.language,
            },
        )
        return self.retriever.search_report_documents(
            request.query,
            top_k=request.top_k,
            company=request.company,
            symbol=request.symbol,
            language=request.language,
        )
