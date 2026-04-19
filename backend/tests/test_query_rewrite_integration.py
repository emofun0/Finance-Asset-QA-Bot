from app.rag.retriever import RetrievalResult
from app.schemas.domain import IntentType, RouteDecision
from app.schemas.request import ChatRequest
from app.services.knowledge_qa_service import KnowledgeQAService


class StubRagSearchTool:
    def __init__(self) -> None:
        self.search_queries: list[str] = []
        self.report_queries: list[str] = []

    def search(self, request) -> list[RetrievalResult]:
        self.search_queries.append(request.query)
        return [
            RetrievalResult(
                chunk_id="doc-1::chunk-0",
                score=0.9,
                content="市盈率是股票价格与每股收益之间的比率。",
                metadata={
                    "title": "市盈率基础定义",
                    "doc_type": "glossary",
                    "url": "https://example.com/pe",
                    "language": "zh",
                },
            )
        ]

    def search_report_documents(self, request) -> list[RetrievalResult]:
        self.report_queries.append(request.query)
        return [
            RetrievalResult(
                chunk_id="report::doc-1::snippet-0",
                score=1.1,
                content="营业收入 1000，同比 10%；净利润 200，同比 8%。",
                metadata={
                    "title": "Tencent quarterly results",
                    "doc_type": "quarterly_results",
                    "url": "https://example.com/report",
                    "language": "zh",
                    "company": "Tencent",
                    "symbol": "0700.HK",
                },
            )
        ]


class StubQueryRewriteService:
    def __init__(self, rewritten_query: str) -> None:
        self.rewritten_query = rewritten_query
        self.calls: list[tuple[str, IntentType]] = []

    def rewrite(self, message: str, route: RouteDecision) -> str:
        self.calls.append((message, route.intent))
        return self.rewritten_query


def test_knowledge_qa_service_uses_query_rewrite_for_finance_knowledge() -> None:
    rag_search_tool = StubRagSearchTool()
    rewrite_service = StubQueryRewriteService("price to earnings ratio definition")
    service = KnowledgeQAService(
        rag_search_tool=rag_search_tool,
        query_rewrite_service=rewrite_service,
    )
    request = ChatRequest(message="什么是市盈率？")
    route = RouteDecision(intent=IntentType.FINANCE_KNOWLEDGE, need_rag=True)

    service.answer(request, route)

    assert rewrite_service.calls == [("什么是市盈率？", IntentType.FINANCE_KNOWLEDGE)]
    assert rag_search_tool.search_queries == ["price to earnings ratio definition"]


def test_knowledge_qa_service_keeps_agent_rewritten_query_priority() -> None:
    rag_search_tool = StubRagSearchTool()
    rewrite_service = StubQueryRewriteService("this should not be used")
    service = KnowledgeQAService(
        rag_search_tool=rag_search_tool,
        query_rewrite_service=rewrite_service,
    )
    request = ChatRequest(
        message="总结一下腾讯最新季度财报",
        metadata={"retrieval_query": "Tencent 0700.HK latest quarterly results earnings release"},
    )
    route = RouteDecision(
        intent=IntentType.REPORT_SUMMARY,
        need_rag=True,
        extracted_company="Tencent",
        extracted_symbol="0700.HK",
    )

    service.answer(request, route)

    assert rewrite_service.calls == []
    assert rag_search_tool.report_queries == ["Tencent 0700.HK latest quarterly results earnings release"]
