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
                chunk_id="doc-1::term-0",
                score=0.9,
                content="术语：市盈率\n定义：市盈率是股票价格与每股收益之间的比率。",
                metadata={
                    "title": "市盈率基础定义",
                    "doc_type": "glossary",
                    "chunk_kind": "glossary_term",
                    "term": "市盈率",
                    "url": "https://example.com/pe",
                    "language": "zh",
                },
            )
        ]

    def search_report_documents(self, request) -> list[RetrievalResult]:
        self.report_queries.append(request.query)
        return [
            RetrievalResult(
                chunk_id="report::doc-1::metric-0",
                score=1.1,
                content="营业收入 1000，同比 10%；净利润 200，同比 8%。",
                metadata={
                    "doc_id": "doc-1",
                    "title": "Tencent quarterly results",
                    "doc_type": "quarterly_results",
                    "chunk_kind": "report_metric",
                    "url": "https://example.com/report",
                    "language": "zh",
                    "company": "Tencent",
                    "symbol": "0700.HK",
                },
            )
        ]


class StubRetriever:
    def get_document_chunks(self, doc_id: str, *, chunk_kinds=None) -> list[RetrievalResult]:
        assert doc_id == "doc-1"
        return [
            RetrievalResult(
                chunk_id="doc-1::profile",
                score=0.0,
                content="公司：Tencent\n股票代码：0700.HK\n标题：Tencent quarterly results",
                metadata={"doc_id": "doc-1", "chunk_kind": "report_profile", "title": "Tencent quarterly results"},
            ),
            RetrievalResult(
                chunk_id="doc-1::metric-0",
                score=0.0,
                content="营业收入 1000，同比 10%。",
                metadata={"doc_id": "doc-1", "chunk_kind": "report_metric", "title": "Tencent quarterly results"},
            ),
            RetrievalResult(
                chunk_id="doc-1::metric-1",
                score=0.0,
                content="净利润 200，同比 8%。",
                metadata={"doc_id": "doc-1", "chunk_kind": "report_metric", "title": "Tencent quarterly results"},
            ),
            RetrievalResult(
                chunk_id="doc-1::table-0",
                score=0.0,
                content="收入 1000 900\n净利润 200 185",
                metadata={"doc_id": "doc-1", "chunk_kind": "report_table", "title": "Tencent quarterly results"},
            ),
        ]


class EmptyRagSearchTool(StubRagSearchTool):
    def search(self, request) -> list[RetrievalResult]:
        self.search_queries.append(request.query)
        return []

    def search_report_documents(self, request) -> list[RetrievalResult]:
        self.report_queries.append(request.query)
        return []


class IrrelevantKnowledgeRagSearchTool(StubRagSearchTool):
    def search(self, request) -> list[RetrievalResult]:
        self.search_queries.append(request.query)
        return [
            RetrievalResult(
                chunk_id="doc-2::chunk-0",
                score=0.88,
                content="盈利能力主要分析指标包括销售毛利率、销售净利率、资产报酬率、净资产收益率等。",
                metadata={
                    "title": "财务分析的基本内容",
                    "doc_type": "knowledge_article",
                    "chunk_kind": "text_chunk",
                    "url": "https://example.com/finance-analysis",
                    "language": "zh",
                },
            )
        ]


class StubWebSearchTool:
    def __init__(self) -> None:
        self.finance_queries: list[str] = []
        self.report_queries: list[str] = []
        self.generic_report_queries: list[str] = []

    def search_finance_knowledge(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        self.finance_queries.append(query)
        return [
            RetrievalResult(
                chunk_id="web::0",
                score=1.0,
                content="PE ratio, or price-to-earnings ratio, measures price relative to earnings.",
                metadata={
                    "title": "Investopedia PE Ratio",
                    "doc_type": "glossary",
                    "url": "https://example.com/pe-web",
                    "language": "en",
                },
            )
        ]

    def search_company_reports(self, query: str, profile, top_k: int = 3) -> list[RetrievalResult]:
        self.report_queries.append(query)
        return [
            RetrievalResult(
                chunk_id="web::1",
                score=1.0,
                content="Revenue was RMB 1000 and net profit was RMB 200.",
                metadata={
                    "title": "Tencent Investor Relations",
                    "doc_type": "web_search",
                    "url": "https://example.com/report-web",
                    "language": "en",
                    "company": "Tencent",
                    "symbol": "0700.HK",
                },
            )
        ]

    def search_company_reports_by_query(
        self,
        query: str,
        *,
        company: str | None = None,
        symbol: str | None = None,
        top_k: int = 3,
    ) -> list[RetrievalResult]:
        self.generic_report_queries.append(query)
        return [
            RetrievalResult(
                chunk_id="web::2",
                score=1.0,
                content="Revenue was RMB 1000 and net profit was RMB 200.",
                metadata={
                    "title": "Generic Report Search",
                    "doc_type": "web_search",
                    "url": "https://example.com/report-web-generic",
                    "language": "en",
                    "company": company,
                    "symbol": symbol,
                },
            )
        ]


def test_knowledge_qa_service_uses_agent_retrieval_query_for_finance_knowledge() -> None:
    rag_search_tool = StubRagSearchTool()
    service = KnowledgeQAService(rag_search_tool=rag_search_tool)
    request = ChatRequest(
        message="什么是市盈率？",
        metadata={"retrieval_query": "price to earnings ratio definition"},
    )
    route = RouteDecision(intent=IntentType.FINANCE_KNOWLEDGE, need_rag=True)

    service.answer(request, route)

    assert rag_search_tool.search_queries == ["price to earnings ratio definition"]


def test_knowledge_qa_service_uses_agent_retrieval_query_for_report_summary() -> None:
    rag_search_tool = StubRagSearchTool()
    service = KnowledgeQAService(rag_search_tool=rag_search_tool)
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

    assert rag_search_tool.report_queries == ["Tencent 0700.HK latest quarterly results earnings release"]


def test_report_summary_includes_report_context_for_llm() -> None:
    rag_search_tool = StubRagSearchTool()
    service = KnowledgeQAService(rag_search_tool=rag_search_tool, retriever=StubRetriever())
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

    answer = service.answer(request, route)

    report_context = answer.objective_data["report_context"]
    assert report_context["doc_id"] == "doc-1"
    assert "营业收入 1000，同比 10%。" in report_context["metric_lines"]
    assert "净利润 200，同比 8%。" in report_context["metric_lines"]
    assert any("收入 1000 900" in row for row in report_context["table_rows"])


def test_knowledge_qa_service_falls_back_to_web_for_finance_knowledge_when_rag_fails() -> None:
    rag_search_tool = EmptyRagSearchTool()
    web_search_tool = StubWebSearchTool()
    service = KnowledgeQAService(rag_search_tool=rag_search_tool, web_search_tool=web_search_tool)
    request = ChatRequest(
        message="什么是市盈率？",
        metadata={"retrieval_query": "price to earnings ratio definition"},
    )
    route = RouteDecision(intent=IntentType.FINANCE_KNOWLEDGE, need_rag=True)

    answer = service.answer(request, route)

    assert rag_search_tool.search_queries == ["price to earnings ratio definition"]
    assert web_search_tool.finance_queries == ["price to earnings ratio definition"]
    assert answer.objective_data["source_mode"] == "web_fallback"


def test_knowledge_qa_service_falls_back_to_web_for_irrelevant_local_knowledge_hits() -> None:
    rag_search_tool = IrrelevantKnowledgeRagSearchTool()
    web_search_tool = StubWebSearchTool()
    service = KnowledgeQAService(rag_search_tool=rag_search_tool, web_search_tool=web_search_tool)
    request = ChatRequest(
        message="什么是ebita",
        metadata={"retrieval_query": "EBITA definition meaning earnings before interest taxes and amortization"},
    )
    route = RouteDecision(intent=IntentType.FINANCE_KNOWLEDGE, need_rag=True)

    answer = service.answer(request, route)

    assert rag_search_tool.search_queries == ["EBITA definition meaning earnings before interest taxes and amortization"]
    assert web_search_tool.finance_queries == ["EBITA definition meaning earnings before interest taxes and amortization"]
    assert answer.objective_data["source_mode"] == "web_fallback"


def test_knowledge_qa_service_falls_back_to_web_for_report_summary_when_rag_fails() -> None:
    rag_search_tool = EmptyRagSearchTool()
    web_search_tool = StubWebSearchTool()
    service = KnowledgeQAService(rag_search_tool=rag_search_tool, web_search_tool=web_search_tool)
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

    answer = service.answer(request, route)

    assert rag_search_tool.report_queries == ["Tencent 0700.HK latest quarterly results earnings release"]
    assert web_search_tool.report_queries == ["Tencent 0700.HK latest quarterly results earnings release"]
    assert answer.objective_data["source_mode"] == "web_fallback"


def test_knowledge_qa_service_falls_back_to_generic_web_report_search_without_profile() -> None:
    rag_search_tool = EmptyRagSearchTool()
    web_search_tool = StubWebSearchTool()
    service = KnowledgeQAService(rag_search_tool=rag_search_tool, web_search_tool=web_search_tool)
    request = ChatRequest(
        message="总结一下腾讯最新季度财报",
        metadata={"retrieval_query": "腾讯 最新季度财报 revenue net profit"},
    )
    route = RouteDecision(
        intent=IntentType.REPORT_SUMMARY,
        need_rag=True,
        extracted_company="腾讯",
        extracted_symbol=None,
    )

    answer = service.answer(request, route)

    assert rag_search_tool.report_queries == ["腾讯 最新季度财报 revenue net profit"]
    assert web_search_tool.generic_report_queries == ["腾讯 最新季度财报 revenue net profit"]
    assert answer.objective_data["source_mode"] == "web_fallback"
