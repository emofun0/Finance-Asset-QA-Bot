from dataclasses import dataclass

from app.schemas.domain import IntentType, RouteDecision
from app.schemas.request import ChatRequest
from app.services.knowledge_qa_service import KnowledgeQAService


@dataclass(frozen=True)
class FakeRetrievalResult:
    chunk_id: str
    score: float
    content: str
    metadata: dict


class FakeKnowledgeRetriever:
    def search(self, query: str, **kwargs: object) -> list[FakeRetrievalResult]:
        if "市盈率" in query:
            return [
                FakeRetrievalResult(
                    chunk_id="glossary::0",
                    score=0.95,
                    content="市盈率是股票价格除以每股收益后的估值倍数。",
                    metadata={
                        "title": "财务基础词条",
                        "doc_type": "glossary",
                        "url": "https://example.com/glossary",
                    },
                )
            ]

        if "腾讯" in query:
            return [
                FakeRetrievalResult(
                    chunk_id="report::0",
                    score=0.92,
                    content="总收入为人民币7,518亿元，同比增长14%。净利润为人民币2,596亿元，同比增长17%。",
                    metadata={
                        "title": "腾讯公布二零二五年度及第四季业绩",
                        "doc_type": "earnings_release",
                        "url": "https://example.com/tencent-results",
                        "company": "Tencent",
                    },
                )
            ]

        return []


class EmptyKnowledgeRetriever:
    def search(self, query: str, **kwargs: object) -> list[FakeRetrievalResult]:
        return []


class FakeWebSearchTool:
    def search_finance_knowledge(self, query: str, top_k: int = 3) -> list[FakeRetrievalResult]:
        return [
            FakeRetrievalResult(
                chunk_id="web::0",
                score=0.88,
                content="The price-to-earnings ratio compares a company's share price to its earnings per share.",
                metadata={
                    "title": "Investor.gov Glossary - Price to earnings ratio",
                    "doc_type": "glossary",
                    "url": "https://www.investor.gov/example",
                },
            )
        ]

    def search_company_reports(self, query: str, profile: object, top_k: int = 4) -> list[FakeRetrievalResult]:
        return [
            FakeRetrievalResult(
                chunk_id="web::1",
                score=0.91,
                content="Revenue was $100 billion and net income was $20 billion in the latest quarter.",
                metadata={
                    "title": "Official earnings release",
                    "doc_type": "earnings_release",
                    "url": "https://example.com/earnings",
                    "company": "NVIDIA",
                    "symbol": "NVDA",
                },
            )
        ]


def build_service() -> KnowledgeQAService:
    return KnowledgeQAService(retriever=FakeKnowledgeRetriever())


class ReportQueryExpandingRetriever:
    def search(self, query: str, **kwargs: object) -> list[FakeRetrievalResult]:
        if "latest earnings financial results" in query.lower():
            return [
                FakeRetrievalResult(
                    chunk_id="report::tsla",
                    score=0.93,
                    content="Revenue increased to $25 billion while operating income improved in the latest quarter.",
                    metadata={
                        "title": "Tesla Q1 2025 Financial Results",
                        "doc_type": "quarterly_results",
                        "url": "https://www.sec.gov/example-tsla",
                        "company": "Tesla",
                        "symbol": "TSLA",
                    },
                )
            ]
        return []


class QuarterVsAnnualWebSearchTool:
    def search_finance_knowledge(self, query: str, top_k: int = 3) -> list[FakeRetrievalResult]:
        return [
            FakeRetrievalResult(
                chunk_id="web::annual",
                score=0.9,
                content="Annual report summarizes a full fiscal year and often includes audited financial statements.",
                metadata={
                    "title": "Investor.gov Glossary - annual-report",
                    "doc_type": "glossary",
                    "url": "https://www.investor.gov/annual-report",
                },
            ),
            FakeRetrievalResult(
                chunk_id="web::quarterly",
                score=0.88,
                content="Quarterly report focuses on performance in a recent quarter and typically covers a shorter reporting period.",
                metadata={
                    "title": "Investor.gov Glossary - quarterly-report",
                    "doc_type": "glossary",
                    "url": "https://www.investor.gov/quarterly-report",
                },
            ),
        ]


class IrrelevantKnowledgeRetriever:
    def search(self, query: str, **kwargs: object) -> list[FakeRetrievalResult]:
        return [
            FakeRetrievalResult(
                chunk_id="local::0",
                score=0.2,
                content="这是一个和问题无关的财务分析通用段落。",
                metadata={
                    "title": "通用财务分析",
                    "doc_type": "knowledge_article",
                    "url": "https://example.com/local",
                },
            )
        ]


class MixedConceptRetriever:
    def search(self, query: str, **kwargs: object) -> list[FakeRetrievalResult]:
        return [
            FakeRetrievalResult(
                chunk_id="concept::0",
                score=0.91,
                content=(
                    "市盈率，也常写作 PE ratio 或 Price-to-Earnings Ratio，表示股票价格与每股收益之间的倍数关系。"
                    "市盈率通常用来衡量市场愿意为公司当前盈利支付多高的价格。"
                    "收入又称营业收入，是利润计算的起点。净利润是扣除成本和税费后的最终利润。"
                ),
                metadata={
                    "title": "财务基础词条",
                    "doc_type": "glossary",
                    "url": "https://example.com/glossary",
                },
            )
        ]


def test_finance_knowledge_answer_contains_source() -> None:
    service = build_service()

    response = service.answer(
        ChatRequest(message="什么是市盈率？"),
        RouteDecision(
            intent=IntentType.FINANCE_KNOWLEDGE,
            need_rag=True,
            reason="test",
        ),
    )

    assert response.summary.startswith("市盈率通常表示")
    assert response.objective_data["matched_chunks"] == 1
    assert response.sources[0].name == "财务基础词条"


def test_finance_knowledge_excerpt_prefers_concise_local_window() -> None:
    service = KnowledgeQAService(retriever=MixedConceptRetriever())

    response = service.answer(
        ChatRequest(message="什么是市盈率？"),
        RouteDecision(
            intent=IntentType.FINANCE_KNOWLEDGE,
            need_rag=True,
            reason="test",
        ),
    )

    assert "市盈率" in response.analysis[0]
    assert "净利润" not in response.analysis[0]


def test_report_summary_extracts_financial_highlights() -> None:
    service = build_service()

    response = service.answer(
        ChatRequest(message="腾讯最近财报摘要是什么？"),
        RouteDecision(
            intent=IntentType.REPORT_SUMMARY,
            need_rag=True,
            extracted_company="Tencent",
            extracted_symbol="0700.HK",
            reason="test",
        ),
    )

    assert response.summary.startswith("根据检索到的官方财报资料")
    assert any("收入" in item for item in response.analysis)
    assert response.sources[0].type == "earnings_release"


def test_report_summary_normalizes_company_alias_before_local_retrieval() -> None:
    service = build_service()

    response = service.answer(
        ChatRequest(message="腾讯最近财报摘要是什么？"),
        RouteDecision(
            intent=IntentType.REPORT_SUMMARY,
            need_rag=True,
            extracted_company="腾讯",
            reason="test",
        ),
    )

    assert response.route.extracted_company == "Tencent"
    assert response.route.extracted_symbol == "0700.HK"
    assert response.objective_data["source_mode"] == "local_rag"
    assert any("收入" in item for item in response.analysis)


def test_report_summary_keeps_strong_local_report_without_web_fallback() -> None:
    service = KnowledgeQAService(
        retriever=FakeKnowledgeRetriever(),
        web_search_tool=FakeWebSearchTool(),
    )

    response = service.answer(
        ChatRequest(message="腾讯最近财报摘要是什么？"),
        RouteDecision(
            intent=IntentType.REPORT_SUMMARY,
            need_rag=True,
            extracted_company="Tencent",
            extracted_symbol="0700.HK",
            reason="test",
        ),
    )

    assert response.objective_data["source_mode"] == "local_rag"
    assert response.sources[0].name == "腾讯公布二零二五年度及第四季业绩"


def test_finance_knowledge_can_fall_back_to_web_search() -> None:
    service = KnowledgeQAService(
        retriever=EmptyKnowledgeRetriever(),
        web_search_tool=FakeWebSearchTool(),
    )

    response = service.answer(
        ChatRequest(message="什么是市盈率？"),
        RouteDecision(
            intent=IntentType.FINANCE_KNOWLEDGE,
            need_rag=True,
            reason="test",
        ),
    )

    assert response.objective_data["source_mode"] == "web_fallback"
    assert response.sources[0].name.startswith("Investor.gov Glossary")


def test_finance_knowledge_uses_web_when_local_result_is_irrelevant() -> None:
    service = KnowledgeQAService(
        retriever=IrrelevantKnowledgeRetriever(),
        web_search_tool=FakeWebSearchTool(),
    )

    response = service.answer(
        ChatRequest(message="什么是 beta coefficient？"),
        RouteDecision(
            intent=IntentType.FINANCE_KNOWLEDGE,
            need_rag=True,
            reason="test",
        ),
    )

    assert response.objective_data["source_mode"] == "web_fallback"


def test_finance_knowledge_can_answer_quarterly_vs_annual_report_difference() -> None:
    service = KnowledgeQAService(
        retriever=EmptyKnowledgeRetriever(),
        web_search_tool=QuarterVsAnnualWebSearchTool(),
    )

    response = service.answer(
        ChatRequest(message="季度报告和年报有什么区别？"),
        RouteDecision(
            intent=IntentType.FINANCE_KNOWLEDGE,
            need_rag=True,
            reason="test",
        ),
    )

    assert "季度报告通常聚焦最近一个季度" in response.summary
    assert response.objective_data["source_mode"] == "web_fallback"


def test_report_summary_can_fall_back_to_web_search() -> None:
    service = KnowledgeQAService(
        retriever=EmptyKnowledgeRetriever(),
        web_search_tool=FakeWebSearchTool(),
    )

    response = service.answer(
        ChatRequest(message="英伟达最近财报摘要是什么？"),
        RouteDecision(
            intent=IntentType.REPORT_SUMMARY,
            need_rag=True,
            extracted_company="NVIDIA",
            extracted_symbol="NVDA",
            reason="test",
        ),
    )

    assert response.objective_data["source_mode"] == "web_fallback"
    assert response.sources[0].name == "Official earnings release"


def test_report_summary_uses_expanded_query_for_latest_earnings_questions() -> None:
    service = KnowledgeQAService(retriever=ReportQueryExpandingRetriever())

    response = service.answer(
        ChatRequest(message="Tesla latest earnings summary"),
        RouteDecision(
            intent=IntentType.REPORT_SUMMARY,
            need_rag=True,
            extracted_company="Tesla",
            extracted_symbol="TSLA",
            reason="test",
        ),
    )

    assert response.objective_data["matched_chunks"] == 1
    assert response.sources[0].name == "Tesla Q1 2025 Financial Results"
