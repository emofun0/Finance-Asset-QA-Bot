from pydantic import BaseModel

from app.llm.client import BaseLLMClient
from app.llm.contracts import GeneratedAnswerSections, QueryRewriteResult, VerificationResult
from app.schemas.domain import IntentType, RouteDecision
from app.schemas.response import AnswerPayload, SourceItem
from app.services.answer_generation_service import AnswerGenerationService
from app.services.query_rewrite_service import QueryRewriteService
from app.services.verification_service import VerificationService


class FakeLLMClient(BaseLLMClient):
    def __init__(self, result: BaseModel) -> None:
        self.result = result

    def is_enabled(self) -> bool:
        return True

    def generate_structured(self, *, system_prompt: str, user_prompt: str, schema: type[BaseModel]):
        assert system_prompt
        assert user_prompt
        return self.result

    def generate_text_stream(self, *, system_prompt: str, user_prompt: str):
        assert system_prompt
        assert user_prompt
        yield ""


def build_answer_payload() -> AnswerPayload:
    return AnswerPayload(
        question_type=IntentType.ASSET_PRICE,
        request_message="英伟达股价是多少？",
        summary="NVDA 最新价格约为 100.0 USD",
        objective_data={"symbol": "NVDA", "latest_price": 100.0},
        analysis=["原始分析"],
        sources=[SourceItem(type="market_data", name="Yahoo Finance", value="NVDA")],
        limitations=["原始限制"],
        route=RouteDecision(intent=IntentType.ASSET_PRICE, need_market_data=True, extracted_symbol="NVDA", reason="test"),
    )


def build_report_payload() -> AnswerPayload:
    return AnswerPayload(
        question_type=IntentType.REPORT_SUMMARY,
        request_message="英伟达最近财报摘要是什么？",
        summary="根据检索到的官方财报资料，NVIDIA 最近披露材料的核心财务亮点已提取。",
        objective_data={
            "retrieval_enabled": True,
            "source_mode": "web_fallback",
            "matched_chunks": 3,
            "company": "NVIDIA",
            "symbol": "NVDA",
        },
        analysis=["营收同比增长。"],
        sources=[SourceItem(type="earnings_release", name="NVIDIA earnings release", value="https://example.com/nvda")],
        limitations=["当前摘要基于检索到的财报片段，不保证覆盖整份报告全部重点。"],
        route=RouteDecision(intent=IntentType.REPORT_SUMMARY, need_rag=True, extracted_company="NVIDIA", extracted_symbol="NVDA", reason="test"),
    )


def test_query_rewrite_service_returns_rewritten_query() -> None:
    service = QueryRewriteService(
        llm_client=FakeLLMClient(
            QueryRewriteResult(
                rewritten_query="NVIDIA NVDA latest earnings release quarterly report",
                search_keywords=["NVIDIA", "NVDA"],
                notes=[],
            )
        )
    )

    rewritten = service.rewrite(
        "英伟达最近财报摘要是什么？",
        RouteDecision(intent=IntentType.REPORT_SUMMARY, need_rag=True, extracted_company="NVIDIA", extracted_symbol="NVDA", reason="test"),
    )

    assert rewritten.startswith("NVIDIA NVDA")


def test_query_rewrite_service_skips_simple_finance_knowledge() -> None:
    service = QueryRewriteService(
        llm_client=FakeLLMClient(
            QueryRewriteResult(
                rewritten_query="不应被使用的改写",
                search_keywords=[],
                notes=[],
            )
        )
    )

    rewritten = service.rewrite(
        "什么是市盈率？",
        RouteDecision(intent=IntentType.FINANCE_KNOWLEDGE, need_rag=True, reason="test"),
    )

    assert rewritten == "什么是市盈率？"


def test_answer_generation_service_replaces_summary_and_sections() -> None:
    service = AnswerGenerationService(
        llm_client=FakeLLMClient(
            GeneratedAnswerSections(
                summary="NVDA 当前价格维持在高位区间。",
                analysis=["价格高于前收盘价。", "当前回答基于市场数据。"],
                limitations=["仅反映已获取的市场数据。"],
            )
        )
    )

    result = service.generate(
        request_message="英伟达最近财报摘要是什么？",
        route=RouteDecision(intent=IntentType.REPORT_SUMMARY, need_rag=True, extracted_company="NVIDIA", extracted_symbol="NVDA", reason="test"),
        draft_answer=build_report_payload(),
    )

    assert result.summary == "NVDA 当前价格维持在高位区间。"
    assert result.analysis[0] == "价格高于前收盘价。"


def test_answer_generation_service_skips_asset_answers() -> None:
    payload = build_answer_payload()
    service = AnswerGenerationService(
        llm_client=FakeLLMClient(
            GeneratedAnswerSections(
                summary="模型改写后的价格摘要。",
                analysis=["模型改写后的分析。"],
                limitations=["模型改写后的限制。"],
            )
        )
    )

    result = service.generate(
        request_message=payload.request_message,
        route=payload.route,
        draft_answer=payload,
    )

    assert result.summary == payload.summary
    assert result.analysis == payload.analysis


def test_verification_service_applies_llm_corrections() -> None:
    service = VerificationService(
        llm_client=FakeLLMClient(
            VerificationResult(
                is_valid=False,
                issues=["摘要过长"],
                corrected_summary="NVDA：价格信息已校正。",
                corrected_analysis=["校验后的分析"],
                corrected_limitations=["校验后的限制"],
            )
        )
    )

    verified = service.verify(build_answer_payload())

    assert verified.summary == "NVDA：价格信息已校正。"
    assert verified.analysis == ["校验后的分析"]
    assert verified.limitations == ["校验后的限制"]


def test_answer_generation_service_skips_generation_without_evidence() -> None:
    payload = AnswerPayload(
        question_type=IntentType.FINANCE_KNOWLEDGE,
        request_message="什么是市盈率？",
        summary="知识库中未检索到足够依据，当前无法可靠回答该问题。",
        objective_data={"retrieval_enabled": True, "source_mode": "not_found", "matched_chunks": 0},
        analysis=["当前问题未在本地知识库或官方网页检索结果中找到高相关片段。"],
        sources=[],
        limitations=["当前回答依赖本地知识库和官方网页检索，不会在无依据时自由生成结论。"],
        route=RouteDecision(intent=IntentType.FINANCE_KNOWLEDGE, need_rag=True, reason="test"),
    )
    service = AnswerGenerationService(
        llm_client=FakeLLMClient(
            GeneratedAnswerSections(
                summary="市盈率是一个重要估值指标。",
                analysis=["模型自由发挥的分析。"],
                limitations=["模型自由发挥的限制。"],
            )
        )
    )

    result = service.generate(
        request_message="什么是市盈率？",
        route=payload.route,
        draft_answer=payload,
    )

    assert result.summary == payload.summary
    assert result.sources == []


def test_answer_generation_service_skips_finance_knowledge_answers() -> None:
    payload = AnswerPayload(
        question_type=IntentType.FINANCE_KNOWLEDGE,
        request_message="什么是市盈率？",
        summary="市盈率通常表示股票价格相对于每股收益的估值倍数。",
        objective_data={"retrieval_enabled": True, "source_mode": "local_rag", "matched_chunks": 1},
        analysis=["市盈率常用于衡量市场如何给公司盈利定价。"],
        sources=[SourceItem(type="glossary", name="财务基础词条", value="https://example.com/glossary")],
        limitations=["当前版本基于检索片段做抽取式归纳。"],
        route=RouteDecision(intent=IntentType.FINANCE_KNOWLEDGE, need_rag=True, reason="test"),
    )
    service = AnswerGenerationService(
        llm_client=FakeLLMClient(
            GeneratedAnswerSections(
                summary="模型改写后的解释。",
                analysis=["模型改写后的分析。"],
                limitations=["模型改写后的限制。"],
            )
        )
    )

    result = service.generate(
        request_message=payload.request_message,
        route=payload.route,
        draft_answer=payload,
    )

    assert result.summary == payload.summary
    assert result.analysis == payload.analysis


def test_verification_service_forces_insufficient_evidence_for_empty_source_knowledge_answer() -> None:
    payload = AnswerPayload(
        question_type=IntentType.FINANCE_KNOWLEDGE,
        request_message="什么是市盈率？",
        summary="市盈率是股票价格除以每股收益。",
        objective_data={"retrieval_enabled": True, "source_mode": "not_found", "matched_chunks": 0},
        analysis=["这是模型在无来源时补充的解释。"],
        sources=[],
        limitations=["原始限制。"],
        route=RouteDecision(intent=IntentType.FINANCE_KNOWLEDGE, need_rag=True, reason="test"),
    )

    verified = VerificationService().verify(payload)

    assert "无法可靠回答" in verified.summary
    assert verified.sources == []


def test_verification_service_removes_conflicting_insufficient_evidence_limitations_when_sources_exist() -> None:
    payload = build_report_payload()
    payload.limitations = ["依据不足、无法可靠回答", "source_mode=not_found 或 sources 为空"]

    verified = VerificationService().verify(payload)

    assert all("依据不足" not in item for item in verified.limitations)
    assert all("source_mode=" not in item for item in verified.limitations)
