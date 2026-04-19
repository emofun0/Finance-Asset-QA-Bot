from dataclasses import dataclass
from app.schemas.request import ChatRequest
from app.schemas.domain import IntentType
from app.services.answer_service import AnswerService
from app.services.answer_generation_service import AnswerGenerationService
from app.services.asset_qa_service import AssetQAService
from app.services.chat_presenter_service import ChatPresenterService
from app.services.knowledge_qa_service import KnowledgeQAService
from app.services.router_service import RouterService
from app.services.verification_service import VerificationService
from app.tools.market_data_tool import MarketSnapshot, PricePoint

from datetime import UTC, datetime, timedelta


class FakeMarketDataTool:
    def get_snapshot(self, symbol: str) -> MarketSnapshot:
        return MarketSnapshot(
            symbol=symbol,
            company_name="Alibaba",
            currency="USD",
            exchange="NYSE",
            latest_price=110.0,
            latest_timestamp=datetime(2026, 4, 18, 16, 0, tzinfo=UTC),
            previous_close=105.0,
        )

    def get_history(self, symbol: str, days: int) -> list[PricePoint]:
        start = datetime(2026, 4, 10, 16, 0, tzinfo=UTC)
        prices = [100.0, 102.0, 101.0, 106.0, 108.0, 110.0]
        return [
            PricePoint(timestamp=start + timedelta(days=index), close=price)
            for index, price in enumerate(prices)
        ]

    def get_history_range(self, symbol: str, start_at: datetime, end_at: datetime) -> list[PricePoint]:
        prices = [100.0, 102.0, 108.0]
        return [
            PricePoint(timestamp=start_at + timedelta(days=index), close=price)
            for index, price in enumerate(prices)
        ]


@dataclass(frozen=True)
class FakeRetrievalResult:
    chunk_id: str
    score: float
    content: str
    metadata: dict


class FakeKnowledgeRetriever:
    def search(self, query: str, **_: object) -> list[FakeRetrievalResult]:
        if "市盈率" in query:
            return [
                FakeRetrievalResult(
                    chunk_id="glossary::0",
                    score=0.9,
                    content="市盈率通常表示股票价格相对于每股收益的估值倍数。",
                    metadata={
                        "title": "项目内置财务基础词条",
                        "doc_type": "glossary",
                        "url": None,
                    },
                )
            ]
        return []


class FailIfCalledAnswerGenerationService(AnswerGenerationService):
    def generate(self, request_message: str, route, draft_answer):  # type: ignore[override]
        raise AssertionError("资产问答不应进入回答生成阶段。")


class FailIfCalledVerificationService(VerificationService):
    def verify(self, answer):  # type: ignore[override]
        raise AssertionError("资产问答不应进入校验阶段。")


class RecordingAnswerGenerationService(AnswerGenerationService):
    def __init__(self) -> None:
        super().__init__()
        self.called = False

    def generate(self, request_message: str, route, draft_answer):  # type: ignore[override]
        self.called = True
        generated = draft_answer.model_copy(deep=True)
        generated.summary = "生成阶段已处理摘要。"
        return generated


class RecordingVerificationService(VerificationService):
    def __init__(self) -> None:
        super().__init__()
        self.called = False

    def verify(self, answer):  # type: ignore[override]
        self.called = True
        verified = answer.model_copy(deep=True)
        verified.limitations = ["校验阶段已处理限制。"]
        return verified


def build_answer_service() -> AnswerService:
    asset_qa_service = AssetQAService(market_data_tool=FakeMarketDataTool())
    return AnswerService(
        router_service=RouterService(),
        asset_qa_service=asset_qa_service,
        knowledge_qa_service=KnowledgeQAService(retriever=FakeKnowledgeRetriever()),
        answer_generation_service=AnswerGenerationService(),
        verification_service=VerificationService(),
        chat_presenter_service=ChatPresenterService(asset_qa_service=asset_qa_service),
    )


def test_answer_service_returns_asset_price_answer() -> None:
    service = build_answer_service()

    response = service.answer(ChatRequest(message="阿里巴巴当前股价是多少？"))

    assert response.question_type == "asset_price"
    assert response.objective_data["symbol"] == "BABA"
    assert response.objective_data["latest_price"] == 110.0


def test_answer_service_returns_knowledge_stub() -> None:
    service = build_answer_service()

    response = service.answer(ChatRequest(message="什么是市盈率？"))

    assert response.question_type == IntentType.FINANCE_KNOWLEDGE
    assert response.objective_data["retrieval_enabled"] is True
    assert response.sources[0].name == "项目内置财务基础词条"


def test_answer_service_skips_generation_and_verification_for_asset_answers() -> None:
    asset_qa_service = AssetQAService(market_data_tool=FakeMarketDataTool())
    service = AnswerService(
        router_service=RouterService(),
        asset_qa_service=asset_qa_service,
        knowledge_qa_service=KnowledgeQAService(retriever=FakeKnowledgeRetriever()),
        answer_generation_service=FailIfCalledAnswerGenerationService(),
        verification_service=FailIfCalledVerificationService(),
        chat_presenter_service=ChatPresenterService(asset_qa_service=asset_qa_service),
    )

    response = service.answer(ChatRequest(message="阿里巴巴当前股价是多少？"))

    assert response.question_type == IntentType.ASSET_PRICE
    assert response.objective_data["latest_price"] == 110.0


def test_answer_service_runs_generation_and_verification_for_knowledge_answers() -> None:
    generation_service = RecordingAnswerGenerationService()
    verification_service = RecordingVerificationService()
    asset_qa_service = AssetQAService(market_data_tool=FakeMarketDataTool())
    service = AnswerService(
        router_service=RouterService(),
        asset_qa_service=asset_qa_service,
        knowledge_qa_service=KnowledgeQAService(retriever=FakeKnowledgeRetriever()),
        answer_generation_service=generation_service,
        verification_service=verification_service,
        chat_presenter_service=ChatPresenterService(asset_qa_service=asset_qa_service),
    )

    response = service.answer(ChatRequest(message="什么是市盈率？"))

    assert generation_service.called is True
    assert verification_service.called is True
    assert response.question_type == IntentType.FINANCE_KNOWLEDGE
    assert response.summary == "生成阶段已处理摘要。"
    assert response.limitations == ["校验阶段已处理限制。"]


def test_chat_presenter_only_keeps_web_sources() -> None:
    presenter = ChatPresenterService(asset_qa_service=AssetQAService(market_data_tool=FakeMarketDataTool()))
    payload = build_answer_service().answer(ChatRequest(message="什么是市盈率？"))
    payload.sources = [
        payload.sources[0].model_copy(update={"name": "本地词条", "value": "manual://zh-core-finance-concepts"}),
        payload.sources[0].model_copy(update={"name": "证监会文章", "value": "https://www.csrc.gov.cn/example"}),
    ]

    message = presenter.build_message(payload)

    assert [source.name for source in message.sources] == ["证监会文章"]
