import asyncio

from app.api.routes import chat as chat_route
from app.schemas.request import ChatRequest, LLMSelection
from app.schemas.domain import IntentType, RouteDecision
from app.schemas.response import AnswerPayload, ChatMessagePayload, SourceItem
from app.llm.contracts import AgentPlanningResult
from app.services.agent_service import AgentService, AgentStreamEvent, AgentToolExecutor
from app.services.chat_presenter_service import ChatPresenterService


class FakeAgentService:
    def answer(self, request):  # type: ignore[no-untyped-def]
        return AnswerPayload(
            question_type=IntentType.ASSET_PRICE,
            request_message=request.message,
            summary="BABA 最新价格约为 110.00 USD",
            objective_data={"symbol": "BABA", "latest_price": 110.0},
            analysis=["当前价格高于前收盘价。"],
            sources=[SourceItem(type="market_data", name="Yahoo Finance", value="https://finance.yahoo.com/quote/BABA")],
            limitations=["当前价格可能不是逐笔实时成交价。"],
            route=RouteDecision(intent=IntentType.ASSET_PRICE, need_market_data=True, extracted_symbol="BABA", reason="agent"),
        )

    def stream_chat(self, request):  # type: ignore[no-untyped-def]
        yield AgentStreamEvent(type="status", payload={"text": "正在判断问题意图..."})
        yield AgentStreamEvent(type="thought", payload={"text": "先确定用户在问价格。", "tool_name": "asset_price"})
        yield AgentStreamEvent(type="status", payload={"text": "正在查询最新市场价格..."})
        yield AgentStreamEvent(type="tool", payload={"tool_name": "asset_price", "summary": "已查询到 BABA 的最新价格。"})
        yield AgentStreamEvent(type="status", payload={"text": "正在整理最终回答..."})
        yield AgentStreamEvent(
            type="final",
            payload=ChatMessagePayload(
                text="BABA 最新价格约为 110.00 USD。\n\n- 当前价格高于前收盘价。",
                sources=[SourceItem(type="market_data", name="Yahoo Finance", value="https://finance.yahoo.com/quote/BABA")],
                chart=None,
            ).model_dump(mode="json"),
        )


class FakeFallbackAnswerService:
    def answer(self, request):  # type: ignore[no-untyped-def]
        return AnswerPayload(
            question_type=IntentType.FINANCE_KNOWLEDGE,
            request_message=request.message,
            summary="回退链路已执行。",
            objective_data={"source_mode": "fallback"},
            analysis=["因为 agent 不可用，所以走旧链路。"],
            sources=[],
            limitations=["这是测试专用回退结果。"],
            route=RouteDecision(intent=IntentType.FINANCE_KNOWLEDGE, need_rag=True, reason="fallback"),
        )

    def answer_chat(self, request):  # type: ignore[no-untyped-def]
        return ChatMessagePayload(text="回退链路已执行。", sources=[], chart=None)


class DummyToolService:
    def answer(self, request, route):  # type: ignore[no-untyped-def]
        raise AssertionError("fallback test should not reach tool services")


class RecordingAssetQAService:
    def __init__(self) -> None:
        self.last_route = None

    def answer(self, request, route):  # type: ignore[no-untyped-def]
        self.last_route = route
        return AnswerPayload(
            question_type=route.intent,
            request_message=request.message,
            summary="ok",
            objective_data={"time_range_days": route.time_range_days},
            analysis=[],
            sources=[],
            limitations=[],
            route=route,
        )

    def get_price_history(self, symbol: str, days: int):  # type: ignore[no-untyped-def]
        return {
            "symbol": symbol,
            "trend": "上涨",
            "change_pct": 10.0,
            "points": [
                {"timestamp": "2026-04-01T00:00:00+00:00", "close": 10.0},
                {"timestamp": "2026-04-02T00:00:00+00:00", "close": 11.0},
            ],
        }


def test_chat_endpoint_uses_agent_service(monkeypatch) -> None:
    monkeypatch.setattr(chat_route, "get_agent_service", lambda provider=None, model=None: FakeAgentService())
    response = chat_route.chat(
        ChatRequest(
            message="阿里巴巴当前股价是多少？",
            llm=LLMSelection(provider="ollama", model="test-model"),
        )
    )

    payload = response.model_dump(mode="json")
    assert payload["success"] is True
    assert payload["data"]["question_type"] == "asset_price"
    assert payload["data"]["objective_data"]["symbol"] == "BABA"


def test_chat_stream_endpoint_emits_agent_events(monkeypatch) -> None:
    monkeypatch.setattr(chat_route, "get_agent_service", lambda provider=None, model=None: FakeAgentService())

    async def collect_stream() -> str:
        response = await chat_route.chat_stream(
            ChatRequest(
                message="阿里巴巴当前股价是多少？",
                llm=LLMSelection(provider="ollama", model="test-model"),
            )
        )
        chunks: list[str] = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    body = asyncio.run(collect_stream())

    assert "event: meta" in body
    assert "event: status" in body
    assert "event: thought" in body
    assert "event: tool" in body
    assert "event: delta" in body
    assert "event: done" in body
    assert "BABA 最新价格约为 110.00 USD" in body


def test_agent_service_falls_back_when_graph_is_disabled() -> None:
    service = AgentService(
        provider="ollama",
        model="test-model",
        asset_qa_service=DummyToolService(),  # type: ignore[arg-type]
        knowledge_qa_service=DummyToolService(),  # type: ignore[arg-type]
        chat_presenter_service=None,  # type: ignore[arg-type]
        fallback_answer_service=FakeFallbackAnswerService(),  # type: ignore[arg-type]
    )
    service.graph = None
    service.model = None

    answer = service.answer(type("Request", (), {"message": "什么是市盈率？"})())
    message = service.answer_chat(type("Request", (), {"message": "什么是市盈率？"})())

    assert answer.summary == "回退链路已执行。"
    assert message.text == "回退链路已执行。"


def test_agent_tool_executor_converts_years_to_days() -> None:
    asset_service = RecordingAssetQAService()
    executor = AgentToolExecutor(
        asset_qa_service=asset_service,  # type: ignore[arg-type]
        knowledge_qa_service=DummyToolService(),  # type: ignore[arg-type]
    )

    result = executor.run(
        ChatRequest(message="intel最近三年股价", llm=LLMSelection(provider="ollama", model="test-model")),
        AgentPlanningResult(
            tool_name="asset_trend",
            thought="获取 Intel 最近三年的股价趋势",
            company="Intel",
            symbol="INTC",
            time_length=3,
            time_unit="year",
            reason="test",
        ),
    )

    assert asset_service.last_route is not None
    assert asset_service.last_route.time_range_days == 1095
    assert result.objective_data["time_range_days"] == 1095


def test_agent_service_normalizes_price_query_with_time_range_to_trend() -> None:
    service = AgentService(
        provider="ollama",
        model="test-model",
        asset_qa_service=DummyToolService(),  # type: ignore[arg-type]
        knowledge_qa_service=DummyToolService(),  # type: ignore[arg-type]
        chat_presenter_service=ChatPresenterService(asset_qa_service=RecordingAssetQAService()),  # type: ignore[arg-type]
        fallback_answer_service=FakeFallbackAnswerService(),  # type: ignore[arg-type]
    )

    plan = service._normalize_plan(  # type: ignore[attr-defined]
        "intel过去三年股价",
        AgentPlanningResult(
            tool_name="asset_price",
            thought="获取价格",
            company="Intel",
            symbol="INTC",
            time_length=3,
            time_unit="year",
            reason="",
        ),
    )

    assert plan.tool_name == "asset_trend"


def test_chat_presenter_uses_plain_text_numbered_points() -> None:
    presenter = ChatPresenterService(asset_qa_service=RecordingAssetQAService())  # type: ignore[arg-type]
    payload = AnswerPayload(
        question_type=IntentType.ASSET_TREND,
        request_message="intel过去三年股价",
        summary="INTC 最近 1095 天整体上涨。",
        objective_data={"symbol": "INTC", "time_range_days": 1095},
        analysis=["样本起点价格约为 31.83，终点价格约为 68.50。"],
        sources=[],
        limitations=["当前趋势结论基于日线收盘数据。"],
        route=RouteDecision(intent=IntentType.ASSET_TREND, need_market_data=True, extracted_symbol="INTC", reason="test"),
    )

    message = presenter.build_message(payload)

    assert "要点：" in message.text
    assert "\n1. 样本起点价格约为 31.83，终点价格约为 68.50。" in message.text
    assert "- " not in message.text
