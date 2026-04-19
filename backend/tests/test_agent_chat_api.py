import asyncio

from app.api.routes import chat as chat_route
from app.schemas.request import ChatRequest, LLMSelection, SessionResetRequest
from app.schemas.domain import IntentType, RouteDecision
from app.schemas.response import AnswerPayload, ChatMessagePayload, SourceItem
from app.llm.contracts import AgentPlanningResult
from app.services.agent_service import AgentService, AgentStreamEvent, AgentToolExecutor
from app.services.chat_presenter_service import ChatPresenterService
from app.services.session_memory_service import SessionMemoryService


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
        session_memory_service=SessionMemoryService(),
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
        session_memory_service=SessionMemoryService(),
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


def test_reset_chat_session_clears_memory(monkeypatch) -> None:
    memory = SessionMemoryService()
    monkeypatch.setattr(chat_route, "get_session_memory_service", lambda: memory)

    cached_answer = AnswerPayload(
        question_type=IntentType.ASSET_TREND,
        request_message="intel股价怎么样",
        summary="INTC 最近 30 天整体上涨。",
        objective_data={"symbol": "INTC", "time_range_days": 30},
        analysis=[],
        sources=[],
        limitations=[],
        route=RouteDecision(intent=IntentType.ASSET_TREND, need_market_data=True, extracted_company="Intel", extracted_symbol="INTC", reason="test"),
    )
    plan = AgentPlanningResult(
        tool_name="asset_trend",
        thought="查询 Intel 股价趋势",
        company="Intel",
        symbol="INTC",
        time_length=30,
        time_unit="day",
        reason="test",
    )
    memory.remember("session-1", plan, cached_answer)

    response = chat_route.reset_chat_session(SessionResetRequest(session_id="session-1"))

    assert response["success"] is True
    assert response["data"]["cleared"] is True
    assert memory.get("session-1") is None


def test_session_memory_fills_subject_and_reuses_cached_answer() -> None:
    memory = SessionMemoryService()
    cached_answer = AnswerPayload(
        question_type=IntentType.ASSET_TREND,
        request_message="intel股价怎么样",
        summary="INTC 最近 30 天整体上涨。",
        objective_data={"symbol": "INTC", "time_range_days": 30},
        analysis=["样本起点价格约为 50，终点价格约为 60。"],
        sources=[],
        limitations=[],
        route=RouteDecision(intent=IntentType.ASSET_TREND, need_market_data=True, extracted_company="Intel", extracted_symbol="INTC", reason="test"),
    )
    first_plan = AgentPlanningResult(
        tool_name="asset_trend",
        thought="查询 Intel 股价趋势",
        company="Intel",
        symbol="INTC",
        time_length=30,
        time_unit="day",
        reason="test",
    )
    memory.remember("session-1", first_plan, cached_answer)

    followup_plan = AgentPlanningResult(
        tool_name="asset_trend",
        thought="延续上文主体",
        company=None,
        symbol=None,
        time_length=30,
        time_unit="day",
        reason="test",
    )
    filled_plan = memory.fill_subject_from_memory("session-1", followup_plan)

    assert filled_plan.company == "Intel"
    assert filled_plan.symbol == "INTC"
    assert filled_plan.time_length == 30
    assert filled_plan.time_unit == "day"
    assert memory.get_cached_answer("session-1", filled_plan) is not None


def test_agent_service_normalizes_colloquial_price_question_to_trend() -> None:
    service = AgentService(
        provider="ollama",
        model="test-model",
        asset_qa_service=DummyToolService(),  # type: ignore[arg-type]
        knowledge_qa_service=DummyToolService(),  # type: ignore[arg-type]
        chat_presenter_service=ChatPresenterService(asset_qa_service=RecordingAssetQAService()),  # type: ignore[arg-type]
        fallback_answer_service=FakeFallbackAnswerService(),  # type: ignore[arg-type]
        session_memory_service=SessionMemoryService(),
    )

    plan = service._normalize_plan(  # type: ignore[attr-defined]
        "intel股价怎么样",
        AgentPlanningResult(
            tool_name="asset_price",
            thought="查看 Intel 股价",
            company="Intel",
            symbol="INTC",
            reason="",
        ),
    )

    assert plan.tool_name == "asset_trend"
    assert plan.time_length == 30
    assert plan.time_unit == "day"


def test_agent_service_inherits_previous_time_range_for_followup_reason_question() -> None:
    memory = SessionMemoryService()
    previous_answer = AnswerPayload(
        question_type=IntentType.ASSET_TREND,
        request_message="最近一年呢",
        summary="INTC 最近 365 天整体上涨。",
        objective_data={"symbol": "INTC", "time_range_days": 365},
        analysis=[],
        sources=[],
        limitations=[],
        route=RouteDecision(intent=IntentType.ASSET_TREND, need_market_data=True, extracted_company="Intel", extracted_symbol="INTC", reason="test"),
    )
    previous_plan = AgentPlanningResult(
        tool_name="asset_trend",
        thought="查询 Intel 最近一年走势",
        company="Intel",
        symbol="INTC",
        time_length=1,
        time_unit="year",
        reason="test",
    )
    memory.remember("session-1", previous_plan, previous_answer)

    service = AgentService(
        provider="ollama",
        model="test-model",
        asset_qa_service=DummyToolService(),  # type: ignore[arg-type]
        knowledge_qa_service=DummyToolService(),  # type: ignore[arg-type]
        chat_presenter_service=ChatPresenterService(asset_qa_service=RecordingAssetQAService()),  # type: ignore[arg-type]
        fallback_answer_service=FakeFallbackAnswerService(),  # type: ignore[arg-type]
        session_memory_service=memory,
    )

    inherited = service._inherit_time_range_from_memory(  # type: ignore[attr-defined]
        ChatRequest(message="为啥涨这么多", session_id="session-1"),
        AgentPlanningResult(
            tool_name="asset_event_analysis",
            thought="分析上涨原因",
            company="Intel",
            symbol="INTC",
            time_length=1,
            time_unit="month",
            reason="test",
        ),
    )

    assert inherited.time_length == 365
    assert inherited.time_unit == "day"
