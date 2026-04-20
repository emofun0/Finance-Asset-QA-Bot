from app.llm.client import NullLLMClient
from app.llm.contracts import AgentPlanningResult
from app.schemas.domain import IntentType, RouteDecision
from app.schemas.response import AnswerPayload
from app.services.agent_service import AgentService
from app.services.chat_presenter_service import ChatPresenterService
from app.services.session_memory_service import SessionMemoryService


class StubAssetQAService:
    def answer(self, request, route: RouteDecision) -> AnswerPayload:
        return AnswerPayload(
            question_type=IntentType.ASSET_TREND,
            request_message=request.message,
            summary=f"{route.extracted_symbol} 最近 {route.time_range_days} 天走势已查询。",
            objective_data={"time_range_days": route.time_range_days},
            analysis=[],
            sources=[],
            limitations=[],
            route=route,
        )


class StubKnowledgeQAService:
    def answer(self, request, route: RouteDecision) -> AnswerPayload:
        raise AssertionError("knowledge service should not be called in this test")


class StubFallbackAnswerService:
    def answer(self, request):
        raise AssertionError("fallback answer should not be called in this test")

    def answer_chat(self, request):
        raise AssertionError("fallback answer_chat should not be called in this test")


def test_session_memory_does_not_fill_time_fields() -> None:
    memory_service = SessionMemoryService()
    previous_plan = AgentPlanningResult(
        tool_name="asset_trend",
        thought="查询 30 天走势。",
        company="Alibaba",
        symbol="BABA",
        time_length=30,
        time_unit="day",
        event_date=None,
        rewritten_query=None,
        direct_response=None,
        reason="上一轮查询 30 天。",
    )
    previous_answer = AnswerPayload(
        question_type=IntentType.ASSET_TREND,
        request_message="BABA 最近 30 天涨跌情况如何？",
        summary="BABA 最近 30 天整体上涨。",
        objective_data={"time_range_days": 30},
        analysis=[],
        sources=[],
        limitations=[],
        route=RouteDecision(intent=IntentType.ASSET_TREND, extracted_company="Alibaba", extracted_symbol="BABA"),
    )
    memory_service.remember("session-1", previous_plan, previous_answer)

    plan = AgentPlanningResult(
        tool_name="asset_trend",
        thought="查询近期走势。",
        company=None,
        symbol=None,
        time_length=None,
        time_unit=None,
        event_date=None,
        rewritten_query=None,
        direct_response=None,
        reason="追问走势。",
    )

    filled = memory_service.fill_plan_from_memory("session-1", plan)

    assert filled.symbol == "BABA"
    assert filled.company == "Alibaba"
    assert filled.time_length is None
    assert filled.time_unit is None


def test_agent_uses_only_agent_supplied_time_fields() -> None:
    agent_service = AgentService(
        asset_qa_service=StubAssetQAService(),
        knowledge_qa_service=StubKnowledgeQAService(),
        chat_presenter_service=ChatPresenterService(),
        fallback_answer_service=StubFallbackAnswerService(),
        session_memory_service=SessionMemoryService(),
        llm_client=NullLLMClient(),
    )

    plan = AgentPlanningResult(
        tool_name="asset_trend",
        thought="查询最近半年走势。",
        company="Alibaba",
        symbol="BABA",
        time_length=6,
        time_unit="month",
        event_date=None,
        rewritten_query=None,
        direct_response=None,
        reason="Agent 指定半年。",
    )

    route = agent_service.tool_executor._build_route(
        intent=IntentType.ASSET_TREND,
        plan=plan,
        need_market_data=True,
        time_range_days=agent_service.tool_executor._resolve_time_range_days(plan, default_days=30),
    )

    assert route.time_range_days == 180
