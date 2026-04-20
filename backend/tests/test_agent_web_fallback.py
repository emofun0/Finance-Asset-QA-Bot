from app.llm.client import NullLLMClient
from app.llm.contracts import AgentPlanningResult
from app.schemas.domain import IntentType, RouteDecision
from app.schemas.request import ChatRequest
from app.schemas.response import AnswerPayload, SourceItem
from app.services.agent_service import AgentService
from app.services.chat_presenter_service import ChatPresenterService
from app.services.session_memory_service import SessionMemoryService


class StubKnowledgeQAService:
    def __init__(self) -> None:
        self.backends: list[str] = []

    def answer(self, request: ChatRequest, route: RouteDecision) -> AnswerPayload:
        backend = str(request.metadata.get("search_backend") or "rag")
        self.backends.append(backend)
        if backend == "web":
            return AnswerPayload(
                question_type=IntentType.FINANCE_KNOWLEDGE,
                request_message=request.message,
                summary="EBITA 是未计利息、税项及摊销前利润。",
                objective_data={
                    "source_mode": "web_fallback",
                    "matched_terms": ["EBITA"],
                },
                analysis=["网页结果明确给出了 EBITA 的术语定义。"],
                sources=[SourceItem(type="web_search", name="Investopedia", value="https://example.com/ebita")],
                limitations=[],
                route=route,
            )

        return AnswerPayload(
            question_type=IntentType.FINANCE_KNOWLEDGE,
            request_message=request.message,
            summary="当前检索结果没有包含这一术语的解释，只命中了泛化的财务分析文章。",
            objective_data={
                "source_mode": "local_rag",
                "matched_terms": [],
            },
            analysis=["当前本地命中内容在讲一般财务分析指标，没有直接解释 EBITA。"],
            sources=[SourceItem(type="knowledge_article", name="财务分析的基本内容", value="https://example.com/local")],
            limitations=[],
            route=route,
        )


class StubAssetQAService:
    def answer(self, request: ChatRequest, route: RouteDecision) -> AnswerPayload:
        raise AssertionError("asset service should not be called in this test")


class StubFallbackAnswerService:
    def answer(self, request: ChatRequest) -> AnswerPayload:
        raise AssertionError("fallback answer should not be called in this test")

    def answer_chat(self, request: ChatRequest):
        raise AssertionError("fallback answer_chat should not be called in this test")


def test_agent_retries_with_web_when_local_rag_misses_core_term() -> None:
    knowledge_service = StubKnowledgeQAService()
    agent_service = AgentService(
        asset_qa_service=StubAssetQAService(),
        knowledge_qa_service=knowledge_service,
        chat_presenter_service=ChatPresenterService(),
        fallback_answer_service=StubFallbackAnswerService(),
        session_memory_service=SessionMemoryService(),
        llm_client=NullLLMClient(),
    )

    request = ChatRequest(message="什么是ebita", session_id="test-session")
    plan = AgentPlanningResult(
        tool_name="finance_knowledge",
        thought="先用本地知识库检索 EBITA 定义。",
        company=None,
        symbol=None,
        time_length=None,
        time_unit=None,
        event_date=None,
        rewritten_query="什么是 EBITA",
        direct_response=None,
        reason="概念解释优先本地 RAG。",
    )

    answer = agent_service._run_tool(request, plan)

    assert knowledge_service.backends == ["rag", "web"]
    assert answer.objective_data["source_mode"] == "web_fallback"
    assert "未计利息、税项及摊销前利润" in answer.summary
