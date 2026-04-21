from app.llm.client import AgentToolCall, AgentToolDefinition, AgentToolTurn, BaseLLMClient
from app.rag.retriever import RetrievalResult
from app.schemas.request import ChatRequest
from app.services.agent_service import AgentService
from app.services.chat_presenter_service import ChatPresenterService
from app.services.session_memory_service import SessionMemoryService


class StubLLMClient(BaseLLMClient):
    def __init__(self, turns: list[AgentToolTurn]) -> None:
        self.turns = turns

    def is_enabled(self) -> bool:
        return True

    def generate_structured(self, *, system_prompt: str, user_prompt: str, schema):
        raise AssertionError("structured generation should not be used")

    def generate_text_stream(self, *, system_prompt: str, user_prompt: str):
        raise AssertionError("text stream should not be used")

    def generate_tool_turn(
        self,
        *,
        system_prompt: str,
        messages: list[dict],
        tools: list[AgentToolDefinition],
    ) -> AgentToolTurn:
        assert tools
        return self.turns.pop(0)


class StubRagSearchTool:
    def search(self, request) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id="local::1",
                score=0.61,
                content="一篇泛化财务分析文章，未直接给出 EBITA 定义。",
                metadata={"title": "财务分析基础", "doc_type": "knowledge_article", "chunk_kind": "text_chunk"},
            )
        ]

    def search_report_documents(self, request) -> list[RetrievalResult]:
        return []


class StubWebSearchTool:
    def search_finance_knowledge(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id="web::1",
                score=0.93,
                content="EBITA 指未计利息、税项及摊销前利润。",
                metadata={
                    "title": "Official EBITA Definition",
                    "doc_type": "web_search",
                    "url": "https://example.com/ebita",
                },
            )
        ]


class StubKnowledgeQAService:
    def __init__(self) -> None:
        self.rag_search_tool = StubRagSearchTool()
        self.web_search_tool = StubWebSearchTool()
        self.retriever = None


class StubAssetQAService:
    web_search_tool = None

    def get_price_history(self, symbol: str, days: int) -> dict:
        return {
            "symbol": symbol,
            "time_range_days": days,
            "trend": "上涨",
            "change_pct": 1.2,
            "points": [
                {"timestamp": "2026-01-01T00:00:00+00:00", "close": 100.0},
                {"timestamp": "2026-01-02T00:00:00+00:00", "close": 101.2},
            ],
        }


class StubFallbackAnswerService:
    def answer(self, request):
        raise AssertionError("fallback should not be used")

    def answer_chat(self, request):
        raise AssertionError("fallback should not be used")


def test_agent_uses_model_driven_tool_loop_until_final_answer() -> None:
    llm_client = StubLLMClient(
        turns=[
            AgentToolTurn(
                text="先查本地资料。",
                tool_calls=[
                    AgentToolCall(
                        id="tool-1",
                        name="search_local_knowledge",
                        arguments={"query": "什么是 EBITA", "top_k": 3},
                    )
                ],
                assistant_message={"role": "assistant", "content": "先查本地资料。", "tool_calls": [{"id": "tool-1"}]},
            ),
            AgentToolTurn(
                text="本地结果不够，我再查网页。",
                tool_calls=[
                    AgentToolCall(
                        id="tool-2",
                        name="search_web_knowledge",
                        arguments={"query": "EBITA official definition", "top_k": 3},
                    )
                ],
                assistant_message={"role": "assistant", "content": "本地结果不够，我再查网页。", "tool_calls": [{"id": "tool-2"}]},
            ),
            AgentToolTurn(
                text="EBITA 通常指未计利息、税项及摊销前利润。当前网页结果直接给出了这一定义。",
                tool_calls=[],
                assistant_message={"role": "assistant", "content": "EBITA 通常指未计利息、税项及摊销前利润。当前网页结果直接给出了这一定义。"},
            ),
        ]
    )

    agent_service = AgentService(
        asset_qa_service=StubAssetQAService(),
        knowledge_qa_service=StubKnowledgeQAService(),
        chat_presenter_service=ChatPresenterService(asset_qa_service=StubAssetQAService()),
        fallback_answer_service=StubFallbackAnswerService(),
        session_memory_service=SessionMemoryService(),
        llm_client=llm_client,
    )

    answer = agent_service.answer(ChatRequest(message="什么是 EBITA", session_id="s1"))

    assert "未计利息、税项及摊销前利润" in answer.summary
    assert answer.objective_data["used_tools"] == ["search_local_knowledge", "search_web_knowledge"]
    assert answer.objective_data["source_mode"] == "web"

