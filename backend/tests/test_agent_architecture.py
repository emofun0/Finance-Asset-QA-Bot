from app.llm.client import NullLLMClient
from app.rag.retriever import KnowledgeRetriever
from app.services.agent_service import AgentService
from app.services.chat_presenter_service import ChatPresenterService
from app.services.knowledge_qa_service import KnowledgeQAService
from app.services.session_memory_service import SessionMemoryService
from app.tools.rag_search_tool import RagSearchTool


class StubAssetQAService:
    web_search_tool = None

    def get_price_history(self, symbol: str, days: int) -> dict:
        return {
            "symbol": symbol,
            "time_range_days": days,
            "trend": "持平",
            "change_pct": 0.0,
            "points": [
                {"timestamp": "2026-01-01T00:00:00+00:00", "close": 1.0},
                {"timestamp": "2026-01-02T00:00:00+00:00", "close": 1.0},
            ],
        }


class StubFallbackAnswerService:
    def answer(self, request):
        raise AssertionError("should not be called")

    def answer_chat(self, request):
        raise AssertionError("should not be called")


def test_native_agent_prompt_and_tools_prefer_local_search_first() -> None:
    asset_service = StubAssetQAService()
    agent_service = AgentService(
        asset_qa_service=asset_service,
        knowledge_qa_service=KnowledgeQAService(rag_search_tool=RagSearchTool(KnowledgeRetriever("backend/data/knowledge"))),
        chat_presenter_service=ChatPresenterService(asset_qa_service=asset_service),
        fallback_answer_service=StubFallbackAnswerService(),
        session_memory_service=SessionMemoryService(),
        llm_client=NullLLMClient(),
    )

    system_prompt = agent_service._build_system_prompt()
    tools = {tool.name: tool for tool in agent_service._build_tools()}

    assert "优先使用本地检索工具" in system_prompt
    assert "只有在本地结果明显不足" in system_prompt
    assert "不要只用单个中文名" in system_prompt
    assert "search_local_knowledge" in system_prompt
    assert "search_local_reports" in system_prompt
    assert "默认先用这个工具" in tools["search_local_knowledge"].description
    assert "默认先用这个工具" in tools["search_local_reports"].description
    assert "英文名、股票代码和简称" in tools["search_local_reports"].description
    assert "仅在本地知识不足" in tools["search_web_knowledge"].description


def test_knowledge_qa_service_accepts_rag_search_tool() -> None:
    retriever = KnowledgeRetriever("backend/data/knowledge")
    service = KnowledgeQAService(rag_search_tool=RagSearchTool(retriever))

    assert service.rag_search_tool is not None


def test_agent_expands_company_search_query_with_aliases() -> None:
    asset_service = StubAssetQAService()
    agent_service = AgentService(
        asset_qa_service=asset_service,
        knowledge_qa_service=KnowledgeQAService(rag_search_tool=RagSearchTool(KnowledgeRetriever("backend/data/knowledge"))),
        chat_presenter_service=ChatPresenterService(asset_qa_service=asset_service),
        fallback_answer_service=StubFallbackAnswerService(),
        session_memory_service=SessionMemoryService(),
        llm_client=NullLLMClient(),
    )

    profile, _ = agent_service._resolve_subject({"company": "腾讯控股"}, require_symbol=False)
    query = agent_service._expand_search_query("最新财报 摘要", profile, "腾讯控股")

    assert "最新财报 摘要" in query
    assert "Tencent" in query
    assert "腾讯控股" in query
    assert "0700.HK" in query
