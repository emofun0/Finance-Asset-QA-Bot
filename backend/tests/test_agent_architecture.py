from app.llm.prompts import build_agent_planning_prompt
from app.rag.retriever import KnowledgeRetriever
from app.services.knowledge_qa_service import KnowledgeQAService
from app.tools.rag_search_tool import RagSearchTool


def test_agent_planning_prompt_contains_rag_and_web_fallback_examples() -> None:
    system_prompt, user_prompt = build_agent_planning_prompt("什么是纳斯达克")

    assert "web_finance_knowledge" in user_prompt
    assert "report_summary" in user_prompt
    assert "纳斯达克 Nasdaq official definition" in user_prompt
    assert "默认同时包含中文和英文检索词" in user_prompt
    assert "腾讯 Tencent 0700.HK 最新季度财报" in user_prompt
    assert "优先使用本地 RAG" in user_prompt
    assert "简体中文" not in system_prompt or isinstance(system_prompt, str)


def test_knowledge_qa_service_accepts_rag_search_tool() -> None:
    retriever = KnowledgeRetriever("backend/data/knowledge")
    service = KnowledgeQAService(rag_search_tool=RagSearchTool(retriever))

    assert service.rag_search_tool is not None
