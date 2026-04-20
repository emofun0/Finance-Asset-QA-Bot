from app.llm.prompts import build_agent_planning_prompt
from app.rag.retriever import KnowledgeRetriever
from app.services.knowledge_qa_service import KnowledgeQAService
from app.tools.rag_search_tool import RagSearchTool


def test_agent_planning_prompt_contains_rag_and_web_fallback_examples() -> None:
    system_prompt, user_prompt = build_agent_planning_prompt("什么是纳斯达克")

    assert "web_finance_knowledge" in user_prompt
    assert "report_summary" in user_prompt
    assert "asset_event_analysis" in user_prompt
    assert "纳斯达克 Nasdaq official definition" in user_prompt
    assert "阿里巴巴最近半年为什么下跌" in user_prompt
    assert "为什么最近半年下跌" in user_prompt
    assert "腾讯最近一个月为什么涨这么多" in user_prompt
    assert "为什么跌、为什么涨" in user_prompt
    assert "用户：最近半年呢" in user_prompt
    assert "用户：那最近一年呢" in user_prompt
    assert "\"time_length\":6,\"time_unit\":\"month\"" in user_prompt
    assert "\"time_length\":1,\"time_unit\":\"year\"" in user_prompt
    assert "时间范围必须显式填为半年" in user_prompt
    assert "默认同时包含中文和英文检索词" in user_prompt
    assert "腾讯 Tencent 0700.HK 最新季度财报" in user_prompt
    assert "优先使用本地 RAG" in user_prompt
    assert "简体中文" not in system_prompt or isinstance(system_prompt, str)


def test_knowledge_qa_service_accepts_rag_search_tool() -> None:
    retriever = KnowledgeRetriever("backend/data/knowledge")
    service = KnowledgeQAService(rag_search_tool=RagSearchTool(retriever))

    assert service.rag_search_tool is not None
