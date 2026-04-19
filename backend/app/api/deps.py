from pathlib import Path

from app.core.config import settings
from app.llm.client import BaseLLMClient, build_llm_client
from app.rag.ingest import KnowledgeBaseBuilder
from app.rag.retriever import KnowledgeRetriever
from app.services.answer_service import AnswerService
from app.services.agent_service import AgentService
from app.services.answer_generation_service import AnswerGenerationService
from app.services.asset_qa_service import AssetQAService
from app.services.chat_presenter_service import ChatPresenterService
from app.services.knowledge_qa_service import KnowledgeQAService
from app.services.llm_catalog_service import LLMCatalogService
from app.services.router_service import RouterService
from app.services.session_memory_service import SessionMemoryService
from app.services.verification_service import VerificationService
from app.tools.market_data_tool import MarketDataTool
from app.tools.rag_search_tool import RagSearchTool
from app.tools.web_search_tool import OfficialWebSearchTool


_session_memory_service = SessionMemoryService()


def get_settings():
    return settings


def get_router_service(provider: str | None = None, model: str | None = None) -> RouterService:
    return RouterService(llm_client=get_llm_client(provider=provider, model=model))


def get_asset_qa_service(provider: str | None = None, model: str | None = None) -> AssetQAService:
    return AssetQAService(
        market_data_tool=get_market_data_tool(),
        web_search_tool=get_web_search_tool(),
        llm_client=get_llm_client(provider=provider, model=model),
    )


def get_knowledge_qa_service(provider: str | None = None, model: str | None = None) -> KnowledgeQAService:
    return KnowledgeQAService(
        rag_search_tool=get_rag_search_tool(),
        web_search_tool=get_web_search_tool(),
    )


def get_answer_service(provider: str | None = None, model: str | None = None) -> AnswerService:
    asset_qa_service = get_asset_qa_service(provider=provider, model=model)
    return AnswerService(
        router_service=get_router_service(provider=provider, model=model),
        asset_qa_service=asset_qa_service,
        knowledge_qa_service=get_knowledge_qa_service(provider=provider, model=model),
        answer_generation_service=get_answer_generation_service(provider=provider, model=model),
        verification_service=get_verification_service(provider=provider, model=model),
        chat_presenter_service=get_chat_presenter_service(asset_qa_service=asset_qa_service),
    )


def get_agent_service(provider: str | None = None, model: str | None = None) -> AgentService:
    asset_qa_service = get_asset_qa_service(provider=provider, model=model)
    knowledge_qa_service = get_knowledge_qa_service(provider=provider, model=model)
    chat_presenter_service = get_chat_presenter_service(asset_qa_service=asset_qa_service)
    fallback_answer_service = AnswerService(
        router_service=get_router_service(provider=provider, model=model),
        asset_qa_service=asset_qa_service,
        knowledge_qa_service=knowledge_qa_service,
        answer_generation_service=get_answer_generation_service(provider=provider, model=model),
        verification_service=get_verification_service(provider=provider, model=model),
        chat_presenter_service=chat_presenter_service,
    )
    return AgentService(
        asset_qa_service=asset_qa_service,
        knowledge_qa_service=knowledge_qa_service,
        chat_presenter_service=chat_presenter_service,
        fallback_answer_service=fallback_answer_service,
        session_memory_service=get_session_memory_service(),
        llm_client=get_llm_client(provider=provider, model=model),
    )


def get_market_data_tool() -> MarketDataTool:
    return MarketDataTool()


def get_knowledge_base_builder() -> KnowledgeBaseBuilder:
    return KnowledgeBaseBuilder(Path(settings.knowledge_base_dir))


def get_knowledge_retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever(settings.knowledge_base_dir)


def get_rag_search_tool() -> RagSearchTool:
    return RagSearchTool(get_knowledge_retriever())


def get_web_search_tool() -> OfficialWebSearchTool:
    return OfficialWebSearchTool()


def get_llm_client(provider: str | None = None, model: str | None = None) -> BaseLLMClient:
    return build_llm_client(provider=provider, model=model)


def get_answer_generation_service(provider: str | None = None, model: str | None = None) -> AnswerGenerationService:
    return AnswerGenerationService(llm_client=get_llm_client(provider=provider, model=model))


def get_verification_service(provider: str | None = None, model: str | None = None) -> VerificationService:
    return VerificationService(llm_client=get_llm_client(provider=provider, model=model))


def get_llm_catalog_service() -> LLMCatalogService:
    return LLMCatalogService()


def get_chat_presenter_service(asset_qa_service: AssetQAService | None = None) -> ChatPresenterService:
    return ChatPresenterService(asset_qa_service=asset_qa_service or get_asset_qa_service())


def get_session_memory_service() -> SessionMemoryService:
    return _session_memory_service
