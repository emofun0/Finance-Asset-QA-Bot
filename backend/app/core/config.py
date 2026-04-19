from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Finance Asset QA System")
    app_env: str = os.getenv("APP_ENV", "development")
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    frontend_port: int = int(os.getenv("FRONTEND_PORT", "5173"))
    knowledge_base_dir: str = os.getenv("KNOWLEDGE_BASE_DIR", str(BASE_DIR / "data" / "knowledge"))
    trace_log_dir: str = os.getenv("TRACE_LOG_DIR", str(BASE_DIR / "logs" / "traces"))
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    llm_enable_routing: bool = os.getenv("LLM_ENABLE_ROUTING", "true").lower() == "true"
    llm_enable_generation: bool = os.getenv("LLM_ENABLE_GENERATION", "true").lower() == "true"
    llm_enable_query_rewrite: bool = os.getenv("LLM_ENABLE_QUERY_REWRITE", "true").lower() == "true"
    llm_enable_verification: bool = os.getenv("LLM_ENABLE_VERIFICATION", "true").lower() == "true"
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.1")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_reasoning_effort: str = os.getenv("OPENAI_REASONING_EFFORT", "medium")
    rag_vector_db_dir: str = os.getenv("RAG_VECTOR_DB_DIR", str(BASE_DIR / "data" / "knowledge" / "chroma"))
    rag_collection_name: str = os.getenv("RAG_COLLECTION_NAME", "finance_knowledge")
    rag_embedding_provider: str = os.getenv("RAG_EMBEDDING_PROVIDER", "sentence_transformers")
    rag_embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    rag_embedding_batch_size: int = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "32"))


settings = Settings()
