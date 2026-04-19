from fastapi import APIRouter, Depends

from app.api.deps import get_knowledge_base_builder
from app.rag.ingest import KnowledgeBaseBuilder

router = APIRouter(prefix="/api/v1/rag")


@router.post("/ingest")
def ingest_knowledge_base(builder: KnowledgeBaseBuilder = Depends(get_knowledge_base_builder)) -> dict:
    stats = builder.build()
    return {
        "status": "ok",
        "message": "知识库索引已重建。",
        "stats": stats,
    }
