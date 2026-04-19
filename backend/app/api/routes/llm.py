from fastapi import APIRouter

from app.api.deps import get_llm_catalog_service
from app.schemas.response import LLMModelCatalogResponse

router = APIRouter(prefix="/api/v1/llm")


@router.get("/models", response_model=LLMModelCatalogResponse)
def list_llm_models() -> LLMModelCatalogResponse:
    return get_llm_catalog_service().list_models()
