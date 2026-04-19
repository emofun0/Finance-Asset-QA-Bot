from fastapi import APIRouter

from app.api.routes.assets import router as assets_router
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.llm import router as llm_router
from app.api.routes.rag import router as rag_router
from app.api.routes.traces import router as traces_router

api_router = APIRouter()
api_router.include_router(assets_router, tags=["assets"])
api_router.include_router(chat_router, tags=["chat"])
api_router.include_router(health_router, tags=["health"])
api_router.include_router(llm_router, tags=["llm"])
api_router.include_router(rag_router, tags=["rag"])
api_router.include_router(traces_router, tags=["traces"])
