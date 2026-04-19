from uuid import uuid4

from fastapi import APIRouter

from app.observability.request_trace import request_trace
from app.api.deps import get_answer_service
from app.schemas.request import ChatRequest
from app.schemas.response import StandardResponse

router = APIRouter(prefix="/api/v1")


@router.post("/chat", response_model=StandardResponse)
def chat(request: ChatRequest) -> StandardResponse:
    request_id = str(uuid4())
    llm_selection = request.llm
    answer_service = get_answer_service(
        provider=llm_selection.provider if llm_selection else None,
        model=llm_selection.model if llm_selection else None,
    )
    with request_trace(request_id, request.model_dump(mode="json"), "/api/v1/chat") as trace:
        response = StandardResponse(
            request_id=request_id,
            success=True,
            data=answer_service.answer(request),
        )
        trace.finalize(status="success", response=response)
        return response
