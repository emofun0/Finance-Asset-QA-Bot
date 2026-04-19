import asyncio
import json
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.observability.request_trace import request_trace
from app.core.errors import AppError
from app.api.deps import get_answer_service
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse, StandardResponse

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


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chunk_text(text: str, chunk_size: int = 24) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        chunks.append(normalized[start:end])
        start = end
    return chunks


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    request_id = str(uuid4())
    llm_selection = request.llm
    answer_service = get_answer_service(
        provider=llm_selection.provider if llm_selection else None,
        model=llm_selection.model if llm_selection else None,
    )

    async def event_generator():
        yield _sse("meta", {"request_id": request_id})
        with request_trace(request_id, request.model_dump(mode="json"), "/api/v1/chat/stream") as trace:
            try:
                stream_plan = answer_service.stream_chat(request)
            except AppError as exc:
                error_payload = {
                    "request_id": request_id,
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
                trace.finalize(status="error", error=error_payload)
                yield _sse("error", error_payload)
                return
            except Exception as exc:
                error_payload = {
                    "request_id": request_id,
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "服务器内部错误。",
                    "details": {"type": exc.__class__.__name__},
                }
                trace.finalize(status="error", error=error_payload)
                yield _sse("error", error_payload)
                return

            collected_chunks: list[str] = []
            try:
                for chunk in stream_plan.chunks:
                    if not chunk:
                        continue
                    collected_chunks.append(chunk)
                    yield _sse("delta", {"text": chunk})
                    await asyncio.sleep(0.01)
            except AppError as exc:
                error_payload = {
                    "request_id": request_id,
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
                trace.finalize(status="error", error=error_payload)
                yield _sse("error", error_payload)
                return
            except Exception as exc:
                error_payload = {
                    "request_id": request_id,
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "服务器内部错误。",
                    "details": {"type": exc.__class__.__name__},
                }
                trace.finalize(status="error", error=error_payload)
                yield _sse("error", error_payload)
                return

            final_text = "".join(collected_chunks).strip() or stream_plan.fallback_text
            message = answer_service.chat_presenter_service.build_message(
                stream_plan.answer,
                text_override=final_text,
            )
            response = ChatResponse(request_id=request_id, message=message)
            trace.finalize(status="success", response=response)
            yield _sse("done", response.model_dump(mode="json"))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
