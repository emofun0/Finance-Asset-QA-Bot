import asyncio
import json
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.observability.request_trace import request_trace
from app.core.errors import AppError
from app.api.deps import get_agent_service
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse, StandardResponse

router = APIRouter(prefix="/api/v1")


@router.post("/chat", response_model=StandardResponse)
def chat(request: ChatRequest) -> StandardResponse:
    request_id = str(uuid4())
    llm_selection = request.llm
    agent_service = get_agent_service(
        provider=llm_selection.provider if llm_selection else None,
        model=llm_selection.model if llm_selection else None,
    )
    with request_trace(request_id, request.model_dump(mode="json"), "/api/v1/chat") as trace:
        response = StandardResponse(
            request_id=request_id,
            success=True,
            data=agent_service.answer(request),
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
    agent_service = get_agent_service(
        provider=llm_selection.provider if llm_selection else None,
        model=llm_selection.model if llm_selection else None,
    )

    async def event_generator():
        yield _sse("meta", {"request_id": request_id})
        with request_trace(request_id, request.model_dump(mode="json"), "/api/v1/chat/stream") as trace:
            try:
                stream_events = agent_service.stream_chat(request)
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
                final_message_payload: dict | None = None
                for event in stream_events:
                    if event.type == "status":
                        yield _sse("status", event.payload)
                        await asyncio.sleep(0.01)
                        continue
                    if event.type == "thought":
                        yield _sse("thought", event.payload)
                        await asyncio.sleep(0.01)
                        continue
                    if event.type == "tool":
                        yield _sse("tool", event.payload)
                        await asyncio.sleep(0.01)
                        continue
                    if event.type == "final":
                        final_message_payload = event.payload
                        for chunk in _chunk_text(str(final_message_payload.get("text") or "")):
                            collected_chunks.append(chunk)
                            yield _sse("delta", {"text": chunk})
                            await asyncio.sleep(0.01)
                if final_message_payload is None:
                    raise RuntimeError("Agent stream returned no final message.")
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

            final_text = "".join(collected_chunks).strip() or str(final_message_payload.get("text") or "")
            response = ChatResponse(
                request_id=request_id,
                message={
                    **final_message_payload,
                    "text": final_text,
                },
            )
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
