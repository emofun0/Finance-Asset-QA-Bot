from fastapi import APIRouter, HTTPException, Query

from app.observability.request_trace import list_traces, read_trace

router = APIRouter(prefix="/api/v1/traces")


@router.get("")
def get_trace_list(limit: int = Query(default=20, ge=1, le=100)) -> list[dict]:
    return list_traces(limit=limit)


@router.get("/{request_id}")
def get_trace_detail(request_id: str) -> dict:
    try:
        return read_trace(request_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Trace not found.") from exc
