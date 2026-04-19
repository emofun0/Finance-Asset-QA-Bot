from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel

from app.core.config import settings


_current_trace: ContextVar["RequestTraceRecorder | None"] = ContextVar("current_trace", default=None)


class RequestTraceRecorder:
    def __init__(self, request_id: str, request_payload: dict[str, Any], route_path: str, session_id: str | None = None) -> None:
        self.request_id = request_id
        self.session_id = session_id
        self._path = self._build_trace_path(request_id=request_id, session_id=session_id)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {
            "request_id": request_id,
            "session_id": session_id,
            "route_path": route_path,
            "started_at": self._now(),
            "request": self._serialize(request_payload),
            "status": "running",
            "events": [],
        }
        self._flush()

    def record(self, stage: str, payload: Any) -> None:
        self._data["events"].append(
            {
                "timestamp": self._now(),
                "stage": stage,
                "payload": self._serialize(payload),
            }
        )
        self._flush()

    def finalize(self, status: str, response: Any | None = None, error: Any | None = None) -> None:
        self._data["status"] = status
        self._data["finished_at"] = self._now()
        if response is not None:
            self._data["response"] = self._serialize(response)
        if error is not None:
            self._data["error"] = self._serialize(error)
        self._flush()

    def _flush(self) -> None:
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _build_trace_path(self, *, request_id: str, session_id: str | None) -> Path:
        session_key = self._normalize_session_key(session_id)
        return Path(settings.trace_log_dir) / session_key / f"{request_id}.json"

    def _normalize_session_key(self, session_id: str | None) -> str:
        if not session_id:
            return "__anonymous__"
        safe = "".join(ch for ch in session_id if ch.isalnum() or ch in {"-", "_"})
        return safe or "__anonymous__"

    def _serialize(self, value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, datetime):
            return value.astimezone(UTC).isoformat()
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value):
            return self._serialize(asdict(value))
        if isinstance(value, dict):
            return {str(key): self._serialize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._serialize(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()


@contextmanager
def request_trace(
    request_id: str,
    request_payload: dict[str, Any],
    route_path: str,
    session_id: str | None = None,
) -> Iterator[RequestTraceRecorder]:
    recorder = RequestTraceRecorder(
        request_id=request_id,
        request_payload=request_payload,
        route_path=route_path,
        session_id=session_id,
    )
    token = _current_trace.set(recorder)
    recorder.record("request.received", request_payload)
    try:
        yield recorder
    except Exception as exc:
        recorder.finalize(
            status="error",
            error={"type": exc.__class__.__name__, "message": str(exc)},
        )
        raise
    finally:
        _current_trace.reset(token)


def trace_event(stage: str, payload: Any) -> None:
    recorder = _current_trace.get()
    if recorder is None:
        return
    recorder.record(stage, payload)


def get_trace_file_path(request_id: str) -> Path:
    trace_dir = Path(settings.trace_log_dir)
    direct_path = trace_dir / f"{request_id}.json"
    if direct_path.exists():
        return direct_path

    for path in trace_dir.rglob(f"{request_id}.json"):
        return path
    return trace_dir / f"{request_id}.json"


def read_trace(request_id: str) -> dict[str, Any]:
    path = get_trace_file_path(request_id)
    return json.loads(path.read_text(encoding="utf-8"))


def list_traces(limit: int = 20) -> list[dict[str, Any]]:
    trace_dir = Path(settings.trace_log_dir)
    if not trace_dir.exists():
        return []

    files = sorted(trace_dir.rglob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]
    summaries: list[dict[str, Any]] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summaries.append(
            {
                "request_id": payload.get("request_id"),
                "session_id": payload.get("session_id"),
                "started_at": payload.get("started_at"),
                "finished_at": payload.get("finished_at"),
                "status": payload.get("status"),
                "message": payload.get("request", {}).get("message"),
            }
        )
    return summaries
