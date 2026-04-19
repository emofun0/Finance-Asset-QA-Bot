from pathlib import Path

from app.observability.request_trace import list_traces, read_trace, request_trace, trace_event
from app.core.config import settings


def test_request_trace_persists_events(tmp_path: Path) -> None:
    original_trace_dir = settings.trace_log_dir
    object.__setattr__(settings, "trace_log_dir", str(tmp_path))
    try:
        with request_trace(
            "trace-test",
            {"message": "什么是市盈率？", "session_id": "session-1"},
            "/api/v1/chat",
            session_id="session-1",
        ) as trace:
            trace_event("router.final", {"intent": "finance_knowledge"})
            trace.finalize(status="success", response={"success": True})

        payload = read_trace("trace-test")
        assert payload["status"] == "success"
        assert payload["session_id"] == "session-1"
        assert payload["request"]["message"] == "什么是市盈率？"
        assert any(event["stage"] == "router.final" for event in payload["events"])
        assert (tmp_path / "session-1" / "trace-test.json").exists()

        traces = list_traces(limit=5)
        assert traces[0]["request_id"] == "trace-test"
        assert traces[0]["session_id"] == "session-1"
    finally:
        object.__setattr__(settings, "trace_log_dir", original_trace_dir)
