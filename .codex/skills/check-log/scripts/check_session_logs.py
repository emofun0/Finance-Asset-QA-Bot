#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_TRACE_ROOTS = (
    "backend/logs/traces",
    "logs/traces",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 session id 查找 trace 日志，并输出该 session 下的全部 JSON 记录。"
    )
    parser.add_argument("session_id", help="目标 session id")
    parser.add_argument(
        "--trace-root",
        action="append",
        default=[],
        help="trace 根目录，可重复传入；默认依次尝试 backend/logs/traces 和 logs/traces",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="只输出每个 JSON 的摘要，不打印完整 JSON 内容",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = Path.cwd()
    trace_roots = [workspace / item for item in (args.trace_root or list(DEFAULT_TRACE_ROOTS))]

    session_dir = find_session_dir(trace_roots, args.session_id)
    if session_dir is None:
        print(f"未找到 session 目录：{args.session_id}", file=sys.stderr)
        print("已检查路径：", file=sys.stderr)
        for root in trace_roots:
            print(f"- {root}", file=sys.stderr)
        return 1

    json_files = sorted(session_dir.glob("*.json"), key=sort_key)
    if not json_files:
        print(f"session 目录存在，但没有 JSON 文件：{session_dir}", file=sys.stderr)
        return 1

    print(f"session_id: {args.session_id}")
    print(f"session_dir: {session_dir}")
    print(f"json_count: {len(json_files)}")

    for index, json_file in enumerate(json_files, start=1):
        payload = load_json(json_file)
        print()
        print(f"===== [{index}/{len(json_files)}] {json_file.name} =====")
        print(build_summary(payload, json_file))
        if args.summary_only:
            continue
        print("--- json ---")
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 0


def find_session_dir(trace_roots: list[Path], session_id: str) -> Path | None:
    for root in trace_roots:
        candidate = root / session_id
        if candidate.is_dir():
            return candidate
    return None


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"JSON 解析失败：{path} -> {exc}") from exc


def sort_key(path: Path) -> tuple[str, str]:
    payload = load_json(path)
    started_at = str(payload.get("started_at") or "")
    return (started_at, path.name)


def build_summary(payload: dict[str, Any], path: Path) -> str:
    request = payload.get("request") or {}
    response = payload.get("response") or {}
    message = request.get("message")
    status = payload.get("status")
    started_at = payload.get("started_at")
    finished_at = payload.get("finished_at")
    route_path = payload.get("route_path")
    request_id = payload.get("request_id")
    events = payload.get("events") or []
    final_text = extract_final_text(response)

    lines = [
        f"path: {path}",
        f"request_id: {request_id}",
        f"route_path: {route_path}",
        f"status: {status}",
        f"started_at: {started_at}",
        f"finished_at: {finished_at}",
        f"user_message: {message}",
        f"event_count: {len(events)}",
    ]
    if final_text:
        lines.append(f"final_text: {truncate(one_line(final_text), 200)}")
    return "\n".join(lines)


def extract_final_text(response: dict[str, Any]) -> str:
    if not isinstance(response, dict):
        return ""
    message = response.get("message")
    if isinstance(message, dict):
        return str(message.get("text") or "")
    data = response.get("data")
    if isinstance(data, dict):
        return str(data.get("summary") or "")
    return ""


def one_line(value: str) -> str:
    return " ".join(str(value).split()).strip()


def truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(limit - 1, 0)].rstrip() + "…"


if __name__ == "__main__":
    raise SystemExit(main())
