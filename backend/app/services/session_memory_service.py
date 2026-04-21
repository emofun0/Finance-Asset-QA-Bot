from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

from app.schemas.response import AnswerPayload


@dataclass
class CompressedTurn:
    user_text: str
    assistant_text: str
    tool_notes: list[str] = field(default_factory=list)


@dataclass
class SessionMemory:
    summary_lines: list[str] = field(default_factory=list)
    recent_turns: list[CompressedTurn] = field(default_factory=list)
    last_answer: AnswerPayload | None = None


class SessionMemoryService:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionMemory] = {}
        self._lock = RLock()

    def get(self, session_id: str | None) -> SessionMemory | None:
        if not session_id:
            return None
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_create(self, session_id: str | None) -> SessionMemory | None:
        if not session_id:
            return None
        with self._lock:
            return self._sessions.setdefault(session_id, SessionMemory())

    def clear(self, session_id: str | None) -> bool:
        if not session_id:
            return False
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def build_context_messages(self, session_id: str | None) -> list[dict[str, str]]:
        memory = self.get(session_id)
        if memory is None:
            return []

        messages: list[dict[str, str]] = []
        summary_text = self._build_summary_text(memory)
        if summary_text:
            messages.append(
                {
                    "role": "assistant",
                    "content": summary_text,
                }
            )

        for turn in memory.recent_turns:
            messages.append({"role": "user", "content": turn.user_text})
            assistant_parts = [turn.assistant_text]
            if turn.tool_notes:
                assistant_parts.append("当时保留的工具线索：\n" + "\n".join(f"- {item}" for item in turn.tool_notes[:3]))
            messages.append({"role": "assistant", "content": "\n\n".join(part for part in assistant_parts if part).strip()})
        return messages

    def remember(
        self,
        session_id: str | None,
        *,
        user_message: str,
        answer: AnswerPayload,
        tool_notes: list[str],
    ) -> None:
        memory = self.get_or_create(session_id)
        if memory is None:
            return

        turn = CompressedTurn(
            user_text=self._trim(user_message, 220),
            assistant_text=self._trim(answer.summary, 320),
            tool_notes=[self._trim(item, 180) for item in tool_notes if item.strip()][:3],
        )
        memory.recent_turns.append(turn)
        memory.last_answer = answer

        while len(memory.recent_turns) > 2:
            archived = memory.recent_turns.pop(0)
            memory.summary_lines.append(self._turn_to_summary_line(archived))
        memory.summary_lines = memory.summary_lines[-8:]

    def _build_summary_text(self, memory: SessionMemory) -> str:
        parts: list[str] = []
        if memory.summary_lines:
            parts.append("以下是此前会话的压缩记忆，仅用于延续上下文：")
            parts.extend(f"- {item}" for item in memory.summary_lines[-8:])
        return "\n".join(parts).strip()

    def _turn_to_summary_line(self, turn: CompressedTurn) -> str:
        tool_text = ""
        if turn.tool_notes:
            tool_text = f"；工具线索：{'；'.join(turn.tool_notes[:2])}"
        return f"用户问：{self._trim(turn.user_text, 80)}；回答结论：{self._trim(turn.assistant_text, 100)}{tool_text}"

    def _trim(self, value: str, limit: int) -> str:
        normalized = " ".join(str(value or "").split()).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(limit - 1, 0)].rstrip() + "…"
