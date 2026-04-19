from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

from app.llm.contracts import AgentPlanningResult
from app.schemas.response import AnswerPayload


@dataclass
class SessionMemory:
    last_company: str | None = None
    last_symbol: str | None = None
    last_question_type: str | None = None
    recent_summaries: list[str] = field(default_factory=list)
    recent_answers: list[AnswerPayload] = field(default_factory=list)
    cached_answers: dict[str, AnswerPayload] = field(default_factory=dict)


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

    def describe_context(self, session_id: str | None) -> str:
        memory = self.get(session_id)
        if memory is None:
            return ""

        parts: list[str] = []
        if memory.last_company or memory.last_symbol:
            subject = f"{memory.last_company or ''} {memory.last_symbol or ''}".strip()
            if subject:
                parts.append(f"上一轮主要资产：{subject}")
        if memory.last_question_type:
            parts.append(f"上一轮问题类型：{memory.last_question_type}")
        if memory.recent_summaries:
            parts.append("最近工具结果摘要：")
            parts.extend(f"- {item}" for item in memory.recent_summaries[-2:])
        return "\n".join(parts).strip()

    def get_cached_answer(self, session_id: str | None, plan: AgentPlanningResult) -> AnswerPayload | None:
        memory = self.get(session_id)
        if memory is None:
            return None
        return memory.cached_answers.get(self._build_cache_key(plan))

    def remember(self, session_id: str | None, plan: AgentPlanningResult, answer: AnswerPayload) -> None:
        memory = self.get_or_create(session_id)
        if memory is None:
            return

        memory.last_company = answer.route.extracted_company or plan.company or memory.last_company
        memory.last_symbol = answer.route.extracted_symbol or plan.symbol or memory.last_symbol
        question_type = answer.question_type.value if hasattr(answer.question_type, "value") else str(answer.question_type)
        memory.last_question_type = question_type
        if answer.summary:
            memory.recent_summaries.append(answer.summary.strip())
            memory.recent_summaries = memory.recent_summaries[-4:]
        memory.recent_answers.append(answer)
        memory.recent_answers = memory.recent_answers[-4:]
        memory.cached_answers[self._build_cache_key(plan)] = answer

    def fill_plan_from_memory(self, session_id: str | None, plan: AgentPlanningResult) -> AgentPlanningResult:
        memory = self.get(session_id)
        if memory is None:
            return plan

        normalized = plan.model_copy(deep=True)
        if not normalized.company and memory.last_company:
            normalized.company = memory.last_company
        if not normalized.symbol and memory.last_symbol:
            normalized.symbol = memory.last_symbol
        if normalized.time_length is None:
            related_answer = self.get_related_answer(session_id, normalized)
            time_range_days = self._extract_time_range_days(related_answer)
            if time_range_days is not None:
                normalized.time_length = time_range_days
                normalized.time_unit = "day"
        return normalized

    def fill_subject_from_memory(self, session_id: str | None, plan: AgentPlanningResult) -> AgentPlanningResult:
        return self.fill_plan_from_memory(session_id, plan)

    def get_related_answer(self, session_id: str | None, plan: AgentPlanningResult) -> AnswerPayload | None:
        memory = self.get(session_id)
        if memory is None:
            return None

        target_company = (plan.company or memory.last_company or "").strip().lower()
        target_symbol = (plan.symbol or memory.last_symbol or "").strip().lower()
        if not target_company and not target_symbol:
            return memory.recent_answers[-1] if memory.recent_answers else None

        for answer in reversed(memory.recent_answers):
            answer_company = (answer.route.extracted_company or "").strip().lower()
            answer_symbol = (answer.route.extracted_symbol or "").strip().lower()
            if target_symbol and answer_symbol == target_symbol:
                return answer
            if target_company and answer_company == target_company:
                return answer
        return memory.recent_answers[-1] if memory.recent_answers else None

    def _extract_time_range_days(self, answer: AnswerPayload | None) -> int | None:
        if answer is None:
            return None
        value = answer.objective_data.get("time_range_days")
        if isinstance(value, int) and value > 0:
            return value
        return None

    def _build_cache_key(self, plan: AgentPlanningResult) -> str:
        return "|".join(
            [
                plan.tool_name,
                (plan.company or "").strip().lower(),
                (plan.symbol or "").strip().lower(),
                str(plan.time_length or ""),
                str(plan.time_unit or ""),
                str(plan.event_date or ""),
            ]
        )
