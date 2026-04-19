from __future__ import annotations

from app.core.config import settings
from app.llm.client import BaseLLMClient, NullLLMClient
from app.llm.contracts import GeneratedAnswerSections
from app.llm.prompts import build_answer_generation_prompt
from app.observability.request_trace import trace_event
from app.schemas.domain import IntentType
from app.schemas.domain import RouteDecision
from app.schemas.response import AnswerPayload


class AnswerGenerationService:
    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.llm_client = llm_client or NullLLMClient()

    def generate(
        self,
        request_message: str,
        route: RouteDecision,
        draft_answer: AnswerPayload,
    ) -> AnswerPayload:
        if not settings.llm_enable_generation or not self.llm_client.is_enabled():
            trace_event("answer_generation.skipped", {"reason": "llm_disabled"})
            return draft_answer

        if self._should_skip_generation(draft_answer):
            trace_event("answer_generation.skipped", {"reason": "insufficient_evidence"})
            return draft_answer

        system_prompt, user_prompt = build_answer_generation_prompt(
            request_message=request_message,
            route=route,
            draft_answer=draft_answer,
        )
        trace_event("answer_generation.prompt", {"system_prompt": system_prompt, "user_prompt": user_prompt})

        try:
            generated = self.llm_client.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=GeneratedAnswerSections,
            )
        except Exception as exc:
            trace_event("answer_generation.error", {"type": exc.__class__.__name__, "message": str(exc)})
            return draft_answer

        answer = draft_answer.model_copy(deep=True)
        answer.summary = generated.summary.strip() or answer.summary
        answer.analysis = [item.strip() for item in generated.analysis if item.strip()] or answer.analysis
        answer.limitations = [item.strip() for item in generated.limitations if item.strip()] or answer.limitations
        trace_event("answer_generation.output", generated)
        return answer

    def _should_skip_generation(self, draft_answer: AnswerPayload) -> bool:
        source_mode = str(draft_answer.objective_data.get("source_mode") or "").strip().lower()
        question_type = draft_answer.question_type
        question_type_value = question_type.value if isinstance(question_type, IntentType) else str(question_type)
        is_retrieval_answer = question_type_value in {
            IntentType.FINANCE_KNOWLEDGE.value,
            IntentType.REPORT_SUMMARY.value,
        }
        is_asset_answer = question_type_value in {
            IntentType.ASSET_PRICE.value,
            IntentType.ASSET_TREND.value,
            IntentType.ASSET_EVENT_ANALYSIS.value,
        }

        if is_asset_answer:
            return True
        if source_mode == "not_found":
            return True
        if is_retrieval_answer and not draft_answer.sources:
            return True
        if "无法可靠回答" in draft_answer.summary or "暂未检索到" in draft_answer.summary:
            return True
        return False
