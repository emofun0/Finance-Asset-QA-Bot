from __future__ import annotations

import re

from app.core.config import settings
from app.llm.client import BaseLLMClient, NullLLMClient
from app.llm.contracts import VerificationResult
from app.llm.prompts import build_verification_prompt
from app.observability.request_trace import trace_event
from app.schemas.response import AnswerPayload


class VerificationService:
    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.llm_client = llm_client or NullLLMClient()

    def verify(self, answer: AnswerPayload) -> AnswerPayload:
        normalized = self._apply_deterministic_checks(answer)
        trace_event("verification.normalized", normalized)
        if not settings.llm_enable_verification or not self.llm_client.is_enabled():
            trace_event("verification.skipped", {"reason": "llm_disabled"})
            return normalized

        system_prompt, user_prompt = build_verification_prompt(normalized)
        trace_event("verification.prompt", {"system_prompt": system_prompt, "user_prompt": user_prompt})
        try:
            result = self.llm_client.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=VerificationResult,
            )
        except Exception as exc:
            trace_event("verification.error", {"type": exc.__class__.__name__, "message": str(exc)})
            return normalized

        if result.is_valid:
            trace_event("verification.output", result)
            return normalized

        patched = normalized.model_copy(deep=True)
        if result.corrected_summary:
            patched.summary = result.corrected_summary.strip()
        if result.corrected_analysis:
            patched.analysis = [item.strip() for item in result.corrected_analysis if item.strip()]
        if result.corrected_limitations:
            patched.limitations = [item.strip() for item in result.corrected_limitations if item.strip()]
        trace_event("verification.output", result)
        return self._apply_deterministic_checks(patched)

    def _apply_deterministic_checks(self, answer: AnswerPayload) -> AnswerPayload:
        normalized = answer.model_copy(deep=True)
        normalized.summary = normalized.summary.strip()
        normalized.analysis = self._normalize_text_list(normalized.analysis, fallback="当前回答未生成足够分析内容。")
        normalized.limitations = self._normalize_text_list(
            normalized.limitations,
            fallback="当前结果仅供信息检索与学习参考，不构成投资建议。",
        )
        normalized.sources = self._dedupe_sources(normalized)

        if not normalized.summary:
            normalized.summary = "当前已生成回答，但摘要为空。"

        if self._should_force_insufficient_evidence(normalized):
            normalized.summary = "知识库和官方网页检索均未提供足够依据，当前无法可靠回答该问题。"
            normalized.analysis = [
                "当前问题未命中本地知识库，也未从官方网页检索到足够相关片段。",
                "系统已按保守策略拒绝在无依据时补充常识性结论。",
            ]
            normalized.limitations = [
                "当前回答严格受限于已检索到的证据，不会在依据不足时自由生成。",
            ]
            return normalized

        if self._looks_like_asset_answer(normalized):
            symbol = str(normalized.objective_data.get("symbol") or "").strip()
            if symbol and symbol not in normalized.summary:
                normalized.summary = f"{symbol}：{normalized.summary}"

        normalized.limitations = self._sanitize_limitations(normalized)
        return normalized

    def _normalize_text_list(self, values: list[str], fallback: str) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            compact = re.sub(r"\s+", " ", value).strip()
            if not compact or compact in seen:
                continue
            seen.add(compact)
            normalized.append(compact)
        return normalized[:4] or [fallback]

    def _sanitize_limitations(self, answer: AnswerPayload) -> list[str]:
        question_type = answer.question_type.value if hasattr(answer.question_type, "value") else str(answer.question_type)
        cleaned: list[str] = []
        for item in answer.limitations:
            if self._looks_like_internal_message(item):
                continue
            if self._has_sufficient_evidence(answer) and self._looks_like_insufficient_evidence_message(item):
                continue
            cleaned.append(item)

        fallback = self._default_limitations_for(question_type)
        return self._normalize_text_list(cleaned, fallback=fallback)

    def _dedupe_sources(self, answer: AnswerPayload):
        deduped = []
        seen: set[tuple[str, str | None]] = set()
        for source in answer.sources:
            key = (source.name.strip(), source.value)
            if key in seen or not source.name.strip():
                continue
            seen.add(key)
            deduped.append(source)
        return deduped[:6]

    def _looks_like_asset_answer(self, answer: AnswerPayload) -> bool:
        return answer.question_type in {"asset_price", "asset_trend", "asset_event_analysis"} or answer.question_type.value in {
            "asset_price",
            "asset_trend",
            "asset_event_analysis",
        }

    def _has_sufficient_evidence(self, answer: AnswerPayload) -> bool:
        question_type = answer.question_type.value if hasattr(answer.question_type, "value") else str(answer.question_type)
        if question_type not in {"finance_knowledge", "report_summary"}:
            return False
        source_mode = str(answer.objective_data.get("source_mode") or "").strip().lower()
        return source_mode != "not_found" and bool(answer.sources)

    def _looks_like_internal_message(self, text: str) -> bool:
        lowered = text.lower()
        return "校验提示" in text or "source_mode=" in lowered or "sources 为空" in text

    def _looks_like_insufficient_evidence_message(self, text: str) -> bool:
        return "依据不足" in text or "无法可靠回答" in text

    def _default_limitations_for(self, question_type: str) -> str:
        if question_type == "report_summary":
            return "当前摘要基于检索到的财报片段，不保证覆盖整份报告全部重点。"
        if question_type == "finance_knowledge":
            return "当前版本基于检索片段做抽取式归纳，不等同于完整教材解释。"
        return "当前结果仅供信息检索与学习参考，不构成投资建议。"

    def _should_force_insufficient_evidence(self, answer: AnswerPayload) -> bool:
        question_type = answer.question_type.value if hasattr(answer.question_type, "value") else str(answer.question_type)
        if question_type not in {"finance_knowledge", "report_summary"}:
            return False

        source_mode = str(answer.objective_data.get("source_mode") or "").strip().lower()
        if source_mode == "not_found":
            return True
        return not answer.sources
