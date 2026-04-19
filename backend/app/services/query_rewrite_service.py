from __future__ import annotations

import re

from app.core.config import settings
from app.llm.client import BaseLLMClient, NullLLMClient
from app.llm.contracts import QueryRewriteResult
from app.llm.prompts import build_query_rewrite_prompt
from app.observability.request_trace import trace_event
from app.schemas.domain import IntentType, RouteDecision


class QueryRewriteService:
    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.llm_client = llm_client or NullLLMClient()

    def rewrite(self, message: str, route: RouteDecision) -> str:
        if not settings.llm_enable_query_rewrite or not self.llm_client.is_enabled():
            trace_event("query_rewrite.skipped", {"message": message, "reason": "llm_disabled"})
            return message
        if self._should_skip_rewrite(message, route):
            trace_event(
                "query_rewrite.skipped",
                {
                    "message": message,
                    "reason": "simple_finance_knowledge",
                    "intent": route.intent,
                },
            )
            return message

        system_prompt, user_prompt = build_query_rewrite_prompt(message, route)
        trace_event("query_rewrite.prompt", {"system_prompt": system_prompt, "user_prompt": user_prompt})
        try:
            result = self.llm_client.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=QueryRewriteResult,
            )
        except Exception as exc:
            trace_event("query_rewrite.error", {"type": exc.__class__.__name__, "message": str(exc)})
            return message

        rewritten = result.rewritten_query.strip() or message
        trace_event(
            "query_rewrite.output",
            {
                "rewritten_query": rewritten,
                "search_keywords": result.search_keywords,
                "notes": result.notes,
            },
        )
        return rewritten

    def _should_skip_rewrite(self, message: str, route: RouteDecision) -> bool:
        if route.intent != IntentType.FINANCE_KNOWLEDGE:
            return False
        if route.extracted_company or route.extracted_symbol:
            return False
        if self._is_short_term_question(message):
            return False

        lowered = message.lower()
        concept_keywords = (
            "什么是",
            "是什么意思",
            "定义",
            "概念",
            "区别",
            "市盈率",
            "市净率",
            "净利润",
            "营收",
            "收入",
            "pe ratio",
            "price-to-earnings",
            "price to earnings",
            "beta",
        )
        return any(keyword in lowered for keyword in concept_keywords)

    def _is_short_term_question(self, message: str) -> bool:
        lowered = message.lower()
        asks_definition = any(token in lowered for token in ["什么是", "是什么意思", "定义", "解释", "什么叫", "啥是"])
        if not asks_definition:
            return False

        core = re.sub(r"^\s*(请)?解释(一下)?", "", message)
        core = re.sub(r"^\s*请问", "", core)
        core = re.sub(r"^\s*什么是", "", core)
        core = re.sub(r"^\s*什么叫", "", core)
        core = re.sub(r"^\s*啥是", "", core)
        core = re.sub(r"是什么意思|的定义|定义|解释|一下|？|\?|。", "", core)
        compact = re.sub(r"\s+", "", core)
        return 0 < len(compact) <= 12
