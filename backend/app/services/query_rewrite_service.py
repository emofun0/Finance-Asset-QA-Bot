from __future__ import annotations

from app.core.config import settings
from app.llm.client import BaseLLMClient, NullLLMClient
from app.llm.contracts import QueryRewriteResult
from app.llm.prompts import build_query_rewrite_prompt
from app.observability.request_trace import trace_event
from app.schemas.domain import RouteDecision


class QueryRewriteService:
    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.llm_client = llm_client or NullLLMClient()

    def rewrite(self, message: str, route: RouteDecision) -> str:
        if not settings.llm_enable_query_rewrite or not self.llm_client.is_enabled():
            trace_event("query_rewrite.skipped", {"message": message, "reason": "llm_disabled"})
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
