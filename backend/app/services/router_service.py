from __future__ import annotations

from app.core.config import settings
from app.llm.client import BaseLLMClient, NullLLMClient
from app.llm.contracts import RoutingDecisionResult
from app.llm.prompts import build_router_prompt
from app.observability.request_trace import trace_event
from app.schemas.domain import IntentType, RouteDecision
from app.schemas.request import ChatRequest


class RouterService:
    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.llm_client = llm_client or NullLLMClient()

    def route(self, request: ChatRequest) -> RouteDecision:
        baseline = RouteDecision(intent=IntentType.UNKNOWN, decision_source="router_baseline", reason="No heuristic routing is used.")
        if not settings.llm_enable_routing or not self.llm_client.is_enabled():
            trace_event("router.llm_skipped", {"reason": "llm_disabled"})
            return baseline

        system_prompt, user_prompt = build_router_prompt(request.message, baseline)
        trace_event(
            "router.prompt",
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            },
        )
        try:
            llm_result = self.llm_client.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=RoutingDecisionResult,
            )
        except Exception as exc:
            trace_event("router.llm_error", {"type": exc.__class__.__name__, "message": str(exc)})
            return baseline

        route = RouteDecision(
            intent=llm_result.intent,
            need_market_data=llm_result.need_market_data,
            need_rag=llm_result.need_rag,
            need_news=llm_result.need_news,
            extracted_symbol=llm_result.extracted_symbol,
            extracted_company=llm_result.extracted_company,
            time_range_days=llm_result.time_range_days,
            event_date=llm_result.event_date,
            event_date_is_inferred=False,
            decision_source="llm",
            reason=llm_result.reason or "LLM router classified the question.",
        )
        trace_event("router.final", route)
        return self._normalize_route(route)

    def _normalize_route(self, route: RouteDecision) -> RouteDecision:
        normalized = route.model_copy(deep=True)
        if normalized.intent == IntentType.ASSET_PRICE:
            normalized.need_market_data = True
            normalized.need_rag = False
            normalized.need_news = False
        elif normalized.intent == IntentType.ASSET_TREND:
            normalized.need_market_data = True
            normalized.need_rag = False
            normalized.need_news = False
            normalized.time_range_days = normalized.time_range_days or 30
        elif normalized.intent == IntentType.ASSET_EVENT_ANALYSIS:
            normalized.need_market_data = True
            normalized.need_news = True
            normalized.need_rag = False
            normalized.time_range_days = normalized.time_range_days or 30
        elif normalized.intent in {IntentType.FINANCE_KNOWLEDGE, IntentType.REPORT_SUMMARY}:
            normalized.need_rag = True
            normalized.need_market_data = False
            normalized.need_news = False
        return normalized
