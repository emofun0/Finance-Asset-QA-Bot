from __future__ import annotations

from datetime import datetime
import re

from app.core.company_catalog import build_company_alias_map
from app.core.config import settings
from app.llm.client import BaseLLMClient, NullLLMClient
from app.llm.contracts import RoutingDecisionResult
from app.llm.prompts import build_router_prompt
from app.observability.request_trace import trace_event
from app.schemas.domain import IntentType, RouteDecision
from app.schemas.request import ChatRequest


class RouterService:
    _company_aliases = build_company_alias_map()
    _finance_terms = {"PE", "EPS", "ROE", "ROA", "EBITDA"}

    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.llm_client = llm_client or NullLLMClient()

    def route(self, request: ChatRequest) -> RouteDecision:
        heuristic_route = self._route_with_rules(request.message)
        trace_event("router.heuristic", heuristic_route)

        if self._should_skip_llm_route(heuristic_route):
            trace_event(
                "router.llm_skipped",
                {"reason": "heuristic_confident", "intent": heuristic_route.intent},
            )
            final_route = heuristic_route
        else:
            llm_route = self._route_with_llm(request.message, heuristic_route)
            if llm_route is not None:
                trace_event("router.llm", llm_route)
                final_route = self._merge_routes(heuristic_route, llm_route)
            else:
                final_route = heuristic_route

        final_route = self._normalize_route(final_route)
        trace_event("router.final", final_route)
        return final_route

    def _route_with_rules(self, message: str) -> RouteDecision:
        lowered = message.lower()
        company, symbol = self._extract_company_and_symbol(message)
        time_range_days = self._extract_time_range_days(message)
        event_date, event_date_is_inferred = self._extract_event_date(message)

        if self._looks_like_knowledge_question(lowered):
            return RouteDecision(
                intent=IntentType.FINANCE_KNOWLEDGE,
                need_rag=True,
                extracted_symbol=symbol,
                extracted_company=company,
                time_range_days=time_range_days,
                event_date=event_date,
                event_date_is_inferred=event_date_is_inferred,
                decision_source="rule",
                reason="Matched finance concept or explanatory question keywords.",
            )

        if any(keyword in message for keyword in ["财报摘要", "业绩摘要", "年报摘要", "季报摘要", "半年报摘要", "季度财报"]) or any(
            keyword in lowered for keyword in ["quarterly report", "earnings summary", "annual report summary", "quarterly earnings"]
        ):
            return RouteDecision(
                intent=IntentType.REPORT_SUMMARY,
                need_rag=True,
                extracted_symbol=symbol,
                extracted_company=company,
                time_range_days=time_range_days,
                event_date=event_date,
                event_date_is_inferred=event_date_is_inferred,
                decision_source="rule",
                reason="Matched report summary keywords.",
            )

        if self._looks_like_event_question(message, lowered):
            return RouteDecision(
                intent=IntentType.ASSET_EVENT_ANALYSIS,
                need_market_data=True,
                need_news=True,
                extracted_symbol=symbol,
                extracted_company=company,
                time_range_days=time_range_days,
                event_date=event_date,
                event_date_is_inferred=event_date_is_inferred,
                decision_source="rule",
                reason="Matched event analysis keywords.",
            )

        if self._looks_like_trend_question(message, lowered, company, symbol):
            return RouteDecision(
                intent=IntentType.ASSET_TREND,
                need_market_data=True,
                extracted_symbol=symbol,
                extracted_company=company,
                time_range_days=time_range_days,
                event_date=event_date,
                event_date_is_inferred=event_date_is_inferred,
                decision_source="rule",
                reason="Matched trend or time-range keywords.",
            )

        if self._looks_like_price_question(message, lowered):
            return RouteDecision(
                intent=IntentType.ASSET_PRICE,
                need_market_data=True,
                extracted_symbol=symbol,
                extracted_company=company,
                time_range_days=time_range_days,
                event_date=event_date,
                event_date_is_inferred=event_date_is_inferred,
                decision_source="rule",
                reason="Matched price lookup keywords.",
            )

        return RouteDecision(
            intent=IntentType.UNKNOWN,
            extracted_symbol=symbol,
            extracted_company=company,
            time_range_days=time_range_days,
            event_date=event_date,
            event_date_is_inferred=event_date_is_inferred,
            decision_source="rule",
            reason="No strong asset or knowledge signal was found.",
        )

    def _route_with_llm(self, message: str, heuristic_route: RouteDecision) -> RouteDecision | None:
        if not settings.llm_enable_routing or not self.llm_client.is_enabled():
            return None

        system_prompt, user_prompt = build_router_prompt(message, heuristic_route)
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
            return None

        return RouteDecision(
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

    def _merge_routes(self, heuristic_route: RouteDecision, llm_route: RouteDecision) -> RouteDecision:
        use_llm_intent = llm_route.intent != IntentType.UNKNOWN or heuristic_route.intent == IntentType.UNKNOWN

        merged = heuristic_route.model_copy(deep=True)
        if use_llm_intent:
            merged.intent = llm_route.intent
            merged.need_market_data = llm_route.need_market_data
            merged.need_rag = llm_route.need_rag
            merged.need_news = llm_route.need_news
            merged.reason = llm_route.reason or heuristic_route.reason
            merged.decision_source = "llm+rule"

        merged.extracted_symbol = llm_route.extracted_symbol or heuristic_route.extracted_symbol
        merged.extracted_company = llm_route.extracted_company or heuristic_route.extracted_company
        merged.time_range_days = llm_route.time_range_days or heuristic_route.time_range_days
        merged.event_date = llm_route.event_date or heuristic_route.event_date
        merged.event_date_is_inferred = (
            heuristic_route.event_date_is_inferred and merged.event_date == heuristic_route.event_date
        )
        return merged

    def _should_skip_llm_route(self, heuristic_route: RouteDecision) -> bool:
        return heuristic_route.intent != IntentType.UNKNOWN

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

    def _looks_like_knowledge_question(self, lowered_message: str) -> bool:
        keywords = [
            "什么是",
            "区别",
            "市盈率",
            "净利润",
            "收入",
            "每股收益",
            "financial",
            "pe ratio",
            "revenue",
            "profit",
            "earnings per share",
        ]
        return any(keyword in lowered_message for keyword in keywords)

    def _looks_like_event_question(self, message: str, lowered_message: str) -> bool:
        phrases = [
            "为什么涨",
            "为什么跌",
            "为何上涨",
            "为何下跌",
            "为何大涨",
            "为何大跌",
            "受什么影响",
            "是什么原因",
            "原因",
            "催化剂",
        ]
        english = ["why did", "why was", "why rose", "why fell", "what drove", "what caused"]
        if any(phrase in message for phrase in phrases) or any(phrase in lowered_message for phrase in english):
            return True
        if any(keyword in message for keyword in ["为什么", "为何", "原因"]) and any(
            keyword in message for keyword in ["涨", "跌", "上涨", "下跌", "大涨", "大跌"]
        ):
            return True
        return False

    def _looks_like_trend_question(
        self,
        message: str,
        lowered_message: str,
        company: str | None,
        symbol: str | None,
    ) -> bool:
        phrases = ["走势", "涨跌", "最近", "近期", "这周", "这一个月", "表现如何", "波动", "7天", "30天", "7 天", "30 天"]
        english = ["trend", "change", "last 7 days", "last 30 days", "recent performance", "movement"]
        if any(keyword in message for keyword in phrases) or any(keyword in lowered_message for keyword in english):
            return True
        if (company or symbol) and any(keyword in message for keyword in ["近期如何", "最近怎么样", "最近表现"]):
            return True
        return False

    def _looks_like_price_question(self, message: str, lowered_message: str) -> bool:
        return any(keyword in message for keyword in ["股价", "价格", "报价", "现价", "现在多少钱", "当前多少钱"]) or any(
            keyword in lowered_message for keyword in ["price", "quote", "how much now"]
        )

    def _extract_company_and_symbol(self, message: str) -> tuple[str | None, str | None]:
        for symbol_match in re.finditer(r"\b[A-Z][A-Z0-9.-]{0,9}\b", message):
            symbol = symbol_match.group(0)
            if self._is_non_symbol_token(symbol):
                continue
            for _, (company, mapped_symbol) in self._company_aliases.items():
                if mapped_symbol == symbol:
                    return company, symbol
            return None, symbol

        lowered = message.lower()
        for alias, (company, symbol) in self._company_aliases.items():
            if alias in lowered or alias in message:
                return company, symbol

        return None, None

    def _is_non_symbol_token(self, token: str) -> bool:
        if token in self._finance_terms:
            return True
        if re.fullmatch(r"Q[1-4]", token):
            return True
        if re.fullmatch(r"H[12]", token):
            return True
        if re.fullmatch(r"FY\d{2,4}", token):
            return True
        if token in {"TTM", "YOY", "QOQ"}:
            return True
        return False

    def _extract_time_range_days(self, message: str) -> int | None:
        lowered = message.lower()
        if re.search(r"\b90\s*day(s)?\b", lowered) or re.search(r"90\s*天", message):
            return 90
        if re.search(r"\b30\s*day(s)?\b", lowered) or re.search(r"30\s*天", message) or "一个月" in message:
            return 30
        if re.search(r"\b7\s*day(s)?\b", lowered) or re.search(r"7\s*天", message) or "这周" in message:
            return 7
        return None

    def _extract_event_date(self, message: str) -> tuple[str | None, bool]:
        year_month_day = re.search(r"(?P<year>20\d{2})[/-](?P<month>\d{1,2})[/-](?P<day>\d{1,2})", message)
        if year_month_day:
            return (
                self._to_iso_date(
                    int(year_month_day.group("year")),
                    int(year_month_day.group("month")),
                    int(year_month_day.group("day")),
                ),
                False,
            )

        chinese_date = re.search(r"(?:(?P<year>20\d{2})年)?(?P<month>\d{1,2})月(?P<day>\d{1,2})日", message)
        if chinese_date:
            has_explicit_year = chinese_date.group("year") is not None
            year = int(chinese_date.group("year")) if has_explicit_year else self._infer_year(
                int(chinese_date.group("month")),
                int(chinese_date.group("day")),
            )
            return (
                self._to_iso_date(year, int(chinese_date.group("month")), int(chinese_date.group("day"))),
                not has_explicit_year,
            )

        slash_date = re.search(r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})\b", message)
        if slash_date:
            year = self._infer_year(int(slash_date.group("month")), int(slash_date.group("day")))
            return (
                self._to_iso_date(year, int(slash_date.group("month")), int(slash_date.group("day"))),
                True,
            )

        month_names = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        english_date = re.search(
            r"\b(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
            r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(?P<day>\d{1,2})(?:,\s*(?P<year>20\d{2}))?\b",
            message,
            re.IGNORECASE,
        )
        if english_date:
            month = month_names[english_date.group("month").lower()]
            day = int(english_date.group("day"))
            has_explicit_year = english_date.group("year") is not None
            year = int(english_date.group("year")) if has_explicit_year else self._infer_year(month, day)
            return self._to_iso_date(year, month, day), not has_explicit_year

        return None, False

    def _infer_year(self, month: int, day: int) -> int:
        today = datetime.now().date()
        if (month, day) <= (today.month, today.day):
            return today.year
        return today.year - 1

    def _to_iso_date(self, year: int, month: int, day: int) -> str | None:
        try:
            return datetime(year, month, day).date().isoformat()
        except ValueError:
            return None
