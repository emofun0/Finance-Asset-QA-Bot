from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.core.company_catalog import find_company_profile
from app.core.errors import AppError
from app.observability.request_trace import trace_event
from app.schemas.domain import IntentType, RouteDecision
from app.schemas.request import ChatRequest
from app.schemas.response import AnswerPayload, SourceItem
from app.tools.market_data_tool import MarketDataTool, PricePoint
from app.tools.web_search_tool import OfficialWebSearchTool


class AssetQAService:
    _event_move_threshold_pct = 3.0

    def __init__(
        self,
        market_data_tool: MarketDataTool,
        web_search_tool: OfficialWebSearchTool | None = None,
    ) -> None:
        self.market_data_tool = market_data_tool
        self.web_search_tool = web_search_tool

    def answer(self, request: ChatRequest, route: RouteDecision) -> AnswerPayload:
        symbol = self._resolve_symbol(route)
        time_range = route.time_range_days or 7

        if route.intent == IntentType.ASSET_PRICE:
            answer = self._build_price_answer(request, route, symbol)
            trace_event("asset.answer", {"intent": route.intent, "summary": answer.summary})
            return answer

        if route.intent == IntentType.ASSET_TREND:
            answer = self._build_trend_answer(request, route, symbol, time_range)
            trace_event("asset.answer", {"intent": route.intent, "summary": answer.summary})
            return answer

        if route.intent == IntentType.ASSET_EVENT_ANALYSIS:
            answer = self._build_event_analysis_answer(request, route, symbol, time_range)
            trace_event("asset.answer", {"intent": route.intent, "summary": answer.summary})
            return answer

        raise AppError(
            code="UNSUPPORTED_ASSET_INTENT",
            message="当前资产服务暂不支持该类型问题。",
            status_code=400,
            details={"intent": route.intent},
        )

    def get_price_snapshot(self, symbol: str) -> dict:
        snapshot = self.market_data_tool.get_snapshot(symbol)
        daily_change_pct = self._compute_change_percent(snapshot.latest_price, snapshot.previous_close)

        return {
            "symbol": snapshot.symbol,
            "company": snapshot.company_name,
            "currency": snapshot.currency,
            "exchange": snapshot.exchange,
            "latest_price": round(snapshot.latest_price, 4),
            "previous_close": round(snapshot.previous_close, 4) if snapshot.previous_close is not None else None,
            "daily_change_pct": round(daily_change_pct, 4) if daily_change_pct is not None else None,
            "as_of": snapshot.latest_timestamp.isoformat(),
        }

    def get_price_history(self, symbol: str, days: int) -> dict:
        history = self.market_data_tool.get_history(symbol, days)
        trend = self._summarize_trend(history)

        return {
            "symbol": symbol,
            "time_range_days": days,
            "trend": trend["trend_label"],
            "change_pct": trend["change_pct"],
            "points": [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "close": round(point.close, 4),
                }
                for point in history
            ],
        }

    def _build_price_answer(self, request: ChatRequest, route: RouteDecision, symbol: str) -> AnswerPayload:
        snapshot = self.market_data_tool.get_snapshot(symbol)
        daily_change_pct = self._compute_change_percent(snapshot.latest_price, snapshot.previous_close)

        change_sentence = "暂无前一交易日数据。"
        if daily_change_pct is not None:
            direction = "上涨" if daily_change_pct > 0 else "下跌" if daily_change_pct < 0 else "持平"
            change_sentence = f"相对前一交易日收盘价，当前约 {direction} {abs(daily_change_pct):.2f}%。"

        return AnswerPayload(
            question_type=route.intent,
            request_message=request.message,
            summary=f"{symbol} 最新价格约为 {snapshot.latest_price:.2f} {snapshot.currency or ''}".strip(),
            objective_data={
                "symbol": symbol,
                "company": route.extracted_company,
                "currency": snapshot.currency,
                "exchange": snapshot.exchange,
                "latest_price": round(snapshot.latest_price, 4),
                "previous_close": round(snapshot.previous_close, 4) if snapshot.previous_close is not None else None,
                "daily_change_pct": round(daily_change_pct, 4) if daily_change_pct is not None else None,
                "as_of": snapshot.latest_timestamp.isoformat(),
            },
            analysis=[
                change_sentence,
                "当前版本的价格问答基于 Yahoo Finance 的最近市场数据。",
            ],
            sources=[
                SourceItem(type="market_data", name="Yahoo Finance", value=symbol),
                SourceItem(type="system", name="router_service", value=route.reason),
            ],
            limitations=[
                "当前价格可能为最近可得市场价格，并非逐笔实时成交价。",
                "当前版本不输出任何投资建议。",
            ],
            route=route,
        )

    def _build_trend_answer(
        self,
        request: ChatRequest,
        route: RouteDecision,
        symbol: str,
        time_range: int,
    ) -> AnswerPayload:
        snapshot = self.market_data_tool.get_snapshot(symbol)
        history = self.market_data_tool.get_history(symbol, time_range)
        trend = self._summarize_trend(history)

        return AnswerPayload(
            question_type=route.intent,
            request_message=request.message,
            summary=f"{symbol} 最近 {time_range} 天整体{trend['trend_label']}，区间涨跌幅约 {trend['change_pct']:.2f}%。",
            objective_data={
                "symbol": symbol,
                "company": route.extracted_company,
                "currency": snapshot.currency,
                "current_price": round(snapshot.latest_price, 4),
                "time_range_days": time_range,
                "change_pct": round(trend["change_pct"], 4),
                "start_price": round(trend["start_price"], 4),
                "end_price": round(trend["end_price"], 4),
                "high_price": round(trend["high_price"], 4),
                "low_price": round(trend["low_price"], 4),
                "as_of": snapshot.latest_timestamp.isoformat(),
                "points": [
                    {
                        "timestamp": point.timestamp.isoformat(),
                        "close": round(point.close, 4),
                    }
                    for point in history
                ],
            },
            analysis=[
                f"样本起点价格约为 {trend['start_price']:.2f}，终点价格约为 {trend['end_price']:.2f}。",
                f"区间最高价约为 {trend['high_price']:.2f}，最低价约为 {trend['low_price']:.2f}。",
                f"根据区间涨跌幅和波动范围，当前归类为“{trend['trend_label']}”。",
            ],
            sources=[
                SourceItem(type="market_data", name="Yahoo Finance", value=symbol),
                SourceItem(type="system", name="router_service", value=route.reason),
            ],
            limitations=[
                "当前趋势结论基于最近可得日线收盘数据，不代表盘中波动。",
                "当前版本仅做历史数据描述，不预测未来走势。",
            ],
            route=route,
        )

    def _build_event_analysis_answer(
        self,
        request: ChatRequest,
        route: RouteDecision,
        symbol: str,
        time_range: int,
    ) -> AnswerPayload:
        snapshot = self.market_data_tool.get_snapshot(symbol)
        if route.event_date and route.event_date_is_inferred:
            return self._build_event_date_confirmation_answer(request, route, symbol, snapshot.currency, time_range)

        profile = find_company_profile(route.extracted_company, symbol)
        event_window = self._build_event_window(symbol, route.event_date, time_range)
        if not self._event_has_significant_move(event_window):
            return self._build_no_significant_move_answer(request, route, symbol, snapshot.currency, time_range, event_window)

        event_results = []
        if self.web_search_tool and profile:
            raw_results = self.web_search_tool.search_company_events(
                request.message,
                profile,
                top_k=6,
                event_date=route.event_date,
            )
            event_results = self._filter_event_results(
                raw_results,
                symbol=symbol,
                company=route.extracted_company or profile.canonical_name,
            )[:4]

        trace_event(
            "asset.event_analysis",
            {
                "symbol": symbol,
                "event_date": route.event_date,
                "event_window": event_window,
                "result_count": len(event_results),
            },
        )

        if not event_results:
            summary = self._build_event_summary(symbol, route.event_date, event_window, found_sources=False)
            return AnswerPayload(
                question_type=route.intent,
                request_message=request.message,
                summary=summary,
                objective_data={
                    "symbol": symbol,
                    "company": route.extracted_company,
                    "currency": snapshot.currency,
                    "event_date": route.event_date,
                    "time_range_days": time_range,
                    "source_mode": "insufficient_event_evidence",
                    **event_window,
                },
                analysis=[
                    self._build_event_move_line(event_window),
                    "当前已获取价格异动窗口，但未检索到足够可靠的公告或新闻证据来解释涨跌原因。",
                ],
                sources=[
                    SourceItem(type="market_data", name="Yahoo Finance", value=symbol),
                    SourceItem(type="system", name="router_service", value=route.reason),
                ],
                limitations=[
                    "事件归因依赖外部新闻、公告和财报材料；当前证据不足时不会强行解释原因。",
                    "该回答只描述已观测到的价格波动，不构成投资建议。",
                ],
                route=route,
            )

        analysis = [self._build_event_move_line(event_window)]
        analysis.extend(self._extract_event_observations(event_results))
        summary = self._build_event_summary(symbol, route.event_date, event_window, found_sources=True)

        return AnswerPayload(
            question_type=route.intent,
            request_message=request.message,
            summary=summary,
            objective_data={
                "symbol": symbol,
                "company": route.extracted_company,
                "currency": snapshot.currency,
                "event_date": route.event_date,
                "time_range_days": time_range,
                "source_mode": "web_event_search",
                **event_window,
            },
            analysis=analysis[:4],
            sources=[SourceItem(type="market_data", name="Yahoo Finance", value=symbol)] + self._to_event_sources(event_results[:4]),
            limitations=[
                "事件归因基于已检索到的公告或新闻线索，只能提供高概率解释而非确定因果。",
                "若需更严谨结论，应结合完整公告、财报电话会和宏观背景交叉验证。",
            ],
            route=route,
        )

    def _build_event_window(self, symbol: str, event_date: str | None, time_range: int) -> dict:
        if event_date:
            center = datetime.fromisoformat(event_date).replace(tzinfo=UTC)
            history = self.market_data_tool.get_history_range(
                symbol,
                start_at=center - timedelta(days=3),
                end_at=center + timedelta(days=3),
            )
            matched_point = next(
                (point for point in history if point.timestamp.date().isoformat() == event_date),
                self._closest_point(history, center),
            )
            previous_point = self._previous_point(history, matched_point)
            reference_price = previous_point.close if previous_point else history[0].close
            change_pct = self._compute_change_percent(matched_point.close, reference_price)
            return {
                "analysis_mode": "event_date",
                "event_date": matched_point.timestamp.date().isoformat(),
                "reference_price": round(reference_price, 4),
                "event_close": round(matched_point.close, 4),
                "event_change_pct": round(change_pct or 0.0, 4),
                "points": [
                    {"timestamp": point.timestamp.isoformat(), "close": round(point.close, 4)}
                    for point in history
                ],
            }

        history = self.market_data_tool.get_history(symbol, time_range)
        trend = self._summarize_trend(history)
        return {
            "analysis_mode": "recent_window",
            "event_date": None,
            "time_range_days": time_range,
            "reference_price": round(trend["start_price"], 4),
            "event_close": round(trend["end_price"], 4),
            "event_change_pct": round(trend["change_pct"], 4),
            "points": [
                {"timestamp": point.timestamp.isoformat(), "close": round(point.close, 4)}
                for point in history
            ],
        }

    def _build_event_summary(
        self,
        symbol: str,
        event_date: str | None,
        event_window: dict,
        *,
        found_sources: bool,
    ) -> str:
        if event_date:
            if found_sources:
                return (
                    f"{symbol} 在 {event_window['event_date']} 附近出现约 {event_window['event_change_pct']:.2f}% 的单日异动，"
                    "系统已检索到可能相关的公告或新闻线索。"
                )
            return (
                f"{symbol} 在 {event_window['event_date']} 附近出现约 {event_window['event_change_pct']:.2f}% 的单日异动，"
                "但当前未找到足够可靠的事件依据。"
            )

        if found_sources:
            return (
                f"{symbol} 最近 {event_window.get('time_range_days') or '一段时间'} 的价格异动已有候选事件线索，"
                "系统基于价格窗口和网页检索做了归因整理。"
            )
        return (
            f"{symbol} 最近一段时间的价格异动已识别，"
            "但当前未找到足够可靠的事件依据。"
        )

    def _build_event_move_line(self, event_window: dict) -> str:
        if event_window.get("analysis_mode") == "event_date":
            return (
                f"目标交易日 {event_window['event_date']} 的收盘价约为 {event_window['event_close']:.2f}，"
                f"相对前一参考价变动约 {event_window['event_change_pct']:.2f}%。"
            )
        return (
            f"最近窗口内价格由约 {event_window['reference_price']:.2f} 变动到 {event_window['event_close']:.2f}，"
            f"区间涨跌幅约 {event_window['event_change_pct']:.2f}%。"
        )

    def _extract_event_observations(self, results: list) -> list[str]:
        observations: list[str] = []
        for result in results[:3]:
            title = str(result.metadata.get("title") or "候选事件")
            excerpt = self._trim_excerpt(result.content)
            observations.append(f"{title}：{excerpt}")
        return observations

    def _to_event_sources(self, results: list) -> list[SourceItem]:
        return [
            SourceItem(
                type=str(result.metadata.get("doc_type") or "event_news"),
                name=str(result.metadata.get("title") or result.metadata.get("source_name") or "事件来源"),
                value=str(result.metadata.get("url") or ""),
            )
            for result in results
        ]

    def _build_event_date_confirmation_answer(
        self,
        request: ChatRequest,
        route: RouteDecision,
        symbol: str,
        currency: str | None,
        time_range: int,
    ) -> AnswerPayload:
        return AnswerPayload(
            question_type=route.intent,
            request_message=request.message,
            summary="问题中的事件日期未包含年份，系统当前不会自动猜测具体年份来做涨跌归因。",
            objective_data={
                "symbol": symbol,
                "company": route.extracted_company,
                "currency": currency,
                "event_date": route.event_date,
                "time_range_days": time_range,
                "source_mode": "event_date_needs_confirmation",
            },
            analysis=[
                f"系统可将“{route.event_date}”识别为候选日期，但该日期缺少显式年份。",
                "请补充完整年份后再做事件归因，以避免把不同年份的异动混为同一次事件。",
            ],
            sources=[SourceItem(type="system", name="router_service", value=route.reason)],
            limitations=[
                "事件归因必须依赖准确日期和可靠公告/新闻证据；当前日期信息不足时系统不会强行解释原因。",
            ],
            route=route,
        )

    def _build_no_significant_move_answer(
        self,
        request: ChatRequest,
        route: RouteDecision,
        symbol: str,
        currency: str | None,
        time_range: int,
        event_window: dict,
    ) -> AnswerPayload:
        return AnswerPayload(
            question_type=route.intent,
            request_message=request.message,
            summary=(
                f"{symbol} 在 {event_window.get('event_date') or '目标窗口'} 的价格变动约为 "
                f"{event_window['event_change_pct']:.2f}%，未达到系统用于“大涨/大跌归因”的显著异动阈值。"
            ),
            objective_data={
                "symbol": symbol,
                "company": route.extracted_company,
                "currency": currency,
                "event_date": event_window.get("event_date"),
                "time_range_days": time_range,
                "source_mode": "price_move_below_threshold",
                **event_window,
            },
            analysis=[
                self._build_event_move_line(event_window),
                f"当前系统要求单日或目标窗口变动绝对值至少达到 {self._event_move_threshold_pct:.1f}% 才进入“大涨/大跌原因”归因流程。",
            ],
            sources=[
                SourceItem(type="market_data", name="Yahoo Finance", value=symbol),
                SourceItem(type="system", name="router_service", value=route.reason),
            ],
            limitations=[
                "当价格异动不显著时，系统不会把普通波动强行解释为事件驱动的大涨或大跌。",
            ],
            route=route,
        )

    def _resolve_symbol(self, route: RouteDecision) -> str:
        if route.extracted_symbol:
            return route.extracted_symbol
        profile = find_company_profile(company=route.extracted_company)
        if profile:
            return profile.symbol
        raise AppError(
            code="MISSING_SYMBOL",
            message="当前无法识别问题中的资产代码。",
            status_code=400,
            details={"company": route.extracted_company},
        )

    def _summarize_trend(self, history: list[PricePoint]) -> dict:
        if len(history) < 2:
            raise AppError(
                code="INSUFFICIENT_HISTORY",
                message="历史数据不足，无法计算趋势。",
                status_code=502,
                details={"points": len(history)},
            )

        start_price = history[0].close
        end_price = history[-1].close
        high_price = max(point.close for point in history)
        low_price = min(point.close for point in history)
        change_pct = self._compute_change_percent(end_price, start_price)
        if change_pct is None:
            raise AppError(
                code="INVALID_HISTORY_DATA",
                message="历史价格无效，无法计算涨跌幅。",
                status_code=502,
            )

        if change_pct >= 2:
            trend_label = "上涨"
        elif change_pct <= -2:
            trend_label = "下跌"
        else:
            trend_label = "震荡"

        return {
            "trend_label": trend_label,
            "change_pct": change_pct,
            "start_price": start_price,
            "end_price": end_price,
            "high_price": high_price,
            "low_price": low_price,
        }

    def _compute_change_percent(self, current: float, reference: float | None) -> float | None:
        if reference is None or reference == 0:
            return None
        return ((current - reference) / reference) * 100

    def _closest_point(self, history: list[PricePoint], center: datetime) -> PricePoint:
        return min(history, key=lambda point: abs(point.timestamp - center))

    def _previous_point(self, history: list[PricePoint], target: PricePoint) -> PricePoint | None:
        previous_candidates = [point for point in history if point.timestamp < target.timestamp]
        if not previous_candidates:
            return None
        return previous_candidates[-1]

    def _trim_excerpt(self, text: str, limit: int = 160) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return f"{compact[: limit - 1]}…"

    def _event_has_significant_move(self, event_window: dict) -> bool:
        return abs(float(event_window.get("event_change_pct") or 0.0)) >= self._event_move_threshold_pct

    def _filter_event_results(
        self,
        results: list,
        *,
        symbol: str,
        company: str | None,
    ) -> list:
        filtered = []
        for result in results:
            title = str(result.metadata.get("title") or "")
            url = str(result.metadata.get("url") or "")
            content = str(result.content or "")
            haystack = f"{title}\n{content}".lower()
            company_term = (company or "").lower()
            symbol_term = symbol.lower()

            if not any(term and term in haystack for term in [company_term, symbol_term]):
                continue
            if self._is_low_signal_event_result(title, url):
                continue
            if not self._contains_event_signal(haystack):
                continue
            filtered.append(result)
        return filtered

    def _is_low_signal_event_result(self, title: str, url: str) -> bool:
        lowered = f"{title} {url}".lower()
        low_signal_markers = [
            "stock price",
            "stock quote",
            "/quote/",
            "/quotes/",
            "earnings call transcripts",
            "earnings-calls",
            "investing/stock/",
        ]
        return any(marker in lowered for marker in low_signal_markers)

    def _contains_event_signal(self, haystack: str) -> bool:
        signals = [
            "earnings",
            "results",
            "guidance",
            "announcement",
            "shares rose",
            "shares fell",
            "jump",
            "drop",
            "财报",
            "业绩",
            "公告",
            "上涨",
            "下跌",
            "大涨",
            "大跌",
        ]
        return any(signal in haystack for signal in signals)
