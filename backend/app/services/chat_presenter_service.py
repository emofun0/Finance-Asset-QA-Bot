from __future__ import annotations

from app.schemas.response import AnswerPayload, ChatChartData, ChatChartPoint, ChatMessagePayload, SourceItem
from app.services.asset_qa_service import AssetQAService


class ChatPresenterService:
    def __init__(self, asset_qa_service: AssetQAService) -> None:
        self.asset_qa_service = asset_qa_service

    def build_message(self, answer: AnswerPayload, text_override: str | None = None) -> ChatMessagePayload:
        return ChatMessagePayload(
            text=(text_override.strip() if text_override else self._compose_text(answer)),
            sources=self._filter_sources(answer.sources),
            chart=self._build_chart(answer),
        )

    def _compose_text(self, answer: AnswerPayload) -> str:
        blocks: list[str] = []
        summary = answer.summary.strip()
        if summary:
            blocks.append(summary)

        analysis = [item.strip() for item in answer.analysis if item.strip()]
        if analysis:
            blocks.append("要点：\n" + "\n".join(f"{index}. {item}" for index, item in enumerate(analysis, start=1)))

        limitations = [item.strip() for item in answer.limitations if item.strip()]
        if limitations:
            blocks.append("补充说明：\n" + "\n".join(f"{index}. {item}" for index, item in enumerate(limitations, start=1)))

        return "\n\n".join(blocks).strip() or "当前未生成回答。"

    def _filter_sources(self, sources: list[SourceItem]) -> list[SourceItem]:
        filtered: list[SourceItem] = []
        seen: set[tuple[str, str | None]] = set()
        for source in sources:
            if source.type == "system":
                continue
            name = source.name.strip()
            if not name:
                continue
            if not self._is_web_url(source.value):
                continue
            key = (name, source.value)
            if key in seen:
                continue
            seen.add(key)
            filtered.append(source)
        return filtered[:6]

    def _is_web_url(self, value: str | None) -> bool:
        normalized = str(value or "").strip().lower()
        return normalized.startswith("http://") or normalized.startswith("https://")

    def _build_chart(self, answer: AnswerPayload) -> ChatChartData | None:
        question_type = answer.question_type.value if hasattr(answer.question_type, "value") else str(answer.question_type)
        if question_type not in {"asset_price", "asset_trend", "asset_event_analysis"}:
            return None

        symbol = str(answer.objective_data.get("symbol") or answer.route.extracted_symbol or "").strip()
        if not symbol:
            return None

        days = answer.objective_data.get("time_range_days")
        time_range_days = days if isinstance(days, int) and days > 0 else 30
        history = self.asset_qa_service.get_price_history(symbol, time_range_days)
        raw_points = history.get("points", [])
        if not raw_points:
            return None

        points: list[ChatChartPoint] = []
        for item in raw_points:
            timestamp = str(item.get("timestamp") or "").strip()
            close = item.get("close")
            if not timestamp or not isinstance(close, int | float):
                continue
            points.append(ChatChartPoint(timestamp=timestamp, close=float(close)))

        if len(points) < 2:
            return None

        trend = history.get("trend")
        change_pct = history.get("change_pct")
        return ChatChartData(
            symbol=symbol,
            trend=str(trend) if isinstance(trend, str) else None,
            change_pct=float(change_pct) if isinstance(change_pct, int | float) else None,
            points=points,
        )
