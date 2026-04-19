from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import yfinance as yf

from app.core.company_catalog import find_company_profile
from app.core.errors import AppError, UpstreamServiceError
from app.observability.request_trace import trace_event


@dataclass(frozen=True)
class PricePoint:
    timestamp: datetime
    close: float


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    company_name: str | None
    currency: str | None
    exchange: str | None
    latest_price: float
    latest_timestamp: datetime
    previous_close: float | None


class MarketDataTool:
    def get_snapshot(self, symbol: str) -> MarketSnapshot:
        ticker = yf.Ticker(symbol)
        trace_event("market.snapshot.request", {"symbol": symbol})

        try:
            info = ticker.fast_info or {}
            history = ticker.history(period="5d", interval="1d", auto_adjust=False)
        except Exception as exc:  # pragma: no cover - upstream/network variability
            raise UpstreamServiceError(
                message="获取市场数据失败。",
                details={"symbol": symbol, "type": exc.__class__.__name__},
            ) from exc

        if history.empty:
            raise AppError(
                code="ASSET_NOT_FOUND",
                message=f"未找到资产 `{symbol}` 的市场数据。",
                status_code=404,
                details={"symbol": symbol},
            )

        latest_row = history.iloc[-1]
        previous_close = None
        if len(history) >= 2:
            previous_close = float(history.iloc[-2]["Close"])

        latest_ts = self._to_utc_datetime(history.index[-1].to_pydatetime())
        profile = find_company_profile(symbol=symbol)
        company_name = profile.canonical_name if profile else None
        exchange = None
        currency = None

        try:
            metadata = ticker.history_metadata or {}
            exchange = metadata.get("exchangeName") or None
            currency = metadata.get("currency") or None
        except Exception:
            pass

        snapshot = MarketSnapshot(
            symbol=symbol,
            company_name=company_name,
            currency=currency or info.get("currency"),
            exchange=exchange,
            latest_price=float(latest_row["Close"]),
            latest_timestamp=latest_ts,
            previous_close=previous_close,
        )
        trace_event(
            "market.snapshot.response",
            {
                "symbol": snapshot.symbol,
                "currency": snapshot.currency,
                "exchange": snapshot.exchange,
                "latest_price": snapshot.latest_price,
                "previous_close": snapshot.previous_close,
                "latest_timestamp": snapshot.latest_timestamp,
            },
        )
        return snapshot

    def get_history(self, symbol: str, days: int) -> list[PricePoint]:
        trace_event("market.history.request", {"symbol": symbol, "days": days})
        if days <= 0:
            raise AppError(
                code="INVALID_TIME_RANGE",
                message="时间范围必须大于 0。",
                status_code=400,
                details={"days": days},
            )

        ticker = yf.Ticker(symbol)
        fetch_period_days = max(days * 3, 30)

        try:
            history = ticker.history(period=f"{fetch_period_days}d", interval="1d", auto_adjust=False)
        except Exception as exc:  # pragma: no cover - upstream/network variability
            raise UpstreamServiceError(
                message="获取历史市场数据失败。",
                details={"symbol": symbol, "days": days, "type": exc.__class__.__name__},
            ) from exc

        if history.empty:
            raise AppError(
                code="ASSET_NOT_FOUND",
                message=f"未找到资产 `{symbol}` 的历史数据。",
                status_code=404,
                details={"symbol": symbol, "days": days},
            )

        cutoff = datetime.now(UTC) - timedelta(days=days + 3)
        points = [
            PricePoint(
                timestamp=self._to_utc_datetime(index.to_pydatetime()),
                close=float(row["Close"]),
            )
            for index, row in history.iterrows()
            if self._to_utc_datetime(index.to_pydatetime()) >= cutoff
        ]

        if len(points) < 2:
            points = [
                PricePoint(
                    timestamp=self._to_utc_datetime(index.to_pydatetime()),
                    close=float(row["Close"]),
                )
                for index, row in history.tail(min(len(history), max(days, 2))).iterrows()
            ]

        trace_event(
            "market.history.response",
            {
                "symbol": symbol,
                "days": days,
                "points": len(points),
                "start_at": points[0].timestamp if points else None,
                "end_at": points[-1].timestamp if points else None,
            },
        )
        return points

    def get_history_range(self, symbol: str, start_at: datetime, end_at: datetime) -> list[PricePoint]:
        ticker = yf.Ticker(symbol)
        start_utc = self._to_utc_datetime(start_at)
        end_utc = self._to_utc_datetime(end_at)
        trace_event(
            "market.history_range.request",
            {
                "symbol": symbol,
                "start_at": start_utc,
                "end_at": end_utc,
            },
        )

        try:
            history = ticker.history(
                start=start_utc.date().isoformat(),
                end=(end_utc + timedelta(days=1)).date().isoformat(),
                interval="1d",
                auto_adjust=False,
            )
        except Exception as exc:  # pragma: no cover - upstream/network variability
            raise UpstreamServiceError(
                message="获取区间市场数据失败。",
                details={
                    "symbol": symbol,
                    "start_at": start_utc.isoformat(),
                    "end_at": end_utc.isoformat(),
                    "type": exc.__class__.__name__,
                },
            ) from exc

        if history.empty:
            raise AppError(
                code="ASSET_NOT_FOUND",
                message=f"未找到资产 `{symbol}` 在目标区间内的历史数据。",
                status_code=404,
                details={
                    "symbol": symbol,
                    "start_at": start_utc.isoformat(),
                    "end_at": end_utc.isoformat(),
                },
            )

        points = [
            PricePoint(
                timestamp=self._to_utc_datetime(index.to_pydatetime()),
                close=float(row["Close"]),
            )
            for index, row in history.iterrows()
        ]
        trace_event(
            "market.history_range.response",
            {
                "symbol": symbol,
                "start_at": start_utc,
                "end_at": end_utc,
                "points": len(points),
            },
        )
        return points

    def _to_utc_datetime(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
