from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.schemas.domain import IntentType, RouteDecision
from app.schemas.request import ChatRequest
from app.services.asset_qa_service import AssetQAService


@dataclass(frozen=True)
class FakeSnapshot:
    symbol: str
    company_name: str | None
    currency: str | None
    exchange: str | None
    latest_price: float
    latest_timestamp: datetime
    previous_close: float | None


@dataclass(frozen=True)
class FakePoint:
    timestamp: datetime
    close: float


class FakeMarketDataTool:
    def get_snapshot(self, symbol: str) -> FakeSnapshot:
        return FakeSnapshot(
            symbol=symbol,
            company_name="Alibaba",
            currency="USD",
            exchange="NYSE",
            latest_price=110.0,
            latest_timestamp=datetime(2026, 4, 18, 16, 0, tzinfo=UTC),
            previous_close=105.0,
        )

    def get_history(self, symbol: str, days: int) -> list[FakePoint]:
        start = datetime(2026, 4, 10, 16, 0, tzinfo=UTC)
        return [
            FakePoint(timestamp=start + timedelta(days=index), close=price)
            for index, price in enumerate([100.0, 102.0, 101.0, 106.0, 108.0, 110.0])
        ]


def build_asset_service() -> AssetQAService:
    return AssetQAService(market_data_tool=FakeMarketDataTool())


def test_asset_price_answer_contains_market_data() -> None:
    service = build_asset_service()

    response = service.answer(
        ChatRequest(message="阿里巴巴当前股价是多少？"),
        RouteDecision(
            intent=IntentType.ASSET_PRICE,
            need_market_data=True,
            extracted_symbol="BABA",
            extracted_company="Alibaba",
            reason="test",
        ),
    )

    assert response.objective_data["latest_price"] == 110.0
    assert response.objective_data["daily_change_pct"] > 0
    assert response.sources[0].name == "Yahoo Finance"


def test_asset_trend_answer_contains_points_and_change() -> None:
    service = build_asset_service()

    response = service.answer(
        ChatRequest(message="BABA 最近 7 天涨跌情况如何？"),
        RouteDecision(
            intent=IntentType.ASSET_TREND,
            need_market_data=True,
            extracted_symbol="BABA",
            extracted_company="Alibaba",
            time_range_days=7,
            reason="test",
        ),
    )

    assert response.objective_data["change_pct"] == 10.0
    assert response.objective_data["points"]
    assert "上涨" in response.summary
