from datetime import UTC, datetime, timedelta

from app.schemas.domain import IntentType, RouteDecision
from app.schemas.request import ChatRequest
from app.services.asset_qa_service import AssetQAService
from app.tools.market_data_tool import MarketSnapshot, PricePoint


class FakeMarketDataTool:
    def get_snapshot(self, symbol: str) -> MarketSnapshot:
        return MarketSnapshot(
            symbol=symbol,
            company_name="Alibaba",
            currency="USD",
            exchange="NYSE",
            latest_price=110.0,
            latest_timestamp=datetime(2026, 4, 18, 16, 0, tzinfo=UTC),
            previous_close=105.0,
        )

    def get_history(self, symbol: str, days: int) -> list[PricePoint]:
        start = datetime(2026, 4, 10, 16, 0, tzinfo=UTC)
        prices = [100.0, 102.0, 101.0, 106.0, 108.0, 110.0]
        return [PricePoint(timestamp=start + timedelta(days=index), close=price) for index, price in enumerate(prices)]

    def get_history_range(self, symbol: str, start_at: datetime, end_at: datetime) -> list[PricePoint]:
        base = datetime(2026, 1, 13, 16, 0, tzinfo=UTC)
        prices = [100.0, 101.0, 109.0, 108.0]
        return [PricePoint(timestamp=base + timedelta(days=index), close=price) for index, price in enumerate(prices)]


class FakeWebSearchResult:
    def __init__(self, title: str, url: str, content: str) -> None:
        self.chunk_id = "web::0"
        self.score = 1.0
        self.content = content
        self.metadata = {
            "title": title,
            "url": url,
            "doc_type": "event_news",
            "source_name": "reuters.com",
            "company": "Alibaba",
            "symbol": "BABA",
        }


class FakeWebSearchTool:
    def search_company_events(self, query: str, profile: object, top_k: int = 4, event_date: str | None = None):
        return [
            FakeWebSearchResult(
                title="Alibaba shares jump after strong quarterly results",
                url="https://www.reuters.com/example-alibaba",
                content="Alibaba shares rose after quarterly results beat expectations and management highlighted stronger cloud demand.",
            )
        ]


class FakeLowSignalWebSearchTool:
    def search_company_events(self, query: str, profile: object, top_k: int = 4, event_date: str | None = None):
        return [
            FakeWebSearchResult(
                title="Alibaba Group Official Website",
                url="https://www.alibabagroup.com/",
                content="Alibaba Group official website home page and company overview.",
            )
        ]


class FakeFlatMoveMarketDataTool(FakeMarketDataTool):
    def get_history_range(self, symbol: str, start_at: datetime, end_at: datetime) -> list[PricePoint]:
        base = datetime(2026, 1, 13, 16, 0, tzinfo=UTC)
        prices = [100.0, 101.0, 101.5, 101.2]
        return [PricePoint(timestamp=base + timedelta(days=index), close=price) for index, price in enumerate(prices)]


def test_asset_event_analysis_returns_sources_and_event_metrics() -> None:
    service = AssetQAService(
        market_data_tool=FakeMarketDataTool(),
        web_search_tool=FakeWebSearchTool(),
    )

    response = service.answer(
        ChatRequest(message="阿里巴巴最近为何1月15日大涨？"),
        RouteDecision(
            intent=IntentType.ASSET_EVENT_ANALYSIS,
            need_market_data=True,
            need_news=True,
            extracted_company="Alibaba",
            extracted_symbol="BABA",
            event_date="2026-01-15",
            time_range_days=30,
            reason="test",
        ),
    )

    assert response.question_type == IntentType.ASSET_EVENT_ANALYSIS
    assert response.objective_data["event_date"] == "2026-01-15"
    assert response.objective_data["event_change_pct"] > 0
    assert any(source.name.startswith("Alibaba shares jump") for source in response.sources)
    assert any("季度" in item or "quarterly" in item.lower() for item in response.analysis)


def test_asset_event_analysis_requires_explicit_year_for_inferred_dates() -> None:
    service = AssetQAService(
        market_data_tool=FakeMarketDataTool(),
        web_search_tool=FakeWebSearchTool(),
    )

    response = service.answer(
        ChatRequest(message="阿里巴巴最近为何1月15日大涨？"),
        RouteDecision(
            intent=IntentType.ASSET_EVENT_ANALYSIS,
            need_market_data=True,
            need_news=True,
            extracted_company="Alibaba",
            extracted_symbol="BABA",
            event_date="2026-01-15",
            event_date_is_inferred=True,
            time_range_days=30,
            reason="test",
        ),
    )

    assert response.objective_data["source_mode"] == "event_date_needs_confirmation"
    assert "未包含年份" in response.summary


def test_asset_event_analysis_rejects_small_price_moves_as_major_events() -> None:
    service = AssetQAService(
        market_data_tool=FakeFlatMoveMarketDataTool(),
        web_search_tool=FakeWebSearchTool(),
    )

    response = service.answer(
        ChatRequest(message="阿里巴巴 2026-01-15 为什么大涨？"),
        RouteDecision(
            intent=IntentType.ASSET_EVENT_ANALYSIS,
            need_market_data=True,
            need_news=True,
            extracted_company="Alibaba",
            extracted_symbol="BABA",
            event_date="2026-01-15",
            time_range_days=30,
            reason="test",
        ),
    )

    assert response.objective_data["source_mode"] == "price_move_below_threshold"
    assert "达到 3.0%" in response.analysis[1]


def test_asset_event_analysis_filters_low_signal_web_results() -> None:
    service = AssetQAService(
        market_data_tool=FakeMarketDataTool(),
        web_search_tool=FakeLowSignalWebSearchTool(),
    )

    response = service.answer(
        ChatRequest(message="阿里巴巴 2026-01-15 为什么大涨？"),
        RouteDecision(
            intent=IntentType.ASSET_EVENT_ANALYSIS,
            need_market_data=True,
            need_news=True,
            extracted_company="Alibaba",
            extracted_symbol="BABA",
            event_date="2026-01-15",
            time_range_days=30,
            reason="test",
        ),
    )

    assert response.objective_data["source_mode"] == "insufficient_event_evidence"
    assert "未找到足够可靠的事件依据" in response.summary
