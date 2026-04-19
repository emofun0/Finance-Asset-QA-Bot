from enum import StrEnum

from pydantic import BaseModel, Field


class IntentType(StrEnum):
    ASSET_PRICE = "asset_price"
    ASSET_TREND = "asset_trend"
    ASSET_EVENT_ANALYSIS = "asset_event_analysis"
    FINANCE_KNOWLEDGE = "finance_knowledge"
    REPORT_SUMMARY = "report_summary"
    UNKNOWN = "unknown"


class RouteDecision(BaseModel):
    intent: IntentType
    need_market_data: bool = False
    need_rag: bool = False
    need_news: bool = False
    extracted_symbol: str | None = None
    extracted_company: str | None = None
    time_range_days: int | None = None
    event_date: str | None = None
    event_date_is_inferred: bool = False
    decision_source: str = Field(default="rule", description="Which router produced the final decision.")
    reason: str = Field(default="", description="Why the router made this decision.")
