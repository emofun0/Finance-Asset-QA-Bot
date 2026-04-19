from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.domain import IntentType


class RoutingDecisionResult(BaseModel):
    intent: IntentType
    need_market_data: bool = False
    need_rag: bool = False
    need_news: bool = False
    extracted_symbol: str | None = Field(default=None, max_length=16)
    extracted_company: str | None = Field(default=None, max_length=120)
    time_range_days: int | None = Field(default=None, ge=1, le=365)
    event_date: str | None = Field(default=None, max_length=32)
    reason: str = Field(default="", max_length=240)


class QueryRewriteResult(BaseModel):
    rewritten_query: str = Field(min_length=1, max_length=500)
    search_keywords: list[str] = Field(default_factory=list, max_length=12)
    notes: list[str] = Field(default_factory=list, max_length=6)


class GeneratedAnswerSections(BaseModel):
    summary: str = Field(min_length=1, max_length=400)
    analysis: list[str] = Field(default_factory=list, max_length=4)
    limitations: list[str] = Field(default_factory=list, max_length=4)


class VerificationResult(BaseModel):
    is_valid: bool
    issues: list[str] = Field(default_factory=list, max_length=6)
    corrected_summary: str | None = Field(default=None, max_length=400)
    corrected_analysis: list[str] = Field(default_factory=list, max_length=4)
    corrected_limitations: list[str] = Field(default_factory=list, max_length=4)


class AgentPlanningResult(BaseModel):
    tool_name: Literal[
        "asset_price",
        "asset_trend",
        "asset_event_analysis",
        "finance_knowledge",
        "web_finance_knowledge",
        "report_summary",
        "web_report_summary",
        "direct_response",
    ]
    thought: str = Field(min_length=1, max_length=240)
    company: str | None = Field(default=None, max_length=120)
    symbol: str | None = Field(default=None, max_length=16)
    time_length: int | None = Field(default=None, ge=1)
    time_unit: Literal["day", "week", "month", "year"] | None = None
    event_date: str | None = Field(default=None, max_length=32)
    rewritten_query: str | None = Field(default=None, max_length=500)
    direct_response: str | None = Field(default=None, max_length=500)
    reason: str = Field(default="", max_length=240)


class EventObservationResult(BaseModel):
    observations: list[str] = Field(default_factory=list, max_length=3)
