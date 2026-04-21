from __future__ import annotations

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


class EventObservationResult(BaseModel):
    observations: list[str] = Field(default_factory=list, max_length=3)
