from typing import Any

from pydantic import BaseModel, Field

from app.schemas.domain import IntentType, RouteDecision


class SourceItem(BaseModel):
    type: str
    name: str
    value: str | None = None


class AnswerPayload(BaseModel):
    question_type: IntentType
    request_message: str
    summary: str
    objective_data: dict[str, Any] = Field(default_factory=dict)
    analysis: list[str] = Field(default_factory=list)
    sources: list[SourceItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    route: RouteDecision


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class StandardResponse(BaseModel):
    request_id: str
    success: bool
    data: AnswerPayload | None = None
    error: ErrorDetail | None = None


class LLMModelItem(BaseModel):
    id: str


class LLMProviderCatalog(BaseModel):
    provider: str
    label: str
    enabled: bool
    default_model: str | None = None
    models: list[LLMModelItem] = Field(default_factory=list)
    error: str | None = None


class LLMModelCatalogResponse(BaseModel):
    default_provider: str
    providers: list[LLMProviderCatalog] = Field(default_factory=list)


class ChatChartPoint(BaseModel):
    timestamp: str
    close: float


class ChatChartData(BaseModel):
    symbol: str
    trend: str | None = None
    change_pct: float | None = None
    points: list[ChatChartPoint] = Field(default_factory=list)


class ChatMessagePayload(BaseModel):
    text: str
    sources: list[SourceItem] = Field(default_factory=list)
    chart: ChatChartData | None = None


class ChatResponse(BaseModel):
    request_id: str
    message: ChatMessagePayload
