export type IntentType =
  | "asset_price"
  | "asset_trend"
  | "asset_event_analysis"
  | "finance_knowledge"
  | "report_summary"
  | "unknown";

export interface RouteDecision {
  intent: IntentType;
  need_market_data: boolean;
  need_rag: boolean;
  need_news: boolean;
  extracted_symbol: string | null;
  extracted_company: string | null;
  time_range_days: number | null;
  reason: string;
}

export interface SourceItem {
  type: string;
  name: string;
  value: string | null;
}

export interface AnswerPayload {
  question_type: IntentType;
  request_message: string;
  summary: string;
  objective_data: Record<string, unknown>;
  analysis: string[];
  sources: SourceItem[];
  limitations: string[];
  route: RouteDecision;
}

export interface ErrorDetail {
  code: string;
  message: string;
  details?: Record<string, unknown> | null;
}

export interface StandardResponse {
  request_id: string;
  success: boolean;
  data: AnswerPayload | null;
  error: ErrorDetail | null;
}

export interface LLMSelection {
  provider: "ollama" | "openai";
  model: string;
}

export interface LLMModelItem {
  id: string;
}

export interface LLMProviderCatalog {
  provider: "ollama" | "openai";
  label: string;
  enabled: boolean;
  default_model: string | null;
  models: LLMModelItem[];
  error: string | null;
}

export interface LLMModelCatalogResponse {
  default_provider: "ollama" | "openai";
  providers: LLMProviderCatalog[];
}

export interface HistoryPoint {
  timestamp: string;
  close: number;
}

export interface AssetHistoryResponse {
  symbol: string;
  time_range_days: number;
  trend: string;
  change_pct: number;
  points: HistoryPoint[];
}

export interface ChatChartData {
  symbol: string;
  trend: string | null;
  change_pct: number | null;
  points: HistoryPoint[];
}

export interface ChatMessagePayload {
  text: string;
  sources: SourceItem[];
  chart: ChatChartData | null;
}

export interface ChatResponse {
  request_id: string;
  message: ChatMessagePayload;
}

export type ChatStreamEvent =
  | { type: "meta"; request_id: string }
  | { type: "delta"; text: string }
  | { type: "done"; request_id: string; message: ChatMessagePayload }
  | { type: "error"; request_id: string; code: string; message: string; details?: Record<string, unknown> | null };
