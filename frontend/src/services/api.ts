import axios from "axios";

import type { AssetHistoryResponse, LLMModelCatalogResponse, LLMSelection, StandardResponse } from "../types/api";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "",
  timeout: 120000,
});

export async function chat(message: string, llm: LLMSelection): Promise<StandardResponse> {
  const response = await api.post<StandardResponse>("/api/v1/chat", {
    message,
    llm,
  });
  return response.data;
}

export async function fetchLLMModels(): Promise<LLMModelCatalogResponse> {
  const response = await api.get<LLMModelCatalogResponse>("/api/v1/llm/models");
  return response.data;
}

export async function fetchAssetHistory(symbol: string, days = 30): Promise<AssetHistoryResponse> {
  const response = await api.get<AssetHistoryResponse>(`/api/v1/assets/${symbol}/history`, {
    params: { days },
  });
  return response.data;
}
