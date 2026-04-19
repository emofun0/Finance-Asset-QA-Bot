import axios from "axios";

import type {
  AssetHistoryResponse,
  ChatResponse,
  ChatStreamEvent,
  LLMModelCatalogResponse,
  LLMSelection,
  StandardResponse,
} from "../types/api";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "",
  timeout: 120000,
});

const streamBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "";

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

interface StreamChatParams {
  message: string;
  llm: LLMSelection;
}

interface StreamChatHandlers {
  onEvent: (event: ChatStreamEvent) => void;
}

interface ChatStreamErrorPayload {
  request_id: string;
  code: string;
  message: string;
  details?: Record<string, unknown> | null;
}

function parseSSEChunk(chunk: string): ChatStreamEvent[] {
  const events: ChatStreamEvent[] = [];
  const blocks = chunk.split("\n\n").filter(Boolean);

  for (const block of blocks) {
    const lines = block.split("\n");
    let eventName = "";
    const dataLines: string[] = [];

    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    }

    if (!eventName || !dataLines.length) {
      continue;
    }

    const payload = JSON.parse(dataLines.join("\n")) as ChatResponse | ChatStreamEvent;
    if (eventName === "meta") {
      events.push({ type: "meta", request_id: (payload as { request_id: string }).request_id });
    } else if (eventName === "status") {
      const statusPayload = payload as { text: string };
      events.push({ type: "status", text: statusPayload.text });
    } else if (eventName === "thought") {
      const thoughtPayload = payload as { text: string; tool_name?: string };
      events.push({ type: "thought", text: thoughtPayload.text, tool_name: thoughtPayload.tool_name });
    } else if (eventName === "tool") {
      const toolPayload = payload as { tool_name: string; summary: string };
      events.push({ type: "tool", tool_name: toolPayload.tool_name, summary: toolPayload.summary });
    } else if (eventName === "delta") {
      events.push({ type: "delta", text: (payload as { text: string }).text });
    } else if (eventName === "done") {
      const donePayload = payload as ChatResponse;
      events.push({ type: "done", request_id: donePayload.request_id, message: donePayload.message });
    } else if (eventName === "error") {
      const errorPayload = payload as ChatStreamErrorPayload;
      events.push({
        type: "error",
        request_id: errorPayload.request_id,
        code: errorPayload.code,
        message: errorPayload.message,
        details: errorPayload.details,
      });
    }
  }

  return events;
}

export async function streamChat(params: StreamChatParams, handlers: StreamChatHandlers): Promise<void> {
  const response = await fetch(`${streamBaseUrl}/api/v1/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(params),
  });

  if (!response.ok || !response.body) {
    throw new Error("流式连接失败。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });

    const boundary = buffer.lastIndexOf("\n\n");
    if (boundary >= 0) {
      const complete = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      for (const event of parseSSEChunk(complete)) {
        handlers.onEvent(event);
        if (event.type === "error") {
          throw new Error(event.message);
        }
      }
    }

    if (done) {
      if (buffer.trim()) {
        for (const event of parseSSEChunk(buffer)) {
          handlers.onEvent(event);
          if (event.type === "error") {
            throw new Error(event.message);
          }
        }
      }
      return;
    }
  }
}

export async function fetchAssetHistory(symbol: string, days = 30): Promise<AssetHistoryResponse> {
  const response = await api.get<AssetHistoryResponse>(`/api/v1/assets/${symbol}/history`, {
    params: { days },
  });
  return response.data;
}
