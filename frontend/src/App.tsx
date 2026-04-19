import { useEffect, useRef, useState } from "react";
import { Alert, Button, Empty, Input, Select, Spin } from "antd";

import { TrendChart } from "./components/TrendChart";
import { fetchLLMModels, resetChatSession, streamChat } from "./services/api";
import type {
  ChatMessagePayload,
  ChatStreamEvent,
  LLMModelCatalogResponse,
  LLMProviderCatalog,
  LLMSelection,
  SourceItem,
} from "./types/api";

const EXAMPLE_QUESTIONS = [
  "BABA 最近 30 天涨跌情况如何？",
  "腾讯最近财报摘要是什么？",
  "什么是市盈率？",
];

type ChatRole = "user" | "assistant";

interface ConversationMessage {
  id: string;
  role: ChatRole;
  text: string;
  sources: SourceItem[];
  chart: ChatMessagePayload["chart"];
  status: "streaming" | "done" | "error";
  transient?: boolean;
}

function createId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getOrCreateSessionId(): string {
  const storageKey = "finance-qa-session-id";
  const existing = window.localStorage.getItem(storageKey);
  if (existing) {
    return existing;
  }
  const generated =
    typeof window.crypto?.randomUUID === "function"
      ? window.crypto.randomUUID()
      : `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  window.localStorage.setItem(storageKey, generated);
  return generated;
}

function createSessionId(): string {
  return typeof window.crypto?.randomUUID === "function"
    ? window.crypto.randomUUID()
    : `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function ChatBubble({ message }: { message: ConversationMessage }) {
  const isAssistant = message.role === "assistant";

  return (
    <div className={`message-row ${isAssistant ? "assistant" : "user"}`}>
      <div className={`message-bubble ${isAssistant ? "assistant" : "user"}`}>
        <div className="message-text">{message.text || (message.status === "streaming" ? "正在生成..." : "")}</div>
        {isAssistant && message.chart ? (
          <div className="message-attachment">
            <TrendChart
              symbol={message.chart.symbol}
              points={message.chart.points}
              trend={message.chart.trend}
              changePct={message.chart.change_pct}
            />
          </div>
        ) : null}
        {isAssistant && message.sources.length ? (
          <div className="message-attachment">
            <div className="source-list">
              {message.sources.map((source, index) => (
                <a
                  key={`${source.name}-${source.value ?? index}`}
                  className="source-chip"
                  href={source.value ?? undefined}
                  target={source.value ? "_blank" : undefined}
                  rel={source.value ? "noreferrer" : undefined}
                >
                  <span>{source.name}</span>
                </a>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [catalogErrorMessage, setCatalogErrorMessage] = useState("");
  const [submitErrorMessage, setSubmitErrorMessage] = useState("");
  const [catalog, setCatalog] = useState<LLMModelCatalogResponse | null>(null);
  const [selectedProvider, setSelectedProvider] = useState<LLMSelection["provider"]>("ollama");
  const [selectedModel, setSelectedModel] = useState("");
  const [sending, setSending] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const sessionIdRef = useRef<string>(getOrCreateSessionId());

  useEffect(() => {
    let active = true;

    async function loadCatalog() {
      setCatalogLoading(true);
      try {
        const response = await fetchLLMModels();
        if (!active) {
          return;
        }
        const enabledProviders = response.providers.filter((provider) => provider.enabled && provider.models.length > 0);
        const preferredOpenAI = enabledProviders.find(
          (provider) => provider.provider === "openai" && provider.models.some((model) => model.id === "gpt-5.4-mini"),
        );
        const initialProvider =
          preferredOpenAI?.provider ??
          enabledProviders.find((provider) => provider.provider === response.default_provider)?.provider ??
          enabledProviders[0]?.provider ??
          response.default_provider;
        const providerCatalog = response.providers.find((provider) => provider.provider === initialProvider) ?? null;
        const initialModel =
          initialProvider === "openai" && providerCatalog?.models.some((model) => model.id === "gpt-5.4-mini")
            ? "gpt-5.4-mini"
            : providerCatalog?.default_model ?? providerCatalog?.models[0]?.id ?? "";
        setCatalog(response);
        setSelectedProvider(initialProvider);
        setSelectedModel(initialModel);
        setCatalogErrorMessage("");
      } catch (error) {
        setCatalog(null);
        setSelectedModel("");
        setCatalogErrorMessage(error instanceof Error ? error.message : "模型列表加载失败。");
      } finally {
        if (active) {
          setCatalogLoading(false);
        }
      }
    }

    void loadCatalog();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const viewport = scrollRef.current;
    if (!viewport) {
      return;
    }
    viewport.scrollTop = viewport.scrollHeight;
  }, [messages, sending]);

  const activeProviderCatalog: LLMProviderCatalog | null =
    catalog?.providers.find((provider) => provider.provider === selectedProvider) ?? null;

  useEffect(() => {
    if (!activeProviderCatalog) {
      return;
    }
    if (activeProviderCatalog.models.some((model) => model.id === selectedModel)) {
      return;
    }
    setSelectedModel(activeProviderCatalog.default_model ?? activeProviderCatalog.models[0]?.id ?? "");
  }, [activeProviderCatalog, selectedModel]);

  function updateAssistantMessage(messageId: string, updater: (current: ConversationMessage) => ConversationMessage) {
    setMessages((current) =>
      current.map((message) => {
        if (message.id !== messageId) {
          return message;
        }
        return updater(message);
      }),
    );
  }

  async function handleSend(nextText?: string) {
    const text = (nextText ?? input).trim();
    if (!text || sending) {
      return;
    }
    if (!selectedModel) {
      setSubmitErrorMessage("当前没有可用模型。");
      return;
    }

    const userMessage: ConversationMessage = {
      id: createId("user"),
      role: "user",
      text,
      sources: [],
      chart: null,
      status: "done",
    };
    const assistantMessageId = createId("assistant");
    const assistantPlaceholder: ConversationMessage = {
      id: assistantMessageId,
      role: "assistant",
      text: "",
      sources: [],
      chart: null,
      status: "streaming",
      transient: true,
    };

    setMessages((current) => [...current, userMessage, assistantPlaceholder]);
    setInput("");
    setSending(true);
    setSubmitErrorMessage("");

    try {
      await streamChat(
        {
          message: text,
          session_id: sessionIdRef.current,
          llm: { provider: selectedProvider, model: selectedModel },
        },
        {
          onEvent: (event: ChatStreamEvent) => {
            if (event.type === "meta") {
              return;
            }

            if (event.type === "delta") {
              updateAssistantMessage(assistantMessageId, (current) => ({
                ...current,
                text: current.transient ? event.text : current.text + event.text,
                transient: false,
              }));
              return;
            }

            if (event.type === "status") {
              updateAssistantMessage(assistantMessageId, (current) => ({
                ...current,
                text: event.text,
                transient: true,
              }));
              return;
            }

            if (event.type === "thought") {
              updateAssistantMessage(assistantMessageId, (current) => ({
                ...current,
                text: `正在思考：${event.text}`,
                transient: true,
              }));
              return;
            }

            if (event.type === "tool") {
              updateAssistantMessage(assistantMessageId, (current) => ({
                ...current,
                text: `正在执行：${event.tool_name}\n${event.summary}`,
                transient: true,
              }));
              return;
            }

            if (event.type === "done") {
              updateAssistantMessage(assistantMessageId, (current) => ({
                ...current,
                text: event.message.text,
                sources: event.message.sources,
                chart: event.message.chart,
                status: "done",
                transient: false,
              }));
              return;
            }

            updateAssistantMessage(assistantMessageId, (current) => ({
              ...current,
              text: event.message,
              status: "error",
            }));
            setSubmitErrorMessage(event.message);
          },
        },
      );
    } catch (error) {
      const nextError = error instanceof Error ? error.message : "发送失败。";
      updateAssistantMessage(assistantMessageId, (current) => ({
        ...current,
        text: nextError,
        status: "error",
      }));
      setSubmitErrorMessage(nextError);
    } finally {
      setSending(false);
    }
  }

  function handleProviderChange(provider: LLMSelection["provider"]) {
    setSelectedProvider(provider);
    const providerCatalog = catalog?.providers.find((item) => item.provider === provider) ?? null;
    const nextModel =
      provider === "openai" && providerCatalog?.models.some((model) => model.id === "gpt-5.4-mini")
        ? "gpt-5.4-mini"
        : providerCatalog?.default_model ?? providerCatalog?.models[0]?.id ?? "";
    setSelectedModel(nextModel);
  }

  async function handleResetSession() {
    if (sending) {
      return;
    }

    setSubmitErrorMessage("");
    try {
      await resetChatSession(sessionIdRef.current);
    } catch (error) {
      setSubmitErrorMessage(error instanceof Error ? error.message : "重启会话失败。");
      return;
    }

    const nextSessionId = createSessionId();
    window.localStorage.setItem("finance-qa-session-id", nextSessionId);
    sessionIdRef.current = nextSessionId;
    setMessages([]);
    setInput("");
  }

  return (
    <div className="chat-shell">
      <div className="chat-frame">
        <header className="chat-header">
          <div>
            <h1>金融问答</h1>
          </div>
          <div className="header-controls">
            <div className="session-pill">当前 Session: {sessionIdRef.current}</div>
            <Button onClick={() => void handleResetSession()} disabled={sending}>
              重启会话
            </Button>
          </div>
        </header>

        {catalogErrorMessage ? <Alert type="warning" message={catalogErrorMessage} showIcon className="top-alert" /> : null}
        {!catalogErrorMessage && activeProviderCatalog?.error ? (
          <Alert type="warning" message={activeProviderCatalog.error} showIcon className="top-alert" />
        ) : null}
        {submitErrorMessage ? <Alert type="error" message={submitErrorMessage} showIcon className="top-alert" /> : null}

        <main className="chat-history" ref={scrollRef}>
          {!messages.length ? (
            <div className="empty-state">
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="开始一段对话" />
              <div className="example-list">
                {EXAMPLE_QUESTIONS.map((question) => (
                  <button key={question} type="button" className="example-chip" onClick={() => void handleSend(question)}>
                    {question}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((message) => <ChatBubble key={message.id} message={message} />)
          )}
        </main>

        <footer className="composer-bar">
          <div className="composer-inner">
            <Input.TextArea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              autoSize={{ minRows: 1, maxRows: 6 }}
              placeholder="输入你的问题"
              onPressEnter={(event) => {
                if (event.shiftKey) {
                  return;
                }
                event.preventDefault();
                void handleSend();
              }}
            />
            <div className="composer-actions">
              <Select
                value={selectedProvider}
                loading={catalogLoading}
                className="composer-select provider-select"
                options={(catalog?.providers ?? []).map((provider) => ({
                  value: provider.provider,
                  label: provider.label,
                  disabled: !provider.enabled || provider.models.length === 0,
                }))}
                onChange={handleProviderChange}
              />
              <Select
                value={selectedModel || undefined}
                loading={catalogLoading}
                className="composer-select model-select"
                placeholder="选择模型"
                options={(activeProviderCatalog?.models ?? []).map((model) => ({
                  value: model.id,
                  label: model.id,
                }))}
                onChange={(value) => setSelectedModel(value)}
              />
              <Spin spinning={sending} size="small" />
              <Button type="primary" onClick={() => void handleSend()} disabled={!input.trim() || !selectedModel} loading={sending}>
                发送
              </Button>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
