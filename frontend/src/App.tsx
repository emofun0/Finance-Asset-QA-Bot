import { startTransition, useEffect, useMemo, useState } from "react";
import { Alert, Button, Input, Layout, Select, Space, Spin, Tag } from "antd";

import { AnswerView } from "./components/AnswerView";
import { TrendChart } from "./components/TrendChart";
import { chat, fetchAssetHistory, fetchLLMModels } from "./services/api";
import type {
  AnswerPayload,
  AssetHistoryResponse,
  HistoryPoint,
  LLMModelCatalogResponse,
  LLMProviderCatalog,
  LLMSelection,
} from "./types/api";

const EXAMPLE_QUESTIONS = [
  "BABA 当前股价是多少？",
  "BABA 最近 30 天涨跌情况如何？",
  "什么是市盈率？",
  "腾讯最近财报摘要是什么？",
];

const ASSET_INTENTS = new Set(["asset_price", "asset_trend", "asset_event_analysis"]);

function normalizeChartData(answer: AnswerPayload, history: AssetHistoryResponse | null) {
  const objectivePoints = answer.objective_data.points;
  const directPoints = Array.isArray(objectivePoints) ? (objectivePoints as HistoryPoint[]) : [];
  const points = history?.points ?? directPoints;
  if (!points.length) {
    return null;
  }

  const trend = history?.trend ?? (typeof answer.objective_data.trend === "string" ? answer.objective_data.trend : null);
  const rawChange = history?.change_pct ?? answer.objective_data.change_pct;
  const changePct = typeof rawChange === "number" ? rawChange : null;
  const symbol =
    history?.symbol ??
    (typeof answer.objective_data.symbol === "string" ? answer.objective_data.symbol : answer.route.extracted_symbol);

  if (!symbol) {
    return null;
  }

  return { symbol, points, trend, changePct };
}

async function resolveHistory(answer: AnswerPayload): Promise<AssetHistoryResponse | null> {
  if (!ASSET_INTENTS.has(answer.question_type)) {
    return null;
  }

  const symbol =
    typeof answer.objective_data.symbol === "string" ? answer.objective_data.symbol : answer.route.extracted_symbol;
  if (!symbol) {
    return null;
  }

  const days =
    typeof answer.objective_data.time_range_days === "number" ? answer.objective_data.time_range_days : 30;
  const objectivePoints = answer.objective_data.points;
  const directPoints = Array.isArray(objectivePoints) ? (objectivePoints as HistoryPoint[]) : [];
  const directTrend = typeof answer.objective_data.trend === "string" ? answer.objective_data.trend : "";
  const directChangePct = typeof answer.objective_data.change_pct === "number" ? answer.objective_data.change_pct : 0;

  if (directPoints.length) {
    return {
      symbol,
      time_range_days: days,
      trend: directTrend,
      change_pct: directChangePct,
      points: directPoints,
    };
  }

  return fetchAssetHistory(symbol, days);
}

function App() {
  const [message, setMessage] = useState(EXAMPLE_QUESTIONS[0]);
  const [answer, setAnswer] = useState<AnswerPayload | null>(null);
  const [history, setHistory] = useState<AssetHistoryResponse | null>(null);
  const [lastRequestId, setLastRequestId] = useState<string>("");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [chartErrorMessage, setChartErrorMessage] = useState<string>("");
  const [catalogErrorMessage, setCatalogErrorMessage] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [catalog, setCatalog] = useState<LLMModelCatalogResponse | null>(null);
  const [selectedProvider, setSelectedProvider] = useState<LLMSelection["provider"]>("ollama");
  const [selectedModel, setSelectedModel] = useState("");

  const activeProviderCatalog = useMemo<LLMProviderCatalog | null>(() => {
    return catalog?.providers.find((provider) => provider.provider === selectedProvider) ?? null;
  }, [catalog, selectedProvider]);

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
        const initialProvider =
          enabledProviders.find((provider) => provider.provider === response.default_provider)?.provider ??
          enabledProviders[0]?.provider ??
          response.default_provider;
        const initialCatalog = response.providers.find((provider) => provider.provider === initialProvider) ?? response.providers[0] ?? null;
        const initialModel = initialCatalog?.default_model ?? initialCatalog?.models[0]?.id ?? "";

        startTransition(() => {
          setCatalog(response);
          setSelectedProvider(initialProvider);
          setSelectedModel(initialModel);
          setCatalogErrorMessage("");
        });
      } catch (error) {
        const nextError = error instanceof Error ? error.message : "模型列表加载失败。";
        if (!active) {
          return;
        }
        startTransition(() => {
          setCatalog(null);
          setSelectedModel("");
          setCatalogErrorMessage(nextError);
        });
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
    if (!activeProviderCatalog) {
      return;
    }
    const hasCurrentModel = activeProviderCatalog.models.some((model) => model.id === selectedModel);
    if (hasCurrentModel) {
      return;
    }
    setSelectedModel(activeProviderCatalog.default_model ?? activeProviderCatalog.models[0]?.id ?? "");
  }, [activeProviderCatalog, selectedModel]);

  async function hydrateHistory(answerPayload: AnswerPayload) {
    try {
      const nextHistory = await resolveHistory(answerPayload);
      startTransition(() => {
        setHistory(nextHistory);
        setChartErrorMessage("");
      });
    } catch (error) {
      const nextError = error instanceof Error ? error.message : "走势图加载失败。";
      startTransition(() => {
        setHistory(null);
        setChartErrorMessage(nextError);
      });
    }
  }

  async function handleSubmit(nextMessage?: string) {
    const text = (nextMessage ?? message).trim();
    if (!text) {
      setErrorMessage("请输入问题。");
      return;
    }
    if (!selectedModel) {
      setErrorMessage("当前没有可用模型，请先检查 Ollama 或 OpenAI 配置。");
      return;
    }

    setLoading(true);
    setErrorMessage("");
    setChartErrorMessage("");

    try {
      const response = await chat(text, {
        provider: selectedProvider,
        model: selectedModel,
      });
      if (!response.success || !response.data) {
        throw new Error(response.error?.message ?? "问答接口返回失败。");
      }

      const nextAnswer = response.data;

      startTransition(() => {
        setMessage(text);
        setAnswer(nextAnswer);
        setHistory(null);
        setLastRequestId(response.request_id);
      });
      void hydrateHistory(nextAnswer);
    } catch (error) {
      const nextError = error instanceof Error ? error.message : "请求失败，请检查后端服务是否已启动。";
      startTransition(() => {
        setAnswer(null);
        setHistory(null);
        setLastRequestId("");
        setErrorMessage(nextError);
        setChartErrorMessage("");
      });
    } finally {
      setLoading(false);
    }
  }

  const chartData = answer ? normalizeChartData(answer, history) : null;

  return (
    <Layout className="page-shell">
      <main className="page-content">
        <section className="masthead">
          <div>
            <p className="eyebrow">Phase 5</p>
            <h1>金融资产问答系统</h1>
            <p className="description">
              一个轻量但结构清晰的前端演示页，展示金融问答、RAG 证据和价格走势。
            </p>
          </div>
          <div className="hero-tags">
            <Tag color="blue">React</Tag>
            <Tag color="gold">FastAPI</Tag>
            <Tag color="green">RAG</Tag>
            <Tag color="purple">Ollama / OpenAI</Tag>
          </div>
        </section>

        <section className="composer panel">
          <div className="panel-heading">
            <div>
              <p className="section-kicker">提问区</p>
              <h3>输入问题并触发后端主链路</h3>
            </div>
          </div>
          <div className="model-selector-row">
            <div className="model-selector-group">
              <span className="selector-label">服务商</span>
              <Select
                value={selectedProvider}
                loading={catalogLoading}
                options={(catalog?.providers ?? []).map((provider) => ({
                  value: provider.provider,
                  label: provider.label,
                  disabled: !provider.enabled || provider.models.length === 0,
                }))}
                onChange={(value) => setSelectedProvider(value)}
              />
            </div>
            <div className="model-selector-group model-selector-model">
              <span className="selector-label">模型</span>
              <Select
                value={selectedModel || undefined}
                loading={catalogLoading}
                placeholder="请选择模型"
                options={(activeProviderCatalog?.models ?? []).map((model) => ({
                  value: model.id,
                  label: model.id,
                }))}
                onChange={(value) => setSelectedModel(value)}
              />
            </div>
          </div>
          {catalogErrorMessage ? <Alert type="warning" message={catalogErrorMessage} showIcon className="catalog-banner" /> : null}
          {!catalogErrorMessage && activeProviderCatalog?.error ? (
            <Alert type="warning" message={activeProviderCatalog.error} showIcon className="catalog-banner" />
          ) : null}
          <Input.TextArea
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            autoSize={{ minRows: 3, maxRows: 6 }}
            placeholder="例如：BABA 最近 30 天涨跌情况如何？"
          />
          <div className="composer-actions">
            <Space wrap>
              {EXAMPLE_QUESTIONS.map((question) => (
                <Button key={question} onClick={() => void handleSubmit(question)}>
                  {question}
                </Button>
              ))}
            </Space>
            <Button
              type="primary"
              size="large"
              loading={loading}
              disabled={!selectedModel}
              onClick={() => void handleSubmit()}
            >
              发送问题
            </Button>
          </div>
          {lastRequestId ? (
            <p className="trace-hint">
              最近一次请求 ID：{lastRequestId} · 当前模型：{selectedProvider}/{selectedModel}
            </p>
          ) : null}
        </section>

        {errorMessage ? <Alert type="error" message={errorMessage} showIcon className="error-banner" /> : null}

        <Spin spinning={loading} tip="正在调用后端并整理结构化结果...">
          <section className="content-grid">
            <div className="primary-column">
              <AnswerView answer={answer} />
            </div>
            <aside className="secondary-column">
              {chartData ? (
                <TrendChart
                  symbol={chartData.symbol}
                  points={chartData.points}
                  trend={chartData.trend}
                  changePct={chartData.changePct}
                />
              ) : (
                <section className="panel chart-empty">
                  <p className="section-kicker">价格走势</p>
                  <h3>当前问题没有可绘制的资产价格数据</h3>
                  <p>当回答类型为股价、趋势或事件归因时，这里会自动展示对应标的的走势图。</p>
                  {chartErrorMessage ? <Alert type="warning" showIcon message={chartErrorMessage} className="chart-warning" /> : null}
                </section>
              )}
            </aside>
          </section>
        </Spin>
      </main>
    </Layout>
  );
}

export default App;
