# 金融资产问答系统总体方案

## 1. 项目目标

本项目目标是在较短时间内交付一个可演示、结构清晰、回答可靠的金融资产问答系统。系统重点不是覆盖所有金融场景，而是围绕以下能力做出一条完整、可验证的链路：

- 资产价格与近期涨跌分析
- 基于 RAG 的金融知识问答
- 结构化回答生成
- 输出校验与约束
- 工具调用与任务路由
- 分层、解耦、模块化设计

系统默认不做未来走势预测，不输出投资建议。

## 2. 设计原则

- 简单优先：先做单后端服务编排，不引入复杂多 Agent 框架
- 结构优先：所有核心链路都输出统一 JSON，再由前端渲染
- 可验证优先：关键数字、来源、字段结构都需要校验
- 解耦优先：路由、工具、RAG、生成、校验分层设计
- 可扩展优先：模型供应商、数据源、向量库通过抽象接口隔离

## 3. 技术栈

### 前端

- React
- TypeScript
- Vite
- Ant Design
- ECharts
- axios

选择原因：

- 搭建速度快
- 组件成熟，适合快速实现结构化展示
- 图表能力足够支撑价格趋势展示

### 后端

- Python 3.11
- FastAPI
- Pydantic
- LangChain（只使用基础编排与检索封装）
- Chroma
- yfinance
- pytest

选择原因：

- FastAPI + Pydantic 非常适合做强约束 API 和结构化返回
- Python 生态对 LLM、RAG、数据处理更直接
- 该作业第一版无需 Go

### 何时考虑 Go

仅在后续出现以下强需求时再考虑：

- 高频并发行情聚合
- 大量异步爬取任务
- 独立行情网关或任务分发服务

在当前作业范围内，Go 不是必要项。

## 4. 推荐项目结构

```text
finance-asset-qa/
├─ frontend/
│  ├─ src/
│  │  ├─ pages/
│  │  ├─ components/
│  │  ├─ services/
│  │  ├─ hooks/
│  │  ├─ types/
│  │  └─ utils/
│  └─ package.json
├─ backend/
│  ├─ app/
│  │  ├─ api/
│  │  │  ├─ routes/
│  │  │  └─ deps.py
│  │  ├─ core/
│  │  │  ├─ config.py
│  │  │  ├─ logger.py
│  │  │  └─ constants.py
│  │  ├─ schemas/
│  │  │  ├─ request.py
│  │  │  ├─ response.py
│  │  │  └─ domain.py
│  │  ├─ llm/
│  │  │  ├─ client.py
│  │  │  ├─ prompts.py
│  │  │  └─ output_parser.py
│  │  ├─ tools/
│  │  │  ├─ market_data_tool.py
│  │  │  ├─ news_tool.py
│  │  │  ├─ rag_search_tool.py
│  │  │  └─ earnings_tool.py
│  │  ├─ rag/
│  │  │  ├─ ingest.py
│  │  │  ├─ retriever.py
│  │  │  ├─ chunker.py
│  │  │  └─ vector_store.py
│  │  ├─ services/
│  │  │  ├─ router_service.py
│  │  │  ├─ asset_qa_service.py
│  │  │  ├─ knowledge_qa_service.py
│  │  │  ├─ answer_service.py
│  │  │  └─ verification_service.py
│  │  ├─ adapters/
│  │  │  ├─ market/
│  │  │  ├─ llm/
│  │  │  └─ storage/
│  │  └─ main.py
│  ├─ data/
│  │  ├─ knowledge/
│  │  ├─ reports/
│  │  └─ chroma/
│  ├─ tests/
│  └─ requirements.txt
├─ docs/
│  ├─ project-plan.md
│  └─ working-plan.md
└─ README.md
```

## 5. 系统分层设计

### 5.1 API 层

负责接收请求、参数校验、返回统一响应格式。

建议接口：

- `POST /api/v1/chat`
- `POST /api/v1/rag/ingest`
- `GET /api/v1/assets/{symbol}/price`
- `GET /api/v1/assets/{symbol}/history?days=7`
- `GET /api/v1/health`

### 5.2 Router 层

负责识别用户问题属于哪类任务，并决定是否调用工具或检索模块。

建议任务类型：

- `asset_price`
- `asset_trend`
- `asset_event_analysis`
- `finance_knowledge`
- `report_summary`

第一版不做复杂 agent graph，只做规则 + 轻量 LLM 路由。

### 5.3 Tool 层

每个工具单一职责，统一输入输出接口。

示例：

- `MarketDataTool.get_price(symbol)`
- `MarketDataTool.get_history(symbol, days)`
- `RAGSearchTool.search(query, top_k)`
- `NewsTool.search(symbol, date_range)`

### 5.4 RAG 层

负责文档导入、切块、向量化、检索。

知识库建议先覆盖：

- 金融基础术语
- 常见财务指标解释
- 示例公司财报摘要
- 示例公司业务背景

### 5.5 Generation 层

负责把工具结果和检索上下文转换为结构化回答。

重点：

- Prompt 模板职责单一
- 输出 JSON 固定
- 区分客观数据和分析性描述

### 5.6 Verification 层

负责回答生成后的二次校验。

校验内容：

- JSON 结构是否完整
- 数值字段是否存在冲突
- 来源字段是否缺失
- 分析内容是否越界到无依据推断

## 6. 核心请求链路

### 6.1 资产问答链路

1. 用户输入问题
2. Router 识别为价格、趋势或事件分析
3. 提取 `symbol`、`company`、`time_range`
4. 调用市场数据工具
5. 若问题涉及原因分析，再补新闻或事件检索
6. 生成结构化回答
7. 执行输出校验
8. 返回统一 JSON 给前端

### 6.2 知识问答链路

1. 用户输入问题
2. Router 识别为知识类问题
3. 对问题做检索改写
4. 调用 RAG 检索相关片段
5. 基于检索结果生成回答
6. 校验引用和结构
7. 返回统一 JSON 给前端

## 7. Prompt 模板设计

不建议使用一个大 Prompt 处理所有场景。建议拆成以下模板：

### 7.1 意图路由模板

职责：

- 识别任务类型
- 识别是否需要工具
- 提取关键参数

输出建议：

```json
{
  "intent": "asset_trend",
  "symbol": "BABA",
  "time_range_days": 7,
  "need_market_data": true,
  "need_rag": false,
  "need_news": false
}
```

### 7.2 检索改写模板

职责：

- 把用户自然语言问题改写成检索更稳定的 query
- 尽量补齐公司英文名、ticker、财报、指标等关键词

### 7.3 回答生成模板

职责：

- 基于工具结果或检索结果回答
- 明确分离客观数据与分析性结论
- 强制输出结构化 JSON

### 7.4 输出校验模板

职责：

- 检查字段是否缺失
- 检查是否存在无依据结论
- 检查数值与摘要是否一致

## 8. 统一返回结构

建议所有问答都先返回统一结构：

```json
{
  "question_type": "asset_trend",
  "summary": "BABA 近 7 日整体震荡偏强。",
  "objective_data": {
    "symbol": "BABA",
    "current_price": 83.21,
    "change_7d": 4.32,
    "change_30d": -1.14
  },
  "analysis": [
    "近 7 日价格呈现反弹。",
    "短期波动可能与财报预期及市场情绪有关。"
  ],
  "sources": [
    {"type": "market_data", "name": "Yahoo Finance"},
    {"type": "document", "name": "Quarterly Report"}
  ],
  "limitations": [
    "事件归因具有不确定性，不构成投资建议。"
  ]
}
```

该结构由 `Pydantic` 强校验，前端不直接消费自由文本。

## 9. RAG 设计方案

### 9.1 数据范围

第一版只做小型知识库，不追求全量金融文档。

建议材料：

- 金融术语说明文档
- 财务指标解释文档
- 选定 2 到 5 家公司的财报摘要
- 选定公司简介和业务背景

### 9.2 切块策略

- 按自然段切块
- 每块保留元数据：`source`、`company`、`date`、`doc_type`
- 控制 chunk 长度，避免过短或过长

### 9.3 检索策略

第一版使用：

- 向量检索 `top_k`

如效果不稳，再增加：

- BM25 + 向量混合检索

### 9.4 回答约束

- 仅在检索到充分证据时回答
- 检索不足时明确说明依据不足
- 财报数字和结论必须可回溯到检索片段

## 10. 前端页面设计

第一版页面只保留一个核心工作台：

- 顶部：问题输入框
- 左侧或上方：结构化回答卡片
- 右侧或下方：价格走势图

回答展示区分为：

- 结论摘要
- 客观数据
- 分析说明
- 来源引用
- 限制说明

这样既适合作业演示，也能突出“结构化”和“可验证”。

## 11. 风险与控制

### 风险 1：LLM 输出不稳定

控制方式：

- JSON schema 约束
- Pydantic 校验
- 输出失败时自动降级为模板化错误响应

### 风险 2：行情数据源波动或字段变化

控制方式：

- 数据访问通过 adapter 封装
- 对缺失数据做兜底提示

### 风险 3：RAG 检索相关性不稳定

控制方式：

- 做 query rewrite
- 控制知识库范围
- 为检索结果保留来源元数据

### 风险 4：回答看起来完整但依据不足

控制方式：

- 引用来源强制输出
- 对分析内容增加 limitations
- 校验阶段识别“无依据扩写”

## 12. 结论

该项目最合适的第一版路线是：

- 前端使用 React 快速完成结构化展示
- 后端使用 FastAPI + Pydantic 建立清晰 API 与强约束数据结构
- 用小型 RAG 覆盖知识问答
- 用工具调用覆盖价格与走势分析
- 用统一 JSON + 校验层提升回答稳定性

这套方案足够简单，且能够完整体现本作业最看重的 Prompt 设计、RAG、结构约束、任务路由和模块化架构能力。
