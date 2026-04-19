# 金融资产问答系统

一个基于 React + FastAPI 的全栈金融问答项目，覆盖两类核心能力：

- 资产问答：价格查询、区间走势、事件归因，实际依赖市场数据与网页检索。
- 金融知识问答：术语解释、财报摘要，实际依赖本地知识库检索与受约束生成。

本 README 只依据当前仓库代码整理，目标是与实现保持一致，并补齐题目 [金融资产问答系统（Financial Asset QA System）.docx](/home/ypw/Codes/Finance_Asset_QA_System/金融资产问答系统（Financial Asset QA System）.docx:1) 中要求 README 回答的验收问题。

## 当前实现概览

- 前端：React 19 + TypeScript + Vite + Ant Design，提供聊天界面、模型选择、会话重置、趋势图展示。
- 后端：FastAPI，提供聊天、资产行情、知识库重建、模型列表、trace 查询接口。
- 资产链路：`MarketDataTool` 通过 `yfinance` 获取价格与历史行情，事件归因通过 `ddgs` 检索新闻和官网网页。
- 知识链路：本地知识库使用 `ChromaDB + dense embedding` 检索；财报、术语库和普通文本采用不同的结构化入库方式，检索不足时支持网页回退。
- LLM：同时支持 OpenAI 与 Ollama。常规问答链路使用自定义 `BaseLLMClient` 抽象；Agent 链路额外使用 LangChain/LangGraph。
- 流式输出：后端 `/api/v1/chat/stream` 返回 SSE，前端使用 `fetch + ReadableStream` 手动消费。
- 观测性：每次聊天请求都会写 trace，可通过 `/api/v1/traces` 查看。

## 目录结构

```text
.
├── backend
│   ├── app
│   │   ├── api            # FastAPI 路由与依赖注入
│   │   ├── core           # 配置、错误、公司目录
│   │   ├── llm            # LLM 客户端、LangChain 工厂、提示词、结构契约
│   │   ├── observability  # trace 记录
│   │   ├── rag            # 分块、索引、检索
│   │   ├── schemas        # 请求/响应模型
│   │   ├── services       # Agent、Answer、Router、Asset、Knowledge 等服务
│   │   └── tools          # 市场数据、网页检索、RAG 检索工具
│   ├── data/knowledge     # source_manifest、raw、processed、index
│   ├── logs/traces        # 请求 trace
│   ├── scripts            # 知识库构建、离线评估脚本
│   └── tests
├── frontend
│   └── src
│       ├── components     # 图表组件
│       ├── services       # API 调用
│       ├── types          # 前端类型定义
│       └── App.tsx        # 单页聊天入口
└── 金融资产问答系统（Financial Asset QA System）.docx
```

## 技术栈

- 前端：React 19、TypeScript、Vite、Ant Design、ECharts
- 后端：FastAPI、Pydantic、Uvicorn
- 市场数据：`yfinance`
- 网页检索：`ddgs`
- 文档处理：BeautifulSoup、PyPDF、`pdftotext`
- 本地检索：`chromadb`、`sentence-transformers`
- LLM 集成：OpenAI、Ollama、LangChain、LangGraph

## 快速启动

### 1. 安装依赖

后端：

```bash
cd backend
pip install -r requirements.txt
```

前端：

```bash
cd frontend
npm install
```

### 2. 配置环境变量

后端配置来自 [backend/app/core/config.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/app/core/config.py:1)，默认会从仓库根目录 `.env` 读取：

```env
APP_NAME=Finance Asset QA System
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
FRONTEND_PORT=5173
KNOWLEDGE_BASE_DIR=backend/data/knowledge
TRACE_LOG_DIR=backend/logs/traces

LLM_PROVIDER=ollama
LLM_ENABLE_ROUTING=true
LLM_ENABLE_GENERATION=true
LLM_ENABLE_VERIFICATION=true
LLM_TIMEOUT_SECONDS=120

OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:8b

OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.1
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_REASONING_EFFORT=medium

RAG_VECTOR_DB_DIR=backend/data/knowledge/chroma
RAG_COLLECTION_NAME=finance_knowledge
RAG_EMBEDDING_PROVIDER=sentence_transformers
RAG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RAG_EMBEDDING_BATCH_SIZE=32
```

前端：

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

### 3. 启动服务

启动后端：

```bash
uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000
```

启动前端：

```bash
cd frontend
npm run dev
```

### 4. 构建或重建知识库

离线全量构建：

```bash
python backend/scripts/build_knowledge_base.py
```

在线仅重建索引：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/rag/ingest
```

两者区别：

- `build_knowledge_base.py`：读取 `source_manifest.json`，下载或读取原始资料，抽取文本，生成 `processed/`，再构建 `index/`。
- `/api/v1/rag/ingest`：只对现有 `processed/*.json` 重新做结构化切块、dense embedding 和 Chroma 落盘，不负责下载原始资料。

### 5. 运行测试

```bash
cd backend
pytest
```

## 主要接口

- `GET /`：根路径提示
- `GET /api/v1/health`：健康检查
- `GET /api/v1/llm/models`：返回前端可选模型
- `POST /api/v1/chat`：非流式问答
- `POST /api/v1/chat/stream`：SSE 流式问答
- `POST /api/v1/chat/session/reset`：清空会话缓存
- `GET /api/v1/assets/{symbol}/price`：价格快照
- `GET /api/v1/assets/{symbol}/history?days=30`：区间历史价格
- `POST /api/v1/rag/ingest`：重建知识库索引
- `GET /api/v1/traces`：trace 列表
- `GET /api/v1/traces/{request_id}`：trace 详情

## 与题目功能要求的对应关系

### 资产问答

题目要求的“资产价格与涨跌分析”已经在代码中实现：

- 价格查询：`AssetQAService._build_price_answer()` 调 `MarketDataTool.get_snapshot()`。
- 区间走势：`AssetQAService._build_trend_answer()` 调 `MarketDataTool.get_history()`，计算涨跌幅与趋势标签。
- 事件归因：`AssetQAService._build_event_analysis_answer()` 先判断价格异动，再调用 `OfficialWebSearchTool.search_company_events()` 检索新闻/网页证据。
- 客观数据与分析描述分离：`AnswerPayload` 中明确拆成 `summary`、`objective_data`、`analysis`、`limitations`。
- 不预测未来走势：趋势回答只描述历史窗口，限制说明里明确不做未来预测。

### 金融知识问答

题目要求的“基于 RAG 的金融知识问答”已经实现：

- 小型知识库：数据放在 `backend/data/knowledge`。
- 文档分块：`KnowledgeBaseBuilder` 会按文档类型分别做结构化切块。
- 向量化：`SentenceTransformerEmbeddingFunction` / 其他 embedding provider 生成 dense embedding。
- 向量检索：`ChromaVectorStore` 持久化存储，`KnowledgeRetriever.search()` 与 `search_report_documents()` 负责召回和重排。
- Web Search 补充：当本地知识或财报材料覆盖不足时，`KnowledgeQAService` 会回退到 `OfficialWebSearchTool`。
- 不完全依赖自由生成：知识和财报回答先检索，再做受限整理；证据不足时会返回“无法可靠回答”。

### 市场数据集成

题目要求“资产问答必须使用外部行情 API”也已满足：

- 当前接入的数据源是 Yahoo Finance，经 `yfinance` 封装在 `MarketDataTool` 中。
- 资产问题不会走本地知识库回答价格和走势。

### 技术要求

题目要求中的四个模块都具备：

- Web 前端界面：`frontend/src/App.tsx`
- 后端 API 服务：`backend/app/main.py`
- LLM 集成模块：`backend/app/llm`
- 向量检索模块：`backend/app/rag`

## README 验收问题回答

题目文档要求 README 包含“系统架构图、技术选型说明、Prompt 设计思路、数据来源说明、优化与扩展思考”。下面按当前代码逐项回答。

### 1. 系统架构图

![架构图](Architecture_diagram.png)

对应代码：

- 应用入口：[backend/app/main.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/app/main.py:1)
- 聊天路由：[backend/app/api/routes/chat.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/app/api/routes/chat.py:17)
- 依赖注入：[backend/app/api/deps.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/app/api/deps.py:48)
- Agent 主链路：[backend/app/services/agent_service.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/app/services/agent_service.py:186)
- 回退链路：[backend/app/services/answer_service.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/app/services/answer_service.py:24)

### 2. 技术选型说明

- React + Vite：当前前端是单页问答应用，包含聊天、模型选择、趋势图和 SSE 消费，React/Vite 足够轻量。
- FastAPI：路由、类型校验、错误处理和流式响应都实现得直接清晰，适合这个项目的 API 组织方式。
- `yfinance`：用于价格快照、区间历史、事件窗口行情，满足资产问答必须依赖市场数据的要求。
- `ddgs`：用于新闻检索、官方站点检索和财报网页 fallback。
- `chromadb`：持久化存储 dense 向量与 metadata，支持过滤检索。
- `sentence-transformers`：默认本地 dense embedding 实现，可替换为 OpenAI 或 Ollama embedding。
- OpenAI / Ollama 双实现：支持本地模型和云模型切换。
- LangChain / LangGraph：只在 Agent 规划链路里用，不是整个后端的统一抽象层。

### 3. Prompt 设计思路

提示词集中在 [backend/app/llm/prompts.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/app/llm/prompts.py:1)，总体原则是“结构化约束优先，生成只在证据边界内进行”。

- 路由提示词：把问题分类为价格、趋势、事件归因、金融知识、财报摘要或 unknown，要求保守，不编造公司、代码、日期。
- Agent 规划提示词：让模型先选工具，再产出结构化参数；同时统一生成 `rewritten_query`，补全正式英文公司名、股票代码、报告类型和关键指标；如果信息不足则输出 `direct_response`。
- 回答生成提示词：只允许基于草稿中的 `objective_data`、`sources` 和已检索证据润色。
- 聊天回答提示词：把工具结果整理成面向用户的正文，不输出 JSON。
- 校验提示词：在知识/财报场景下检查越界推断和证据不足；财报场景要求区分年度与季度口径，不能混写不同 period 的数字。
- 事件归因提示词：只允许根据标题、来源和摘录生成中文归因观察。

### 4. 数据来源说明

当前代码实际使用的数据来源如下：

- 市场行情：Yahoo Finance，经 `yfinance` 调用。
- 本地知识库原始资料：`backend/data/knowledge/source_manifest.json` 中登记的 PDF、HTML 和内联文本。目前本地知识库的缺乏是影响rag效果的主要因素，由于我对金融知识库了解不多，仅用爬虫抓取了一些文件和财报，需要后续补充。
- 网页检索：DuckDuckGo Search，经 `ddgs` 调用。
- 事件/财报网页过滤：结合公司官网域名白名单和新闻站点白名单。

知识库处理流程：

1. `source_manifest.json` 描述文档元数据。
2. `backend/scripts/build_knowledge_base.py` 下载或读取原始资料。
3. HTML/PDF 转文本后写入 `raw/` 与 `processed/`。
4. `KnowledgeBaseBuilder` 进行结构化切块、写入 `chunks.jsonl`、生成 dense embedding，并写入 `ChromaDB`。
5. `KnowledgeRetriever` 在运行时执行向量检索、metadata 过滤与重排。

RAG 细节：

- 当前默认使用 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 生成 dense embedding，也可以通过配置切换到 OpenAI 或 Ollama embedding。
- 向量库存储在 `backend/data/knowledge/chroma/`，collection 名默认是 `finance_knowledge`。
- 检索时会同时利用向量相似度和 metadata 过滤，例如 `company`、`symbol`、`doc_type`、`language`、`chunk_kind`。
- 索引落盘文件包括：
  - `backend/data/knowledge/index/chunks.jsonl`
  - `backend/data/knowledge/index/manifest.json`
  - `backend/data/knowledge/chroma/`
- 其中 `chunks.jsonl` 存 chunk 文本和元数据，`manifest.json` 存构建统计与 embedding 配置，`chroma/` 存持久化向量库。

切块策略：

- 财报文档：
  - 会优先抽成 `report_profile`、`report_metric`、`report_table` 三类 chunk。
  - `report_metric` 只保留带关键财务词和数字的高信号行。
  - `report_table` 会尽量保留表格型行，便于后续把数字上下文整体交给大模型。
- 术语库：
  - 优先抽取 `term -> definition` 结构，入库为 `glossary_term`。
  - 无法稳定抽取时才退化为较短的 `glossary_text`。
- 普通知识文本：
  - 使用更短的文本 chunk，默认 `320` 字符，overlap 更小。
  - 目标是提升概念定义和教材类内容的召回精度。

运行时检索：

- 一般知识问答走 `KnowledgeRetriever.search()`，以 dense 向量为主，叠加少量规则重排。
- 财报问答走 `KnowledgeRetriever.search_report_documents()`，优先召回 `report_metric / report_table / report_profile`。
- 财报命中后，`KnowledgeQAService` 还会按 `doc_id` 回收整份报告的主要数值块和表格块，组装为 `objective_data.report_context`，交给后续 LLM 归纳。
- 这样做的目标不是只让模型看 top-k 碎片，而是让它在一份报告的完整数字上下文内做摘要，并明确区分全年和单季口径。

财报摘要专用检索：

- 流程：
  - 先召回最相关的财报结构化 chunk。
  - 再按命中的主文档 `doc_id` 把同一份报告中的主要数字和表格整体取回。
  - 把这些内容写入 `objective_data.report_context`。
  - 之后由回答生成提示词要求模型只基于这份报告里的数字和表格归纳，并区分年度与季度数据。
- 本地财报证据不足时，`KnowledgeQAService` 会自动回退到 `OfficialWebSearchTool.search_company_reports()`。

### 5. 优化与扩展思考

以下是基于现有代码结构能直接落地的扩展方向：

- 市场数据多源化：加入 Alpha Vantage、Polygon 等，并对价格结果做交叉校验。
- 本地金融数据库：当前检索了一些公司财报和属于网站，知识库较小，不能有效覆盖大部分金融术语，依赖外部检索回退。
- 检索升级：可以增加 reranker 或 cross-encoder 做二次精排。
- 财报摘要增强：当前已经支持结构化数字块和表格块回收，后续还可以进一步做更严格的“年度/季度分栏摘要”和表格列语义解析。
- 事件归因增强：目前只基于新闻标题/摘要和少量网页文本，可扩展为公告正文抓取与时间线对齐。
- Query Rewrite 增强：当前已统一由 Agent 规划阶段生成 `rewritten_query`，后续可继续加入更严格的时间归一化和指标模板。
- 前端体验增强：当前只有单页聊天和趋势图，可增加 trace 可视化、工具步骤面板、对话历史持久化。

## 关键实现细节

### 请求流与流式返回

- `/api/v1/chat` 和 `/api/v1/chat/stream` 都直接构造 `AgentService`。
- `AgentService` 启用时，会先做 agent planning，再决定调用资产工具还是知识工具。
- Agent 不可用或报错时，回退到 `AnswerService`。
- SSE 在 FastAPI 路由层通过 `StreamingResponse` 建立。
- `AgentService.stream_chat()` 只负责产出内部事件：`status`、`thought`、`tool`、`final`。

### 路由机制

- `RouterService.route()` 使用 `llm_client.generate_structured(..., schema=RoutingDecisionResult)` 做结构化路由。
- 当 LLM 未启用或调用失败时，只返回 `IntentType.UNKNOWN` baseline。
- README 里不能再写“规则路由 + LLM 增强”；当前实现是 LLM 主导路由。

### 回答生成与校验

- 资产问答：直接返回工具结果，不走自由生成。
- 金融知识问答：`AnswerService` 下会进入 `KnowledgeQAService`，但 `AnswerGenerationService` 对 `finance_knowledge` 默认跳过生成，保持检索式回答。
- 财报摘要：如果检索到足够证据，会把 `report_context` 一并传给生成与校验阶段。
- 证据不足时，`VerificationService` 会强制改写为保守结论。

## 已实现的前端交互

- 模型列表拉取与切换
- 聊天输入与示例问题
- SSE 流式文本展示
- 会话 ID 持久化到 `localStorage`
- 会话重置
- 资产问题趋势图展示
- 来源链接展示

## 已知边界

- 价格和历史行情来自最近可得市场数据，不保证逐笔实时。
- 走势分析基于日线历史数据，不覆盖盘中波动。
- 事件归因只给高概率解释，不给确定性因果结论。
- 财报摘要和知识问答受限于本地资料与网页检索覆盖度，资料不足时会拒答。
- `/api/v1/rag/ingest` 不能替代离线全量抓取脚本，它只重建索引。
