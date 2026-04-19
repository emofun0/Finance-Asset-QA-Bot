# 金融资产问答系统

基于当前仓库代码实现的全栈金融问答项目。系统把问题分成两条主链路：

- 资产问答：价格、走势、异动原因分析，走市场数据与网页检索链路。
- 金融知识问答：术语解释、财报摘要，走本地知识库检索与受约束生成链路。

项目包含 React + Vite 前端、FastAPI 后端、RAG 知识库、可切换的 LLM 提供方，以及面向调试的 trace 日志接口。

## 当前已实现能力

- 聊天式问答界面，支持流式输出、模型选择、会话重置。
- 行情接口：`/api/v1/assets/{symbol}/price`、`/api/v1/assets/{symbol}/history`
- 问答接口：`/api/v1/chat`、`/api/v1/chat/stream`
- 路由能力：价格、趋势、事件归因、金融知识、财报摘要
- 本地知识库构建与重建：`backend/scripts/build_knowledge_base.py`、`/api/v1/rag/ingest`
- 网页检索补充：财报摘要与事件归因支持官方站点/新闻站点检索回退
- Trace 查看：`/api/v1/traces`

## 目录结构

```text
.
├── backend
│   ├── app
│   │   ├── api            # FastAPI 路由
│   │   ├── core           # 配置、错误、公司别名表
│   │   ├── llm            # LLM 客户端、提示词、结构化输出契约
│   │   ├── rag            # 分块、索引、检索
│   │   ├── services       # 路由、资产问答、知识问答、Agent、校验等
│   │   ├── tools          # 市场数据、网页检索
│   │   └── observability  # 请求 trace
│   ├── data/knowledge     # source_manifest、raw、processed、index
│   ├── scripts            # 知识库构建与评估脚本
│   └── tests
├── frontend
│   ├── src/components     # 图表等组件
│   ├── src/services       # 前端 API 调用
│   └── src/types
└── 金融资产问答系统（Financial Asset QA System）.docx
```

## 技术栈

- 前端：React 19、TypeScript、Vite、Ant Design、ECharts
- 后端：FastAPI、Pydantic、Uvicorn
- 市场数据：`yfinance`
- 网页检索：`ddgs`
- 文档处理：BeautifulSoup、PyPDF、`pdftotext`
- 向量检索：scikit-learn 本地向量化与稀疏矩阵索引
- LLM 集成：OpenAI、Ollama、LangChain、LangGraph

## 运行方式

### 1. 后端依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 前端依赖

```bash
cd frontend
npm install
```

### 3. 环境变量

后端通过仓库根目录 `.env` 读取配置，代码里实际使用到的核心变量来自 [backend/app/core/config.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/app/core/config.py:1)：

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
LLM_ENABLE_QUERY_REWRITE=true
LLM_ENABLE_VERIFICATION=true
LLM_TIMEOUT_SECONDS=120
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:8b
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.1
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_REASONING_EFFORT=medium
```

前端还需要配置：

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

### 4. 启动后端

```bash
uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000
```

### 5. 启动前端

```bash
cd frontend
npm run dev
```

### 6. 重建知识库

如果修改了 `backend/data/knowledge/source_manifest.json` 或原始资料，需要重新构建：

```bash
python backend/scripts/build_knowledge_base.py
```

也可以在服务启动后调用：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/rag/ingest
```

### 7. 运行测试

```bash
cd backend
pytest
```

## 主要接口

- `GET /api/v1/health`：健康检查
- `GET /api/v1/llm/models`：返回前端可选模型列表
- `POST /api/v1/chat`：非流式问答
- `POST /api/v1/chat/stream`：SSE 流式问答
- `POST /api/v1/chat/session/reset`：清空当前会话缓存
- `GET /api/v1/assets/{symbol}/price`：最新价格快照
- `GET /api/v1/assets/{symbol}/history?days=30`：区间历史价格
- `POST /api/v1/rag/ingest`：重建知识库索引
- `GET /api/v1/traces`：查看请求 trace 列表
- `GET /api/v1/traces/{request_id}`：查看单次请求 trace 详情

## 作业要求对应说明

### 1. 系统架构图

```text
前端 React/Vite
   │
   ├── /api/v1/llm/models
   ├── /api/v1/chat/stream
   └── /api/v1/assets/{symbol}/history
   │
FastAPI API 层
   │
   └── AgentService / AnswerService
        │
        ├── RouterService
        │    ├── 规则路由
        │    └── 可选 LLM 路由增强
        │
        ├── 资产问答链路 AssetQAService
        │    ├── MarketDataTool -> yfinance
        │    └── OfficialWebSearchTool -> ddgs 新闻/网页检索
        │
        └── 知识问答链路 KnowledgeQAService
             ├── QueryRewriteService
             ├── KnowledgeRetriever
             ├── LocalVectorStore
             └── OfficialWebSearchTool（检索不足时回退）
```

对应代码位置：

- 应用入口：[backend/app/main.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/app/main.py:1)
- API 路由聚合：[backend/app/api/routes/__init__.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/app/api/routes/__init__.py:1)
- Agent 主链路：[backend/app/services/agent_service.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/app/services/agent_service.py:1)
- 传统回答链路：[backend/app/services/answer_service.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/app/services/answer_service.py:1)

### 2. 技术选型说明

- 前端选择 React + Vite，是因为当前仓库需要一个轻量的单页问答界面，并且已经实现了流式 SSE 消费、模型切换与图表展示。
- 后端选择 FastAPI，适合快速组织问答、资产、RAG、trace 等多类接口，且与 Pydantic 模型配合紧密。
- 资产行情选择 `yfinance`，代码中的价格快照、历史价格与区间行情均通过 [backend/app/tools/market_data_tool.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/app/tools/market_data_tool.py:1) 获取。
- RAG 没有引入外部向量数据库，而是采用本地分块 + scikit-learn 向量化 + 稀疏矩阵索引，落地简单，适合作业规模与离线构建。
- 网页检索使用 `ddgs`，一方面给事件归因补新闻线索，另一方面给知识问答和财报摘要提供本地资料不足时的补充来源。
- LLM 侧同时兼容 OpenAI 与 Ollama，前端通过 `/api/v1/llm/models` 动态拉取可用模型列表。

### 3. Prompt 设计思路

当前仓库的提示词集中在 [backend/app/llm/prompts.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/app/llm/prompts.py:1)，设计原则是“先约束边界，再输出结构化结果”。

- 路由提示词：要求模型只在 `asset_price`、`asset_trend`、`asset_event_analysis`、`finance_knowledge`、`report_summary`、`unknown` 中做保守分类，不允许编造公司、代码和日期。
- 查询改写提示词：仅改写检索表达，不改变问题意图，重点补齐公司英文名、股票代码、报告类型和财务指标。
- 回答生成提示词：严格要求只能基于草稿中的 `objective_data`、`sources` 和检索证据润色，禁止补数字和补事实。
- 校验提示词：专门检查越界推断、结构缺失、来源不足、数字冲突；如果知识/财报类问题没证据，必须改写成“依据不足”的保守回答。
- Agent 规划提示词：先选工具，再给参数；若系统无法可靠调用工具，则必须退回 `direct_response`。
- 事件归因提示词：只允许基于标题、来源和摘录做中文归因观察，证据不足时只能写“可能与……有关”。

这套设计与代码实现是一致的：资产问答直接返回工具结果，不经过回答生成与校验；知识问答与财报摘要则进入生成和校验阶段。[backend/tests/test_asset_qa_service.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/tests/test_asset_qa_service.py:1) 和 [backend/tests/test_chat_api.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/tests/test_chat_api.py:1) 已覆盖其中一部分行为。

### 4. 数据来源说明

当前仓库里能确认的真实数据来源如下：

- 市场行情：Yahoo Finance，经由 `yfinance` 调用。
- 本地知识库原始资料：`backend/data/knowledge/source_manifest.json` 中登记的 PDF、HTML 和内联文本，处理后写入 `raw/`、`processed/`、`index/`。
- 网页检索：DuckDuckGo Search，经由 `ddgs` 调用。
- 财报/事件检索约束域名：公司 `official_domains` 与若干新闻站点白名单，见 [backend/app/core/company_catalog.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/app/core/company_catalog.py:1) 与 [backend/app/tools/web_search_tool.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/app/tools/web_search_tool.py:1)。

知识库中已经入库的资料类型包括：

- 财报、业绩快报、季度演示材料
- 金融基础概念与投资入门材料
- 中英文混合资料，检索时会结合语言偏好与文档类型做排序

对应处理流程：

1. `source_manifest.json` 定义来源元数据
2. `build_knowledge_base.py` 下载或读取原始资料
3. HTML/PDF 转文本后写入 `processed/`
4. `KnowledgeBaseBuilder` 进行分块、向量化、索引落盘
5. `KnowledgeRetriever` 执行本地检索与重排

### 5. 优化与扩展思考

以下内容只基于当前代码能直接看出的扩展方向，不代表已经实现：

- 市场数据层目前只接了 Yahoo Finance，可继续增加 Alpha Vantage、Polygon 等多数据源，并做交叉校验与失败切换。
- 本地检索目前是轻量级稀疏向量方案，若知识库规模继续扩大，可以升级到专用向量库并加入更细粒度的召回/重排策略。
- 事件归因仍依赖网页标题和摘要，后续可以增加正文抓取、公告抽取和时间线对齐，提升归因质量。
- `company_catalog` 目前维护的是有限公司映射表，后续可扩展为外部配置或数据库，降低代码维护成本。
- 目前已有 trace 接口，但缺少更系统的离线评估报表，可以把 `backend/scripts/run_phase6_eval.py` 进一步做成稳定的评测流程。

## 与题目要求的对应关系

- 资产问答必须使用市场数据 API：已满足，资产链路走 `MarketDataTool`
- 金融知识问答基于 RAG：已满足，包含分块、向量化、检索与网页回退
- Web 前端界面：已满足
- 后端 API 服务：已满足
- LLM 集成模块：已满足，支持 OpenAI 与 Ollama
- 向量检索模块：已满足

当前仍需你在交付阶段自行补齐的内容不在代码里：

- 3 分钟演示视频

## 已知边界

- 资产价格与走势使用最近可得市场数据，不保证逐笔实时。
- 事件归因只提供高概率解释，不做确定性因果判断。
- 财报摘要与知识问答严格受检索证据约束，资料不足时会返回保守结论。
- 仓库当前工作区中 `.env.example` 与旧版 `README.md` 曾被删除；本 README 仅基于现有代码重新整理。
