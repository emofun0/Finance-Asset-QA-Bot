# 金融资产问答系统

一个基于 `React + FastAPI` 的全栈金融问答项目，当前仓库已经实现两条主链路：

- 资产问答：查询股票价格、区间走势、涨跌幅，并结合网页检索给出事件归因。
- 金融知识问答：基于本地知识库检索回答术语、财报和基础金融概念问题，必要时可回退到网页检索。

这份 README 只描述当前仓库里的实际实现，不再保留早期设计阶段的过时说明。

## 当前能力

- 前端：React 19 + TypeScript + Vite + Ant Design，提供单页聊天界面、模型切换、会话重置、趋势图展示。
- 后端：FastAPI，提供聊天、资产行情、知识库重建、模型列表、健康检查、trace 查询等接口。
- 模型接入：同时支持 `Ollama` 和 `OpenAI`。
- 市场数据：通过 `yfinance` 获取价格快照和历史行情。
- 网页检索：通过 `ddgs` 做新闻/网页补充检索。
- 本地知识库：使用 `ChromaDB + sentence-transformers` 做向量检索。
- 流式响应：`/api/v1/chat/stream` 使用 SSE，前端手动消费流式事件。
- 可观测性：每次聊天请求都会落一份 trace 到本地目录。

## 项目结构

```text
.
├── backend
│   ├── app
│   │   ├── api            # FastAPI 路由与依赖装配
│   │   ├── core           # 配置、错误、公司目录
│   │   ├── llm            # LLM 客户端、提示词、结构契约
│   │   ├── observability  # trace 记录
│   │   ├── rag            # 知识库构建、向量检索
│   │   ├── schemas        # 请求/响应模型
│   │   ├── services       # Agent、资产问答、知识问答等服务
│   │   └── tools          # 市场数据、网页检索、RAG 工具
│   ├── data/knowledge     # 知识库原始资料、处理结果、索引
│   ├── logs/traces        # 请求 trace
│   ├── scripts            # 知识库构建、评估脚本
│   └── tests
├── frontend
│   └── src
│       ├── components
│       ├── services
│       ├── types
│       └── App.tsx
├── Architecture_diagram.png
└── 金融资产问答系统（Financial Asset QA System）.docx
```

## 技术栈

- 前端：React 19、TypeScript、Vite、Ant Design、ECharts
- 后端：FastAPI、Pydantic、Uvicorn
- LLM：OpenAI、Ollama
- Agent 与工具调用：自定义 LLM 抽象 + 工具执行链
- 行情数据：`yfinance`
- 网页检索：`ddgs`
- 文档处理：BeautifulSoup、PyPDF
- 向量检索：`chromadb`、`sentence-transformers`

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

后端配置定义在 [backend/app/core/config.py](/home/ypw/Codes/Finance_Asset_QA_System/backend/app/core/config.py:1)。默认通过 `python-dotenv` 读取 `.env`。

可参考下面这份最小配置：

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
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-5.1
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

说明：

- 如果只使用 `Ollama`，可以不配置 `OPENAI_API_KEY`。
- 如果要使用 OpenAI，后端会从 `/models` 拉取可用模型列表，前端会优先显示 `gpt-5.4-mini`（若账号可用）。
- `KNOWLEDGE_BASE_DIR` 和 `TRACE_LOG_DIR` 默认都指向 `backend/` 下的本地目录。

### 3. 启动后端

在仓库根目录执行：

```bash
uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000
```

后端启动后可访问：

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/api/v1/health`

### 4. 启动前端

```bash
cd frontend
npm run dev
```

默认开发地址通常是：

- `http://127.0.0.1:5173`
- `http://localhost:5173`

后端 CORS 也默认只放行这两个前端地址。

## 知识库构建

当前知识库目录在 `backend/data/knowledge`，包含：

- `source_manifest.json`：知识源定义
- `raw/`：下载或抽取后的原始文本/文件
- `processed/`：结构化处理结果
- `chroma/`：向量库持久化目录
- `index/`：构建后的索引产物

### 全量构建

```bash
python backend/scripts/build_knowledge_base.py
```

这个脚本会：

1. 读取 `source_manifest.json`
2. 下载或读取原始资料
3. 解析 HTML / PDF / inline_text
4. 写入 `raw/` 和 `processed/`
5. 调用 `KnowledgeBaseBuilder` 重建向量索引

### 仅重建索引

```bash
curl -X POST http://127.0.0.1:8000/api/v1/rag/ingest
```

这个接口只会基于已有 `processed/*.json` 重建索引，不负责重新下载资料。

## 主要接口

- `GET /`：根路径探活
- `GET /api/v1/health`：健康检查
- `GET /api/v1/llm/models`：列出当前可用模型
- `POST /api/v1/chat`：非流式聊天
- `POST /api/v1/chat/stream`：SSE 流式聊天
- `POST /api/v1/chat/session/reset`：清空当前会话上下文
- `GET /api/v1/assets/{symbol}/price`：获取价格快照
- `GET /api/v1/assets/{symbol}/history?days=7`：获取历史价格
- `POST /api/v1/rag/ingest`：重建知识库索引
- `GET /api/v1/traces`：查看 trace 列表
- `GET /api/v1/traces/{request_id}`：查看 trace 详情

## 问答链路说明

### 资产问答

资产相关问题会优先走市场数据与事件检索链路：

- 价格查询：读取实时/最近价格快照
- 走势分析：读取区间历史行情并计算涨跌幅
- 事件归因：结合公司新闻或网页检索补充背景

价格和走势问题不依赖本地知识库回答。

### 金融知识问答

知识类问题会优先走本地知识库检索：

- 金融术语解释
- 基础投资知识
- 财报摘要与报告内容问答

当本地知识不足时，服务会按场景回退到网页检索，并尽量返回带来源的保守回答。

### 流式聊天

流式接口会按 SSE 发送以下事件：

- `meta`：返回 `request_id`
- `agent`：返回中间推理/工具执行进度文本
- `delta`：正文增量片段
- `done`：最终完整消息
- `error`：错误信息

前端当前通过 `fetch + ReadableStream` 手动解析这些事件。

## 会话与 Trace

- 会话 ID 由前端保存在 `localStorage`
- 后端会基于 `session_id` 维护短期上下文
- `POST /api/v1/chat/session/reset` 可清空会话记忆
- 每次聊天请求都会在 `backend/logs/traces` 下落一份 trace

如果你要排查某次回答过程，优先看：

- `GET /api/v1/traces`
- `GET /api/v1/traces/{request_id}`

## 验收问题

根据 [金融资产问答系统（Financial Asset QA System）.docx](金融资产问答系统（Financial Asset QA System）.docx) 中给出的功能示例，验收时可优先使用以下问题：

### 资产问答示例

- 阿里巴巴当前股价是多少？
- BABA 最近 7 天涨跌情况如何？
- 阿里巴巴最近为何 1 月 15 日大涨？
- 特斯拉近期走势如何？

这些问题对应文档中的资产问答能力要求：

- 获取实时或近期价格数据
- 计算涨跌幅，如 7 日、30 日
- 对趋势进行结构化总结，如上涨、下跌、震荡
- 对可能影响因素进行分析，如财报、宏观事件、新闻等

### RAG 示例

- 什么是市盈率？
- 收入和净利润的区别是什么？
- 某公司最近季度财报摘要是什么？

这些问题对应文档中的金融知识问答要求：

- 构建小型金融知识库
- 实现文档分块与向量化
- 完成向量检索
- 在知识不足时结合 Web Search 检索结果生成回答

说明：

- 资产相关问题应优先调用市场数据链路，不应依赖本地知识库直接生成。
- 知识类问题应优先命中本地 RAG；若本地证据不足，再回退到网页检索。
- 文档中还要求回答尽量减少 hallucination，并区分“客观数据”和“分析性描述”。

## 测试

当前仓库提供的是后端测试：

```bash
cd backend
pytest
```

已存在的测试主要覆盖：

- 健康检查
- trace 记录
- Agent 架构
- Agent 的时间字段处理
- Agent 在网页回退场景下的行为

## 已知边界

- 市场数据依赖 Yahoo Finance，可用性受外部服务影响。
- 网页检索依赖 `ddgs`，结果质量会受公开网页覆盖度影响。
- 本地知识库的效果强依赖 `source_manifest.json` 的资料质量与覆盖范围。
- OpenAI 模型列表需要有效 `OPENAI_API_KEY`；未配置时前端只能使用 Ollama。
- 若本机安装了 `pdftotext`，PDF 抽取效果通常比纯 Python 回退更好，但不是硬依赖。

## 架构图

![系统架构图](Architecture_diagram.png)
