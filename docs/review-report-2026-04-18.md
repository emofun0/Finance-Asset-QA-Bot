# 审查报告（2026-04-18）

## 审查方式

- 依据作业说明：`金融资产问答系统（Financial Asset QA System）.docx`
- 自动化测试：`conda run -n finance-qa python -m pytest backend/tests -q`
- 后端联调：`conda run -n finance-qa uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000`
- 前端验证：`cd frontend && npm run build`
- 人工用例覆盖：
  - `阿里巴巴当前股价是多少？`
  - `BABA 最近 30 天涨跌情况如何？`
  - `什么是市盈率？`
  - `腾讯最近财报摘要是什么？`
  - `阿里巴巴最近为何1月15日大涨？`
  - `阿里巴巴最近为何2025年1月15日大涨？`
  - `特斯拉近期走势如何？`
  - `/api/v1/assets/*`
  - `/api/v1/traces`
  - `/api/v1/rag/ingest`

## 结果摘要

- 自动化测试通过：`28 passed in 0.76s`
- 前端构建通过，但产物存在大 chunk 警告
- 健康检查、价格接口、历史接口、问答主链路、trace 查询、RAG 重建都能跑通
- 资产问答主链路可正确返回结构化行情数据
- 知识问答与财报摘要可返回来源，但检索质量还有明显改进空间

## 已验证通过的功能

- `GET /api/v1/health`
- `GET /api/v1/assets/BABA/price`
- `GET /api/v1/assets/BABA/history?days=7`
- `POST /api/v1/chat`
  - `阿里巴巴当前股价是多少？`
  - `BABA 最近 30 天涨跌情况如何？`
  - `什么是市盈率？`
  - `腾讯最近财报摘要是什么？`
  - `阿里巴巴最近为何1月15日大涨？`
  - `阿里巴巴最近为何2025年1月15日大涨？`
  - `特斯拉近期走势如何？`
- 异常处理
  - `GET /api/v1/assets/BABA/history?days=0` 返回 `422`
  - `GET /api/v1/assets/INVALID123/price` 返回 `404`
  - `POST /api/v1/chat` 缺少 `message` 返回 `422`
  - `POST /api/v1/chat` 输入 `帮我推荐一只股票` 返回 `400`
- `GET /api/v1/traces?limit=3`
- `POST /api/v1/rag/ingest`

## 主要问题

### 1. `AnswerGenerationService` 和 `VerificationService` 没有真正接入运行链路

- 严重程度：高
- 现象：
  - 依赖注入里已经创建了 `AnswerGenerationService`、`VerificationService`
  - 但 `AnswerService.answer()` 在拿到草稿答案后直接返回，没有调用 `.generate()` 或 `.verify()`
  - 这导致 README 和代码结构里声明的“生成/校验层”在运行时实际上不生效
- 代码位置：
  - `backend/app/services/answer_service.py:13-48`
- 影响：
  - `LLM_ENABLE_GENERATION`、`LLM_ENABLE_VERIFICATION` 这两个开关当前是死配置
  - Phase 4 的能力只在单元测试里存在，没有进入真实请求链路
- 修改建议：
  - 在 `AnswerService.answer()` 中显式串接：
    - 先拿 `draft_answer`
    - 对知识问答按条件执行 `answer_generation_service.generate()`
    - 再执行 `verification_service.verify()`
  - 如果当前刻意禁用这两层，就应删除无效依赖和 README 中对应描述，避免“代码声称有、运行时没有”

### 2. 财报摘要对“单一高质量本地文档”判定为证据不足，导致不必要地退化到网页搜索

- 严重程度：高
- 复现：
  - `腾讯最近财报摘要是什么？`
- 实测：
  - 本地检索能命中 6 个腾讯年报/业绩 release chunk，且都来自同一份官方 PDF
  - 但系统仍返回 `source_mode=web_fallback`，结果里混入了通用的 `Investors - Tencent 腾讯` 页面
- 根因：
  - `_has_sufficient_report_coverage()` 强制要求 `len(results) >= 3` 且 `unique_titles >= 2`
  - 当本地只有“一份很强的官方材料”时，会被误判为覆盖不足
- 代码位置：
  - `backend/app/services/knowledge_qa_service.py:69-85`
  - `backend/app/services/knowledge_qa_service.py:440-444`
- 影响：
  - 明明已有本地官方资料，回答却退化到网页抓取
  - 来源质量下降，摘要容易混入低信号导航页
- 修改建议：
  - 把“是否足够”从“标题数”改为“文档质量 + 相关 chunk 密度”
  - 若同一 official PDF 命中多个高分 chunk，应直接视为可用
  - 合并结果时优先保留 `earnings_release`、`annual_report`，弱化 generic IR 页面

### 3. 金融知识问答会把无关概念一起带进分析文本，回答不够干净

- 严重程度：中
- 复现：
  - `什么是市盈率？`
- 实测：
  - 摘要是正确的
  - 但 `analysis` 第一条同时混入了“收入”“净利润”的说明，第二条又是较长的原文摘录
- 根因：
  - `_extract_relevant_excerpt()` 只要命中关键词就截长片段，找不到合适句子时还会直接回退整段 chunk
  - 当前 chunk 粒度偏大，同一 chunk 内可能混有多个术语
- 代码位置：
  - `backend/app/services/knowledge_qa_service.py:312-335`
  - `backend/app/services/knowledge_qa_service.py:381-409`
  - `backend/app/rag/chunker.py:4-35`
- 影响：
  - 用户会看到“摘要正确，但分析夹带无关知识点”的现象
  - 降低作业要求里的“专业、数据驱动、减少 hallucination”的观感
- 修改建议：
  - 术语类知识源按“单概念卡片”切块，不要把多个定义合并进一个 chunk
  - `analysis` 只返回 1 到 2 条高相关句，不要回退整段原文
  - 对概念问答增加基于标题/首句的精确抽取规则

### 4. 直接价格接口缺少公司名字段

- 严重程度：中
- 复现：
  - `GET /api/v1/assets/BABA/price`
- 实测：
  - 返回里 `company: null`
  - 但 `/api/v1/chat` 同类问题可以给出 `company: "Alibaba"`
- 根因：
  - `MarketDataTool.get_snapshot()` 中 `company_name` 被初始化为 `None`，后续未填充
- 代码位置：
  - `backend/app/tools/market_data_tool.py:56-76`
- 影响：
  - 直接资产接口的数据完整性不如问答接口
  - 不利于前端直接消费或做统一展示
- 修改建议：
  - 从 `ticker.info`、`quoteType`、或本地公司目录中补齐 `company_name`
  - 若行情源取不到，则优先使用内部别名表兜底

### 5. 前端把“辅助历史曲线请求”当成主链路的一部分，失败时会把已成功的回答一并清空

- 严重程度：中
- 现象：
  - `handleSubmit()` 先调用 `/api/v1/chat`
  - 然后无条件等待 `resolveHistory()`
  - 只要第二个请求报错，就会进入 `catch`，把已经拿到的 `answer` 清空
- 代码位置：
  - `frontend/src/App.tsx:40-53`
  - `frontend/src/App.tsx:74-98`
- 影响：
  - 聊天主回答本来成功，但只要历史曲线接口抖动，前端就展示成“整个请求失败”
  - 用户体验上会误判系统不可用
- 修改建议：
  - 将 `/chat` 结果先落 UI，再异步补拉历史数据
  - 对 `resolveHistory()` 单独做容错，不要影响主答案展示
  - 如果 `answer.objective_data.points` 已存在，优先直接画图，避免重复请求

### 6. 前端来源列表不可点击，证据可追溯性打折

- 严重程度：低
- 现象：
  - `引用来源` 只把 `name + url` 渲染成纯文本
  - 用户不能直接打开财报、公告或网页来源
- 代码位置：
  - `frontend/src/components/AnswerView.tsx:91-105`
- 影响：
  - 作业要求强调来源与准确性控制，但当前 UI 无法快速复核证据
- 修改建议：
  - 对 URL 来源渲染为可点击链接
  - 区分 `market_data`、`glossary`、`earnings_release`、`web_search` 的展示样式

## 结论

当前项目已经满足“可运行、可演示”的基本要求，资产问答、RAG、网页回退、trace 和前端页面都已经具备。但如果按 reviewer 标准看，主要短板不在“功能缺失”，而在“证据质量控制”和“运行链路一致性”。

建议优先级：

1. 先修 `AnswerService` 的生成/校验层接入问题，明确真实运行链路。
2. 再修财报摘要的本地 RAG 判定逻辑，避免无意义网页回退。
3. 然后收紧知识问答的摘录策略和 chunk 粒度。
4. 最后处理前端容错和来源可点击性。
