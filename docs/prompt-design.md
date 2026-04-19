# Prompt 与结构化生成设计

## 1. 目标

本阶段的目标不是把所有问答都交给大模型自由发挥，而是让模型只在明确边界内工作：

- 检索前做查询改写
- 有证据时做结构化归纳
- 输出后做二次校验
- 无证据时明确拒答或降级

## 2. 提供方抽象

系统当前支持两类 LLM 提供方：

- `ollama`
- `openai`

统一接口定义在 `backend/app/llm/client.py`：

- `BaseLLMClient.is_enabled()`
- `BaseLLMClient.generate_structured(...)`

这样做的目的：

- 上层服务不依赖具体模型厂商
- 本地演示可直接使用 Ollama
- 线上或后续扩展可切换到 OpenAI

## 3. OpenAI 接口选择

OpenAI 接口当前采用官方 `openai` Python SDK，并优先使用新的 `Responses API`，而不是旧的 Chat Completions 作为主实现。

当前代码路径：

- `OpenAILLMClient.generate_structured(...)`

实现方式：

- `client.responses.parse(...)`
- `text_format=<Pydantic Schema>`

这样做的理由：

- 与官方新接口方向一致
- 更适合结构化输出
- 方便直接把结果解析成 Pydantic 模型

## 4. Prompt 模板

### 4.1 查询改写

文件：

- `backend/app/llm/prompts.py`

职责：

- 补齐英文公司名、ticker、报告类型、关键指标
- 把中文问题改写成更适合检索和网页搜索的查询
- 不允许添加未经输入支持的事实

输出结构：

- `rewritten_query`
- `search_keywords`
- `notes`

### 4.2 回答生成

职责：

- 对服务层的草稿回答做结构化润色
- 不改变 `objective_data`
- 不发明新的数值、来源或公司事实

输出结构：

- `summary`
- `analysis`
- `limitations`

关键约束：

- 如果 `source_mode=not_found`
- 或知识/财报类回答没有 `sources`

则必须保留“依据不足”的结论，不能用常识补答。

### 4.3 校验

职责：

- 检查结构完整性
- 检查来源边界
- 检查数字与摘要是否冲突
- 检查是否出现无依据推断

输出结构：

- `is_valid`
- `issues`
- `corrected_summary`
- `corrected_analysis`
- `corrected_limitations`

## 5. 确定性保护规则

除了 LLM 校验，系统还保留了确定性规则。

主要实现位置：

- `backend/app/services/answer_generation_service.py`
- `backend/app/services/verification_service.py`

当前保护策略：

- 检索型回答在 `source_mode=not_found` 时，跳过生成润色
- 检索型回答在 `sources=[]` 时，强制改写为“依据不足”
- 资产类回答允许润色摘要，但不允许修改行情数字
- 校验层会去重 `sources`、`analysis`、`limitations`

## 6. 当前运行策略

默认建议：

- 本地开发：`LLM_PROVIDER=ollama`
- 使用模型：`phi4-reasoning:14b` 或其他本地模型

可选配置：

- `LLM_ENABLE_QUERY_REWRITE=true`
- `LLM_ENABLE_GENERATION=true`
- `LLM_ENABLE_VERIFICATION=true`

如果不配置 OpenAI key：

- OpenAI 客户端会安全降级，不会阻塞系统运行

## 7. 已验证结果

已完成以下验证：

- 本地 Ollama 可输出可解析 JSON
- `BABA 当前股价是多少？` 可走完整生成与校验链路
- `什么是市盈率？` 可从英文官方词条回退并输出中文答案
- 无依据的知识问答不会再被 LLM 常识性补答

## 8. 后续可改进项

- 为 `query_rewrite` 增加更强的术语双语词典
- 为 `verification` 增加更细的数值交叉检查
- 为网页检索增加时间过滤与官方站内优先级
- 为前端展示增加 `source_mode` 和校验提示可视化
