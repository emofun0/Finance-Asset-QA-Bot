from __future__ import annotations

import json

from app.llm.contracts import AgentPlanningResult, EventObservationResult
from app.schemas.domain import RouteDecision
from app.schemas.response import AnswerPayload


def build_router_prompt(message: str, heuristic_route: RouteDecision) -> tuple[str, str]:
    system_prompt = (
        "你是金融资产问答系统中的路由器。"
        "你负责把用户问题分类为资产价格、资产趋势、资产事件归因、金融知识问答、财报摘要或 unknown。"
        "你必须优先做保守判断，不要凭空发明公司、股票代码或日期。"
        "不要依赖规则关键词复述，要根据语义判断。"
    )
    user_prompt = (
        "请根据用户问题输出结构化 JSON。\n"
        "few-shot 参考：\n"
        "用户：阿里巴巴最近半年股价怎么样\n"
        "输出：{\"intent\":\"asset_trend\",\"need_market_data\":true,\"need_rag\":false,\"need_news\":false,\"extracted_symbol\":\"BABA\",\"extracted_company\":\"Alibaba\",\"time_range_days\":180,\"event_date\":null,\"reason\":\"用户在询问特定资产的时间区间表现。\"}\n"
        "用户：什么是纳斯达克综合指数\n"
        "输出：{\"intent\":\"finance_knowledge\",\"need_market_data\":false,\"need_rag\":true,\"need_news\":false,\"extracted_symbol\":null,\"extracted_company\":null,\"time_range_days\":null,\"event_date\":null,\"reason\":\"用户在询问金融概念定义。\"}\n"
        "用户：总结一下腾讯最新季度财报\n"
        "输出：{\"intent\":\"report_summary\",\"need_market_data\":false,\"need_rag\":true,\"need_news\":false,\"extracted_symbol\":\"0700.HK\",\"extracted_company\":\"Tencent\",\"time_range_days\":null,\"event_date\":null,\"reason\":\"用户要财报摘要，应走报告检索链路。\"}\n"
        f"用户问题：{message}\n"
        f"当前参考上下文：{json.dumps(heuristic_route.model_dump(mode='json'), ensure_ascii=False)}\n"
    )
    return system_prompt, user_prompt


def build_query_rewrite_prompt(message: str, route: RouteDecision) -> tuple[str, str]:
    system_prompt = (
        "你是金融问答系统中的查询改写器。"
        "你的任务是把用户问题改写为更适合检索的查询，并补齐公司英文名、股票代码、报告类型和关键财务指标。"
        "你必须保留原问题意图，不得凭空添加事实。"
    )
    user_prompt = (
        "请根据以下输入返回结构化 JSON。\n"
        "字段要求：rewritten_query, search_keywords, notes。\n"
        f"原始问题：{message}\n"
        f"路由结果：{json.dumps(route.model_dump(mode='json'), ensure_ascii=False)}\n"
    )
    return system_prompt, user_prompt


def build_answer_generation_prompt(
    request_message: str,
    route: RouteDecision,
    draft_answer: AnswerPayload,
) -> tuple[str, str]:
    system_prompt = (
        "你是金融资产问答系统中的回答生成器。"
        "你只能基于给定的客观数据、检索证据和系统草稿做归纳，不得编造数字、来源或公司事实。"
        "你的输出必须是结构化 JSON，且 summary、analysis、limitations 都要信息充分、严格受证据约束。"
    )
    user_prompt = (
        "请对以下金融问答草稿进行结构化润色，输出 JSON。\n"
        "要求：\n"
        "1. 保留 objective_data 中的客观信息边界，不要发明新数字。\n"
        "2. 如果 objective_data.source_mode=not_found，或者知识/财报类回答没有 sources，你必须保留“依据不足、无法可靠回答”的结论，不得用常识补答。\n"
        "3. summary 要直接回答问题，并尽量保留草稿中的关键事实、数字和时间信息。\n"
        "4. analysis 每条都必须基于已给证据，优先保留关键经营指标、同比变化和业务线亮点。\n"
        "5. limitations 只保留真正有必要的边界说明，不要写空泛套话。\n"
        f"用户问题：{request_message}\n"
        f"路由：{json.dumps(route.model_dump(mode='json'), ensure_ascii=False)}\n"
        f"草稿回答：{json.dumps(draft_answer.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n"
    )
    return system_prompt, user_prompt


def build_chat_response_prompt(
    request_message: str,
    route: RouteDecision,
    draft_answer: AnswerPayload,
) -> tuple[str, str]:
    system_prompt = (
        "你是金融资产问答助手。"
        "你只能基于给定草稿中的事实、来源边界和限制来回答，不得编造数字、时间、结论或额外来源。"
        "你要直接输出给用户看的最终正文，不要输出 JSON，不要输出标题，不要解释你的思考过程。"
        "如果证据不足，必须明确说依据不足，不能用常识补充。"
    )
    user_prompt = (
        "请把下面的金融问答草稿整理成一条适合聊天界面的最终回答。\n"
        "要求：\n"
        "1. 直接输出正文。\n"
        "2. 先给出明确结论，再展开说明有依据的关键信息，不要压缩成关键词式或过短句式回答。\n"
        "3. 如有必要，用“补充说明”引出边界或限制，但不要重复空话。\n"
        "4. 不要出现“summary”“analysis”“limitations”“objective_data”等字段名。\n"
        "5. 不要添加草稿里没有的新事实。\n"
        "6. 如果草稿里有数字、同比变化、日期或业务分部，请优先保留，不要泛化改写成空洞结论。\n"
        f"用户问题：{request_message}\n"
        f"路由：{json.dumps(route.model_dump(mode='json'), ensure_ascii=False)}\n"
        f"草稿回答：{json.dumps(draft_answer.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n"
    )
    return system_prompt, user_prompt


def build_verification_prompt(answer_payload: AnswerPayload) -> tuple[str, str]:
    system_prompt = (
        "你是金融问答系统中的结构校验器。"
        "你的任务是检查回答是否存在越界推断、结构缺失、来源不足、数字与结论冲突。"
        "输出必须是结构化 JSON。"
    )
    user_prompt = (
        "请校验以下回答，并给出是否通过。\n"
        "字段要求：is_valid, issues, corrected_summary, corrected_analysis, corrected_limitations。\n"
        "如果回答属于知识问答或财报摘要，且 source_mode=not_found 或 sources 为空，应判定为未通过，并改写成依据不足的保守回答。\n"
        f"回答：{json.dumps(answer_payload.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n"
    )
    return system_prompt, user_prompt


def build_agent_planning_prompt(message: str, conversation_context: str | None = None) -> tuple[str, str]:
    system_prompt = (
        "你是金融问答系统中的智能代理规划器。"
        "你必须先判断用户问题最适合调用哪个工具，再给出保守、可执行的参数。"
        "不要编造不存在的公司、股票代码、日期或财报。"
        "优先依赖你的金融常识把中文公司名、英文公司名与股票代码补齐到检索查询中。"
        "如果问题信息不足、超出系统范围，使用 direct_response。"
    )
    schema_json = json.dumps(AgentPlanningResult.model_json_schema(), ensure_ascii=False)
    user_prompt = (
        "请根据用户问题输出结构化 JSON。\n"
        "可用 tool_name：asset_price, asset_trend, asset_event_analysis, finance_knowledge, web_finance_knowledge, report_summary, web_report_summary, direct_response。\n"
        "规划原则：\n"
        "1. 价格/走势/事件归因属于市场数据工具。\n"
        "2. 概念解释、公司披露、财报摘要优先使用本地 RAG：finance_knowledge / report_summary。\n"
        "3. 只有在用户明确表示本地检索不对、要联网搜、要官网/新闻来源，或会话上下文显示上一轮 RAG 证据不足时，才使用 web_finance_knowledge / web_report_summary。\n"
        "4. rewritten_query 要写成真正可检索的查询，优先补上英文公司名、股票代码、报告类型、常见英文术语。\n"
        "5. thought 要简短准确，不要输出长链路推理。\n"
        "6. 日期用 ISO 格式，如 2026-01-15；不确定时留空。\n"
        "7. 涉及时间范围时，不要换算成天数；请填写 time_length 和 time_unit，例如最近三年 => time_length=3, time_unit=year。\n"
        "8. time_unit 只能是 day, week, month, year 之一；没有明确时间时可留空。\n"
        "few-shot 参考：\n"
        "用户：阿里巴巴最近半年股价怎么样\n"
        "输出：{\"tool_name\":\"asset_trend\",\"thought\":\"查询 Alibaba 过去半年的股价趋势。\",\"company\":\"Alibaba\",\"symbol\":\"BABA\",\"time_length\":6,\"time_unit\":\"month\",\"event_date\":null,\"rewritten_query\":null,\"direct_response\":null,\"reason\":\"用户在问特定资产的区间走势。\"}\n"
        "用户：什么是纳斯达克\n"
        "输出：{\"tool_name\":\"finance_knowledge\",\"thought\":\"先用本地知识库检索纳斯达克定义。\",\"company\":null,\"symbol\":null,\"time_length\":null,\"time_unit\":null,\"event_date\":null,\"rewritten_query\":\"Nasdaq Composite index definition stock exchange meaning\",\"direct_response\":null,\"reason\":\"概念解释问题先走 RAG。\"}\n"
        "用户：上一个解释不对，去网上搜纳斯达克官方定义\n"
        "输出：{\"tool_name\":\"web_finance_knowledge\",\"thought\":\"改用网页检索查找更可靠的官方定义。\",\"company\":null,\"symbol\":null,\"time_length\":null,\"time_unit\":null,\"event_date\":null,\"rewritten_query\":\"Nasdaq official definition stock exchange index glossary\",\"direct_response\":null,\"reason\":\"用户明确要求联网检索并对上一轮本地结果不满意。\"}\n"
        "用户：总结一下腾讯最新季度财报\n"
        "输出：{\"tool_name\":\"report_summary\",\"thought\":\"先检索 Tencent 最近季度财报材料并提炼摘要。\",\"company\":\"Tencent\",\"symbol\":\"0700.HK\",\"time_length\":null,\"time_unit\":null,\"event_date\":null,\"rewritten_query\":\"Tencent 0700.HK latest quarterly results earnings release quarterly presentation\",\"direct_response\":null,\"reason\":\"财报摘要优先本地披露材料。\"}\n"
        "用户：本地财报搜得不对，去官网找腾讯最新财报\n"
        "输出：{\"tool_name\":\"web_report_summary\",\"thought\":\"改用官网网页检索寻找最新财报材料。\",\"company\":\"Tencent\",\"symbol\":\"0700.HK\",\"time_length\":null,\"time_unit\":null,\"event_date\":null,\"rewritten_query\":\"Tencent 0700.HK latest quarterly results investor relations official\",\"direct_response\":null,\"reason\":\"用户明确要求网页/官网 fallback。\"}\n"
        + (f"会话上下文：\n{conversation_context}\n" if conversation_context else "")
        + f"用户问题：{message}\n"
        "请严格输出 JSON，禁止附加说明。JSON Schema 如下：\n"
        f"{schema_json}"
    )
    return system_prompt, user_prompt


def build_agent_response_prompt(
    request_message: str,
    answer_payload: AnswerPayload,
) -> tuple[str, str]:
    system_prompt = (
        "你是金融问答系统中的最终回答代理。"
        "你只能基于给定工具结果回答，不得编造任何数字、来源、日期或结论。"
        "必须始终使用简体中文回答。"
        "直接输出给用户的正文，不要输出 JSON，不要解释你的思考过程。"
        "不要把输入当成 JSON/对象说明文来解读，不要描述字段结构。"
        "不要输出 Markdown 标题、粗体、代码块、项目符号。"
    )
    user_prompt = (
        "请把下面的工具结果整理成最终回答。\n"
        "要求：\n"
        "1. 全文必须是简体中文。\n"
        "2. 输出纯文本，不要用 Markdown。\n"
        "3. 先给出直接结论，再充分展开有依据的事实，不要压缩成关键词、口号或过短句式回答。\n"
        "4. 如有必要，用“补充说明：”单独说明限制。\n"
        "5. 不要添加工具结果中没有的新事实。\n"
        "6. 如果工具结果里包含关键数字、同比变化、日期、财务指标或业务分部，请保留这些信息，不要泛化改写。\n"
        f"用户问题：{request_message}\n"
        f"工具结果：{json.dumps(answer_payload.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n"
    )
    return system_prompt, user_prompt


def build_event_observation_prompt(
    *,
    request_message: str,
    symbol: str,
    company: str | None,
    event_window: dict,
    event_results: list[dict],
) -> tuple[str, str]:
    system_prompt = (
        "你是金融事件归因助手。"
        "你只能基于给定新闻标题、来源和摘录，总结股价异动的可能原因。"
        "必须始终使用简体中文。"
        "不要照抄英文原文，不要输出 Markdown，不要输出字段解释。"
        "如果证据不足，只能保守表达为“可能与…有关”。"
    )
    schema_json = json.dumps(EventObservationResult.model_json_schema(), ensure_ascii=False)
    user_prompt = (
        "请阅读以下网页检索结果，输出 2-3 条中文归因观察。\n"
        "要求：\n"
        "1. 每条都要是完整中文句子。\n"
        "2. 不要直接复制英文标题或长摘录。\n"
        "3. 如果来源彼此重复，要做合并归纳。\n"
        "4. 不要超出材料本身的证据范围。\n"
        f"用户问题：{request_message}\n"
        f"资产：{company or symbol} ({symbol})\n"
        f"价格窗口：{json.dumps(event_window, ensure_ascii=False)}\n"
        f"网页结果：{json.dumps(event_results, ensure_ascii=False, indent=2)}\n"
        "请严格输出 JSON，禁止附加说明。JSON Schema 如下：\n"
        f"{schema_json}"
    )
    return system_prompt, user_prompt
