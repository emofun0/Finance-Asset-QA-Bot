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
    )
    user_prompt = (
        "请根据用户问题和已有的启发式抽取结果，输出结构化 JSON。\n"
        "规则：\n"
        "0. intent 只能是以下值之一：asset_price, asset_trend, asset_event_analysis, finance_knowledge, report_summary, unknown。\n"
        "1. 价格/走势/事件归因属于资产问答，需要 need_market_data=true。\n"
        "2. 金融概念、财报摘要属于知识/RAG 问答，需要 need_rag=true。\n"
        "3. 如果问题是“为什么涨跌”“为何大涨/大跌”，通常 need_news=true。\n"
        "4. event_date 使用 ISO 日期格式，如 2026-01-15；若未明确则置空。\n"
        f"用户问题：{message}\n"
        f"启发式结果：{json.dumps(heuristic_route.model_dump(mode='json'), ensure_ascii=False)}\n"
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
        "你的输出必须是结构化 JSON，且 summary 简洁、analysis 只写有依据的分析、limitations 明确边界。"
    )
    user_prompt = (
        "请对以下金融问答草稿进行结构化润色，输出 JSON。\n"
        "要求：\n"
        "1. 保留 objective_data 中的客观信息边界，不要发明新数字。\n"
        "2. 如果 objective_data.source_mode=not_found，或者知识/财报类回答没有 sources，你必须保留“依据不足、无法可靠回答”的结论，不得用常识补答。\n"
        "3. summary 用 1-2 句。\n"
        "4. analysis 2-4 条，每条都必须基于已给证据。\n"
        "5. limitations 1-3 条。\n"
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
        "2. 先给出简洁结论，再给出 2-4 条有依据的要点。\n"
        "3. 如有必要，用“补充说明”引出边界或限制，但不要重复空话。\n"
        "4. 不要出现“summary”“analysis”“limitations”“objective_data”等字段名。\n"
        "5. 不要添加草稿里没有的新事实。\n"
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


def build_agent_planning_prompt(message: str) -> tuple[str, str]:
    system_prompt = (
        "你是金融问答系统中的智能代理规划器。"
        "你必须先判断用户问题最适合调用哪个工具，再给出保守、可执行的参数。"
        "不要编造不存在的公司、股票代码、日期或财报。"
        "如果问题信息不足、超出系统范围，使用 direct_response。"
    )
    schema_json = json.dumps(AgentPlanningResult.model_json_schema(), ensure_ascii=False)
    user_prompt = (
        "请根据用户问题输出结构化 JSON。\n"
        "可用 tool_name：asset_price, asset_trend, asset_event_analysis, finance_knowledge, report_summary, direct_response。\n"
        "规则：\n"
        "1. 股价/现价/报价类问题用 asset_price。\n"
        "2. 趋势/最近表现/区间涨跌类问题用 asset_trend。\n"
        "3. 为什么涨跌/原因/催化剂类问题用 asset_event_analysis。\n"
        "4. 金融概念、术语解释类问题用 finance_knowledge。\n"
        "5. 财报摘要、业绩摘要类问题用 report_summary。\n"
        "6. 若无法可靠调用工具，就用 direct_response 并给出简短说明。\n"
        "7. thought 要简洁描述你准备做什么，不要泄露隐私或长篇推理。\n"
        "8. rewritten_query 仅在 finance_knowledge / report_summary 时可填写。\n"
        "9. 日期用 ISO 格式，如 2026-01-15；不确定时留空。\n"
        "10. 若 company 与 symbol 只确定一个，另一个字段可以留空。\n"
        "11. 涉及时间范围时，不要换算成天数；请填写 time_length 和 time_unit，例如最近三年 => time_length=3, time_unit=year。\n"
        "12. time_unit 只能是 day, week, month, year 之一；没有明确时间时可留空。\n"
        f"用户问题：{message}\n"
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
        "3. 先给出直接结论，再给出 2-4 条有依据的短句。\n"
        "4. 如有必要，用“补充说明：”单独说明限制。\n"
        "5. 不要添加工具结果中没有的新事实。\n"
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
