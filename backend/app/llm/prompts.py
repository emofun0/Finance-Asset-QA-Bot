from __future__ import annotations

import json

from app.llm.contracts import EventObservationResult
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


def build_answer_generation_prompt(
    request_message: str,
    route: RouteDecision,
    draft_answer: AnswerPayload,
) -> tuple[str, str]:
    system_prompt = (
        "你是金融资产问答系统中的回答生成器。"
        "你只能基于给定的客观数据、检索证据和系统草稿做归纳，不得编造数字、来源或公司事实。"
        "如果草稿里包含 report_context、metric_lines、table_rows 等财报上下文，你必须优先依据这些原始数字片段归纳。"
        "如果同一份财报同时包含年度和季度数字，你必须明确区分口径，不能把不同期间的数字混写成一条结论。"
        "你的输出必须是结构化 JSON，且 summary、analysis、limitations 都要信息充分、严格受证据约束。"
    )
    user_prompt = (
        "请对以下金融问答草稿进行结构化润色，输出 JSON。\n"
        "要求：\n"
        "1. 保留 objective_data 中的客观信息边界，不要发明新数字。\n"
        "2. 如果 objective_data.source_mode=not_found，或者知识/财报类回答没有 sources，你必须保留“依据不足、无法可靠回答”的结论，不得用常识补答。\n"
        "3. summary 要直接回答问题，并尽量保留草稿中的关键事实、数字和时间信息。\n"
        "4. analysis 每条都必须基于已给证据，优先保留关键经营指标、同比变化和业务线亮点。\n"
        "5. 如果草稿里有 objective_data.report_context，请优先阅读其中的 metric_lines 和 table_rows，再生成摘要。\n"
        "6. 对财报问题，若证据里同时有全年和单季数据，必须显式说明哪个是全年、哪个是单季。\n"
        "7. limitations 只保留真正有必要的边界说明，不要写空泛套话。\n"
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
        "如果草稿里带有财报的整份数字上下文，你应优先基于这些数字组织摘要，并明确区分年度与季度口径。"
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
