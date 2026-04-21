from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from typing import Any, Callable, Iterator

from app.core.company_catalog import CompanyProfile, find_company_profile, get_company_catalog
from app.llm.client import AgentToolCall, AgentToolDefinition, BaseLLMClient, NullLLMClient
from app.observability.request_trace import trace_event
from app.rag.retriever import RetrievalResult
from app.schemas.domain import IntentType, RouteDecision
from app.schemas.request import ChatRequest
from app.schemas.response import AnswerPayload, ChatMessagePayload, SourceItem
from app.services.answer_service import AnswerService
from app.services.asset_qa_service import AssetQAService
from app.services.chat_presenter_service import ChatPresenterService
from app.services.knowledge_qa_service import KnowledgeQAService
from app.services.session_memory_service import SessionMemoryService


@dataclass
class AgentStreamEvent:
    type: str
    payload: dict[str, Any]


@dataclass
class ToolExecutionResult:
    output: str
    note: str
    sources: list[SourceItem] = field(default_factory=list)
    objective_data: dict[str, Any] = field(default_factory=dict)
    question_type: IntentType | None = None
    route: RouteDecision | None = None


@dataclass
class AgentState:
    request_message: str
    tool_notes: list[str] = field(default_factory=list)
    sources: list[SourceItem] = field(default_factory=list)
    used_tools: list[str] = field(default_factory=list)
    objective_data: dict[str, Any] = field(default_factory=dict)
    last_company: str | None = None
    last_symbol: str | None = None
    last_time_range_days: int | None = None
    last_event_date: str | None = None
    inferred_question_type: IntentType = IntentType.UNKNOWN


class AgentService:
    _max_tool_turns = 8

    def __init__(
        self,
        asset_qa_service: AssetQAService,
        knowledge_qa_service: KnowledgeQAService,
        chat_presenter_service: ChatPresenterService,
        fallback_answer_service: AnswerService,
        session_memory_service: SessionMemoryService,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        self.asset_qa_service = asset_qa_service
        self.knowledge_qa_service = knowledge_qa_service
        self.chat_presenter_service = chat_presenter_service
        self.fallback_answer_service = fallback_answer_service
        self.session_memory_service = session_memory_service
        self.llm_client = llm_client or NullLLMClient()
        self._tool_handlers: dict[str, Callable[[dict[str, Any]], ToolExecutionResult]] = {
            "lookup_company": self._tool_lookup_company,
            "get_price_snapshot": self._tool_get_price_snapshot,
            "get_price_history": self._tool_get_price_history,
            "search_local_knowledge": self._tool_search_local_knowledge,
            "search_local_reports": self._tool_search_local_reports,
            "get_report_document_context": self._tool_get_report_document_context,
            "search_web_knowledge": self._tool_search_web_knowledge,
            "search_web_reports": self._tool_search_web_reports,
            "search_company_news": self._tool_search_company_news,
        }

    def is_enabled(self) -> bool:
        return self.llm_client.is_enabled()

    def answer(self, request: ChatRequest) -> AnswerPayload:
        if not self.is_enabled():
            trace_event("agent.fallback", {"reason": "agent_disabled"})
            return self.fallback_answer_service.answer(request)

        try:
            return self._run_agent_once(request).answer
        except Exception as exc:
            trace_event("agent.fallback", {"reason": "agent_error", "type": exc.__class__.__name__, "message": str(exc)})
            return self.fallback_answer_service.answer(request)

    def answer_chat(self, request: ChatRequest) -> ChatMessagePayload:
        if not self.is_enabled():
            return self.fallback_answer_service.answer_chat(request)

        try:
            return self._run_agent_once(request).message
        except Exception as exc:
            trace_event("agent.fallback", {"reason": "agent_error", "type": exc.__class__.__name__, "message": str(exc)})
            return self.fallback_answer_service.answer_chat(request)

    def stream_chat(self, request: ChatRequest) -> Iterator[AgentStreamEvent]:
        if not self.is_enabled():
            trace_event("agent.fallback", {"reason": "agent_disabled"})
            fallback_message = self.fallback_answer_service.answer_chat(request)
            yield AgentStreamEvent(type="final", payload=fallback_message.model_dump(mode="json"))
            return

        try:
            yield from self._run_agent_stream(request)
        except Exception as exc:
            trace_event("agent.fallback", {"reason": "agent_error", "type": exc.__class__.__name__, "message": str(exc)})
            fallback_message = self.fallback_answer_service.answer_chat(request)
            yield AgentStreamEvent(type="final", payload=fallback_message.model_dump(mode="json"))

    def _run_agent_stream(self, request: ChatRequest) -> Iterator[AgentStreamEvent]:
        messages = self.session_memory_service.build_context_messages(request.session_id)
        messages.append({"role": "user", "content": request.message})
        state = AgentState(request_message=request.message)
        final_text = ""

        trace_event("agent.start", {"message_count": len(messages), "session_id": request.session_id})
        yield AgentStreamEvent(type="agent", payload={"text": "开始分析问题并决定下一步动作。"})

        for turn_index in range(self._max_tool_turns):
            yield AgentStreamEvent(type="agent", payload={"text": f"进入第 {turn_index + 1} 轮决策。"})
            turn = self.llm_client.generate_tool_turn(
                system_prompt=self._build_system_prompt(),
                messages=messages,
                tools=self._build_tools(),
            )
            trace_event(
                "agent.turn.assistant",
                {
                    "turn": turn_index + 1,
                    "text": turn.text,
                    "tool_calls": [call.name for call in turn.tool_calls],
                },
            )
            messages.append(turn.assistant_message)

            if turn.text:
                yield AgentStreamEvent(type="agent", payload={"text": turn.text.strip()})
                final_text = turn.text.strip()

            if not turn.tool_calls:
                break

            for tool_call in turn.tool_calls:
                yield AgentStreamEvent(type="agent", payload={"text": self._format_tool_call_event(tool_call)})
                trace_event(
                    "agent.turn.tool_call",
                    {
                        "turn": turn_index + 1,
                        "tool_name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                )
                result = self._execute_tool(tool_call, state)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result.output,
                    }
                )
                yield AgentStreamEvent(type="agent", payload={"text": self._format_tool_result_event(tool_call.name, result.note)})
                trace_event(
                    "agent.turn.tool_result",
                    {
                        "turn": turn_index + 1,
                        "tool_name": tool_call.name,
                        "note": result.note,
                    },
                )
        else:
            final_text = final_text or "已达到本轮工具调用上限，当前先给出保守回答。"

        final_text = self._sanitize_final_text(final_text) or "当前无法可靠回答该问题。"
        answer = self._build_answer_payload(request, final_text, state)
        self.session_memory_service.remember(
            request.session_id,
            user_message=request.message,
            answer=answer,
            tool_notes=state.tool_notes,
        )
        trace_event(
            "agent.final",
            {
                "question_type": answer.question_type,
                "sources": len(answer.sources),
                "summary": answer.summary,
            },
        )
        message = self.chat_presenter_service.build_message(answer, text_override=answer.summary)
        yield AgentStreamEvent(type="final", payload=message.model_dump(mode="json"))

    def _run_agent_once(self, request: ChatRequest) -> _AgentRunResult:
        messages = self.session_memory_service.build_context_messages(request.session_id)
        messages.append({"role": "user", "content": request.message})
        state = AgentState(request_message=request.message)
        final_text = ""

        trace_event("agent.start", {"message_count": len(messages), "session_id": request.session_id})

        for turn_index in range(self._max_tool_turns):
            turn = self.llm_client.generate_tool_turn(
                system_prompt=self._build_system_prompt(),
                messages=messages,
                tools=self._build_tools(),
            )
            trace_event(
                "agent.turn.assistant",
                {
                    "turn": turn_index + 1,
                    "text": turn.text,
                    "tool_calls": [call.name for call in turn.tool_calls],
                },
            )
            messages.append(turn.assistant_message)

            if turn.text:
                final_text = turn.text.strip()

            if not turn.tool_calls:
                break

            for tool_call in turn.tool_calls:
                trace_event(
                    "agent.turn.tool_call",
                    {
                        "turn": turn_index + 1,
                        "tool_name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                )
                result = self._execute_tool(tool_call, state)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result.output,
                    }
                )
                trace_event(
                    "agent.turn.tool_result",
                    {
                        "turn": turn_index + 1,
                        "tool_name": tool_call.name,
                        "note": result.note,
                    },
                )
        else:
            final_text = final_text or "已达到本轮工具调用上限，当前先给出保守回答。"

        final_text = self._sanitize_final_text(final_text) or "当前无法可靠回答该问题。"
        answer = self._build_answer_payload(request, final_text, state)
        self.session_memory_service.remember(
            request.session_id,
            user_message=request.message,
            answer=answer,
            tool_notes=state.tool_notes,
        )
        trace_event(
            "agent.final",
            {
                "question_type": answer.question_type,
                "sources": len(answer.sources),
                "summary": answer.summary,
            },
        )
        message = self.chat_presenter_service.build_message(answer, text_override=answer.summary)
        return _AgentRunResult(answer=answer, message=message)

    def _execute_tool(self, tool_call: AgentToolCall, state: AgentState) -> ToolExecutionResult:
        handler = self._tool_handlers.get(tool_call.name)
        if handler is None:
            result = ToolExecutionResult(
                output=f"工具 `{tool_call.name}` 不存在。",
                note=f"{tool_call.name} 不存在。",
            )
        else:
            try:
                result = handler(tool_call.arguments)
            except Exception as exc:
                result = ToolExecutionResult(
                    output=f"工具执行失败：{exc}",
                    note=f"{tool_call.name} 执行失败：{exc}",
                )

        state.used_tools.append(tool_call.name)
        if result.note:
            state.tool_notes.append(result.note)
            state.tool_notes = state.tool_notes[-8:]
        if result.sources:
            state.sources.extend(result.sources)
        if result.question_type is not None:
            state.inferred_question_type = result.question_type
        if result.objective_data:
            state.objective_data.update(result.objective_data)
            state.last_company = str(result.objective_data.get("company") or state.last_company or "") or state.last_company
            state.last_symbol = str(result.objective_data.get("symbol") or state.last_symbol or "") or state.last_symbol
            days = result.objective_data.get("time_range_days")
            if isinstance(days, int) and days > 0:
                state.last_time_range_days = days
            event_date = str(result.objective_data.get("event_date") or "").strip()
            if event_date:
                state.last_event_date = event_date
        if result.route is not None:
            state.last_company = result.route.extracted_company or state.last_company
            state.last_symbol = result.route.extracted_symbol or state.last_symbol
            if result.route.time_range_days:
                state.last_time_range_days = result.route.time_range_days
            if result.route.event_date:
                state.last_event_date = result.route.event_date
        return result

    def _build_answer_payload(self, request: ChatRequest, final_text: str, state: AgentState) -> AnswerPayload:
        question_type = self._infer_question_type(state)
        route = RouteDecision(
            intent=question_type,
            need_market_data=question_type in {
                IntentType.ASSET_PRICE,
                IntentType.ASSET_TREND,
                IntentType.ASSET_EVENT_ANALYSIS,
            },
            need_rag=question_type in {IntentType.FINANCE_KNOWLEDGE, IntentType.REPORT_SUMMARY},
            need_news=question_type == IntentType.ASSET_EVENT_ANALYSIS,
            extracted_company=state.last_company,
            extracted_symbol=state.last_symbol,
            time_range_days=state.last_time_range_days,
            event_date=state.last_event_date,
            decision_source="native_agent",
            reason="模型通过原生工具调用完成本轮决策。",
        )
        sources = self._dedupe_sources(state.sources)
        analysis = state.tool_notes[-4:]
        return AnswerPayload(
            question_type=question_type,
            request_message=request.message,
            summary=final_text,
            objective_data={
                **state.objective_data,
                "source_mode": self._infer_source_mode(state),
                "used_tools": state.used_tools,
            },
            analysis=analysis,
            sources=sources,
            limitations=[],
            route=route,
        )

    def _infer_question_type(self, state: AgentState) -> IntentType:
        if state.inferred_question_type != IntentType.UNKNOWN:
            return state.inferred_question_type
        tool_set = set(state.used_tools)
        if "search_company_news" in tool_set and "get_price_history" in tool_set:
            return IntentType.ASSET_EVENT_ANALYSIS
        if "get_price_history" in tool_set:
            return IntentType.ASSET_TREND
        if "get_price_snapshot" in tool_set:
            return IntentType.ASSET_PRICE
        if tool_set & {"search_local_reports", "search_web_reports", "get_report_document_context"}:
            return IntentType.REPORT_SUMMARY
        if tool_set & {"search_local_knowledge", "search_web_knowledge"}:
            return IntentType.FINANCE_KNOWLEDGE
        return IntentType.UNKNOWN

    def _infer_source_mode(self, state: AgentState) -> str:
        if any(tool_name.startswith("search_web") or tool_name == "search_company_news" for tool_name in state.used_tools):
            return "web"
        if any(tool_name.startswith("search_local") or tool_name == "get_report_document_context" for tool_name in state.used_tools):
            return "local_rag"
        if any(tool_name.startswith("get_price") for tool_name in state.used_tools):
            return "market_data"
        return "agent_direct"

    def _build_system_prompt(self) -> str:
        today = datetime.now(UTC).date().isoformat()
        return (
            "你是一个金融问答代理，必须始终使用简体中文回答。"
            f"今天的日期是 {today}。"
            "你以原生工具调用方式工作：可以连续调用多个工具，直到你确认已经收集到足够证据，再直接输出最终回答。"
            "不要输出 JSON，不要解释内部工作流，不要虚构数字、来源、日期或结论。"
            "对于金融概念、公司披露、财报问题，优先使用本地检索工具 search_local_knowledge 或 search_local_reports。"
            "只有在本地结果明显不足、用户明确要求联网/官网/最新外部来源、或你需要补充新闻时，才改用 web 工具。"
            "涉及公司检索时，不要只用单个中文名；优先补全中文名、英文标准名、股票代码和常见简称后再搜索。"
            "对于价格、走势、波动原因，先用市场数据工具；若要分析原因，再结合新闻工具。"
            "如果证据不足，要明确说依据不足。"
            "你可以先输出简短思考或判断，再调用工具；也可以直接调用工具。"
        )

    def _build_tools(self) -> list[AgentToolDefinition]:
        return [
            AgentToolDefinition(
                name="lookup_company",
                description="查找公司目录中的标准公司名、股票代码、别名和官网域名。遇到中文名、简称、英文名混用时，先用它确认标准标的。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "company": {"type": "string"},
                        "symbol": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                },
            ),
            AgentToolDefinition(
                name="get_price_snapshot",
                description="获取单个资产的最新价格快照。需要 symbol，若不确定可先 lookup_company。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "symbol": {"type": "string"},
                    },
                },
            ),
            AgentToolDefinition(
                name="get_price_history",
                description="获取单个资产最近若干天的历史价格摘要。适合回答走势、区间涨跌、高低点问题。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "symbol": {"type": "string"},
                        "days": {"type": "integer", "minimum": 1, "maximum": 3650},
                    },
                    "required": ["days"],
                },
            ),
            AgentToolDefinition(
                name="search_local_knowledge",
                description="搜索本地金融知识库。对于概念解释，默认先用这个工具，而不是 web 搜索。若涉及公司，请尽量把中文名、英文名、股票代码、简称一起带入 query。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "company": {"type": "string"},
                        "symbol": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 8},
                    },
                    "required": ["query"],
                },
            ),
            AgentToolDefinition(
                name="search_local_reports",
                description="搜索本地财报/业绩材料。对于财报摘要、指标、表格问题，默认先用这个工具。不要只搜单个中文名，尽量同时带上英文名、股票代码和简称。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "company": {"type": "string"},
                        "symbol": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 8},
                    },
                    "required": ["query"],
                },
            ),
            AgentToolDefinition(
                name="get_report_document_context",
                description="根据本地财报搜索结果中的 doc_id，提取整份财报的重要指标行和表格行。做财报总结前很有用。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                    },
                    "required": ["doc_id"],
                },
            ),
            AgentToolDefinition(
                name="search_web_knowledge",
                description="搜索网页金融知识。仅在本地知识不足或用户明确要求联网、官方、最新外部来源时使用。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": ["query"],
                },
            ),
            AgentToolDefinition(
                name="search_web_reports",
                description="搜索官网或网页财报材料。仅在本地财报不足或用户明确要求官网/联网时使用。query 里尽量同时带上中英文名、股票代码和简称。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "company": {"type": "string"},
                        "symbol": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": ["query"],
                },
            ),
            AgentToolDefinition(
                name="search_company_news",
                description="搜索公司新闻或事件线索。适合在价格异动原因分析时补充外部证据。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "company": {"type": "string"},
                        "symbol": {"type": "string"},
                        "event_date": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 6},
                    },
                    "required": ["query"],
                },
            ),
        ]

    def _tool_lookup_company(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        query = str(arguments.get("query") or arguments.get("company") or arguments.get("symbol") or "").strip()
        symbol = str(arguments.get("symbol") or "").strip()
        company = str(arguments.get("company") or "").strip()
        limit = int(arguments.get("limit") or 5)
        matches = self._search_company_catalog(query=query, company=company, symbol=symbol, limit=limit)
        payload = {
            "query": query or company or symbol,
            "matches": [
                {
                    "company": item.canonical_name,
                    "symbol": item.symbol,
                    "official_domains": list(item.official_domains),
                }
                for item in matches
            ],
        }
        note = "company_lookup 未命中。" if not matches else "company_lookup 命中：" + "、".join(
            f"{item.canonical_name}({item.symbol})" for item in matches[:3]
        )
        objective_data: dict[str, Any] = {}
        if len(matches) == 1:
            objective_data = {"company": matches[0].canonical_name, "symbol": matches[0].symbol}
        return ToolExecutionResult(
            output=json.dumps(payload, ensure_ascii=False),
            note=note,
            objective_data=objective_data,
        )

    def _tool_get_price_snapshot(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        profile, symbol = self._resolve_subject(arguments)
        payload = self.asset_qa_service.get_price_snapshot(symbol)
        payload["company"] = payload.get("company") or (profile.canonical_name if profile else arguments.get("company"))
        note = f"{symbol} 最新价 {payload.get('latest_price')}，时间 {payload.get('as_of')}。"
        return ToolExecutionResult(
            output=json.dumps(payload, ensure_ascii=False),
            note=note,
            objective_data=payload,
            question_type=IntentType.ASSET_PRICE,
            route=RouteDecision(
                intent=IntentType.ASSET_PRICE,
                need_market_data=True,
                extracted_company=str(payload.get("company") or "") or None,
                extracted_symbol=symbol,
                decision_source="native_agent",
                reason="模型调用市场快照工具。",
            ),
        )

    def _tool_get_price_history(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        days = int(arguments.get("days") or 30)
        profile, symbol = self._resolve_subject(arguments)
        history_payload = self.asset_qa_service.get_price_history(symbol, days)
        points = history_payload.get("points") or []
        high_point = max(points, key=lambda item: item.get("close", float("-inf"))) if points else None
        low_point = min(points, key=lambda item: item.get("close", float("inf"))) if points else None
        output = {
            **history_payload,
            "company": profile.canonical_name if profile else arguments.get("company"),
            "high_point": high_point,
            "low_point": low_point,
        }
        note = (
            f"{symbol} 最近 {days} 天{history_payload.get('trend')}，"
            f"区间涨跌幅约 {history_payload.get('change_pct')}%。"
        )
        return ToolExecutionResult(
            output=json.dumps(output, ensure_ascii=False),
            note=note,
            objective_data=output,
            question_type=IntentType.ASSET_TREND,
            route=RouteDecision(
                intent=IntentType.ASSET_TREND,
                need_market_data=True,
                extracted_company=profile.canonical_name if profile else None,
                extracted_symbol=symbol,
                time_range_days=days,
                decision_source="native_agent",
                reason="模型调用历史行情工具。",
            ),
        )

    def _tool_search_local_knowledge(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        raw_query = str(arguments.get("query") or "").strip()
        top_k = int(arguments.get("top_k") or 5)
        profile, symbol = self._resolve_subject(arguments, require_symbol=False)
        query = self._expand_search_query(raw_query, profile, arguments.get("company"))
        results = self.knowledge_qa_service.rag_search_tool.search(
            self._build_rag_request(
                query=query,
                top_k=top_k,
                company=profile.canonical_name if profile else arguments.get("company"),
                symbol=symbol,
                doc_types=["glossary", "knowledge_article"],
                chunk_kinds=["glossary_term", "glossary_text", "text_chunk"],
            )
        )
        serialized = self._serialize_retrieval_results(results)
        note = "本地知识库无结果。" if not results else f"本地知识库命中 {len(results)} 条。"
        return ToolExecutionResult(
            output=json.dumps({"query": query, "original_query": raw_query, "results": serialized}, ensure_ascii=False),
            note=note,
            sources=self._to_sources(results[:4]),
            objective_data={"company": profile.canonical_name if profile else arguments.get("company"), "symbol": symbol},
            question_type=IntentType.FINANCE_KNOWLEDGE,
        )

    def _tool_search_local_reports(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        raw_query = str(arguments.get("query") or "").strip()
        top_k = int(arguments.get("top_k") or 5)
        profile, symbol = self._resolve_subject(arguments, require_symbol=False)
        query = self._expand_search_query(raw_query, profile, arguments.get("company"))
        results = self.knowledge_qa_service.rag_search_tool.search_report_documents(
            self._build_rag_request(
                query=query,
                top_k=top_k,
                company=profile.canonical_name if profile else arguments.get("company"),
                symbol=symbol,
            )
        )
        serialized = self._serialize_retrieval_results(results)
        note = "本地财报未命中。" if not results else f"本地财报命中 {len(results)} 条。"
        return ToolExecutionResult(
            output=json.dumps({"query": query, "original_query": raw_query, "results": serialized}, ensure_ascii=False),
            note=note,
            sources=self._to_sources(results[:4]),
            objective_data={"company": profile.canonical_name if profile else arguments.get("company"), "symbol": symbol},
            question_type=IntentType.REPORT_SUMMARY,
        )

    def _tool_get_report_document_context(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        doc_id = str(arguments.get("doc_id") or "").strip()
        retriever = self.knowledge_qa_service.retriever
        if retriever is None:
            raise RuntimeError("本地检索器不可用。")
        chunks = retriever.get_document_chunks(
            doc_id,
            chunk_kinds=["report_profile", "report_metric", "report_table"],
        )
        profile = next((item for item in chunks if item.metadata.get("chunk_kind") == "report_profile"), None)
        metric_lines = self._dedupe_texts(
            [self._trim(" ".join(item.content.split()), 260) for item in chunks if item.metadata.get("chunk_kind") == "report_metric"],
            limit=16,
        )
        table_rows = self._dedupe_texts(
            [
                self._trim(" ".join(row.split()), 180)
                for item in chunks
                if item.metadata.get("chunk_kind") == "report_table"
                for row in item.content.splitlines()
            ],
            limit=12,
            min_length=8,
        )
        title = str(chunks[0].metadata.get("title") or "") if chunks else ""
        context = {
            "doc_id": doc_id,
            "title": title,
            "report_period": self._extract_report_period(title),
            "profile": profile.content if profile else None,
            "metric_lines": metric_lines,
            "table_rows": table_rows,
        }
        note = (
            f"提取财报上下文：指标 {len(metric_lines)} 条，表格行 {len(table_rows)} 条。"
            if chunks
            else "未提取到财报上下文。"
        )
        return ToolExecutionResult(
            output=json.dumps(context, ensure_ascii=False),
            note=note,
            objective_data={
                "primary_doc_id": doc_id,
                "primary_doc_title": title,
                "report_context": context,
            },
            question_type=IntentType.REPORT_SUMMARY,
        )

    def _tool_search_web_knowledge(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        raw_query = str(arguments.get("query") or "").strip()
        top_k = int(arguments.get("top_k") or 3)
        query = self._expand_search_query(raw_query, None, arguments.get("company"))
        results = self.knowledge_qa_service.web_search_tool.search_finance_knowledge(query, top_k=top_k)
        note = "网页知识无结果。" if not results else f"网页知识命中 {len(results)} 条。"
        return ToolExecutionResult(
            output=json.dumps({"query": query, "original_query": raw_query, "results": self._serialize_retrieval_results(results)}, ensure_ascii=False),
            note=note,
            sources=self._to_sources(results[:4]),
            question_type=IntentType.FINANCE_KNOWLEDGE,
        )

    def _tool_search_web_reports(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        raw_query = str(arguments.get("query") or "").strip()
        top_k = int(arguments.get("top_k") or 4)
        profile, symbol = self._resolve_subject(arguments, require_symbol=False)
        query = self._expand_search_query(raw_query, profile, arguments.get("company"))
        if profile and self.knowledge_qa_service.web_search_tool:
            results = self.knowledge_qa_service.web_search_tool.search_company_reports(query, profile, top_k=top_k)
        else:
            results = self.knowledge_qa_service.web_search_tool.search_company_reports_by_query(
                query,
                company=str(arguments.get("company") or "").strip() or None,
                symbol=symbol,
                top_k=top_k,
            )
        note = "网页财报无结果。" if not results else f"网页财报命中 {len(results)} 条。"
        return ToolExecutionResult(
            output=json.dumps({"query": query, "original_query": raw_query, "results": self._serialize_retrieval_results(results)}, ensure_ascii=False),
            note=note,
            sources=self._to_sources(results[:4]),
            objective_data={"company": profile.canonical_name if profile else arguments.get("company"), "symbol": symbol},
            question_type=IntentType.REPORT_SUMMARY,
        )

    def _tool_search_company_news(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        query = str(arguments.get("query") or "").strip()
        event_date = str(arguments.get("event_date") or "").strip() or None
        top_k = int(arguments.get("top_k") or 4)
        profile, symbol = self._resolve_subject(arguments)
        if self.asset_qa_service.web_search_tool is None:
            raise RuntimeError("新闻搜索工具不可用。")
        results = self.asset_qa_service.web_search_tool.search_company_events(
            query,
            profile,
            top_k=top_k,
            event_date=event_date,
        )
        note = "新闻线索无结果。" if not results else f"新闻线索命中 {len(results)} 条。"
        return ToolExecutionResult(
            output=json.dumps({"query": query, "results": self._serialize_retrieval_results(results)}, ensure_ascii=False),
            note=note,
            sources=self._to_sources(results[:4]),
            objective_data={
                "company": profile.canonical_name,
                "symbol": symbol,
                "event_date": event_date,
            },
            question_type=IntentType.ASSET_EVENT_ANALYSIS,
        )

    def _search_company_catalog(
        self,
        *,
        query: str,
        company: str,
        symbol: str,
        limit: int,
    ) -> list[CompanyProfile]:
        normalized_candidates = [item.strip().lower() for item in [query, company, symbol] if item and item.strip()]
        if not normalized_candidates:
            return []

        matches: list[tuple[int, CompanyProfile]] = []
        for profile in get_company_catalog():
            haystacks = [profile.canonical_name.lower(), profile.symbol.lower(), *(alias.lower() for alias in profile.aliases)]
            score = 0
            for candidate in normalized_candidates:
                if candidate == profile.symbol.lower() or candidate == profile.canonical_name.lower():
                    score += 4
                elif any(candidate == alias for alias in haystacks):
                    score += 3
                elif any(candidate in alias or alias in candidate for alias in haystacks):
                    score += 1
            if score > 0:
                matches.append((score, profile))
        matches.sort(key=lambda item: (-item[0], item[1].canonical_name))
        return [profile for _, profile in matches[:limit]]

    def _expand_search_query(
        self,
        query: str,
        profile: CompanyProfile | None,
        raw_company: Any | None = None,
    ) -> str:
        values = [str(query or "").strip()]
        if profile:
            values.extend(profile.search_terms())
        elif raw_company:
            values.append(str(raw_company).strip())

        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            compact = " ".join(str(value or "").split()).strip()
            if not compact:
                continue
            normalized = compact.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(compact)
        return " ".join(deduped)

    def _resolve_subject(
        self,
        arguments: dict[str, Any],
        *,
        require_symbol: bool = True,
    ) -> tuple[CompanyProfile | None, str | None]:
        company = str(arguments.get("company") or "").strip() or None
        symbol = str(arguments.get("symbol") or "").strip() or None
        profile = find_company_profile(company=company, symbol=symbol)
        resolved_symbol = profile.symbol if profile else symbol
        if require_symbol and not resolved_symbol:
            raise ValueError("缺少可识别的股票代码，请先 lookup_company 或直接提供 symbol。")
        return profile, resolved_symbol

    def _serialize_retrieval_results(self, results: list[RetrievalResult]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for item in results[:6]:
            serialized.append(
                {
                    "chunk_id": item.chunk_id,
                    "score": round(item.score, 4),
                    "title": item.metadata.get("title"),
                    "doc_id": item.metadata.get("doc_id"),
                    "doc_type": item.metadata.get("doc_type"),
                    "chunk_kind": item.metadata.get("chunk_kind"),
                    "url": item.metadata.get("url"),
                    "content_excerpt": self._trim(item.content, 320),
                }
            )
        return serialized

    def _to_sources(self, results: list[RetrievalResult]) -> list[SourceItem]:
        sources: list[SourceItem] = []
        seen: set[tuple[str, str | None]] = set()
        for item in results:
            name = str(item.metadata.get("title") or item.metadata.get("source_name") or "Document")
            value = item.metadata.get("url")
            key = (name, value)
            if key in seen:
                continue
            seen.add(key)
            sources.append(SourceItem(type=str(item.metadata.get("doc_type") or "document"), name=name, value=value))
        return sources

    def _build_rag_request(
        self,
        *,
        query: str,
        top_k: int,
        company: str | None,
        symbol: str | None,
        doc_types: list[str] | None = None,
        chunk_kinds: list[str] | None = None,
    ):
        from app.tools.rag_search_tool import RagSearchRequest

        return RagSearchRequest(
            query=query,
            top_k=top_k,
            company=company,
            symbol=symbol,
            language="zh" if self._contains_chinese(query) else None,
            doc_types=doc_types,
            chunk_kinds=chunk_kinds,
        )

    def _format_tool_call_event(self, tool_call: AgentToolCall) -> str:
        args_text = json.dumps(tool_call.arguments, ensure_ascii=False)
        if len(args_text) > 220:
            args_text = args_text[:219] + "…"
        return f"调用工具 {tool_call.name}，参数：{args_text}"

    def _format_tool_result_event(self, tool_name: str, note: str) -> str:
        return f"工具 {tool_name} 返回：{note}"

    def _sanitize_final_text(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = cleaned.replace("```", "").replace("**", "")
        if cleaned.lower().startswith("final answer:"):
            cleaned = cleaned.split(":", maxsplit=1)[-1].strip()
        return cleaned

    def _dedupe_sources(self, sources: list[SourceItem]) -> list[SourceItem]:
        deduped: list[SourceItem] = []
        seen: set[tuple[str, str | None]] = set()
        for source in sources:
            key = (source.name.strip(), source.value)
            if key in seen or not source.name.strip():
                continue
            seen.add(key)
            deduped.append(source)
        return deduped[:8]

    def _trim(self, text: str, limit: int) -> str:
        compact = " ".join(str(text or "").split()).strip()
        if len(compact) <= limit:
            return compact
        return compact[: max(limit - 1, 0)].rstrip() + "…"

    def _dedupe_texts(self, values: list[str], *, limit: int, min_length: int = 12) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            compact = value.strip()
            if len(compact) < min_length or compact in seen:
                continue
            seen.add(compact)
            deduped.append(compact)
            if len(deduped) >= limit:
                break
        return deduped

    def _extract_report_period(self, title: str) -> str | None:
        import re

        patterns = [
            r"(20\d{2}年(?:度)?报告)",
            r"(20\d{2}年中期报告)",
            r"(20\d{2}\s*Q[1-4])",
            r"(20\d{2}年(?:第[一二三四1-4]季度))",
        ]
        for pattern in patterns:
            matched = re.search(pattern, title, flags=re.IGNORECASE)
            if matched:
                return matched.group(1).replace(" ", "")
        return None

    def _contains_chinese(self, value: str) -> bool:
        import re

        return bool(re.search(r"[\u4e00-\u9fff]", value))


@dataclass
class _AgentRunResult:
    answer: AnswerPayload
    message: ChatMessagePayload
