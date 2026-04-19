from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterator, TypedDict

from app.core.company_catalog import find_company_profile
from app.core.errors import AppError
from app.llm.contracts import AgentPlanningResult
from app.llm.langchain_factory import build_langchain_chat_model
from app.llm.prompts import build_agent_planning_prompt, build_agent_response_prompt
from app.observability.request_trace import trace_event
from app.schemas.domain import IntentType, RouteDecision
from app.schemas.request import ChatRequest
from app.schemas.response import AnswerPayload, ChatMessagePayload, SourceItem
from app.services.answer_service import AnswerService
from app.services.asset_qa_service import AssetQAService
from app.services.chat_presenter_service import ChatPresenterService
from app.services.knowledge_qa_service import KnowledgeQAService

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:  # pragma: no cover - optional dependency at runtime
    HumanMessage = None
    SystemMessage = None

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - optional dependency at runtime
    END = "__end__"
    START = "__start__"
    StateGraph = None


class AgentGraphState(TypedDict, total=False):
    request_message: str
    plan: dict
    answer_payload: dict
    final_text: str


@dataclass
class AgentStreamEvent:
    type: str
    payload: dict


@dataclass
class AgentRunResult:
    answer: AnswerPayload
    message: ChatMessagePayload


class AgentToolExecutor:
    def __init__(
        self,
        asset_qa_service: AssetQAService,
        knowledge_qa_service: KnowledgeQAService,
    ) -> None:
        self.asset_qa_service = asset_qa_service
        self.knowledge_qa_service = knowledge_qa_service

    def run(self, request: ChatRequest, plan: AgentPlanningResult) -> AnswerPayload:
        if plan.tool_name == "asset_price":
            route = self._build_route(
                intent=IntentType.ASSET_PRICE,
                plan=plan,
                need_market_data=True,
            )
            return self.asset_qa_service.answer(request, route)

        if plan.tool_name == "asset_trend":
            time_range_days = self._resolve_time_range_days(plan, default_days=30)
            route = self._build_route(
                intent=IntentType.ASSET_TREND,
                plan=plan,
                need_market_data=True,
                time_range_days=time_range_days,
            )
            return self.asset_qa_service.answer(request, route)

        if plan.tool_name == "asset_event_analysis":
            time_range_days = self._resolve_time_range_days(plan, default_days=30)
            route = self._build_route(
                intent=IntentType.ASSET_EVENT_ANALYSIS,
                plan=plan,
                need_market_data=True,
                need_news=True,
                time_range_days=time_range_days,
                event_date=plan.event_date,
            )
            return self.asset_qa_service.answer(request, route)

        if plan.tool_name == "finance_knowledge":
            route = self._build_route(
                intent=IntentType.FINANCE_KNOWLEDGE,
                plan=plan,
                need_rag=True,
            )
            return self.knowledge_qa_service.answer(request, route)

        if plan.tool_name == "report_summary":
            route = self._build_route(
                intent=IntentType.REPORT_SUMMARY,
                plan=plan,
                need_rag=True,
            )
            return self.knowledge_qa_service.answer(request, route)

        raise AppError(
            code="UNSUPPORTED_AGENT_TOOL",
            message="当前代理未找到可执行工具。",
            status_code=400,
            details={"tool_name": plan.tool_name},
        )

    def _resolve_time_range_days(self, plan: AgentPlanningResult, default_days: int) -> int:
        if plan.time_length is None:
            return default_days

        unit = (plan.time_unit or "day").strip().lower()
        factor = {
            "day": 1,
            "week": 7,
            "month": 30,
            "year": 365,
        }.get(unit)
        if factor is None:
            raise AppError(
                code="INVALID_TIME_UNIT",
                message="代理返回了不支持的时间单位。",
                status_code=400,
                details={"time_unit": plan.time_unit},
            )
        return plan.time_length * factor

    def _build_route(
        self,
        *,
        intent: IntentType,
        plan: AgentPlanningResult,
        need_market_data: bool = False,
        need_rag: bool = False,
        need_news: bool = False,
        time_range_days: int | None = None,
        event_date: str | None = None,
    ) -> RouteDecision:
        profile = find_company_profile(company=plan.company, symbol=plan.symbol)
        company = plan.company or (profile.canonical_name if profile else None)
        symbol = plan.symbol or (profile.symbol if profile else None)
        return RouteDecision(
            intent=intent,
            need_market_data=need_market_data,
            need_rag=need_rag,
            need_news=need_news,
            extracted_symbol=symbol,
            extracted_company=company,
            time_range_days=time_range_days,
            event_date=event_date,
            decision_source="agent",
            reason=plan.reason or f"Agent selected tool {plan.tool_name}.",
        )


class AgentService:
    def __init__(
        self,
        *,
        provider: str | None,
        model: str | None,
        asset_qa_service: AssetQAService,
        knowledge_qa_service: KnowledgeQAService,
        chat_presenter_service: ChatPresenterService,
        fallback_answer_service: AnswerService,
    ) -> None:
        self.provider = provider
        self.model_name = model
        self.chat_presenter_service = chat_presenter_service
        self.fallback_answer_service = fallback_answer_service
        self.tool_executor = AgentToolExecutor(
            asset_qa_service=asset_qa_service,
            knowledge_qa_service=knowledge_qa_service,
        )
        self.model = build_langchain_chat_model(provider=provider, model=model)
        self.graph = self._build_graph() if self.model is not None and StateGraph is not None else None

    def is_enabled(self) -> bool:
        return self.graph is not None and HumanMessage is not None and SystemMessage is not None

    def answer(self, request: ChatRequest) -> AnswerPayload:
        if not self.is_enabled():
            trace_event("agent.fallback", {"reason": "agent_disabled"})
            return self.fallback_answer_service.answer(request)

        try:
            state = self.graph.invoke({"request_message": request.message})
            return self._build_run_result(request, state).answer
        except Exception as exc:
            trace_event("agent.fallback", {"reason": "agent_error", "type": exc.__class__.__name__, "message": str(exc)})
            return self.fallback_answer_service.answer(request)

    def answer_chat(self, request: ChatRequest) -> ChatMessagePayload:
        if not self.is_enabled():
            return self.fallback_answer_service.answer_chat(request)

        try:
            state = self.graph.invoke({"request_message": request.message})
            return self._build_run_result(request, state).message
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
            yield AgentStreamEvent(type="status", payload={"text": "正在判断问题意图..."})
            plan = self._plan_request(request.message)
            yield AgentStreamEvent(
                type="thought",
                payload={
                    "text": plan.thought,
                    "tool_name": plan.tool_name,
                    "time_length": plan.time_length,
                    "time_unit": plan.time_unit,
                },
            )

            if plan.tool_name == "direct_response":
                yield AgentStreamEvent(type="status", payload={"text": "正在整理最终回答..."})
                run_result = self._build_run_result(
                    request,
                    {
                        "request_message": request.message,
                        "plan": plan.model_dump(mode="json"),
                        "final_text": (plan.direct_response or "当前无法可靠回答该问题。").strip(),
                    },
                )
                yield AgentStreamEvent(type="final", payload=run_result.message.model_dump(mode="json"))
                return

            yield AgentStreamEvent(type="status", payload={"text": self._build_tool_status(plan)})
            answer = self._run_tool(request, plan)
            yield AgentStreamEvent(
                type="tool",
                payload={
                    "tool_name": answer.question_type.value if hasattr(answer.question_type, "value") else str(answer.question_type),
                    "summary": answer.summary,
                },
            )

            yield AgentStreamEvent(type="status", payload={"text": "正在整理最终回答..."})
            final_text = self._render_final_text(request.message, plan, answer)
            run_result = self._build_run_result(
                request,
                {
                    "request_message": request.message,
                    "plan": plan.model_dump(mode="json"),
                    "answer_payload": answer.model_dump(mode="json"),
                    "final_text": final_text,
                },
            )
            yield AgentStreamEvent(type="final", payload=run_result.message.model_dump(mode="json"))
        except Exception as exc:
            trace_event("agent.fallback", {"reason": "agent_error", "type": exc.__class__.__name__, "message": str(exc)})
            fallback_message = self.fallback_answer_service.answer_chat(request)
            yield AgentStreamEvent(type="final", payload=fallback_message.model_dump(mode="json"))

    def _build_graph(self):
        workflow = StateGraph(AgentGraphState)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("call_tool", self._tool_node)
        workflow.add_node("respond", self._respond_node)
        workflow.add_edge(START, "plan")
        workflow.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {
                "call_tool": "call_tool",
                "respond": "respond",
            },
        )
        workflow.add_edge("call_tool", "respond")
        workflow.add_edge("respond", END)
        return workflow.compile()

    def _plan_node(self, state: AgentGraphState) -> AgentGraphState:
        plan = self._plan_request(state["request_message"])
        return {"plan": plan.model_dump(mode="json")}

    def _tool_node(self, state: AgentGraphState) -> AgentGraphState:
        plan = AgentPlanningResult.model_validate(state["plan"])
        request = ChatRequest(message=state["request_message"])
        answer = self._run_tool(request, plan)
        return {"answer_payload": answer.model_dump(mode="json")}

    def _respond_node(self, state: AgentGraphState) -> AgentGraphState:
        plan = AgentPlanningResult.model_validate(state["plan"])
        if plan.tool_name == "direct_response":
            final_text = (plan.direct_response or "当前无法可靠回答该问题。").strip()
            return {"final_text": final_text}

        answer = AnswerPayload.model_validate(state["answer_payload"])
        final_text = self._render_final_text(state["request_message"], plan, answer)
        return {"final_text": final_text}

    def _route_after_plan(self, state: AgentGraphState) -> str:
        plan = AgentPlanningResult.model_validate(state["plan"])
        if plan.tool_name == "direct_response":
            return "respond"
        return "call_tool"

    def _build_run_result(self, request: ChatRequest, state: AgentGraphState) -> AgentRunResult:
        final_text = str(state.get("final_text") or "").strip()
        if "answer_payload" in state:
            answer = AnswerPayload.model_validate(state["answer_payload"])
            message = self.chat_presenter_service.build_message(answer, text_override=final_text or None)
            return AgentRunResult(answer=answer, message=message)

        plan = AgentPlanningResult.model_validate(state["plan"])
        answer = AnswerPayload(
            question_type=IntentType.UNKNOWN,
            request_message=request.message,
            summary=final_text or plan.direct_response or "当前无法可靠回答该问题。",
            objective_data={"source_mode": "agent_direct_response"},
            analysis=[plan.reason or "当前代理未调用后端工具。"],
            sources=[],
            limitations=["当前回答未调用数据工具或知识检索结果，请谨慎参考。"],
            route=RouteDecision(
                intent=IntentType.UNKNOWN,
                decision_source="agent",
                reason=plan.reason or "Agent produced a direct response.",
            ),
        )
        message = self.chat_presenter_service.build_message(answer, text_override=answer.summary)
        return AgentRunResult(answer=answer, message=message)

    def _extract_message_text(self, message) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
            return "".join(parts)
        return str(content or "")

    def _plan_request(self, request_message: str) -> AgentPlanningResult:
        system_prompt, user_prompt = build_agent_planning_prompt(request_message)
        structured_model = self.model.with_structured_output(AgentPlanningResult)
        plan = structured_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        plan = self._normalize_plan(request_message, plan)
        trace_event("agent.plan", plan)
        return plan

    def _run_tool(self, request: ChatRequest, plan: AgentPlanningResult) -> AnswerPayload:
        answer = self.tool_executor.run(request, plan)
        trace_event("agent.tool_result", {"question_type": answer.question_type, "summary": answer.summary})
        return answer

    def _render_final_text(self, request_message: str, plan: AgentPlanningResult, answer: AnswerPayload) -> str:
        question_type = answer.question_type.value if hasattr(answer.question_type, "value") else str(answer.question_type)
        if question_type in {"asset_price", "asset_trend", "asset_event_analysis"}:
            return self.chat_presenter_service.build_message(answer).text

        system_prompt, user_prompt = build_agent_response_prompt(request_message, answer)
        try:
            response = self.model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            final_text = self._extract_message_text(response)
        except Exception as exc:
            trace_event("agent.respond_error", {"type": exc.__class__.__name__, "message": str(exc)})
            final_text = self.chat_presenter_service.build_message(answer).text

        return self._sanitize_final_text(final_text, answer)

    def _build_tool_status(self, plan: AgentPlanningResult) -> str:
        if plan.tool_name == "asset_price":
            return "正在查询最新市场价格..."
        if plan.tool_name == "asset_trend":
            return "正在拉取历史行情并计算走势..."
        if plan.tool_name == "asset_event_analysis":
            return "正在查询行情、检索新闻并归纳上涨原因..."
        if plan.tool_name == "finance_knowledge":
            return "正在检索知识库并整理答案..."
        if plan.tool_name == "report_summary":
            return "正在检索财报材料并提炼摘要..."
        return "正在执行工具..."

    def _normalize_plan(self, request_message: str, plan: AgentPlanningResult) -> AgentPlanningResult:
        normalized = plan.model_copy(deep=True)
        message = request_message.lower()
        has_time_range = normalized.time_length is not None or any(
            token in message
            for token in ["最近", "过去", "近", "年", "月", "周", "天", "走势", "趋势", "表现", "历史"]
        )

        if normalized.tool_name == "asset_price" and has_time_range:
            normalized.tool_name = "asset_trend"
            if not normalized.reason:
                normalized.reason = "后端一致性校验：包含时间范围的股价问题改为趋势查询。"

        return normalized

    def _sanitize_final_text(self, text: str, answer: AnswerPayload) -> str:
        cleaned = text.strip()
        if not cleaned:
            return self.chat_presenter_service.build_message(answer).text

        lower = cleaned.lower()
        bad_signals = [
            "this is a json object",
            "here's a breakdown",
            "**data**",
            "`data`",
            "sources:",
            "route:",
        ]
        if any(signal in lower for signal in bad_signals):
            return self.chat_presenter_service.build_message(answer).text

        cleaned = re.sub(r"[*`#]+", "", cleaned)
        cleaned = re.sub(r"\n\s*[-•]\s+", "\n", cleaned)
        return cleaned.strip() or self.chat_presenter_service.build_message(answer).text
