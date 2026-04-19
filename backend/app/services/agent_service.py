from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterator

from app.core.company_catalog import find_company_profile
from app.core.errors import AppError
from app.llm.client import BaseLLMClient, NullLLMClient
from app.llm.contracts import AgentPlanningResult
from app.llm.prompts import build_agent_planning_prompt, build_agent_response_prompt
from app.observability.request_trace import trace_event
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
        request = self._with_retrieval_query(request, plan)
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

        if plan.tool_name == "web_finance_knowledge":
            route = self._build_route(
                intent=IntentType.FINANCE_KNOWLEDGE,
                plan=plan,
                need_rag=True,
            )
            web_request = request.model_copy(deep=True)
            web_request.metadata["search_backend"] = "web"
            return self.knowledge_qa_service.answer(web_request, route)

        if plan.tool_name == "report_summary":
            route = self._build_route(
                intent=IntentType.REPORT_SUMMARY,
                plan=plan,
                need_rag=True,
            )
            return self.knowledge_qa_service.answer(request, route)

        if plan.tool_name == "web_report_summary":
            route = self._build_route(
                intent=IntentType.REPORT_SUMMARY,
                plan=plan,
                need_rag=True,
            )
            web_request = request.model_copy(deep=True)
            web_request.metadata["search_backend"] = "web"
            return self.knowledge_qa_service.answer(web_request, route)

        raise AppError(
            code="UNSUPPORTED_AGENT_TOOL",
            message="当前代理未找到可执行工具。",
            status_code=400,
            details={"tool_name": plan.tool_name},
        )

    def _with_retrieval_query(self, request: ChatRequest, plan: AgentPlanningResult) -> ChatRequest:
        if not plan.rewritten_query:
            return request
        updated_request = request.model_copy(deep=True)
        updated_request.metadata["retrieval_query"] = plan.rewritten_query
        return updated_request

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
        asset_qa_service: AssetQAService,
        knowledge_qa_service: KnowledgeQAService,
        chat_presenter_service: ChatPresenterService,
        fallback_answer_service: AnswerService,
        session_memory_service: SessionMemoryService,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        self.llm_client = llm_client or NullLLMClient()
        self.chat_presenter_service = chat_presenter_service
        self.fallback_answer_service = fallback_answer_service
        self.session_memory_service = session_memory_service
        self.tool_executor = AgentToolExecutor(
            asset_qa_service=asset_qa_service,
            knowledge_qa_service=knowledge_qa_service,
        )

    def is_enabled(self) -> bool:
        return self.llm_client.is_enabled()

    def answer(self, request: ChatRequest) -> AnswerPayload:
        if not self.is_enabled():
            trace_event("agent.fallback", {"reason": "agent_disabled"})
            return self.fallback_answer_service.answer(request)

        try:
            return self._run_request(request).answer
        except Exception as exc:
            trace_event("agent.fallback", {"reason": "agent_error", "type": exc.__class__.__name__, "message": str(exc)})
            return self.fallback_answer_service.answer(request)

    def answer_chat(self, request: ChatRequest) -> ChatMessagePayload:
        if not self.is_enabled():
            return self.fallback_answer_service.answer_chat(request)

        try:
            return self._run_request(request).message
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
            plan = self._plan_request(request)
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
                self.session_memory_service.remember(request.session_id, plan, run_result.answer)
                yield AgentStreamEvent(type="final", payload=run_result.message.model_dump(mode="json"))
                return

            cached_answer = self.session_memory_service.get_cached_answer(request.session_id, plan)
            if cached_answer is not None:
                yield AgentStreamEvent(type="status", payload={"text": "正在复用上一轮已查询结果..."})
                answer = cached_answer
                trace_event("agent.tool_cache_hit", {"question_type": answer.question_type, "summary": answer.summary})
            else:
                yield AgentStreamEvent(type="status", payload={"text": self._build_tool_status(plan)})
                answer = self._run_tool(request, plan)
                self.session_memory_service.remember(request.session_id, plan, answer)
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

    def _build_run_result(self, request: ChatRequest, state: dict) -> AgentRunResult:
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
                extracted_company=plan.company,
                extracted_symbol=plan.symbol,
                decision_source="agent",
                reason=plan.reason or "Agent produced a direct response.",
            ),
        )
        message = self.chat_presenter_service.build_message(answer, text_override=answer.summary)
        return AgentRunResult(answer=answer, message=message)

    def _run_request(self, request: ChatRequest) -> AgentRunResult:
        plan = self._plan_request(request)
        if plan.tool_name == "direct_response":
            run_result = self._build_run_result(
                request,
                {
                    "request_message": request.message,
                    "plan": plan.model_dump(mode="json"),
                    "final_text": (plan.direct_response or "当前无法可靠回答该问题。").strip(),
                },
            )
            self.session_memory_service.remember(request.session_id, plan, run_result.answer)
            return run_result

        cached_answer = self.session_memory_service.get_cached_answer(request.session_id, plan)
        if cached_answer is not None:
            trace_event("agent.tool_cache_hit", {"question_type": cached_answer.question_type, "summary": cached_answer.summary})
            answer = cached_answer
        else:
            answer = self._run_tool(request, plan)
            self.session_memory_service.remember(request.session_id, plan, answer)

        final_text = self._render_final_text(request.message, plan, answer)
        return self._build_run_result(
            request,
            {
                "request_message": request.message,
                "plan": plan.model_dump(mode="json"),
                "answer_payload": answer.model_dump(mode="json"),
                "final_text": final_text,
            },
        )

    def _plan_request(self, request: ChatRequest) -> AgentPlanningResult:
        context = self.session_memory_service.describe_context(request.session_id)
        system_prompt, user_prompt = build_agent_planning_prompt(request.message, context)
        plan = self.llm_client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=AgentPlanningResult,
        )
        plan = self._normalize_plan(request.message, plan)
        plan = self.session_memory_service.fill_plan_from_memory(request.session_id, plan)
        plan = self._inherit_time_range_from_memory(request, plan)
        trace_event("agent.plan", plan)
        return plan

    def _run_tool(self, request: ChatRequest, plan: AgentPlanningResult) -> AnswerPayload:
        related_answer = self.session_memory_service.get_related_answer(request.session_id, plan)
        tool_request = self._build_tool_request(request, plan, related_answer)
        answer = self.tool_executor.run(tool_request, plan)
        trace_event("agent.tool_result", {"question_type": answer.question_type, "summary": answer.summary})
        return answer

    def _render_final_text(self, request_message: str, plan: AgentPlanningResult, answer: AnswerPayload) -> str:
        question_type = answer.question_type.value if hasattr(answer.question_type, "value") else str(answer.question_type)
        if question_type in {"asset_price", "asset_trend", "asset_event_analysis"}:
            return self.chat_presenter_service.build_message(answer).text

        system_prompt, user_prompt = build_agent_response_prompt(request_message, answer)
        try:
            final_text = "".join(
                self.llm_client.generate_text_stream(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            )
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
            return "正在查询行情、检索新闻并归纳变化原因..."
        if plan.tool_name == "finance_knowledge":
            return "正在检索知识库并整理答案..."
        if plan.tool_name == "web_finance_knowledge":
            return "正在检索网页资料并整理答案..."
        if plan.tool_name == "report_summary":
            return "正在检索财报材料并提炼摘要..."
        if plan.tool_name == "web_report_summary":
            return "正在检索官网或网页财报材料..."
        return "正在执行工具..."

    def _normalize_plan(self, request_message: str, plan: AgentPlanningResult) -> AgentPlanningResult:
        normalized = plan.model_copy(deep=True)
        if normalized.rewritten_query:
            normalized.rewritten_query = normalized.rewritten_query.strip()
        if normalized.direct_response:
            normalized.direct_response = normalized.direct_response.strip()
        return normalized

    def _build_tool_request(
        self,
        request: ChatRequest,
        plan: AgentPlanningResult,
        related_answer: AnswerPayload | None,
    ) -> ChatRequest:
        tool_request = request.model_copy(deep=True)
        if plan.rewritten_query:
            tool_request.metadata["retrieval_query"] = plan.rewritten_query.strip()
        if plan.tool_name in {"web_finance_knowledge", "web_report_summary"}:
            tool_request.metadata["search_backend"] = "web"

        if related_answer is None:
            return tool_request

        same_symbol = (
            not plan.symbol
            or (related_answer.route.extracted_symbol or "").strip().lower() == (plan.symbol or "").strip().lower()
        )
        same_company = (
            not plan.company
            or (related_answer.route.extracted_company or "").strip().lower() == (plan.company or "").strip().lower()
        )
        if not same_symbol and not same_company:
            return tool_request

        tool_request.metadata["memory_context"] = {
            "previous_answer": related_answer.model_dump(mode="json"),
        }
        trace_event(
            "agent.memory_context",
            {
                "session_id": request.session_id,
                "question_type": related_answer.question_type,
                "summary": related_answer.summary,
            },
        )
        return tool_request

    def _inherit_time_range_from_memory(self, request: ChatRequest, plan: AgentPlanningResult) -> AgentPlanningResult:
        if self._message_has_explicit_time_range(request.message):
            return plan

        related_answer = self.session_memory_service.get_related_answer(request.session_id, plan)
        if related_answer is None:
            return plan

        time_range_days = related_answer.objective_data.get("time_range_days")
        if not isinstance(time_range_days, int) or time_range_days <= 0:
            return plan

        normalized = plan.model_copy(deep=True)
        normalized.time_length = time_range_days
        normalized.time_unit = "day"
        if not normalized.reason:
            normalized.reason = "后端一致性校验：当前问题未显式给出时间范围，继承上一轮时间窗口。"
        return normalized

    def _message_has_explicit_time_range(self, message: str) -> bool:
        lowered = message.lower()
        if re.search(r"\d+\s*(day|week|month|year|天|周|星期|个月|月|年)", lowered):
            return True
        return any(token in lowered for token in ["最近一年", "最近一月", "最近一个月", "最近一周", "最近一天", "过去", "近年", "近月", "近周", "近天"])

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
