from app.core.errors import AppError
from app.observability.request_trace import trace_event
from app.schemas.domain import IntentType
from app.schemas.request import ChatRequest
from app.schemas.response import AnswerPayload
from app.services.answer_generation_service import AnswerGenerationService
from app.services.asset_qa_service import AssetQAService
from app.services.knowledge_qa_service import KnowledgeQAService
from app.services.router_service import RouterService
from app.services.verification_service import VerificationService


class AnswerService:
    def __init__(
        self,
        router_service: RouterService,
        asset_qa_service: AssetQAService,
        knowledge_qa_service: KnowledgeQAService,
        answer_generation_service: AnswerGenerationService,
        verification_service: VerificationService,
    ) -> None:
        self.router_service = router_service
        self.asset_qa_service = asset_qa_service
        self.knowledge_qa_service = knowledge_qa_service
        self.answer_generation_service = answer_generation_service
        self.verification_service = verification_service

    def answer(self, request: ChatRequest) -> AnswerPayload:
        trace_event("answer.request", request)
        route = self.router_service.route(request)
        trace_event("answer.route", route)

        if route.intent in {
            IntentType.ASSET_PRICE,
            IntentType.ASSET_TREND,
            IntentType.ASSET_EVENT_ANALYSIS,
        }:
            draft_answer = self.asset_qa_service.answer(request, route)
            trace_event("answer.draft", draft_answer)
            trace_event("answer.final", draft_answer)
            return draft_answer

        if route.intent in {IntentType.FINANCE_KNOWLEDGE, IntentType.REPORT_SUMMARY}:
            draft_answer = self.knowledge_qa_service.answer(request, route)
            trace_event("answer.draft", draft_answer)
            generated_answer = self.answer_generation_service.generate(
                request_message=request.message,
                route=route,
                draft_answer=draft_answer,
            )
            trace_event("answer.generated", generated_answer)
            final_answer = self.verification_service.verify(generated_answer)
            trace_event("answer.final", final_answer)
            return final_answer

        raise AppError(
            code="UNSUPPORTED_QUESTION",
            message="当前无法识别该问题属于资产问答还是知识问答。",
            status_code=400,
            details={
                "hint": "请尝试明确说明你要问价格、走势、原因分析，或金融知识概念。",
                "message": request.message,
            },
        )
