from app.schemas.domain import IntentType, RouteDecision
from app.schemas.response import AnswerPayload
from app.services.session_memory_service import SessionMemoryService


def _build_answer(text: str) -> AnswerPayload:
    return AnswerPayload(
        question_type=IntentType.FINANCE_KNOWLEDGE,
        request_message=text,
        summary=text,
        objective_data={},
        analysis=[],
        sources=[],
        limitations=[],
        route=RouteDecision(intent=IntentType.FINANCE_KNOWLEDGE),
    )


def test_session_memory_compresses_old_turns_aggressively() -> None:
    memory = SessionMemoryService()

    memory.remember(
        "session-1",
        user_message="第一轮：什么是市盈率？",
        answer=_build_answer("市盈率是股价相对每股收益的倍数。"),
        tool_notes=["本地知识库命中 3 条。"],
    )
    memory.remember(
        "session-1",
        user_message="第二轮：那腾讯最近财报呢？",
        answer=_build_answer("已命中腾讯最近财报，并抽出关键数字。"),
        tool_notes=["本地财报命中 4 条。", "提取财报上下文：指标 6 条。"],
    )
    memory.remember(
        "session-1",
        user_message="第三轮：最近一个月股价怎么样？",
        answer=_build_answer("最近一个月整体上涨。"),
        tool_notes=["0700.HK 最近 30 天上涨 5.2%。"],
    )

    context_messages = memory.build_context_messages("session-1")

    assert len(context_messages) == 5
    assert "压缩记忆" in context_messages[0]["content"]
    assert "第一轮：什么是市盈率" in context_messages[0]["content"]
    assert context_messages[1]["content"] == "第二轮：那腾讯最近财报呢？"
    assert context_messages[3]["content"] == "第三轮：最近一个月股价怎么样？"
