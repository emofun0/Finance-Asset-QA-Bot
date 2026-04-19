from app.schemas.domain import IntentType
from app.schemas.request import ChatRequest
from app.services.router_service import RouterService


def build_router() -> RouterService:
    return RouterService()


def test_router_handles_assignment_questions() -> None:
    router = build_router()

    assert router.route(ChatRequest(message="阿里巴巴当前股价是多少？")).intent == IntentType.ASSET_PRICE

    trend_route = router.route(ChatRequest(message="BABA 最近 7 天涨跌情况如何？"))
    assert trend_route.intent == IntentType.ASSET_TREND
    assert trend_route.time_range_days == 7

    event_route = router.route(ChatRequest(message="阿里巴巴最近为何1月15日大涨？"))
    assert event_route.intent == IntentType.ASSET_EVENT_ANALYSIS
    assert event_route.event_date is not None
    assert event_route.event_date.endswith("-01-15")
    assert event_route.event_date_is_inferred is True

    tesla_route = router.route(ChatRequest(message="特斯拉近期走势如何？"))
    assert tesla_route.intent == IntentType.ASSET_TREND
    assert tesla_route.extracted_symbol == "TSLA"

    assert router.route(ChatRequest(message="什么是市盈率？")).intent == IntentType.FINANCE_KNOWLEDGE
    assert router.route(ChatRequest(message="收入和净利润的区别是什么？")).intent == IntentType.FINANCE_KNOWLEDGE
    assert router.route(ChatRequest(message="腾讯最近季度财报摘要是什么？")).intent == IntentType.REPORT_SUMMARY


def test_router_handles_variant_questions() -> None:
    router = build_router()

    price_route = router.route(ChatRequest(message="阿里现在多少钱？"))
    assert price_route.intent == IntentType.ASSET_PRICE
    assert price_route.extracted_symbol == "BABA"

    trend_route = router.route(ChatRequest(message="TSLA 这周表现怎么样？"))
    assert trend_route.intent == IntentType.ASSET_TREND
    assert trend_route.time_range_days == 7

    event_route = router.route(ChatRequest(message="英伟达 2026-01-15 为什么涨？"))
    assert event_route.intent == IntentType.ASSET_EVENT_ANALYSIS
    assert event_route.event_date == "2026-01-15"
    assert event_route.event_date_is_inferred is False

    report_route = router.route(ChatRequest(message="Tesla Q1 2025 earnings summary"))
    assert report_route.intent == IntentType.REPORT_SUMMARY
    assert report_route.extracted_company == "Tesla"
    assert report_route.extracted_symbol == "TSLA"
