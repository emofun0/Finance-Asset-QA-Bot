from app.core.company_catalog import CompanyProfile
from app.rag.retriever import RetrievalResult
from app.tools.web_search_tool import OfficialWebSearchTool


def build_tool() -> OfficialWebSearchTool:
    return OfficialWebSearchTool()


def build_result(title: str, url: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="web::0",
        score=1.0,
        content=title,
        metadata={"title": title, "url": url, "doc_type": "event_news"},
    )


def test_search_finance_knowledge_uses_agent_query_directly() -> None:
    tool = build_tool()
    captured: dict[str, object] = {}

    def fake_search_text(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return []

    tool._search_text = fake_search_text  # type: ignore[method-assign]

    tool.search_finance_knowledge("Nasdaq stock exchange definition")

    assert captured["queries"] == ["Nasdaq stock exchange definition"]
    assert captured["search_profile"] == "finance_knowledge"


def test_search_company_reports_builds_site_queries() -> None:
    tool = build_tool()
    captured: dict[str, object] = {}

    def fake_search_text(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return []

    tool._search_text = fake_search_text  # type: ignore[method-assign]

    profile = CompanyProfile(
        canonical_name="Alibaba",
        symbol="BABA",
        country_group="china",
        aliases=("阿里巴巴",),
        official_domains=("alibabagroup.com", "sec.gov"),
    )
    tool.search_company_reports("latest earnings", profile)

    queries = captured["queries"]
    assert isinstance(queries, list)
    assert any("site:alibabagroup.com" in query for query in queries)
    assert "Alibaba" in queries[0]


def test_search_company_events_prefers_news() -> None:
    tool = build_tool()
    calls = {"news": 0, "text": 0}

    def fake_search_news(**kwargs):  # type: ignore[no-untyped-def]
        calls["news"] += 1
        return [build_result("news-result", "https://example.com/news")]

    def fake_search_text(**kwargs):  # type: ignore[no-untyped-def]
        calls["text"] += 1
        return []

    tool._search_news = fake_search_news  # type: ignore[method-assign]
    tool._search_text = fake_search_text  # type: ignore[method-assign]

    profile = CompanyProfile(
        canonical_name="Alibaba",
        symbol="BABA",
        country_group="china",
        aliases=("阿里巴巴",),
        official_domains=("alibabagroup.com",),
    )
    results = tool.search_company_events("why did it jump", profile)

    assert results[0].metadata["title"] == "news-result"
    assert calls == {"news": 1, "text": 0}


def test_search_company_events_falls_back_to_text() -> None:
    tool = build_tool()
    calls = {"news": 0, "text": 0}

    def fake_search_news(**kwargs):  # type: ignore[no-untyped-def]
        calls["news"] += 1
        return []

    def fake_search_text(**kwargs):  # type: ignore[no-untyped-def]
        calls["text"] += 1
        return [build_result("text-result", "https://example.com/text")]

    tool._search_news = fake_search_news  # type: ignore[method-assign]
    tool._search_text = fake_search_text  # type: ignore[method-assign]

    profile = CompanyProfile(
        canonical_name="Alibaba",
        symbol="BABA",
        country_group="china",
        aliases=("阿里巴巴",),
        official_domains=("alibabagroup.com",),
    )
    results = tool.search_company_events("why did it jump", profile)

    assert results[0].metadata["title"] == "text-result"
    assert calls == {"news": 1, "text": 1}
