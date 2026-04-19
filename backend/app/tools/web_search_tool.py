from __future__ import annotations

import re
from urllib.parse import urlparse

from ddgs import DDGS

from app.core.company_catalog import CompanyProfile
from app.observability.request_trace import trace_event
from app.rag.retriever import RetrievalResult


DEFAULT_REGION = "wt-wt"
DEFAULT_SAFETY = "off"
NEWS_SOURCE_DOMAINS = (
    "reuters.com",
    "finance.yahoo.com",
    "cnbc.com",
    "marketwatch.com",
    "bloomberg.com",
    "wsj.com",
)


class OfficialWebSearchTool:
    def __init__(self) -> None:
        self._ddgs = DDGS()

    def search_finance_knowledge(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        results = self._search_text(
            queries=[query],
            top_k=top_k,
            search_profile="finance_knowledge",
        )
        trace_event(
            "web_search.finance_knowledge",
            {
                "query": query,
                "mode": "ddgs_text",
                "results": self._summarize_results(results),
            },
        )
        return results

    def search_company_reports(
        self,
        query: str,
        profile: CompanyProfile,
        top_k: int = 4,
    ) -> list[RetrievalResult]:
        base_query = " ".join(
            part
            for part in [
                profile.canonical_name,
                profile.symbol,
                query,
                "earnings release quarterly results annual report investor relations",
            ]
            if part
        ).strip()
        queries = self._build_site_queries(base_query, profile.official_domains)
        results = self._search_text(
            queries=queries,
            top_k=top_k,
            allowed_domains=profile.official_domains,
            company=profile.canonical_name,
            symbol=profile.symbol,
            doc_type_hint="web_search",
            search_profile="company_reports",
        )
        trace_event(
            "web_search.company_reports",
            {
                "query": base_query,
                "company": profile.canonical_name,
                "symbol": profile.symbol,
                "results": self._summarize_results(results),
            },
        )
        return results

    def search_company_reports_by_query(
        self,
        query: str,
        *,
        company: str | None = None,
        symbol: str | None = None,
        top_k: int = 4,
    ) -> list[RetrievalResult]:
        base_query = " ".join(
            part
            for part in [
                company,
                symbol,
                query,
                "earnings release quarterly results annual report investor relations",
            ]
            if part
        ).strip()
        results = self._search_text(
            queries=[base_query],
            top_k=top_k,
            company=company,
            symbol=symbol,
            doc_type_hint="web_search",
            search_profile="company_reports",
        )
        trace_event(
            "web_search.company_reports_generic",
            {
                "query": base_query,
                "company": company,
                "symbol": symbol,
                "results": self._summarize_results(results),
            },
        )
        return results

    def search_company_events(
        self,
        query: str,
        profile: CompanyProfile,
        top_k: int = 4,
        event_date: str | None = None,
    ) -> list[RetrievalResult]:
        base_query = " ".join(
            part
            for part in [
                profile.canonical_name,
                profile.symbol,
                event_date,
                query,
            ]
            if part
        ).strip()
        news_results = self._search_news(
            query=f"{base_query} stock move reason earnings guidance",
            top_k=top_k,
            company=profile.canonical_name,
            symbol=profile.symbol,
        )
        if news_results:
            trace_event(
                "web_search.company_events",
                {
                    "query": base_query,
                    "company": profile.canonical_name,
                    "symbol": profile.symbol,
                    "event_date": event_date,
                    "mode": "ddgs_news",
                    "results": self._summarize_results(news_results),
                },
            )
            return news_results

        queries = self._build_site_queries(
            f"{base_query} stock move reason earnings guidance news",
            tuple(dict.fromkeys(profile.official_domains + NEWS_SOURCE_DOMAINS)),
        )
        text_results = self._search_text(
            queries=queries,
            top_k=top_k,
            allowed_domains=tuple(dict.fromkeys(profile.official_domains + NEWS_SOURCE_DOMAINS)),
            company=profile.canonical_name,
            symbol=profile.symbol,
            doc_type_hint="event_news",
            search_profile="company_events",
        )
        trace_event(
            "web_search.company_events",
            {
                "query": base_query,
                "company": profile.canonical_name,
                "symbol": profile.symbol,
                "event_date": event_date,
                "mode": "ddgs_text",
                "results": self._summarize_results(text_results),
            },
        )
        return text_results

    def _search_text(
        self,
        *,
        queries: list[str],
        top_k: int,
        allowed_domains: tuple[str, ...] | None = None,
        company: str | None = None,
        symbol: str | None = None,
        doc_type_hint: str | None = None,
        search_profile: str | None = None,
    ) -> list[RetrievalResult]:
        merged: list[RetrievalResult] = []
        seen_urls: set[str] = set()
        for query in queries:
            try:
                raw_results = list(
                    self._ddgs.text(
                        query,
                        region=DEFAULT_REGION,
                        safesearch=DEFAULT_SAFETY,
                        max_results=max(top_k * 3, 8),
                    )
                )
            except Exception as exc:
                trace_event("web_search.error", {"query": query, "type": exc.__class__.__name__, "message": str(exc)})
                continue

            query_terms = self._extract_query_terms(query)
            for rank, item in enumerate(raw_results):
                url = str(item.get("href") or "").strip()
                if not url or url in seen_urls:
                    continue
                if allowed_domains and not self._matches_allowed_domains(url, allowed_domains):
                    continue
                title = self._normalize_whitespace(str(item.get("title") or ""))
                body = self._normalize_whitespace(str(item.get("body") or ""))
                if self._is_low_signal_result(title=title, body=body, url=url, search_profile=search_profile):
                    continue
                content = self._normalize_whitespace(f"{title}\n{body}")
                if len(content) < 40:
                    continue
                seen_urls.add(url)
                merged.append(
                    RetrievalResult(
                        chunk_id=f"web::{len(merged)}",
                        score=self._score_result(title=title, body=body, url=url, query_terms=query_terms, rank=rank),
                        content=content[:2000],
                        metadata={
                            "title": title or url,
                            "url": url,
                            "source_name": urlparse(url).netloc,
                            "doc_type": doc_type_hint or self._guess_doc_type(title, url, body),
                            "language": "zh" if re.search(r"[\u4e00-\u9fff]", content) else "en",
                            "company": company,
                            "symbol": symbol,
                        },
                    )
                )

        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[:top_k]

    def _search_news(
        self,
        *,
        query: str,
        top_k: int,
        company: str | None = None,
        symbol: str | None = None,
    ) -> list[RetrievalResult]:
        try:
            raw_results = list(
                self._ddgs.news(
                    query,
                    region=DEFAULT_REGION,
                    safesearch=DEFAULT_SAFETY,
                    max_results=max(top_k * 2, 6),
                )
            )
        except Exception as exc:
            trace_event("web_search.error", {"query": query, "type": exc.__class__.__name__, "message": str(exc)})
            return []

        query_terms = self._extract_query_terms(query)
        results: list[RetrievalResult] = []
        seen_urls: set[str] = set()
        for rank, item in enumerate(raw_results):
            url = str(item.get("url") or item.get("href") or "").strip()
            if not url or url in seen_urls:
                continue
            title = self._normalize_whitespace(str(item.get("title") or ""))
            body = self._normalize_whitespace(str(item.get("body") or item.get("excerpt") or ""))
            if self._is_low_signal_result(title=title, body=body, url=url, search_profile="company_events"):
                continue
            seen_urls.add(url)
            results.append(
                RetrievalResult(
                    chunk_id=f"news::{len(results)}",
                    score=self._score_result(title=title, body=body, url=url, query_terms=query_terms, rank=rank) + 0.15,
                    content=self._normalize_whitespace(f"{title}\n{body}")[:2000],
                    metadata={
                        "title": title or url,
                        "url": url,
                        "source_name": urlparse(url).netloc,
                        "doc_type": "event_news",
                        "language": "zh" if re.search(r"[\u4e00-\u9fff]", f"{title}{body}") else "en",
                        "company": company,
                        "symbol": symbol,
                    },
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def _build_site_queries(self, base_query: str, domains: tuple[str, ...]) -> list[str]:
        queries = [base_query]
        queries.extend(f"site:{domain} {base_query}" for domain in domains[:3])
        return list(dict.fromkeys(query for query in queries if query.strip()))

    def _extract_query_terms(self, query: str) -> list[str]:
        return [term.lower() for term in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{2,}", query)]

    def _score_result(self, *, title: str, body: str, url: str, query_terms: list[str], rank: int) -> float:
        haystack = f"{title} {body} {url}".lower()
        score = max(1.0 - rank * 0.08, 0.25)
        for term in query_terms:
            if term in haystack:
                score += 0.12
        if any(token in haystack for token in ["earnings", "results", "quarter", "annual report", "财报", "业绩"]):
            score += 0.15
        if any(token in haystack for token in ["why", "jumps", "drops", "surges", "跌", "涨", "原因"]):
            score += 0.12
        if any(token in haystack for token in ["definition", "what is", "术语", "定义", "百科"]):
            score += 0.08
        return score

    def _matches_allowed_domains(self, url: str, allowed_domains: tuple[str, ...]) -> bool:
        netloc = urlparse(url).netloc.lower()
        return any(netloc.endswith(domain.lower()) for domain in allowed_domains)

    def _is_low_signal_result(self, *, title: str, body: str, url: str, search_profile: str | None) -> bool:
        lowered = f"{title} {body} {url}".lower()
        common_tokens = ["login", "sign in", "cookie", "privacy policy", "terms of service"]
        if any(token in lowered for token in common_tokens):
            return True
        if search_profile == "finance_knowledge" and any(token in lowered for token in ["home page", "homepage", "category", "tag archive"]):
            return True
        if search_profile == "company_events" and any(token in lowered for token in ["investor relations", "official website", "home page"]):
            return True
        return False

    def _guess_doc_type(self, title: str, url: str, body: str) -> str:
        lowered = f"{title} {url} {body}".lower()
        if any(token in lowered for token in ["annual report", "10-k", "20-f", "年报"]):
            return "annual_report"
        if any(token in lowered for token in ["quarterly", "quarter results", "10-q", "季报"]):
            return "quarterly_results"
        if any(token in lowered for token in ["earnings release", "financial results", "业绩", "财报"]):
            return "earnings_release"
        if any(token in lowered for token in ["news", "reuters", "cnbc", "marketwatch", "bloomberg"]):
            return "event_news"
        if any(token in lowered for token in ["definition", "what is", "glossary", "术语", "定义", "百科"]):
            return "glossary"
        return "web_search"

    def _normalize_whitespace(self, value: str) -> str:
        value = value.replace("\xa0", " ")
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    def _summarize_results(self, results: list[RetrievalResult]) -> list[dict[str, str | float | None]]:
        return [
            {
                "title": str(result.metadata.get("title")),
                "url": str(result.metadata.get("url")),
                "doc_type": str(result.metadata.get("doc_type")),
                "score": round(result.score, 4),
            }
            for result in results
        ]
