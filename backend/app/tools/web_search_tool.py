from __future__ import annotations

from io import BytesIO
import re
from typing import Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from pypdf import PdfReader

from app.core.company_catalog import CompanyProfile
from app.observability.request_trace import trace_event
from app.rag.retriever import RetrievalResult


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
SEC_USER_AGENT = "Finance Asset QA System research contact finance-qa@example.com"
EVENT_SOURCE_DOMAINS = (
    "reuters.com",
    "finance.yahoo.com",
    "cnbc.com",
    "marketwatch.com",
    "sec.gov",
    "hkexnews.hk",
    "cninfo.com.cn",
)


class OfficialWebSearchTool:
    def __init__(self) -> None:
        self._ddgs = DDGS()

    def search_finance_knowledge(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        direct_results = self._search_investor_glossary(query)
        if direct_results:
            trace_event(
                "web_search.finance_knowledge",
                {
                    "query": query,
                    "mode": "direct_glossary",
                    "results": self._summarize_results(direct_results[:top_k]),
                },
            )
            return direct_results[:top_k]

        results = self._search(
            query=f"{query} investing definition glossary",
            allowed_domains=("investor.gov", "sec.gov"),
            top_k=top_k,
        )
        trace_event(
            "web_search.finance_knowledge",
            {
                "query": query,
                "mode": "search",
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
        report_query = f"{profile.canonical_name} investor relations earnings release annual report quarterly results {query}"
        results = self._search(
            query=report_query,
            allowed_domains=profile.official_domains,
            top_k=top_k,
            company=profile.canonical_name,
            symbol=profile.symbol,
            allow_snippet_fallback=True,
        )
        trace_event(
            "web_search.company_reports",
            {
                "query": report_query,
                "company": profile.canonical_name,
                "symbol": profile.symbol,
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
        date_hint = event_date or ""
        event_query = (
            f"{profile.canonical_name} {profile.symbol} {date_hint} stock jump drop reason earnings "
            f"guidance announcement news {query}"
        ).strip()
        results = self._search(
            query=event_query,
            allowed_domains=tuple(dict.fromkeys(profile.official_domains + EVENT_SOURCE_DOMAINS)),
            top_k=top_k,
            company=profile.canonical_name,
            symbol=profile.symbol,
            doc_type_hint="event_news",
            allow_snippet_fallback=True,
        )
        trace_event(
            "web_search.company_events",
            {
                "query": event_query,
                "company": profile.canonical_name,
                "symbol": profile.symbol,
                "event_date": event_date,
                "results": self._summarize_results(results),
            },
        )
        return results

    def _search(
        self,
        query: str,
        allowed_domains: tuple[str, ...],
        top_k: int,
        company: str | None = None,
        symbol: str | None = None,
        doc_type_hint: str | None = None,
        allow_snippet_fallback: bool = False,
    ) -> list[RetrievalResult]:
        try:
            results = list(self._ddgs.text(query, max_results=max(top_k * 4, 8)))
        except Exception as exc:
            trace_event("web_search.error", {"query": query, "type": exc.__class__.__name__, "message": str(exc)})
            return []

        candidates: list[RetrievalResult] = []
        seen_urls: set[str] = set()
        query_terms = self._extract_query_terms(query)

        for rank, item in enumerate(results):
            url = item.get("href")
            if not url or url in seen_urls:
                continue
            if not self._matches_allowed_domains(url, allowed_domains):
                continue
            seen_urls.add(url)

            title = item.get("title") or url
            body = item.get("body") or ""
            content = ""
            try:
                content = self._fetch_and_extract(url)
            except Exception:
                if allow_snippet_fallback:
                    content = self._normalize_whitespace(f"{title}\n{body}")
                else:
                    continue

            if len(content) < 120 and allow_snippet_fallback:
                content = self._normalize_whitespace(f"{content}\n{title}\n{body}")
            if len(content) < 80:
                continue

            score = self._score_document(title=title, body=body, content=content, query_terms=query_terms, rank=rank)
            candidates.append(
                RetrievalResult(
                    chunk_id=f"web::{rank}",
                    score=score,
                    content=content[:5000],
                    metadata={
                        "title": title,
                        "url": url,
                        "source_name": urlparse(url).netloc,
                        "doc_type": doc_type_hint or self._guess_doc_type(title, url),
                        "language": self._detect_language(content),
                        "company": company,
                        "symbol": symbol,
                    },
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:top_k]

    def _search_investor_glossary(self, query: str) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        for rank, slug in enumerate(self._build_glossary_slugs(query)):
            url = f"https://www.investor.gov/introduction-investing/investing-basics/glossary/{slug}"
            try:
                content = self._fetch_and_extract(url)
            except Exception:
                continue
            if len(content) < 120:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=f"web::glossary::{rank}",
                    score=1.5 - rank * 0.1,
                    content=content[:5000],
                    metadata={
                        "title": f"Investor.gov Glossary - {slug}",
                        "url": url,
                        "source_name": "investor.gov",
                        "doc_type": "glossary",
                        "language": "en",
                        "company": None,
                        "symbol": None,
                    },
                )
            )
        return results

    def _fetch_and_extract(self, url: str) -> str:
        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        if "sec.gov" in url:
            headers["User-Agent"] = SEC_USER_AGENT
            headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if url.lower().endswith(".pdf") or "pdf" in content_type:
            return self._extract_pdf_text(response.content)
        return self._extract_html_text(response.text)

    def _extract_html_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "img", "video", "audio"]):
            tag.decompose()

        root = soup.find("article") or soup.find("main") or soup.find("body") or soup
        lines = self._collect_html_lines(root, ["h1", "h2", "h3", "p", "li", "td"])
        if not lines:
            lines = self._collect_html_lines(root, ["div", "font", "span"])
        return "\n".join(lines)

    def _extract_pdf_text(self, content: bytes) -> str:
        reader = PdfReader(BytesIO(content))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            normalized = self._normalize_whitespace(text)
            if normalized:
                pages.append(normalized)
        return "\n\n".join(pages)

    def _score_document(self, title: str, body: str, content: str, query_terms: list[str], rank: int) -> float:
        score = max(1.0 - rank * 0.08, 0.2)
        lowered_title = title.lower()
        lowered_body = body.lower()
        lowered_content = content.lower()

        for term in query_terms:
            lowered_term = term.lower()
            if lowered_term in lowered_title:
                score += 0.24
            elif lowered_term in lowered_body:
                score += 0.12
            elif lowered_term in lowered_content:
                score += 0.05

        if any(keyword in lowered_title for keyword in ["earnings", "results", "guidance", "announcement", "stock", "shares"]):
            score += 0.1
        return score

    def _extract_query_terms(self, query: str) -> list[str]:
        terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", query)
        return [term.lower() for term in terms]

    def _build_glossary_slugs(self, query: str) -> list[str]:
        normalized = query.lower()
        localized_overrides = {
            "市盈率": "price-to-earnings",
            "本益比": "price-to-earnings",
            "市净率": "price-book-ratio",
            "贝塔": "beta",
            "beta系数": "beta",
            "阿尔法": "alpha",
            "每股收益": "earnings-per-share",
            "股息率": "dividend-yield",
            "净资产收益率": "return-equity-roe",
            "季度报告": "quarterly-report",
            "季报": "quarterly-report",
            "年报": "annual-report",
            "年度报告": "annual-report",
        }
        matched_slugs: list[str] = []
        for phrase, slug in localized_overrides.items():
            if phrase in query:
                matched_slugs.append(slug)
        if matched_slugs:
            return list(dict.fromkeys(matched_slugs))

        normalized = re.sub(r"什么是|what is|define|definition of|coefficient|ratio", " ", normalized)
        tokens = [token for token in re.findall(r"[a-z]{2,}", normalized) if token not in {"the", "and", "for"}]
        if not tokens:
            return []

        overrides = {
            "beta": "beta",
            "pe": "price-earnings-pe-ratio",
            "pe ratio": "price-to-earnings",
            "price earnings ratio": "price-to-earnings",
            "price to earnings ratio": "price-to-earnings",
            "price earnings": "price-earnings-pe-ratio",
        }
        phrases = [" ".join(tokens), tokens[0], tokens[-1]]
        slugs: list[str] = []
        for phrase in phrases:
            if not phrase:
                continue
            slug = overrides.get(phrase, phrase.replace(" ", "-"))
            if slug not in slugs:
                slugs.append(slug)
        return slugs

    def _matches_allowed_domains(self, url: str, allowed_domains: Iterable[str]) -> bool:
        netloc = urlparse(url).netloc.lower()
        return any(netloc.endswith(domain.lower()) for domain in allowed_domains)

    def _guess_doc_type(self, title: str, url: str) -> str:
        lowered = f"{title} {url}".lower()
        if any(keyword in lowered for keyword in ["annual report", "年度报告", "年报", "10-k", "20-f"]):
            return "annual_report"
        if any(keyword in lowered for keyword in ["interim report", "中期报告", "半年报", "half-year"]):
            return "interim_report"
        if any(keyword in lowered for keyword in ["earnings release", "quarter results", "季度业绩", "财报", "8-k", "results"]):
            return "earnings_release"
        if any(keyword in lowered for keyword in ["glossary", "definition", "术语", "what is", "investing basics"]):
            return "glossary"
        if any(keyword in lowered for keyword in ["stock", "shares", "rose", "fell", "guidance", "announcement", "news"]):
            return "event_news"
        return "web_search"

    def _detect_language(self, content: str) -> str:
        return "zh" if re.search(r"[\u4e00-\u9fff]", content) else "en"

    def _normalize_whitespace(self, value: str) -> str:
        value = value.replace("\xa0", " ")
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()

    def _collect_html_lines(self, root: BeautifulSoup, selectors: list[str]) -> list[str]:
        lines: list[str] = []
        seen: set[str] = set()
        for element in root.find_all(selectors):
            text = self._normalize_whitespace(element.get_text(" ", strip=True))
            if len(text) < 8 or text in seen:
                continue
            seen.add(text)
            lines.append(text)
        return lines

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
