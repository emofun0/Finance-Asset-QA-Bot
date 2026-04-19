import re

from app.core.company_catalog import find_company_profile
from app.observability.request_trace import trace_event
from app.schemas.request import ChatRequest
from app.schemas.domain import RouteDecision
from app.schemas.response import AnswerPayload, SourceItem
from app.rag.retriever import KnowledgeRetriever, RetrievalResult
from app.services.query_rewrite_service import QueryRewriteService
from app.tools.web_search_tool import OfficialWebSearchTool


class KnowledgeQAService:
    _report_doc_types = {
        "earnings_release",
        "quarterly_results",
        "quarterly_financial_statements",
        "quarterly_presentation",
        "annual_report",
        "interim_report",
    }

    def __init__(
        self,
        retriever: KnowledgeRetriever,
        web_search_tool: OfficialWebSearchTool | None = None,
        query_rewrite_service: QueryRewriteService | None = None,
    ) -> None:
        self.retriever = retriever
        self.web_search_tool = web_search_tool
        self.query_rewrite_service = query_rewrite_service

    def answer(self, request: ChatRequest, route: RouteDecision) -> AnswerPayload:
        results = self._retrieve(request.message, route)
        trace_event(
            "rag.results",
            {
                "intent": route.intent,
                "count": len(results),
                "results": [
                    {
                        "title": result.metadata.get("title"),
                        "doc_type": result.metadata.get("doc_type"),
                        "url": result.metadata.get("url"),
                        "score": round(result.score, 4),
                    }
                    for result in results[:6]
                ],
            },
        )

        if route.intent.value == "report_summary":
            return self._build_report_summary(request, route, results)
        return self._build_knowledge_answer(request, route, results)

    def _retrieve(self, message: str, route: RouteDecision) -> list[RetrievalResult]:
        rewritten_message = self._rewrite_query(message, route)
        preferred_language = "zh" if self._contains_chinese(message) else None
        trace_event(
            "rag.retrieve",
            {
                "message": message,
                "rewritten_message": rewritten_message,
                "intent": route.intent,
                "preferred_language": preferred_language,
                "company": route.extracted_company,
                "symbol": route.extracted_symbol,
            },
        )

        if route.intent.value == "finance_knowledge":
            results = self._search_local_finance_knowledge(rewritten_message, preferred_language)
            if self._has_sufficient_knowledge_coverage(message, results):
                return results
            if self.web_search_tool:
                return self.web_search_tool.search_finance_knowledge(rewritten_message, top_k=3)

        if route.intent.value == "report_summary":
            local_results = self._search_local_report_summary(rewritten_message, route, preferred_language)
            local_results = self._rerank_report_results(message, local_results)
            if self._has_sufficient_report_coverage(local_results):
                return local_results

            if self.web_search_tool:
                profile = find_company_profile(route.extracted_company, route.extracted_symbol)
                if profile:
                    web_results = self.web_search_tool.search_company_reports(rewritten_message, profile, top_k=3)
                    if local_results:
                        return self._rerank_report_results(message, self._merge_results(local_results, web_results, limit=6))
                    if web_results:
                        return self._rerank_report_results(message, web_results)

            if local_results:
                return local_results

        preferred_doc_type = None
        if any(keyword in message for keyword in ["季度", "季报"]):
            preferred_doc_type = "quarterly_results"
        elif any(keyword in message for keyword in ["半年", "中期"]):
            preferred_doc_type = "interim_report"

        results = self.retriever.search(
            rewritten_message,
            top_k=6,
            company=route.extracted_company,
            symbol=route.extracted_symbol,
            doc_type=preferred_doc_type,
            language=preferred_language,
        )
        if results:
            return results
        return self.retriever.search(
            rewritten_message,
            top_k=6,
            company=route.extracted_company,
            symbol=route.extracted_symbol,
            language=preferred_language,
        )

    def _search_local_finance_knowledge(self, message: str, preferred_language: str | None) -> list[RetrievalResult]:
        candidate_queries = [message]
        lowered = message.lower()
        if "收入" in message and "净利润" in message:
            candidate_queries.append("收入 净利润 区别 营业收入 net income revenue")
        if "市盈率" in message or "pe ratio" in lowered or "price to earnings" in lowered or "price-to-earnings" in lowered:
            candidate_queries.append("市盈率 PE ratio price to earnings price-to-earnings")
        if "季度报告" in message and "年报" in message:
            candidate_queries.append("季度报告 年报 区别 quarterly report annual report difference")
        if "beta" in lowered:
            candidate_queries.append("beta coefficient beta glossary market risk")

        combined: list[RetrievalResult] = []
        for query in candidate_queries:
            combined.extend(
                self.retriever.search(
                    query,
                    top_k=6,
                    doc_types=["knowledge_article", "glossary"],
                    language=preferred_language,
                )
            )
        if not combined:
            for query in candidate_queries:
                combined.extend(
                    self.retriever.search(
                        query,
                        top_k=6,
                        doc_types=["knowledge_article", "glossary"],
                    )
                )
        return self._rerank_finance_results(message, combined)

    def _search_local_report_summary(
        self,
        message: str,
        route: RouteDecision,
        preferred_language: str | None,
    ) -> list[RetrievalResult]:
        preferred_types = ["earnings_release", "quarterly_presentation", "quarterly_results"]
        if any(keyword in message for keyword in ["半年", "中期"]):
            preferred_types.append("interim_report")
        if any(keyword in message for keyword in ["年报", "年度"]):
            preferred_types = ["earnings_release", "annual_report"]

        for query in self._build_report_candidate_queries(message, route):
            results = self.retriever.search(
                query,
                top_k=6,
                company=route.extracted_company,
                symbol=route.extracted_symbol,
                doc_types=preferred_types,
                language=preferred_language,
            )
            if results:
                return results
            results = self.retriever.search(
                query,
                top_k=6,
                company=route.extracted_company,
                symbol=route.extracted_symbol,
                doc_types=preferred_types,
            )
            if results:
                return results

        preferred_doc_type = None
        if any(keyword in message for keyword in ["季度", "季报"]):
            preferred_doc_type = "quarterly_results"
        elif any(keyword in message for keyword in ["半年", "中期"]):
            preferred_doc_type = "interim_report"

        for query in self._build_report_candidate_queries(message, route):
            results = self.retriever.search(
                query,
                top_k=6,
                company=route.extracted_company,
                symbol=route.extracted_symbol,
                doc_type=preferred_doc_type,
                language=preferred_language,
            )
            if results:
                return results
            results = self.retriever.search(
                query,
                top_k=6,
                company=route.extracted_company,
                symbol=route.extracted_symbol,
            )
            if results:
                return results
        return []

    def _build_knowledge_answer(
        self,
        request: ChatRequest,
        route: RouteDecision,
        results: list[RetrievalResult],
    ) -> AnswerPayload:
        if not results:
            return AnswerPayload(
                question_type=route.intent,
                request_message=request.message,
                summary="知识库中未检索到足够依据，当前无法可靠回答该问题。",
                objective_data={
                    "retrieval_enabled": True,
                    "source_mode": "not_found",
                    "matched_chunks": 0,
                },
                analysis=[
                    "当前问题未在本地知识库或官方网页检索结果中找到高相关片段。",
                    "可通过补充术语词条或公司披露材料提升命中率。",
                ],
                sources=[],
                limitations=[
                    "当前回答依赖本地知识库和官方网页检索，不会在无依据时自由生成结论。",
                ],
                route=route,
            )

        summary, analysis = self._summarize_knowledge(request.message, results)
        source_mode = "web_fallback" if self._uses_web_results(results) else "local_rag"
        return AnswerPayload(
            question_type=route.intent,
            request_message=request.message,
            summary=summary,
            objective_data={
                "retrieval_enabled": True,
                "source_mode": source_mode,
                "matched_chunks": len(results),
                "top_doc_titles": [result.metadata.get("title") for result in results[:3]],
            },
            analysis=analysis,
            sources=self._to_sources(results[:3]),
            limitations=[
                "当前版本基于检索片段做抽取式归纳，不等同于完整教材解释。",
                "如问题超出已入库资料范围，系统会优先返回依据不足而非臆测。",
            ],
            route=route,
        )

    def _build_report_summary(
        self,
        request: ChatRequest,
        route: RouteDecision,
        results: list[RetrievalResult],
    ) -> AnswerPayload:
        if not results:
            company = route.extracted_company or route.extracted_symbol or "目标公司"
            return AnswerPayload(
                question_type=route.intent,
                request_message=request.message,
                summary=f"知识库中暂未检索到 {company} 的相关财报资料。",
                objective_data={
                    "retrieval_enabled": True,
                    "source_mode": "not_found",
                    "matched_chunks": 0,
                    "company": route.extracted_company,
                    "symbol": route.extracted_symbol,
                },
                analysis=[
                    "当前公司未在本地知识库中命中，且官方网页检索也未找到可用财报资料，或公司名称未被正确识别。",
                ],
                sources=[],
                limitations=[
                    "当前财报摘要只能基于本地资料或官方网页检索到的年报、中报、业绩材料生成。",
                ],
                route=route,
            )

        company = route.extracted_company or route.extracted_symbol or results[0].metadata.get("company")
        highlights = self._extract_financial_highlights(results)
        source_mode = "web_fallback" if self._uses_web_results(results) else "local_rag"
        summary = (
            f"根据检索到的官方财报资料，{company} 最近披露材料中的核心信息已检索到。"
            if not highlights
            else f"根据检索到的官方财报资料，{company} 最近披露材料的核心财务亮点已提取。"
        )
        analysis = highlights or [self._trim_excerpt(results[0].content)]

        return AnswerPayload(
            question_type=route.intent,
            request_message=request.message,
            summary=summary,
            objective_data={
                "retrieval_enabled": True,
                "source_mode": source_mode,
                "matched_chunks": len(results),
                "company": company,
                "symbol": route.extracted_symbol,
                "top_doc_titles": [result.metadata.get("title") for result in results[:3]],
            },
            analysis=analysis[:4],
            sources=self._to_sources(results[:4]),
            limitations=[
                "当前摘要基于检索到的财报片段，不保证覆盖整份报告全部重点。",
                "若需要更完整结论，应结合完整年报/中报原文阅读。",
            ],
            route=route,
        )

    def _summarize_knowledge(self, message: str, results: list[RetrievalResult]) -> tuple[str, list[str]]:
        lowered = message.lower()

        if "市盈率" in message:
            summary = "市盈率通常表示股票价格相对于每股收益的估值倍数，常用于衡量市场如何给公司盈利定价。"
        elif "pe ratio" in lowered or "price to earnings" in lowered or "price-to-earnings" in lowered:
            summary = "PE ratio 即市盈率，用来衡量股票价格相对于每股收益的估值倍数，是常见的估值指标。"
        elif "beta" in lowered:
            summary = "Beta 系数通常用来衡量某项资产相对于整体市场波动的敏感度，常用于描述系统性风险。"
        elif "季度报告" in message and "年报" in message and "区别" in message:
            summary = "季度报告通常聚焦最近一个季度的经营与财务表现，而年报覆盖完整财年，披露范围更广、内容更完整，且通常包含经审计信息。"
        elif "收入" in message and "净利润" in message and "区别" in message:
            summary = "收入反映公司卖出商品或服务取得的总金额，净利润则是在扣除成本、费用、税项等之后最终留下的利润。"
        elif "净利润" in message:
            summary = "净利润通常指公司在扣除成本、费用、利息和税项之后的最终利润。"
        elif "收入" in message:
            summary = "收入通常指公司通过销售商品或提供服务取得的总金额。"
        elif "quarterly report" in lowered or "10q" in lowered:
            summary = "季度报告通常披露公司最近一个季度未经审计的财务数据和经营情况。"
        else:
            summary = self._trim_excerpt(results[0].content)

        analysis = [self._extract_relevant_excerpt(message, result.content) for result in results[:3]]
        return summary, analysis

    def _extract_financial_highlights(self, results: list[RetrievalResult]) -> list[str]:
        highlights: list[str] = []
        seen: set[str] = set()
        ignored_signals = ["审计", "风险提示", "目录", "图表", "联系我们", "首页", "解决方案", "investor relations", "official website"]
        for result in results:
            sentences = re.split(r"(?<=[。！？.;])\s+|\n", result.content)
            for sentence in sentences:
                normalized = sentence.strip()
                if len(normalized) < 18:
                    continue
                lowered = normalized.lower()
                if any(token in lowered for token in ignored_signals):
                    continue
                has_finance_signal = any(
                    token in lowered
                    for token in [
                        "收入",
                        "营收",
                        "净利润",
                        "revenue",
                        "revenues",
                        "profit",
                        "net income",
                        "operating income",
                        "cash",
                        "cash flow",
                        "gross margin",
                        "deliveries",
                        "现金流",
                        "毛利",
                    ]
                )
                has_number = bool(re.search(r"\d", normalized))
                if has_finance_signal and has_number and normalized not in seen:
                    seen.add(normalized)
                    highlights.append(self._trim_excerpt(normalized))
                if len(highlights) >= 4:
                    return highlights
        return highlights

    def _trim_excerpt(self, text: str, max_length: int = 220) -> str:
        cleaned = " ".join(text.split())
        return cleaned[:max_length] + ("..." if len(cleaned) > max_length else "")

    def _extract_relevant_excerpt(self, message: str, content: str) -> str:
        terms = [term for term in ["市盈率", "每股收益", "营业收入", "收入", "净利润", "盈利能力", "季度报告", "年报"] if term in message]
        terms.extend(token for token in re.findall(r"[A-Za-z]{3,}", message) if token.lower() not in {"what", "define", "coefficient"})
        compact_content = " ".join(content.split())
        sentences = self._split_sentences(content)
        for sentence in sentences:
            if self._is_low_signal_excerpt(sentence):
                continue
            matched_terms = [term for term in terms if term.lower() in sentence.lower()]
            if matched_terms:
                if self._should_shorten_excerpt(message, sentence):
                    shortened = self._extract_keyword_window(compact_content, matched_terms[0])
                    if shortened and not self._is_low_signal_excerpt(shortened):
                        return shortened
                return self._trim_excerpt(sentence)
        for term in terms:
            candidate = self._extract_keyword_window(compact_content, term)
            if candidate and not self._is_low_signal_excerpt(candidate):
                return candidate
        return self._trim_excerpt(sentences[0] if sentences else content)

    def _split_sentences(self, content: str) -> list[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[。！？；;])\s*|(?<=\.)\s+(?=[A-Z])|\n+", content)
            if sentence.strip()
        ]

    def _should_shorten_excerpt(self, message: str, sentence: str) -> bool:
        lowered_message = message.lower()
        is_concept_question = "什么是" in message or "区别" in message or any(
            token in lowered_message
            for token in ["what is", "define", "definition", "beta", "pe ratio", "price to earnings"]
        )
        return is_concept_question and len(" ".join(sentence.split())) > 140

    def _extract_keyword_window(self, compact_content: str, term: str) -> str | None:
        lowered_content = compact_content.lower()
        lowered_term = term.lower()
        position = lowered_content.find(lowered_term)
        if position < 0:
            return None
        start = max(position - 48, 0)
        end = min(position + max(len(term), 18) + 132, len(compact_content))
        return self._trim_excerpt(compact_content[start:end].strip(), max_length=180)

    def _is_low_signal_excerpt(self, text: str) -> bool:
        lowered = " ".join(text.split()).lower()
        low_signal_tokens = [
            "official website of the united states government",
            "here’s how you know",
            "breadcrumb",
            "investor relations",
            "latest investors relations press releases",
            "tesla's mission is to accelerate the world's transition",
        ]
        return any(token in lowered for token in low_signal_tokens)

    def _to_sources(self, results: list[RetrievalResult]) -> list[SourceItem]:
        sources: list[SourceItem] = []
        seen: set[tuple[str, str | None]] = set()
        for result in results:
            key = (result.metadata.get("title", "Unknown Document"), result.metadata.get("url"))
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                SourceItem(
                    type=result.metadata.get("doc_type", "document"),
                    name=result.metadata.get("title", "Unknown Document"),
                    value=result.metadata.get("url"),
                )
            )
        return sources

    def _has_sufficient_report_coverage(self, results: list[RetrievalResult]) -> bool:
        if not results:
            return False
        unique_docs = {
            (result.metadata.get("title"), result.metadata.get("url"))
            for result in results
        }
        if len(unique_docs) >= 2 and len(results) >= 2:
            return True

        top_result = results[0]
        top_doc_type = str(top_result.metadata.get("doc_type") or "")
        return top_doc_type in self._report_doc_types and top_result.score >= 0.55

    def _has_sufficient_knowledge_coverage(self, message: str, results: list[RetrievalResult]) -> bool:
        if not results:
            return False
        finance_terms = [
            "市盈率",
            "本益比",
            "收入",
            "营业收入",
            "净利润",
            "每股收益",
            "beta",
            "pe ratio",
            "price to earnings",
            "price-to-earnings",
            "quarterly report",
            "annual report",
            "财报",
            "季报",
            "年报",
        ]
        key_terms = [term for term in finance_terms if term.lower() in message.lower()]
        if not key_terms:
            key_terms = [term for term in re.findall(r"[A-Za-z]{3,}", message) if len(term) >= 3]
        if not key_terms:
            return bool(results)
        for result in results[:3]:
            haystack = f"{result.metadata.get('title', '')} {result.content}".lower()
            if any(term.lower() in haystack for term in key_terms):
                return True
        return False

    def _merge_results(
        self,
        local_results: list[RetrievalResult],
        web_results: list[RetrievalResult],
        limit: int,
    ) -> list[RetrievalResult]:
        combined = sorted(local_results + web_results, key=lambda item: item.score, reverse=True)
        merged: list[RetrievalResult] = []
        seen: set[tuple[str, str | None]] = set()
        for item in combined:
            key = (item.metadata.get("title", ""), item.metadata.get("url"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= limit:
                break
        return merged

    def _rerank_finance_results(self, message: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        unique: list[RetrievalResult] = []
        seen: set[tuple[str, str | None]] = set()
        lowered_message = message.lower()
        for item in results:
            key = (item.metadata.get("title", ""), item.metadata.get("url"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)

        def boosted_score(item: RetrievalResult) -> float:
            score = item.score
            title = str(item.metadata.get("title", "")).lower()
            content = item.content.lower()
            doc_type = str(item.metadata.get("doc_type", ""))
            source_name = str(item.metadata.get("source_name", ""))
            if doc_type == "glossary":
                score += 0.35
            if source_name == "Project Curated Knowledge":
                score += 0.25
            if "pe ratio" in lowered_message or "price to earnings" in lowered_message or "price-to-earnings" in lowered_message:
                if "市盈率" in title or "pe ratio" in content or "price-to-earnings" in content:
                    score += 0.45
            if "收入" in message and "净利润" in message:
                if "收入" in content and "净利润" in content:
                    score += 0.45
            return score

        return sorted(unique, key=boosted_score, reverse=True)[:6]

    def _rerank_report_results(self, message: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not results:
            return []

        unique: list[RetrievalResult] = []
        seen: set[tuple[str, str | None]] = set()
        lowered_message = message.lower()
        for item in results:
            key = (item.metadata.get("title", ""), item.metadata.get("url"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)

        def boosted_score(item: RetrievalResult) -> float:
            score = item.score
            title = str(item.metadata.get("title", "")).lower()
            url = str(item.metadata.get("url", "")).lower()
            content = item.content.lower()
            doc_type = str(item.metadata.get("doc_type", ""))

            if doc_type in {"earnings_release", "quarterly_results", "quarterly_financial_statements"}:
                score += 0.55
            elif doc_type in {"annual_report", "interim_report", "quarterly_presentation"}:
                score += 0.35
            elif doc_type == "web_search":
                score -= 0.2

            if any(token in title for token in ["financial results", "earnings", "quarter", "annual report", "quarterly"]):
                score += 0.25
            if any(token in url for token in ["/press-release/", "financial-results", "earnings"]):
                score += 0.18
            if "investor relations" in title and not any(token in title for token in ["financial results", "earnings", "quarter", "annual"]):
                score -= 0.3
            if "press releases" in title and "financial results" not in title:
                score -= 0.22
            if any(token in lowered_message for token in ["q1", "q2", "q3", "q4", "quarter", "季度", "季报"]):
                if any(token in f"{title} {content}" for token in ["q1", "q2", "q3", "q4", "quarter", "季度"]):
                    score += 0.22

            return score

        return sorted(unique, key=boosted_score, reverse=True)[:6]

    def _contains_chinese(self, value: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", value))

    def _uses_web_results(self, results: list[RetrievalResult]) -> bool:
        return any(result.chunk_id.startswith("web::") for result in results)

    def _rewrite_query(self, message: str, route: RouteDecision) -> str:
        if self.query_rewrite_service is None:
            return message
        return self.query_rewrite_service.rewrite(message, route)

    def _build_report_candidate_queries(self, message: str, route: RouteDecision) -> list[str]:
        candidates = [message]
        company = route.extracted_company or ""
        symbol = route.extracted_symbol or ""
        base_terms = "earnings release financial results quarterly results annual report interim report"
        if company or symbol:
            candidates.append(" ".join(part for part in [company, symbol, message] if part).strip())
            candidates.append(" ".join(part for part in [company, symbol, base_terms] if part).strip())

        lowered = message.lower()
        if "latest earnings" in lowered or "earnings summary" in lowered:
            candidates.append(" ".join(part for part in [company, symbol, "latest earnings financial results"] if part).strip())
        if any(keyword in message for keyword in ["季度", "季报"]) or "quarter" in lowered:
            candidates.append(" ".join(part for part in [company, symbol, "quarterly results quarterly report"] if part).strip())
        if any(keyword in message for keyword in ["年报", "年度"]) or "annual" in lowered:
            candidates.append(" ".join(part for part in [company, symbol, "annual report annual results"] if part).strip())

        return list(dict.fromkeys(candidate for candidate in candidates if candidate))
