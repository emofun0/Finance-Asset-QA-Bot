import re

from app.core.company_catalog import find_company_profile
from app.observability.request_trace import trace_event
from app.schemas.request import ChatRequest
from app.schemas.domain import RouteDecision
from app.schemas.response import AnswerPayload, SourceItem
from app.rag.retriever import KnowledgeRetriever, RetrievalResult
from app.tools.rag_search_tool import RagSearchRequest, RagSearchTool
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
        rag_search_tool: RagSearchTool | None = None,
        retriever: KnowledgeRetriever | None = None,
        web_search_tool: OfficialWebSearchTool | None = None,
        query_rewrite_service=None,
    ) -> None:
        if rag_search_tool is None and retriever is None:
            raise ValueError("KnowledgeQAService requires either rag_search_tool or retriever.")
        self.rag_search_tool = rag_search_tool or RagSearchTool(retriever)
        self.web_search_tool = web_search_tool
        self.query_rewrite_service = query_rewrite_service

    def answer(self, request: ChatRequest, route: RouteDecision) -> AnswerPayload:
        normalized_route = self._normalize_route_subject(route)
        results = self._retrieve(request, normalized_route)
        trace_event(
            "rag.results",
            {
                "intent": normalized_route.intent,
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

        if normalized_route.intent.value == "report_summary":
            return self._build_report_summary(request, normalized_route, results)
        return self._build_knowledge_answer(request, normalized_route, results)

    def _normalize_route_subject(self, route: RouteDecision) -> RouteDecision:
        profile = find_company_profile(route.extracted_company, route.extracted_symbol)
        if profile is None:
            return route

        normalized = route.model_copy(deep=True)
        normalized.extracted_company = profile.canonical_name
        normalized.extracted_symbol = profile.symbol
        return normalized

    def _retrieve(self, request: ChatRequest, route: RouteDecision) -> list[RetrievalResult]:
        message = request.message
        rewritten_message = self._rewrite_query(request, route)
        preferred_language = "zh" if self._contains_chinese(message) else None
        search_backend = str(request.metadata.get("search_backend") or "rag").strip().lower()
        trace_event(
            "rag.retrieve",
            {
                "message": message,
                "rewritten_message": rewritten_message,
                "search_backend": search_backend,
                "intent": route.intent,
                "preferred_language": preferred_language,
                "company": route.extracted_company,
                "symbol": route.extracted_symbol,
            },
        )

        if route.intent.value == "finance_knowledge":
            if search_backend == "web" and self.web_search_tool:
                return self.web_search_tool.search_finance_knowledge(rewritten_message, top_k=3)
            local_results = self._search_local_finance_knowledge(rewritten_message, preferred_language)
            if self._has_sufficient_knowledge_coverage(message, local_results):
                return local_results
            if self.web_search_tool:
                trace_event(
                    "rag.web_fallback",
                    {
                        "intent": route.intent,
                        "reason": "insufficient_knowledge_coverage",
                        "query": rewritten_message,
                        "local_count": len(local_results),
                    },
                )
                web_results = self.web_search_tool.search_finance_knowledge(rewritten_message, top_k=3)
                if web_results:
                    return web_results
            return local_results

        if route.intent.value == "report_summary":
            local_results = self._search_local_report_summary(rewritten_message, route, preferred_language)
            local_results = self._rerank_report_results(message, local_results)
            if search_backend == "web" and self.web_search_tool:
                profile = find_company_profile(route.extracted_company, route.extracted_symbol)
                if profile:
                    return self.web_search_tool.search_company_reports(rewritten_message, profile, top_k=3)
            if self._has_sufficient_report_coverage(local_results):
                return local_results
            if self.web_search_tool:
                profile = find_company_profile(route.extracted_company, route.extracted_symbol)
                if profile:
                    trace_event(
                        "rag.web_fallback",
                        {
                            "intent": route.intent,
                            "reason": "insufficient_report_coverage",
                            "query": rewritten_message,
                            "company": route.extracted_company,
                            "symbol": route.extracted_symbol,
                            "local_count": len(local_results),
                        },
                    )
                    web_results = self.web_search_tool.search_company_reports(rewritten_message, profile, top_k=3)
                    if web_results:
                        return web_results
            return local_results

        return self.rag_search_tool.search(
            RagSearchRequest(
                query=rewritten_message,
                top_k=6,
                company=route.extracted_company,
                symbol=route.extracted_symbol,
                language=preferred_language,
            )
        )

    def _search_local_finance_knowledge(self, message: str, preferred_language: str | None) -> list[RetrievalResult]:
        results = self.rag_search_tool.search(
            RagSearchRequest(
                query=message,
                top_k=6,
                doc_types=["knowledge_article", "glossary"],
                language=preferred_language,
            )
        )
        if not results and preferred_language:
            results = self.rag_search_tool.search(
                RagSearchRequest(
                    query=message,
                    top_k=6,
                    doc_types=["knowledge_article", "glossary"],
                )
            )
        return self._rerank_finance_results(message, results)

    def _search_local_report_summary(
        self,
        message: str,
        route: RouteDecision,
        preferred_language: str | None,
    ) -> list[RetrievalResult]:
        results = self.rag_search_tool.search_report_documents(
            RagSearchRequest(
                query=message,
                top_k=8,
                company=route.extracted_company,
                symbol=route.extracted_symbol,
                language=preferred_language,
            )
        )
        if not results and preferred_language:
            results = self.rag_search_tool.search_report_documents(
                RagSearchRequest(
                    query=message,
                    top_k=8,
                    company=route.extracted_company,
                    symbol=route.extracted_symbol,
                )
            )
        return results

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
        ordered_results = self._sort_report_results_for_summary(results)
        highlights = self._extract_report_summary_highlights(ordered_results)
        source_mode = "web_fallback" if self._uses_web_results(results) else "local_rag"
        latest_title = str(ordered_results[0].metadata.get("title") or "")
        summary = self._build_report_summary_text(company, latest_title, highlights)
        analysis = highlights or [self._trim_excerpt(ordered_results[0].content)]

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
                "top_doc_titles": [result.metadata.get("title") for result in ordered_results[:3]],
            },
            analysis=analysis[:4],
            sources=self._to_sources(ordered_results[:4]),
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
        elif "做空" in message or "short selling" in lowered:
            summary = "做空通常指先借入并卖出某项资产，待价格下跌后再买回归还，从中赚取差价的交易策略。"
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
            summary = self._build_generic_knowledge_summary(message, results)

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

    def _extract_report_summary_highlights(self, results: list[RetrievalResult]) -> list[str]:
        direct_snippets: list[str] = []
        seen_direct: set[str] = set()
        for result in results:
            if not self._is_direct_report_snippet(result):
                continue
            trimmed = self._normalize_direct_report_snippet(result.content)
            if trimmed in seen_direct:
                continue
            seen_direct.add(trimmed)
            direct_snippets.append(trimmed)
            if len(direct_snippets) >= 4:
                return direct_snippets

        priorities = [
            [r"^收入[為为]?", r"^總收入", r"^总收入", r"^營業收入", r"^营业收入", r"^total net sales", r"^net sales", r"^revenue"],
            [r"經營利潤", r"经营利润", r"經調整EBITA", r"经调整EBITA"],
            [r"淨利潤", r"净利润", r"歸屬於普通股股東的淨利潤", r"归属于普通股股东的净利润", r"^net income"],
            [r"經營活動產生的現金流量淨額", r"经营活动产生的现金流量净额", r"自由現金流", r"自由现金流"],
            [r"客戶管理收入", r"客户管理收入"],
            [r"雲智能集團收入", r"云智能集团收入", r"AI相關產品收入", r"AI相关产品收入"],
            [r"data center revenue", r"aws revenue", r"microsoft cloud revenue", r"google cloud", r"iphone"],
        ]
        highlights: list[str] = []
        seen: set[str] = set()
        for patterns in priorities:
            for result in results:
                sentence = self._find_priority_sentence(result.content, patterns)
                if not sentence:
                    continue
                trimmed = self._trim_excerpt(sentence, max_length=260)
                if trimmed in seen:
                    continue
                seen.add(trimmed)
                highlights.append(trimmed)
                break
            if len(highlights) >= 4:
                return highlights

        fallback = self._extract_financial_highlights(results)
        for item in fallback:
            if item not in seen:
                highlights.append(item)
            if len(highlights) >= 4:
                break
        return direct_snippets + highlights if direct_snippets else highlights

    def _find_priority_sentence(self, content: str, patterns: list[str]) -> str | None:
        candidates: list[tuple[int, str]] = []
        for sentence in self._split_sentences(content):
            normalized = " ".join(sentence.split())
            if len(normalized) < 16:
                continue
            lowered = normalized.lower()
            if any(token in lowered for token in ["首席執行官", "首席财务官", "首席執行官", "董事会", "關於我們", "关于我们"]):
                continue
            if not re.search(r"\d", normalized):
                continue
            for index, pattern in enumerate(patterns):
                if re.search(pattern, normalized, flags=re.IGNORECASE):
                    score = 100 - index * 10
                    if re.search(r"^截至\s*\d{4}\s*年", normalized):
                        score += 20
                    if re.search(r"同比[增长增長下降]", normalized):
                        score += 8
                    if len(normalized) <= 260:
                        score += 5
                    candidates.append((score, normalized))
                    break
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _build_report_summary_text(self, company: str, latest_title: str, highlights: list[str]) -> str:
        period = self._extract_report_period(latest_title)
        if not highlights:
            if period:
                return f"{company} 最近披露的{period}财报已命中，但当前只能确认已检索到官方材料，尚未稳定提炼出关键经营指标。"
            return f"{company} 最近披露的财报材料已命中，但当前只能确认已检索到官方材料，尚未稳定提炼出关键经营指标。"

        opening = f"{company} 最近披露的{period}财报显示：" if period else f"{company} 最近披露的财报显示："
        return f"{opening}{'；'.join(highlights[:2])}"

    def _extract_report_period(self, title: str) -> str | None:
        patterns = [
            r"(\d{4}年\d{1,2}月份季度)",
            r"(\d{4}\s*年\s*\d{1,2}\s*月份季度)",
            r"(\d{4}财年)",
            r"(\d{4}\s*財年)",
            r"(\d{4}年(?:度)?报告)",
            r"(\d{4}年中期报告)",
        ]
        for pattern in patterns:
            matched = re.search(pattern, title)
            if matched:
                return re.sub(r"\s+", "", matched.group(1))
        return None

    def _sort_report_results_for_summary(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        def sort_key(item: RetrievalResult) -> tuple[str, float]:
            published_at = str(item.metadata.get("published_at") or "")
            return (published_at, item.score)

        return sorted(results, key=sort_key, reverse=True)

    def _trim_excerpt(self, text: str, max_length: int = 220) -> str:
        cleaned = " ".join(text.split())
        return cleaned[:max_length] + ("..." if len(cleaned) > max_length else "")

    def _extract_relevant_excerpt(self, message: str, content: str) -> str:
        terms = [
            term
            for term in [
                "市盈率",
                "每股收益",
                "营业收入",
                "收入",
                "净利润",
                "盈利能力",
                "季度报告",
                "年报",
                "做空",
                "卖出",
                "借入",
                "回补",
            ]
            if term in message
        ]
        terms.extend(token for token in re.findall(r"[A-Za-z]{3,}", message) if token.lower() not in {"what", "define", "coefficient"})
        compact_content = " ".join(content.split())
        sentences = self._split_sentences(content)
        for sentence in sentences:
            if self._is_low_signal_excerpt(sentence):
                continue
            if self._is_title_only_excerpt(sentence):
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
            if candidate and not self._is_low_signal_excerpt(candidate) and not self._is_title_only_excerpt(candidate):
                return candidate
        for sentence in sentences:
            if not self._is_low_signal_excerpt(sentence) and not self._is_title_only_excerpt(sentence):
                return self._trim_excerpt(sentence)
        return self._trim_excerpt(sentences[0] if sentences else content)

    def _split_sentences(self, content: str) -> list[str]:
        normalized = (
            content.replace("U.S.", "US")
            .replace("u.s.", "us")
            .replace("No.", "No")
            .replace("no.", "no")
        )
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[。！？；;])\s*|(?<=\.)\s+(?=[A-Z])|\n+", normalized)
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

    def _is_title_only_excerpt(self, text: str) -> bool:
        lowered = " ".join(text.split()).lower()
        return any(
            token in lowered
            for token in [
                "| vantage",
                "| wikipedia",
                "| investopedia",
                "home right arrow terminology",
            ]
        )

    def _build_generic_knowledge_summary(self, message: str, results: list[RetrievalResult]) -> str:
        for result in results:
            excerpt = self._extract_relevant_excerpt(message, result.content)
            if excerpt and not self._is_low_signal_excerpt(excerpt) and not self._is_title_only_excerpt(excerpt):
                return excerpt
        return self._trim_excerpt(results[0].content)

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
        return (
            top_doc_type in self._report_doc_types
            and top_result.score >= 0.85
            and not self._is_garbled_report_result(top_result.content)
        )

    def _has_sufficient_knowledge_coverage(self, message: str, results: list[RetrievalResult]) -> bool:
        if not results:
            return False
        key_terms = self._extract_knowledge_query_terms(message)
        if not key_terms:
            top_result = results[0]
            return top_result.score >= 0.45 and top_result.metadata.get("doc_type") in {"glossary", "knowledge_article"}

        for result in results[:3]:
            haystack = f"{result.metadata.get('title', '')} {result.content}".lower()
            if any(term.lower() in haystack for term in key_terms):
                return True
        return False

    def _extract_knowledge_query_terms(self, message: str) -> list[str]:
        finance_terms = [
            "市盈率",
            "本益比",
            "收入",
            "营业收入",
            "净利润",
            "每股收益",
            "纳斯达克",
            "纳指",
            "nasdaq",
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
        english_terms = [term for term in re.findall(r"[A-Za-z]{3,}", message) if len(term) >= 3]
        chinese_terms = [
            term for term in re.findall(r"[\u4e00-\u9fff]{2,}", message)
            if term not in {"什么是", "有什么", "区别", "定义", "解释", "一下", "一下子"}
        ]
        ordered_terms = key_terms + english_terms + chinese_terms
        deduped: list[str] = []
        seen: set[str] = set()
        for term in ordered_terms:
            normalized = term.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(term.strip())
        return deduped

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
        seen: set[str] = set()
        lowered_message = message.lower()
        for item in results:
            key = item.chunk_id
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
            published_at = str(item.metadata.get("published_at") or "")
            chunk_id = str(item.chunk_id or "")

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
            if published_at:
                score += min(max(int(published_at[:4]) - 2020, 0) * 0.03, 0.3)
                if published_at >= "2025-07-01":
                    score += 0.18
            chunk_match = re.search(r"chunk-(\d+)$", chunk_id)
            if chunk_match:
                chunk_index = int(chunk_match.group(1))
                score += max(0.22 - chunk_index * 0.04, -0.12)
            if any(token in content for token in ["原文下載", "原文下载", "關於阿里巴巴集團", "关于阿里巴巴集团", "支付日期", "股息", "回購", "回购"]):
                score -= 0.16
            if any(token in content for token in ["收入為", "收入 为", "收入为", "經營利潤", "经营利润", "淨利潤", "净利润", "現金流", "现金流", "客戶管理收入", "客户管理收入", "雲智能集團收入", "云智能集团收入"]):
                score += 0.22

            return score

        return sorted(unique, key=boosted_score, reverse=True)[:6]

    def _contains_chinese(self, value: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", value))

    def _uses_web_results(self, results: list[RetrievalResult]) -> bool:
        return any(result.chunk_id.startswith("web::") for result in results)

    def _is_direct_report_snippet(self, result: RetrievalResult) -> bool:
        compact = " ".join(result.content.split())
        lowered = compact.lower()
        if len(compact) < 18 or self._is_garbled_report_result(compact):
            return False
        finance_tokens = [
            "收入",
            "营收",
            "营业收入",
            "净利润",
            "淨利潤",
            "经营利润",
            "經營利潤",
            "现金流",
            "revenue",
            "net sales",
            "net income",
            "operating income",
            "cash flow",
            "aws revenue",
            "data center revenue",
        ]
        return any(token in lowered for token in [item.lower() for item in finance_tokens]) and any(ch.isdigit() for ch in compact)

    def _is_garbled_report_result(self, text: str) -> bool:
        sample = text[:400]
        suspicious = sum(1 for ch in sample if ch in "�ͦʮࣘဳၚᄂ⸶㨡䙆䓳㇬況⃆䙀㔺")
        return suspicious / max(len(sample), 1) >= 0.04

    def _normalize_direct_report_snippet(self, text: str) -> str:
        compact = " ".join(text.split())
        normalized_parts: list[str] = []
        seen: set[str] = set()

        for metric in [
            "总收入",
            "總收入",
            "营业收入",
            "營業收入",
            "净利润",
            "淨利潤",
            "归属于上市公司股东的净利润",
            "归属于本行股东的净利润",
            "本公司权益持有人应占盈利",
            "经营活动产生的现金流量净额",
            "自由现金流",
        ]:
            phrase = self._extract_chinese_metric_phrase(compact, metric)
            if not phrase or phrase in seen:
                continue
            seen.add(phrase)
            normalized_parts.append(phrase)
            if len(normalized_parts) >= 2:
                return "；".join(normalized_parts)

        replacements = [
            (r"(Total net sales|Net sales|Total revenues|Operating income|Net income)\s+\$?\s*([\d,]+(?:\.\d+)?)", lambda m: f"{m.group(1)} {m.group(2)}"),
            (r"(?:reported|revenue of|net sales of)\s+([\d.]+\s+billion\s+U\.?S\.?\s+dollars[^.。；;]*)", lambda m: m.group(1)),
            (r"(AWS revenue reached [^.。；;]+|Data Center revenue reached [^.。；;]+|Microsoft Cloud revenue was [^.。；;]+)", lambda m: m.group(1)),
            (r"(operating income decreased [^.。；;]+|GAAP net income [^.。；;]+|non-GAAP net income[^.。；;]+)", lambda m: m.group(1)),
        ]
        for pattern, formatter in replacements:
            for match in re.finditer(pattern, compact, flags=re.IGNORECASE):
                phrase = self._trim_excerpt(formatter(match), max_length=180)
                if phrase in seen:
                    continue
                seen.add(phrase)
                normalized_parts.append(phrase)
                if len(normalized_parts) >= 2:
                    return "；".join(normalized_parts)
        return self._trim_excerpt(compact, max_length=220)

    def _extract_chinese_metric_phrase(self, compact: str, metric: str) -> str | None:
        position = compact.find(metric)
        if position < 0:
            return None
        window = compact[position: position + 140]
        values = [item for item in re.findall(r"\d[\d,]{2,}(?:\.\d+)?", window) if item not in {"2024", "2025", "2026"}]
        if not values:
            return None
        value = max(values, key=lambda item: len(item.replace(",", "")))
        change_match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", window)
        change = f"，同比 {change_match.group(1)}%" if change_match else ""
        label = metric.replace("归属于上市公司股东的", "").replace("归属于本行股东的", "").strip()
        return f"{label} {value}{change}"

    def _rewrite_query(self, request: ChatRequest, route: RouteDecision) -> str:
        retrieval_query = str(request.metadata.get("retrieval_query") or "").strip()
        if retrieval_query:
            trace_event(
                "query_rewrite.skipped",
                {
                    "message": request.message,
                    "reason": "agent_rewritten_query",
                    "intent": route.intent,
                    "rewritten_query": retrieval_query,
                },
            )
            return retrieval_query
        if self.query_rewrite_service is None:
            return request.message
        return self.query_rewrite_service.rewrite(request.message, route)

    def _build_report_candidate_queries(self, message: str, route: RouteDecision) -> list[str]:
        candidates = [message]
        company = route.extracted_company or ""
        symbol = route.extracted_symbol or ""
        base_terms = "earnings release financial results quarterly results annual report interim report"
        finance_terms = "revenue net profit operating income cash flow"
        if company or symbol:
            candidates.append(" ".join(part for part in [company, symbol, message] if part).strip())
            candidates.append(" ".join(part for part in [company, symbol, base_terms] if part).strip())
            candidates.append(" ".join(part for part in [company, symbol, base_terms, finance_terms] if part).strip())

        lowered = message.lower()
        if "latest earnings" in lowered or "earnings summary" in lowered:
            candidates.append(" ".join(part for part in [company, symbol, "latest earnings financial results"] if part).strip())
        if any(keyword in message for keyword in ["季度", "季报"]) or "quarter" in lowered:
            candidates.append(" ".join(part for part in [company, symbol, "quarterly results quarterly report"] if part).strip())
        if any(keyword in message for keyword in ["年报", "年度"]) or "annual" in lowered:
            candidates.append(" ".join(part for part in [company, symbol, "annual report annual results"] if part).strip())

        return list(dict.fromkeys(candidate for candidate in candidates if candidate))
