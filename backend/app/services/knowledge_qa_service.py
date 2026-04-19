import re

from app.core.company_catalog import find_company_profile
from app.observability.request_trace import trace_event
from app.rag.retriever import KnowledgeRetriever, RetrievalResult
from app.schemas.domain import RouteDecision
from app.schemas.request import ChatRequest
from app.schemas.response import AnswerPayload, SourceItem
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
    ) -> None:
        if rag_search_tool is None and retriever is None:
            raise ValueError("KnowledgeQAService requires either rag_search_tool or retriever.")
        self.retriever = retriever or (getattr(rag_search_tool, "retriever", None) if rag_search_tool is not None else None)
        self.rag_search_tool = rag_search_tool or RagSearchTool(retriever)
        self.web_search_tool = web_search_tool

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
                        "chunk_kind": result.metadata.get("chunk_kind"),
                        "url": result.metadata.get("url"),
                        "score": round(result.score, 4),
                    }
                    for result in results[:8]
                ],
            },
        )

        if normalized_route.intent.value == "report_summary":
            return self._build_report_answer(request, normalized_route, results)
        return self._build_knowledge_answer(request, normalized_route, results)

    def _retrieve(self, request: ChatRequest, route: RouteDecision) -> list[RetrievalResult]:
        query = self._get_retrieval_query(request, route)
        preferred_language = "zh" if self._contains_chinese(request.message) else None
        search_backend = str(request.metadata.get("search_backend") or "rag").strip().lower()
        trace_event(
            "rag.retrieve",
            {
                "message": request.message,
                "retrieval_query": query,
                "search_backend": search_backend,
                "intent": route.intent,
                "company": route.extracted_company,
                "symbol": route.extracted_symbol,
            },
        )

        if route.intent.value == "finance_knowledge":
            local_results = self.rag_search_tool.search(
                RagSearchRequest(
                    query=query,
                    top_k=8,
                    language=preferred_language,
                    doc_types=["glossary", "knowledge_article"],
                    chunk_kinds=["glossary_term", "glossary_text", "text_chunk"],
                )
            )
            if search_backend == "web" and self.web_search_tool:
                web_results = self.web_search_tool.search_finance_knowledge(query, top_k=3)
                return web_results or local_results
            if self._has_sufficient_knowledge_coverage(query, local_results) or not self.web_search_tool:
                return local_results
            trace_event(
                "rag.web_fallback",
                {
                    "intent": route.intent,
                    "reason": "insufficient_knowledge_coverage",
                    "query": query,
                    "local_count": len(local_results),
                },
            )
            web_results = self.web_search_tool.search_finance_knowledge(query, top_k=3)
            return web_results or local_results

        local_results = self.rag_search_tool.search_report_documents(
            RagSearchRequest(
                query=query,
                top_k=8,
                company=route.extracted_company,
                symbol=route.extracted_symbol,
                language=preferred_language,
            )
        )
        if search_backend == "web" and self.web_search_tool:
            web_results = self._search_web_report_fallback(query, route)
            return web_results or local_results
        if self._has_sufficient_report_coverage(local_results) or not self.web_search_tool:
            return local_results
        trace_event(
            "rag.web_fallback",
            {
                "intent": route.intent,
                "reason": "insufficient_report_coverage",
                "query": query,
                "company": route.extracted_company,
                "symbol": route.extracted_symbol,
                "local_count": len(local_results),
            },
        )
        web_results = self._search_web_report_fallback(query, route)
        return web_results or local_results

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
                summary="知识库中没有检索到足够依据，当前无法可靠回答该问题。",
                objective_data={
                    "retrieval_enabled": True,
                    "source_mode": "not_found",
                    "matched_chunks": 0,
                },
                analysis=["当前术语库和普通知识文档都未命中高相关定义。"],
                sources=[],
                limitations=["当前回答严格依赖结构化术语库和知识文档，不做无依据补充。"],
                route=route,
            )

        summary = self._build_knowledge_summary(results)
        analysis = [self._trim(result.content, 200) for result in results[:3]]
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
                "matched_terms": [result.metadata.get("term") for result in results[:3] if result.metadata.get("term")],
            },
            analysis=analysis,
            sources=self._to_sources(results[:4]),
            limitations=[
                "术语解释优先来自结构化词条；若命中的是普通知识段落，表达会更接近原文摘录。",
            ],
            route=route,
        )

    def _build_report_answer(
        self,
        request: ChatRequest,
        route: RouteDecision,
        results: list[RetrievalResult],
    ) -> AnswerPayload:
        company = route.extracted_company or route.extracted_symbol or "目标公司"
        if not results:
            return AnswerPayload(
                question_type=route.intent,
                request_message=request.message,
                summary=f"知识库中暂未检索到 {company} 对应财报材料，当前无法可靠总结。",
                objective_data={
                    "retrieval_enabled": True,
                    "source_mode": "not_found",
                    "matched_chunks": 0,
                    "company": route.extracted_company,
                    "symbol": route.extracted_symbol,
                },
                analysis=["当前未命中该公司的结构化财报指标、表格或报告概要。"],
                sources=[],
                limitations=["财报回答只基于结构化财报片段和官方网页 fallback。"],
                route=route,
            )

        metric_lines = self._collect_report_metrics(results)
        table_lines = self._collect_report_tables(results)
        report_context = self._build_report_context(results)
        source_mode = "web_fallback" if self._uses_web_results(results) else "local_rag"
        latest_title = str(results[0].metadata.get("title") or "")
        summary = self._build_report_summary_text(company, latest_title, report_context)
        analysis = metric_lines[:3]
        if table_lines:
            analysis.append("表格摘录：\n" + "\n".join(table_lines[:4]))
        if not analysis and report_context.get("metric_lines"):
            analysis = [self._trim(item, 220) for item in report_context["metric_lines"][:3]]

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
                "table_hits": len(table_lines),
                "metric_hits": len(metric_lines),
                "primary_doc_id": report_context.get("doc_id"),
                "primary_doc_title": report_context.get("title"),
                "report_period": report_context.get("report_period"),
                "report_context": report_context,
            },
            analysis=analysis[:4],
            sources=self._to_sources(results[:4]),
            limitations=[
                "当前回答会优先基于命中的整份财报数字片段和表格摘录归纳，但仍不保证覆盖全部章节。",
            ],
            route=route,
        )

    def _build_knowledge_summary(self, results: list[RetrievalResult]) -> str:
        top = results[0]
        if top.metadata.get("term"):
            term = str(top.metadata.get("term"))
            definition = top.content.split("定义：", maxsplit=1)[-1].strip()
            return f"{term}：{self._trim(definition, 220)}"
        return self._trim(top.content, 220)

    def _build_report_summary_text(self, company: str, latest_title: str, report_context: dict) -> str:
        period = str(report_context.get("report_period") or self._extract_report_period(latest_title) or "").strip()
        title = str(report_context.get("title") or latest_title).strip()
        period_text = f"{period}" if period else "最近披露的"
        if report_context.get("metric_lines") or report_context.get("table_rows"):
            return f"已命中 {company} {period_text}财报，并提取出该报告中的主要财务数字与表格片段，以下内容应以这份报告原文为准。"
        return f"已命中 {company} 的财报材料《{self._trim(title, 120)}》，但当前仅确认到报告命中，未稳定抽出足够数字。"

    def _build_report_context(self, results: list[RetrievalResult]) -> dict[str, object]:
        if not results:
            return {}
        primary_doc_id = str(results[0].metadata.get("doc_id") or "")
        if not primary_doc_id or self.retriever is None:
            return {}

        document_chunks = self.retriever.get_document_chunks(
            primary_doc_id,
            chunk_kinds=["report_profile", "report_metric", "report_table"],
        )
        profile = next((item for item in document_chunks if item.metadata.get("chunk_kind") == "report_profile"), None)
        metric_lines = self._dedupe_texts(
            [
                self._trim(" ".join(item.content.split()), 260)
                for item in document_chunks
                if item.metadata.get("chunk_kind") == "report_metric"
                and self._is_high_signal_report_metric(item.content)
            ],
            limit=20,
        )
        table_rows = self._dedupe_texts(
            [
                self._trim(" ".join(row.split()), 180)
                for item in document_chunks
                if item.metadata.get("chunk_kind") == "report_table"
                for row in item.content.splitlines()
            ],
            limit=16,
            min_length=8,
        )
        return {
            "doc_id": primary_doc_id,
            "title": results[0].metadata.get("title"),
            "report_period": self._extract_report_period(str(results[0].metadata.get("title") or "")),
            "profile": profile.content if profile else None,
            "metric_lines": metric_lines,
            "table_rows": table_rows,
        }

    def _collect_report_metrics(self, results: list[RetrievalResult]) -> list[str]:
        metrics: list[str] = []
        seen: set[str] = set()
        for result in results:
            if result.metadata.get("chunk_kind") != "report_metric":
                continue
            normalized = self._trim(" ".join(result.content.split()), 220)
            if normalized in seen:
                continue
            seen.add(normalized)
            metrics.append(normalized)
            if len(metrics) >= 6:
                break
        return metrics

    def _collect_report_tables(self, results: list[RetrievalResult]) -> list[str]:
        rows: list[str] = []
        seen: set[str] = set()
        for result in results:
            if result.metadata.get("chunk_kind") != "report_table":
                continue
            for row in result.content.splitlines():
                normalized = self._trim(" ".join(row.split()), 180)
                if len(normalized) < 12 or normalized in seen:
                    continue
                seen.add(normalized)
                rows.append(normalized)
                if len(rows) >= 8:
                    return rows
        return rows

    def _normalize_route_subject(self, route: RouteDecision) -> RouteDecision:
        profile = find_company_profile(route.extracted_company, route.extracted_symbol)
        if profile is None:
            return route
        normalized = route.model_copy(deep=True)
        normalized.extracted_company = profile.canonical_name
        normalized.extracted_symbol = profile.symbol
        return normalized

    def _search_web_report_fallback(self, query: str, route: RouteDecision) -> list[RetrievalResult]:
        if not self.web_search_tool:
            return []
        profile = find_company_profile(route.extracted_company, route.extracted_symbol)
        if profile:
            return self.web_search_tool.search_company_reports(query, profile, top_k=3)
        return self.web_search_tool.search_company_reports_by_query(
            query,
            company=route.extracted_company,
            symbol=route.extracted_symbol,
            top_k=3,
        )

    def _get_retrieval_query(self, request: ChatRequest, route: RouteDecision) -> str:
        retrieval_query = str(request.metadata.get("retrieval_query") or "").strip()
        if retrieval_query:
            trace_event(
                "query_rewrite.agent",
                {
                    "message": request.message,
                    "rewritten_query": retrieval_query,
                    "intent": route.intent,
                },
            )
            return retrieval_query

        extras = [route.extracted_company, route.extracted_symbol, request.message]
        return " ".join(part for part in extras if part).strip()

    def _extract_report_period(self, title: str) -> str | None:
        patterns = [
            r"(20\d{2}年(?:度)?报告)",
            r"(20\d{2}年中期报告)",
            r"(20\d{2}\s*Q[1-4])",
            r"(20\d{2}年(?:第[一二三四1-4]季度))",
        ]
        for pattern in patterns:
            matched = re.search(pattern, title, flags=re.IGNORECASE)
            if matched:
                return re.sub(r"\s+", "", matched.group(1))
        return None

    def _contains_chinese(self, value: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", value))

    def _trim(self, text: str, limit: int) -> str:
        compact = " ".join(text.split())
        return compact[:limit] + ("..." if len(compact) > limit else "")

    def _dedupe_texts(self, values: list[str], *, limit: int, min_length: int = 12) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            compact = value.strip()
            if len(compact) < min_length or compact in seen:
                continue
            seen.add(compact)
            deduped.append(compact)
            if len(deduped) >= limit:
                break
        return deduped

    def _is_high_signal_report_metric(self, text: str) -> bool:
        compact = " ".join(text.split())
        if not re.search(r"\d", compact):
            return False
        finance_keyword = re.search(
            r"(收入|营收|营业收入|總收入|总收入|净利润|淨利潤|应占盈利|经营盈利|经营利润|經營利潤|现金流|自由现金流|毛利|毛利率|经营利润率|revenue|income|profit|margin|cash flow|free cash flow)",
            compact,
            flags=re.IGNORECASE,
        )
        if not finance_keyword:
            return False
        has_finance_number = bool(
            re.search(r"(人民币|亿元|百万元|%|同比|\$|RMB|HK\\$|US\\$)", compact, flags=re.IGNORECASE)
        )
        return has_finance_number

    def _has_sufficient_knowledge_coverage(self, query: str, results: list[RetrievalResult]) -> bool:
        if not results:
            return False
        query_terms = self._extract_knowledge_query_terms(query)
        top = results[0]
        if top.metadata.get("chunk_kind") == "glossary_term":
            return "定义：" in top.content or len(top.content) >= 20

        for item in results[:3]:
            haystack = f"{item.metadata.get('title', '')}\n{item.content}".lower()
            if query_terms and any(term in haystack for term in query_terms):
                return item.score >= 0.5
        return False

    def _has_sufficient_report_coverage(self, results: list[RetrievalResult]) -> bool:
        if not results:
            return False
        metric_hits = self._collect_report_metrics(results)
        if metric_hits:
            return True
        table_hits = self._collect_report_tables(results)
        if len(table_hits) >= 2:
            return True
        top = results[0]
        return top.metadata.get("chunk_kind") in {"report_metric", "report_table"} and top.score >= 0.58

    def _extract_knowledge_query_terms(self, query: str) -> list[str]:
        english_terms = [item.lower() for item in re.findall(r"[A-Za-z]{2,}", query)]
        chinese_terms = [
            item.lower()
            for item in re.findall(r"[\u4e00-\u9fff]{2,}", query)
            if item not in {"什么是", "是什么意思", "定义", "解释", "一下"}
        ]
        normalized = english_terms + chinese_terms
        stopwords = {
            "what",
            "is",
            "the",
            "of",
            "and",
            "meaning",
            "definition",
            "ratio",
            "before",
            "interest",
            "taxes",
            "tax",
            "amortization",
            "earnings",
        }
        deduped: list[str] = []
        seen: set[str] = set()
        for item in normalized:
            token = item.strip().lower()
            if len(token) < 2 or token in stopwords or token in seen:
                continue
            seen.add(token)
            deduped.append(token)
        return deduped

    def _to_sources(self, results: list[RetrievalResult]) -> list[SourceItem]:
        sources: list[SourceItem] = []
        seen: set[tuple[str, str | None]] = set()
        for result in results:
            key = (str(result.metadata.get("title") or "Unknown Document"), result.metadata.get("url"))
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                SourceItem(
                    type=str(result.metadata.get("doc_type") or "document"),
                    name=key[0],
                    value=key[1],
                )
            )
        return sources

    def _uses_web_results(self, results: list[RetrievalResult]) -> bool:
        return any(result.chunk_id.startswith("web::") for result in results)
