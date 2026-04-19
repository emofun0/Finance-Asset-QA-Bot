from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from app.core.config import settings
from app.rag.vector_store import LocalVectorStore


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    score: float
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ProcessedDocument:
    doc_id: str
    title: str
    url: str | None
    text: str
    metadata: dict[str, Any]


class KnowledgeRetriever:
    _report_doc_types = {
        "annual_report",
        "interim_report",
        "quarterly_financial_statements",
        "quarterly_results",
        "earnings_release",
        "quarterly_presentation",
    }

    def __init__(self, knowledge_base_dir: str | Path) -> None:
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.index_dir = self.knowledge_base_dir / "index"
        self.processed_dir = self.knowledge_base_dir / "processed"
        self._chunks = self._load_chunks()
        self._documents = self._load_documents()
        self._store = LocalVectorStore(self.index_dir)
        self._index = None
        if self._store.vectorizer_path.exists() and self._store.matrix_path.exists():
            self._index = self._store.load()

    def search(
        self,
        query: str,
        top_k: int = 5,
        company: str | None = None,
        symbol: str | None = None,
        doc_type: str | None = None,
        doc_types: list[str] | None = None,
        language: str | None = None,
    ) -> list[RetrievalResult]:
        if not self._chunks or self._index is None:
            return []

        query_vector = self._index.vectorizer.transform([query])
        similarities = (self._index.matrix @ query_vector.T).toarray().ravel()
        query_terms = self._extract_query_terms(query)
        is_chinese_query = self._contains_chinese(query)
        doc_type_priority = {item: len(doc_types) - index for index, item in enumerate(doc_types or [])}

        candidates: list[RetrievalResult] = []
        for item, score in zip(self._chunks, similarities):
            if score <= 0:
                continue
            metadata = item["metadata"]
            if company and metadata.get("company") != company:
                continue
            if symbol and metadata.get("symbol") != symbol:
                continue
            if doc_type and metadata.get("doc_type") != doc_type:
                continue
            if doc_types and metadata.get("doc_type") not in doc_types:
                continue
            if language and metadata.get("language") != language:
                continue
            boosted_score = float(score)
            if company and metadata.get("company") == company:
                boosted_score += 0.15
            if symbol and metadata.get("symbol") == symbol:
                boosted_score += 0.2
            if is_chinese_query and metadata.get("language") == "zh":
                boosted_score += 0.08
            boosted_score += doc_type_priority.get(metadata.get("doc_type"), 0) * 0.05
            boosted_score += self._keyword_boost(
                content=item["content"],
                title=metadata.get("title", ""),
                query_terms=query_terms,
            )
            candidates.append(
                RetrievalResult(
                    chunk_id=item["chunk_id"],
                    score=boosted_score,
                    content=item["content"],
                    metadata=metadata,
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:top_k]

    def search_report_documents(
        self,
        query: str,
        *,
        top_k: int = 6,
        company: str | None = None,
        symbol: str | None = None,
        language: str | None = None,
    ) -> list[RetrievalResult]:
        if not self._documents:
            return []

        query_terms = self._extract_query_terms(query)
        candidates: list[RetrievalResult] = []
        for document in self._documents:
            metadata = document.metadata
            if metadata.get("doc_type") not in self._report_doc_types:
                continue
            if company and metadata.get("company") != company:
                continue
            if symbol and metadata.get("symbol") != symbol:
                continue
            if language and metadata.get("language") != language:
                continue

            doc_score = self._score_report_document(document=document, query_terms=query_terms)
            snippets = self._extract_report_snippets(document)
            for index, snippet in enumerate(snippets):
                snippet_score = doc_score + self._score_report_snippet(snippet, query_terms=query_terms, index=index)
                candidates.append(
                    RetrievalResult(
                        chunk_id=f"report::{document.doc_id}::snippet-{index}",
                        score=snippet_score,
                        content=snippet,
                        metadata=metadata,
                    )
                )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:top_k]

    def _load_chunks(self) -> list[dict[str, Any]]:
        chunks_path = self.index_dir / "chunks.jsonl"
        if not chunks_path.exists():
            return []
        return [
            json.loads(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _load_documents(self) -> list[ProcessedDocument]:
        if not self.processed_dir.exists():
            return []

        documents: list[ProcessedDocument] = []
        for path in sorted(self.processed_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            text = str(payload.get("text") or "").strip()
            if not text:
                continue
            metadata = {
                "doc_id": payload.get("doc_id"),
                "title": payload.get("title"),
                "url": payload.get("url"),
                "language": payload.get("language"),
                "source_name": payload.get("source_name"),
                "doc_type": payload.get("doc_type"),
                "published_at": self._resolve_document_date(payload),
                "company": payload.get("company"),
                "symbol": payload.get("symbol"),
            }
            documents.append(
                ProcessedDocument(
                    doc_id=str(payload.get("doc_id") or path.stem),
                    title=str(payload.get("title") or path.stem),
                    url=payload.get("url"),
                    text=text,
                    metadata=metadata,
                )
            )
        return documents

    def _extract_query_terms(self, query: str) -> list[str]:
        known_terms = [
            "市盈率",
            "市净率",
            "净资产收益率",
            "每股收益",
            "净利润",
            "归母净利润",
            "营业收入",
            "收入",
            "营收",
            "市值",
            "成交量",
            "换手率",
            "毛利",
            "现金流",
            "公募基金",
            "股票基金",
            "债券基金",
            "混合型基金",
            "货币型基金",
            "ETF",
            "QDII",
            "FOF",
            "债券",
            "债券收益率",
            "期货",
            "期权",
            "权利金",
            "行权价格",
            "保证金",
            "强行平仓",
            "做市商",
            "蓝筹股",
            "白马股",
            "多头",
            "空头",
            "年报",
            "半年报",
            "中报",
            "季报",
            "季度",
            "财报",
        ]
        matched = [term for term in known_terms if term in query]
        english_terms = re.findall(r"[A-Za-z]{3,}", query.lower())
        return matched + english_terms

    def _keyword_boost(self, content: str, title: str, query_terms: list[str]) -> float:
        boost = 0.0
        lowered_content = content.lower()
        lowered_title = title.lower()
        for term in query_terms:
            lowered_term = term.lower()
            if lowered_term in lowered_title:
                boost += 0.12
            elif lowered_term in lowered_content:
                boost += 0.04
        return min(boost, 0.28)

    def _score_report_document(self, *, document: ProcessedDocument, query_terms: list[str]) -> float:
        metadata = document.metadata
        title = document.title.lower()
        text = document.text
        compact = self._compact_text(text).lower()
        doc_type = str(metadata.get("doc_type") or "")
        score = {
            "earnings_release": 1.35,
            "quarterly_results": 1.28,
            "quarterly_financial_statements": 1.2,
            "interim_report": 1.08,
            "annual_report": 1.0,
            "quarterly_presentation": 0.88,
        }.get(doc_type, 0.6)

        published_at = str(metadata.get("published_at") or "")
        if published_at:
            score += min(max(int(published_at[:4]) - 2023, 0) * 0.08, 0.4)
            if published_at >= "2025-07-01":
                score += 0.18

        for term in query_terms:
            lowered_term = term.lower()
            if lowered_term in title:
                score += 0.2
            elif lowered_term in compact:
                score += 0.06

        if any(token in title for token in ["earnings", "financial results", "业绩", "摘要"]):
            score += 0.16
        if any(token in compact for token in ["收入", "营收", "净利润", "revenue", "net income", "operating income"]):
            score += 0.12

        score -= self._document_noise_penalty(text)
        return score

    def _score_report_snippet(self, snippet: str, *, query_terms: list[str], index: int) -> float:
        compact = self._compact_text(snippet).lower()
        score = max(0.35 - index * 0.05, 0.08)

        priority_terms = [
            "收入",
            "营收",
            "revenue",
            "net sales",
            "净利润",
            "淨利潤",
            "net income",
            "operating income",
            "经营利润",
            "經營利潤",
            "ebita",
            "cash flow",
            "现金流",
            "每股收益",
            "diluted",
        ]
        for position, term in enumerate(priority_terms):
            if term.lower() in compact:
                score += max(0.32 - position * 0.02, 0.08)
        if re.search(r"\d", snippet):
            score += 0.12
        if "%" in snippet or "同比" in snippet or "year over year" in compact:
            score += 0.08
        for term in query_terms:
            if term.lower() in compact:
                score += 0.04
        if len(compact) > 260:
            score -= 0.08
        return score

    def _extract_report_snippets(self, document: ProcessedDocument) -> list[str]:
        metric_patterns = [
            ("revenue", [r"总收入", r"總收入", r"营业收入", r"營業收入", r"收入为", r"收入為", r"total net sales", r"net sales", r"total revenues", r"revenue"]),
            ("operating_profit", [r"经营利润", r"經營利潤", r"经营盈利", r"經營盈利", r"operating income", r"operating profit", r"ebita"]),
            ("net_income", [r"归属于.*净利润", r"本公司权益持有人应占盈利", r"净利润为", r"淨利潤為", r"净利润", r"淨利潤", r"net income", r"profit attributable"]),
            ("cash_flow", [r"经营活动产生的现金流量净额", r"自由现金流", r"free cash flow", r"operating cash flow"]),
            ("eps", [r"每股收益", r"每股基本盈利", r"diluted", r"earnings per share", r"eps"]),
            ("segment", [r"云智能集团收入", r"客户管理收入", r"金融科技及企业服务", r"data center revenue", r"microsoft cloud revenue", r"aws revenue", r"google cloud", r"services revenue", r"iphone"]),
        ]

        snippets: list[str] = []
        seen: set[str] = set()
        lines = self._clean_lines(document.text)
        compact = self._compact_text(document.text)
        for metric_name, patterns in metric_patterns:
            best = self._find_best_metric_snippet(metric_name=metric_name, lines=lines, compact=compact, patterns=patterns)
            if not best:
                continue
            normalized = self._trim_snippet(best)
            if normalized in seen:
                continue
            seen.add(normalized)
            snippets.append(normalized)
            if len(snippets) >= 5:
                break
        return snippets

    def _find_best_metric_snippet(
        self,
        *,
        metric_name: str,
        lines: list[str],
        compact: str,
        patterns: list[str],
    ) -> str | None:
        candidates: list[tuple[float, str]] = []
        for line in lines:
            score = self._score_metric_candidate(line=line, patterns=patterns)
            if score <= 0:
                continue
            candidates.append((score, line))

        for window in self._extract_pattern_windows(compact=compact, metric_name=metric_name, patterns=patterns):
            score = self._score_metric_candidate(line=window, patterns=patterns)
            if score <= 0:
                continue
            candidates.append((score + 0.08, window))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _extract_pattern_windows(self, *, compact: str, metric_name: str, patterns: list[str]) -> list[str]:
        boundary_patterns = {
            "revenue": [r"gross margin", r"cost of sales", r"经营利润", r"經營利潤", r"净利润", r"淨利潤", r"operating income", r"net income"],
            "operating_profit": [r"net income", r"净利润", r"淨利潤", r"cash flow", r"每股收益", r"earnings per share"],
            "net_income": [r"cash flow", r"经营活动产生的现金流量净额", r"每股收益", r"earnings per share", r"diluted"],
            "cash_flow": [r"每股收益", r"earnings per share", r"财务比率", r"capital", r"segment"],
            "eps": [r"shares used", r"net sales by reportable segment", r"财务比率", r"归属于"],
            "segment": [r"operating income", r"净利润", r"淨利潤", r"cash flow", r"经营数据"],
        }
        boundaries = boundary_patterns.get(metric_name, [])
        windows: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, compact, flags=re.IGNORECASE):
                start = match.start()
                default_end = min(start + 240, len(compact))
                end = default_end
                for boundary in boundaries:
                    boundary_match = re.search(boundary, compact[match.end(): default_end], flags=re.IGNORECASE)
                    if not boundary_match:
                        continue
                    candidate_end = match.end() + boundary_match.start()
                    if candidate_end - start >= 18:
                        end = min(end, candidate_end)
                window = compact[start:end].strip(" ：:;，,")
                score = self._score_metric_candidate(line=window, patterns=patterns)
                if score <= 0 or window in seen:
                    continue
                seen.add(window)
                windows.append(window)
        return windows

    def _score_metric_candidate(self, *, line: str, patterns: list[str]) -> float:
        compact = self._compact_text(line)
        lowered = compact.lower()
        if len(compact) < 14 or len(compact) > 340:
            return -1.0
        if not re.search(r"\d", compact):
            return -1.0
        if self._looks_like_noise(compact):
            return -1.0

        score = 0.0
        for index, pattern in enumerate(patterns):
            if re.search(pattern, compact, flags=re.IGNORECASE):
                score += max(0.5 - index * 0.05, 0.15)
        if "%" in compact or "同比" in compact or "year over year" in lowered or "yoy" in lowered:
            score += 0.14
        if any(token in lowered for token in ["人民币", "美元", "u.s. dollars", "rmb", "$"]):
            score += 0.12
        if compact.count("  ") == 0:
            score += 0.05
        if compact.startswith(("营业收入", "營業收入", "总收入", "總收入", "收入为", "收入為", "net sales", "total net sales", "revenue", "operating income", "net income", "净利润", "淨利潤")):
            score += 0.18
        if re.search(
            r"^(营业收入|營業收入|总收入|總收入|收入为|收入為|net sales|total net sales|total revenues|revenue|operating income|net income|净利润|淨利潤)[^0-9]{0,16}[0-9$]",
            compact,
            flags=re.IGNORECASE,
        ):
            score += 0.2
        if any(token in compact for token in ["关键审计事项", "最近三个会计年度", "现金分红", "乡村振兴", "插秧节", "审计应对"]):
            score -= 0.4
        return score

    def _clean_lines(self, text: str) -> list[str]:
        lines: list[str] = []
        for raw_line in text.splitlines():
            line = self._compact_text(raw_line)
            if len(line) < 8:
                continue
            if self._looks_like_noise(line):
                continue
            lines.append(line)
        return lines

    def _compact_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    def _trim_snippet(self, value: str) -> str:
        compact = self._compact_text(value)
        return compact[:280] + ("..." if len(compact) > 280 else "")

    def _looks_like_noise(self, value: str) -> bool:
        lowered = value.lower()
        noise_tokens = [
            "investor relations",
            "annual & interim reports",
            "print page",
            "email alerts",
            "home page",
            "关于我们",
            "加入我們",
            "加入我们",
            "联系我们",
            "目錄",
            "contents",
            "董事",
        ]
        if any(token in lowered for token in noise_tokens):
            return True
        if self._document_noise_penalty(value) >= 0.55:
            return True
        return False

    def _document_noise_penalty(self, text: str) -> float:
        sample = text[:5000]
        if not sample:
            return 1.0
        suspicious_chars = sum(
            1 for ch in sample
            if ch in "�ͦʮࣘဳၚᄂ⸶㨡䙆䓳㇬況⃆䙀㔺"
        )
        ratio = suspicious_chars / max(len(sample), 1)
        return min(ratio * 6, 0.8)

    def _resolve_document_date(self, payload: dict[str, Any]) -> str | None:
        explicit = str(payload.get("published_at") or "").strip()
        if explicit:
            return explicit

        candidates = [
            str(payload.get("url") or ""),
            str(payload.get("title") or ""),
            str(payload.get("doc_id") or ""),
        ]
        for candidate in candidates:
            matched = re.search(r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})", candidate)
            if matched:
                return f"{matched.group(1)}-{int(matched.group(2)):02d}-{int(matched.group(3)):02d}"
        for candidate in candidates:
            matched = re.search(r"(20\d{2})[_ -]q([1-4])", candidate, flags=re.IGNORECASE)
            if matched:
                quarter_end = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}[matched.group(2)]
                return f"{matched.group(1)}-{quarter_end}"
        for candidate in candidates:
            matched = re.search(r"(20\d{2})", candidate)
            if matched:
                return f"{matched.group(1)}-12-31"
        return None

    def _contains_chinese(self, value: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", value))


def get_default_retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever(settings.knowledge_base_dir)
