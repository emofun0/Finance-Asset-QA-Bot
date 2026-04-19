from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.rag.embeddings import build_embedding_debug_info
from app.rag.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    content: str
    metadata: dict[str, Any]


class KnowledgeBaseBuilder:
    _report_doc_types = {
        "annual_report",
        "interim_report",
        "quarterly_financial_statements",
        "quarterly_results",
        "earnings_release",
        "quarterly_presentation",
    }

    def __init__(self, knowledge_base_dir: Path) -> None:
        self.knowledge_base_dir = knowledge_base_dir
        self.processed_dir = knowledge_base_dir / "processed"
        self.index_dir = knowledge_base_dir / "index"
        self.vector_store = ChromaVectorStore(settings.rag_vector_db_dir, settings.rag_collection_name)

    def build(self) -> dict[str, int]:
        documents = self._load_processed_documents()
        chunks = self._build_chunks(documents)
        self._reset_legacy_index()
        self._write_chunks(chunks)
        self.vector_store.reset()
        self.vector_store.upsert(
            [{"chunk_id": chunk.chunk_id, "content": chunk.content, "metadata": chunk.metadata} for chunk in chunks]
        )
        self._write_manifest(documents, chunks)
        return {
            "documents": len(documents),
            "chunks": len(chunks),
        }

    def _load_processed_documents(self) -> list[dict[str, Any]]:
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.processed_dir.glob("*.json"))
        ]

    def _build_chunks(self, documents: list[dict[str, Any]]) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for document in documents:
            if document.get("doc_type") in self._report_doc_types:
                chunks.extend(self._build_report_chunks(document))
                continue
            if document.get("doc_type") == "glossary":
                chunks.extend(self._build_glossary_chunks(document))
                continue
            chunks.extend(self._build_text_chunks(document))
        return chunks

    def _build_report_chunks(self, document: dict[str, Any]) -> list[KnowledgeChunk]:
        text = str(document.get("text") or "")
        metadata_base = self._base_metadata(document, chunk_kind="report_profile")
        chunks: list[KnowledgeChunk] = [
            KnowledgeChunk(
                chunk_id=f"{document['doc_id']}::profile",
                content=self._build_report_profile_text(document),
                metadata=metadata_base,
            )
        ]

        metric_chunks = self._extract_report_metric_chunks(document)
        table_chunks = self._extract_report_table_chunks(document)
        chunks.extend(metric_chunks[:24])
        chunks.extend(table_chunks[:12])
        if len(chunks) == 1 and text.strip():
            chunks.extend(self._build_text_chunks(document, chunk_size=360, overlap=40, chunk_kind="report_text")[:12])
        return chunks

    def _build_glossary_chunks(self, document: dict[str, Any]) -> list[KnowledgeChunk]:
        entries = self._extract_glossary_entries(str(document.get("text") or ""))
        chunks: list[KnowledgeChunk] = []
        for index, entry in enumerate(entries[:160]):
            term = entry["term"]
            definition = entry["definition"]
            metadata = self._base_metadata(document, chunk_kind="glossary_term")
            metadata["term"] = term
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{document['doc_id']}::term-{index}",
                    content=f"术语：{term}\n定义：{definition}",
                    metadata=metadata,
                )
            )
        if chunks:
            return chunks
        return self._build_text_chunks(document, chunk_size=240, overlap=30, chunk_kind="glossary_text")[:40]

    def _build_text_chunks(
        self,
        document: dict[str, Any],
        *,
        chunk_size: int = 320,
        overlap: int = 48,
        chunk_kind: str = "text_chunk",
    ) -> list[KnowledgeChunk]:
        parts = self._split_text(str(document.get("text") or ""), chunk_size=chunk_size, overlap=overlap)
        chunks: list[KnowledgeChunk] = []
        for index, part in enumerate(parts):
            if len(part) < 50:
                continue
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{document['doc_id']}::chunk-{index}",
                    content=part,
                    metadata=self._base_metadata(document, chunk_kind=chunk_kind),
                )
            )
        return chunks

    def _extract_report_metric_chunks(self, document: dict[str, Any]) -> list[KnowledgeChunk]:
        text = str(document.get("text") or "")
        lines = self._clean_lines(text)
        metric_patterns = {
            "revenue": [r"营业收入", r"營業收入", r"总收入", r"總收入", r"\brevenue\b", r"\bnet sales\b"],
            "net_income": [r"净利润", r"淨利潤", r"应占盈利", r"\bnet income\b", r"\bprofit attributable\b"],
            "operating_income": [r"经营利润", r"經營利潤", r"经营盈利", r"\boperating income\b", r"\boperating profit\b", r"\bebita\b"],
            "cash_flow": [r"现金流", r"現金流", r"自由现金流", r"\bcash flow\b"],
            "eps": [r"每股收益", r"每股盈利", r"\beps\b", r"\bearnings per share\b"],
            "margin": [r"毛利率", r"gross margin", r"利润率", r"operating margin", r"net margin"],
            "segment": [
                r"金融科技及企业服务",
                r"客户管理收入",
                r"云智能集团收入",
                r"游戏收入",
                r"广告收入",
                r"segment revenue",
                r"cloud revenue",
            ],
        }

        chunks: list[KnowledgeChunk] = []
        seen: set[str] = set()
        for index, line in enumerate(lines):
            compact = " ".join(line.split())
            if len(compact) < 14 or len(compact) > 320:
                continue
            if not self._looks_numeric(compact):
                continue
            if self._looks_like_report_noise(compact):
                continue
            matched_metric = self._match_metric_name(compact, metric_patterns)
            if not matched_metric:
                continue
            normalized = compact.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            metadata = self._base_metadata(document, chunk_kind="report_metric")
            metadata["metric_name"] = matched_metric
            metadata["report_period"] = self._resolve_report_period(document)
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{document['doc_id']}::metric-{index}",
                    content=compact,
                    metadata=metadata,
                )
            )
        return chunks

    def _extract_report_table_chunks(self, document: dict[str, Any]) -> list[KnowledgeChunk]:
        lines = self._clean_lines(str(document.get("text") or ""))
        tables: list[KnowledgeChunk] = []
        buffer: list[str] = []
        start_index = 0
        for index, line in enumerate(lines):
            if self._looks_like_table_row(line):
                if not buffer:
                    start_index = index
                buffer.append(" ".join(line.split()))
                continue
            if len(buffer) >= 2:
                tables.append(self._build_table_chunk(document, start_index, buffer))
            buffer = []
        if len(buffer) >= 2:
            tables.append(self._build_table_chunk(document, start_index, buffer))
        return tables

    def _build_table_chunk(self, document: dict[str, Any], start_index: int, rows: list[str]) -> KnowledgeChunk:
        metadata = self._base_metadata(document, chunk_kind="report_table")
        metadata["row_count"] = len(rows)
        metadata["report_period"] = self._resolve_report_period(document)
        return KnowledgeChunk(
            chunk_id=f"{document['doc_id']}::table-{start_index}",
            content="\n".join(rows[:8]),
            metadata=metadata,
        )

    def _extract_glossary_entries(self, text: str) -> list[dict[str, str]]:
        lines = self._clean_lines(text)
        entries: list[dict[str, str]] = []
        seen: set[str] = set()
        for index, line in enumerate(lines):
            inline_match = re.match(r"^([\u4e00-\u9fffA-Za-z0-9()\\-／/·\s]{2,36})[：:]\s*(.+)$", line)
            if inline_match:
                term = inline_match.group(1).strip()
                definition = inline_match.group(2).strip()
            elif (
                index + 1 < len(lines)
                and 2 <= len(line) <= 24
                and not self._looks_numeric(line)
                and len(lines[index + 1]) >= 18
            ):
                term = line.strip("：: ")
                definition = lines[index + 1].strip()
            else:
                continue

            if len(definition) < 12:
                continue
            normalized_term = term.lower()
            if normalized_term in seen:
                continue
            seen.add(normalized_term)
            entries.append({"term": term, "definition": definition})
        return entries

    def _base_metadata(self, document: dict[str, Any], *, chunk_kind: str) -> dict[str, Any]:
        return {
            "doc_id": document["doc_id"],
            "title": document["title"],
            "url": document["url"],
            "language": document["language"],
            "source_name": document["source_name"],
            "doc_type": document["doc_type"],
            "published_at": document.get("published_at"),
            "company": document.get("company"),
            "symbol": document.get("symbol"),
            "chunk_kind": chunk_kind,
        }

    def _build_report_profile_text(self, document: dict[str, Any]) -> str:
        details = [
            f"公司：{document.get('company') or '未知'}",
            f"股票代码：{document.get('symbol') or '未知'}",
            f"标题：{document.get('title') or ''}",
            f"报告类型：{document.get('doc_type') or ''}",
            f"披露时间：{document.get('published_at') or self._resolve_report_period(document) or '未知'}",
        ]
        return "\n".join(details)

    def _resolve_report_period(self, document: dict[str, Any]) -> str | None:
        candidates = [
            str(document.get("title") or ""),
            str(document.get("doc_id") or ""),
            str(document.get("url") or ""),
        ]
        patterns = [
            r"(20\d{2}\s*年(?:第[一二三四1-4]季度|中期|年度|年报|半年报))",
            r"(20\d{2}\s*Q[1-4])",
            r"(20\d{2}\s*FY)",
        ]
        for candidate in candidates:
            for pattern in patterns:
                matched = re.search(pattern, candidate, flags=re.IGNORECASE)
                if matched:
                    return re.sub(r"\s+", "", matched.group(1))
        return None

    def _match_metric_name(self, text: str, metric_patterns: dict[str, list[str]]) -> str | None:
        lowered = text.lower()
        for metric_name, patterns in metric_patterns.items():
            if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns):
                return metric_name
        return None

    def _looks_like_table_row(self, value: str) -> bool:
        normalized = value.strip()
        if len(normalized) < 12:
            return False
        if self._looks_like_report_noise(normalized):
            return False
        number_count = len(re.findall(r"\d[\d,]*(?:\.\d+)?%?", normalized))
        has_finance_keyword = bool(
            re.search(
                r"(收入|营收|营业收入|總收入|总收入|净利润|淨利潤|经营利润|經營利潤|现金流|現金流|毛利率|每股收益|revenue|income|profit|margin|cash flow|eps)",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        return number_count >= 2 and has_finance_keyword

    def _looks_numeric(self, value: str) -> bool:
        return bool(re.search(r"\d[\d,]*(?:\.\d+)?", value))

    def _looks_like_report_noise(self, value: str) -> bool:
        lowered = value.lower()
        noise_tokens = [
            "company milestones",
            "business overview",
            "contents",
            "corporate information",
            "investor relations",
            "董事会",
            "公司里程碑",
            "目录",
            "关于我们",
            "joined tencent",
            "founded",
            "listed on hong kong",
        ]
        return any(token in lowered for token in noise_tokens)

    def _clean_lines(self, text: str) -> list[str]:
        return [
            line.strip()
            for line in text.splitlines()
            if len(line.strip()) >= 2
        ]

    def _split_text(self, text: str, *, chunk_size: int, overlap: int) -> list[str]:
        cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if not cleaned:
            return []
        paragraphs = cleaned.split("\n")
        parts: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                parts.append(current)
            if len(paragraph) <= chunk_size:
                current = paragraph
                continue
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                piece = paragraph[start:end].strip()
                if piece:
                    parts.append(piece)
                if end >= len(paragraph):
                    break
                start = max(end - overlap, start + 1)
            current = ""
        if current:
            parts.append(current)
        return parts

    def _write_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = self.index_dir / "chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(
                    json.dumps(
                        {
                            "chunk_id": chunk.chunk_id,
                            "content": chunk.content,
                            "metadata": chunk.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _write_manifest(self, documents: list[dict[str, Any]], chunks: list[KnowledgeChunk]) -> None:
        chunk_kind_counts: dict[str, int] = {}
        for chunk in chunks:
            chunk_kind = str(chunk.metadata.get("chunk_kind") or "unknown")
            chunk_kind_counts[chunk_kind] = chunk_kind_counts.get(chunk_kind, 0) + 1

        manifest = {
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "chunk_kind_counts": chunk_kind_counts,
            "embedding": build_embedding_debug_info(),
            "vector_db_dir": settings.rag_vector_db_dir,
            "collection_name": settings.rag_collection_name,
            "documents": [
                {
                    "doc_id": document["doc_id"],
                    "title": document["title"],
                    "doc_type": document["doc_type"],
                    "language": document["language"],
                    "source_name": document["source_name"],
                    "company": document.get("company"),
                    "symbol": document.get("symbol"),
                    "url": document["url"],
                }
                for document in documents
            ],
        }
        (self.index_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _reset_legacy_index(self) -> None:
        legacy_files = ["vectorizer.joblib", "matrix.npz"]
        for name in legacy_files:
            path = self.index_dir / name
            if path.exists():
                path.unlink()
        chroma_dir = Path(settings.rag_vector_db_dir)
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)
