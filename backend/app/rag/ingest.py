from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.rag.chunker import chunk_text
from app.rag.vector_store import LocalVectorStore


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    content: str
    metadata: dict[str, Any]


class KnowledgeBaseBuilder:
    def __init__(self, knowledge_base_dir: Path) -> None:
        self.knowledge_base_dir = knowledge_base_dir
        self.processed_dir = knowledge_base_dir / "processed"
        self.index_dir = knowledge_base_dir / "index"

    def build(self) -> dict[str, int]:
        documents = self._load_processed_documents()
        chunks = self._build_chunks(documents)
        self._write_chunks(chunks)
        LocalVectorStore(self.index_dir).build([self._build_indexable_text(chunk) for chunk in chunks])
        self._write_manifest(documents, chunks)
        return {
            "documents": len(documents),
            "chunks": len(chunks),
        }

    def _load_processed_documents(self) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for path in sorted(self.processed_dir.glob("*.json")):
            documents.append(json.loads(path.read_text(encoding="utf-8")))
        return documents

    def _build_chunks(self, documents: list[dict[str, Any]]) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for document in documents:
            parts = chunk_text(document["text"])
            for index, part in enumerate(parts):
                if len(part) < 120:
                    continue
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=f"{document['doc_id']}::chunk-{index}",
                        content=part,
                        metadata={
                            "doc_id": document["doc_id"],
                            "title": document["title"],
                            "url": document["url"],
                            "language": document["language"],
                            "source_name": document["source_name"],
                            "doc_type": document["doc_type"],
                            "published_at": document.get("published_at"),
                            "company": document.get("company"),
                            "symbol": document.get("symbol"),
                        },
                    )
                )
        return chunks

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
        manifest = {
            "document_count": len(documents),
            "chunk_count": len(chunks),
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

    def _build_indexable_text(self, chunk: KnowledgeChunk) -> str:
        metadata = chunk.metadata
        header_parts = [
            str(metadata.get("title") or ""),
            str(metadata.get("doc_type") or ""),
            str(metadata.get("source_name") or ""),
            str(metadata.get("company") or ""),
            str(metadata.get("symbol") or ""),
        ]
        header = " ".join(part for part in header_parts if part)
        if not header:
            return chunk.content
        return f"{header}\n{chunk.content}"
