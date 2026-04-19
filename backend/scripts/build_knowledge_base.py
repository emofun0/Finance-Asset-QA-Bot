from __future__ import annotations

import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.rag.ingest import KnowledgeBaseBuilder


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
SEC_USER_AGENT = "Finance Asset QA System research contact finance-qa@example.com"


def normalize_whitespace(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def download_binary(url: str) -> bytes:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    if "sec.gov" in url:
        headers["User-Agent"] = SEC_USER_AGENT
        headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"

    response = requests.get(
        url,
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    return response.content


def extract_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "img", "video", "audio"]):
        tag.decompose()

    root = (
        soup.select_one("div.TRS_Editor")
        or soup.select_one("div.text_content_detail_content")
        or soup.find("article")
        or soup.find("main")
        or soup.find("body")
        or soup
    )
    lines = collect_html_lines(root, ["h1", "h2", "h3", "p", "li", "td", "th"])
    if not lines:
        lines = collect_html_lines(root, ["div", "font", "span"])
    return "\n".join(lines)


def collect_html_lines(root: BeautifulSoup, selectors: list[str]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for element in root.find_all(selectors):
        text = normalize_whitespace(element.get_text(" ", strip=True))
        if not text or len(text) < 2:
            continue
        if text in seen:
            continue
        seen.add(text)
        lines.append(text)
    return lines


def extract_pdf_text(path: Path) -> str:
    pdftotext_binary = shutil.which("pdftotext")
    if pdftotext_binary:
        try:
            completed = subprocess.run(
                [pdftotext_binary, "-layout", str(path), "-"],
                check=True,
                capture_output=True,
                text=True,
            )
            text = normalize_whitespace(completed.stdout)
            if text:
                return text
        except Exception:
            pass

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = normalize_whitespace(text)
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def process_source(source: dict[str, Any], raw_dir: Path, processed_dir: Path) -> None:
    doc_id = source["doc_id"]
    url = source["url"]
    if source.get("inline_text"):
        suffix = ".txt"
    else:
        suffix = ".pdf" if url.lower().endswith(".pdf") else ".html"
    raw_path = raw_dir / f"{doc_id}{suffix}"
    processed_path = processed_dir / f"{doc_id}.json"

    if source.get("inline_text"):
        text = normalize_whitespace(str(source["inline_text"]))
        raw_path.write_text(text, encoding="utf-8")
    else:
        content = download_binary(url)
        raw_path.write_bytes(content)

        if suffix == ".pdf":
            text = extract_pdf_text(raw_path)
        else:
            html = content.decode("utf-8", errors="ignore")
            text = extract_html_text(html)

    processed = {
        **source,
        "raw_path": str(raw_path),
        "text": text,
    }
    processed_path.write_text(
        json.dumps(processed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    knowledge_base_dir = Path(__file__).resolve().parents[1] / "data" / "knowledge"
    raw_dir = knowledge_base_dir / "raw"
    processed_dir = knowledge_base_dir / "processed"
    manifest_path = knowledge_base_dir / "source_manifest.json"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    sources = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_ids = {source["doc_id"] for source in sources}
    for path in processed_dir.glob("*.json"):
        if path.stem not in source_ids:
            path.unlink()
    for path in raw_dir.iterdir():
        if path.is_file() and path.stem not in source_ids:
            path.unlink()

    failures: list[dict[str, str]] = []
    for source in sources:
        print(f"[download] {source['doc_id']}")
        try:
            process_source(source, raw_dir, processed_dir)
        except Exception as exc:
            failures.append({"doc_id": source["doc_id"], "error": str(exc)})
            print(f"[failed] {source['doc_id']}: {exc}")

    stats = KnowledgeBaseBuilder(knowledge_base_dir).build()
    print(json.dumps({"status": "ok", "stats": stats, "failures": failures}, ensure_ascii=False))


if __name__ == "__main__":
    main()
