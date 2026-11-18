"""Turn the source PDFs into page texts and retrieval passages.

Usage:
    python -m src.nutrirag.ingest
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Each source: local file, title, official URL, and the running header to strip.
SOURCES = {
    "spf2019": {
        "file": "spf_recommandations_adultes_2019.pdf",
        "title": "Santé publique France (2019). Recommandations relatives à l'alimentation, "
        "à l'activité physique et à la sédentarité pour les adultes",
        "url": "https://www.santepubliquefrance.fr/sites/default/files/rdd/document/42466_spf00000619.pdf",
        "header": r"Santé publique France / Recommandations relatives à l.alimentation,\s*"
        r"à l.activité physique et à la sédentarité pour les adultes / p\. \d+",
    },
}

TARGET_CHARS = 1200


@dataclass
class Passage:
    id: str
    doc: str
    page: int  # 1-based page of the PDF file
    text: str


def clean_page(text: str, header: str) -> str:
    text = re.sub(header, " ", text)
    text = text.replace("\u00a0", " ").replace("\u2019", "'")
    text = re.sub("[\ue000-\uf8ff]", " ", text)  # private-use bullet glyphs
    text = re.sub(r"-\n(?=[a-zà-ÿ])", "", text)  # words split across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\.{4,}", " ", text)  # table of contents dots
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def split_page(text: str, target: int = TARGET_CHARS) -> list[str]:
    """Group paragraphs into passages of about `target` characters."""
    paragraphs = [re.sub(r"\s*\n\s*", " ", p).strip() for p in text.split("\n\n")]
    paragraphs = [p for p in paragraphs if len(p) > 2]
    out, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) > target:
            out.append(current)
            current = p
        else:
            current = f"{current}\n{p}" if current else p
        while len(current) > 2 * target:  # very long paragraph: cut on sentences
            cut = current.rfind(". ", 0, target) + 1 or target
            out.append(current[:cut].strip())
            current = current[cut:].strip()
    if current:
        out.append(current)
    return out


def ingest() -> tuple[list[dict], list[Passage]]:
    pages, passages = [], []
    for doc_id, src in SOURCES.items():
        path = RAW / src["file"]
        if not path.exists():
            print(f"skipped {doc_id}: {path.name} not found")
            continue
        with pymupdf.open(path) as pdf:
            for i, page in enumerate(pdf, start=1):
                text = clean_page(page.get_text(), src["header"])
                pages.append({"doc": doc_id, "page": i, "text": text})
                for j, chunk in enumerate(split_page(text)):
                    passages.append(Passage(f"{doc_id}-p{i}-{j}", doc_id, i, chunk))
    return pages, passages


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pages, passages = ingest()
    (PROCESSED / "pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
    (PROCESSED / "passages.json").write_text(
        json.dumps([asdict(p) for p in passages], ensure_ascii=False, indent=1), encoding="utf-8"
    )
    lengths = sorted(len(p.text) for p in passages)
    print(f"{len(pages)} pages, {len(passages)} passages, median {lengths[len(lengths) // 2]} chars")


if __name__ == "__main__":
    main()
