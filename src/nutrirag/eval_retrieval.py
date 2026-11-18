"""Retrieval quality on the answerable questions: does a gold page appear in the top k?

Usage:
    python -m src.nutrirag.eval_retrieval
"""

from __future__ import annotations

import json

import pandas as pd

from .ingest import ROOT
from .retrieve import Retriever, load_passages

QUESTIONS = ROOT / "eval" / "questions.jsonl"
RESULTS = ROOT / "reports"
KS = (1, 3, 5, 8)
METHODS = ("bm25", "char_tfidf", "hybrid")


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    retriever = Retriever(load_passages())
    rows = []
    for q in load_questions():
        if not q["pages"]:
            continue
        gold = set(q["pages"])
        for method in METHODS:
            pages = [p["page"] for p in retriever.search(q["question"], k=max(KS), method=method)]
            first = next((i + 1 for i, pg in enumerate(pages) if pg in gold), None)
            rows.append({
                "id": q["id"], "type": q["type"], "method": method,
                **{f"hit@{k}": int(first is not None and first <= k) for k in KS},
                "reciprocal_rank": 1 / first if first else 0.0,
            })
    df = pd.DataFrame(rows)
    summary = df.groupby("method")[[f"hit@{k}" for k in KS] + ["reciprocal_rank"]].mean().round(3)
    by_type = df.pivot_table(index="type", columns="method", values="hit@5", aggfunc="mean").round(3)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS / "retrieval_summary.csv")
    by_type.to_csv(RESULTS / "retrieval_hit5_by_type.csv")
    df.to_csv(RESULTS / "retrieval_per_question.csv", index=False)
    print(summary.to_string())
    print()
    print(by_type.to_string())
    misses = df[(df["method"] == "hybrid") & (df["hit@5"] == 0)]["id"].tolist()
    print("hybrid misses at k=5:", misses)


if __name__ == "__main__":
    main()
