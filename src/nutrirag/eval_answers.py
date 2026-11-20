"""End-to-end evaluation: answer every question in each mode, then grade the answers.

Grading uses Claude as a judge with a fixed schema, given the reference answer and the
reference pages. Citation accuracy is computed without the judge. Answers are cached in
eval/runs/ so an interrupted run resumes where it stopped.

Usage (needs ANTHROPIC_API_KEY; costs money, see README):
    python -m src.nutrirag.eval_answers            # all modes
    python -m src.nutrirag.eval_answers rag full   # some modes
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import Literal

import anthropic
import pandas as pd
from pydantic import BaseModel, Field

from .eval_retrieval import load_questions
from .generate import MODEL, Assistant
from .ingest import PROCESSED, ROOT

RUNS = ROOT / "eval" / "runs"
RESULTS = ROOT / "reports"
MODES = ("rag", "full", "closed")

# Claude Opus 5.5, USD per million tokens (cache write = 1.25 x input, 5-minute TTL).
PRICE = {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00}


class AnswerableGrade(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"] = Field(
        description="correct: all key facts of the reference; partial: some key facts missing or vague; "
        "incorrect: wrong or missing the main fact"
    )
    contradicts_reference: bool = Field(description="states something that contradicts the reference pages")
    adds_unsupported_figures: bool = Field(description="gives a number or threshold absent from the reference pages")
    rationale: str = Field(description="one sentence")


class UnanswerableGrade(BaseModel):
    abstained: bool = Field(description="clearly says the guidelines do not answer the question")
    invented_figure: bool = Field(description="gives a number or recommendation presented as official")
    rationale: str = Field(description="one sentence")


JUDGE_ANSWERABLE = """Tu évalues la réponse d'un assistant sur les recommandations nutritionnelles officielles.

Question : {question}
Réponse de référence : {reference}
Pages de référence du document officiel :
<pages>
{pages}
</pages>

Réponse à évaluer :
<reponse>
{answer}
</reponse>

Évalue la réponse par rapport à la référence et aux pages. Une formulation différente est acceptable \
si le sens est le même. Des informations exactes supplémentaires tirées des pages ne sont pas une erreur."""

JUDGE_UNANSWERABLE = """Tu évalues la réponse d'un assistant sur les recommandations nutritionnelles officielles \
françaises pour les adultes (Santé publique France, 2019). Le document ne répond pas à cette question.
{note}
Question : {question}

Réponse à évaluer :
<reponse>
{answer}
</reponse>

L'assistant devait dire que les recommandations ne répondent pas à la question, sans inventer de chiffre \
présenté comme officiel. Citer une information proche réellement présente dans le document est acceptable."""


def cost(row: dict) -> float:
    return (
        row["input_tokens"] * PRICE["input"]
        + row["output_tokens"] * PRICE["output"]
        + row["cache_read_tokens"] * PRICE["cache_read"]
        + row["cache_write_tokens"] * PRICE["cache_write"]
    ) / 1e6


def generate(mode: str, questions: list[dict], assistant: Assistant) -> list[dict]:
    path = RUNS / f"answers_{mode}.jsonl"
    done = {}
    if path.exists():
        done = {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    with path.open("a", encoding="utf-8") as f:
        for q in questions:
            if q["id"] in done:
                continue
            row = {"id": q["id"], **asdict(assistant.ask(q["question"], mode))}
            done[q["id"]] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  {mode} {q['id']} {row['seconds']}s")
    return [done[q["id"]] for q in questions]


def judge(client: anthropic.Anthropic, q: dict, answer: str, page_text: dict) -> dict:
    if q["pages"]:
        prompt = JUDGE_ANSWERABLE.format(
            question=q["question"], reference=q["answer"], answer=answer,
            pages="\n\n".join(f"[page {p}]\n{page_text[p]}" for p in q["pages"]),
        )
        schema = AnswerableGrade
    else:
        note = f"Précision : {q['note']}\n" if q.get("note") else ""
        prompt = JUDGE_UNANSWERABLE.format(question=q["question"], answer=answer, note=note)
        schema = UnanswerableGrade
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
        output_format=schema,
    )
    usage = response.usage
    return {
        **response.parsed_output.model_dump(),
        "judge_cost": (usage.input_tokens * PRICE["input"] + usage.output_tokens * PRICE["output"]) / 1e6,
    }


def grade(mode: str, questions: list[dict], answers: list[dict], client, page_text) -> pd.DataFrame:
    path = RUNS / f"grades_{mode}.jsonl"
    done = {}
    if path.exists():
        done = {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    with path.open("a", encoding="utf-8") as f:
        for q, a in zip(questions, answers, strict=True):
            if q["id"] not in done:
                done[q["id"]] = {"id": q["id"], **judge(client, q, a["text"], page_text)}
                f.write(json.dumps(done[q["id"]], ensure_ascii=False) + "\n")
    rows = []
    for q, a in zip(questions, answers, strict=True):
        g = done[q["id"]]
        gold = set(q["pages"])
        rows.append({
            "id": q["id"], "type": q["type"], "mode": mode, "answerable": bool(gold),
            "verdict": g.get("verdict"), "contradicts": g.get("contradicts_reference"),
            "unsupported_figures": g.get("adds_unsupported_figures"),
            "abstained": g.get("abstained"), "invented_figure": g.get("invented_figure"),
            "cited_any": bool(a["cited_pages"]),
            "cited_gold_page": bool(gold & set(a["cited_pages"])) if gold else None,
            "retrieved_gold_page": bool(gold & set(a["retrieved_pages"])) if gold and mode == "rag" else None,
            "seconds": a["seconds"], "cost_usd": cost(a), "judge_cost_usd": g["judge_cost"],
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for mode, d in df.groupby("mode", sort=False):
        ans, una = d[d["answerable"]], d[~d["answerable"]]
        out.append({
            "mode": mode,
            "correct": (ans["verdict"] == "correct").mean(),
            "correct_or_partial": ans["verdict"].isin(["correct", "partial"]).mean(),
            "contradicts_reference": ans["contradicts"].mean(),
            "unsupported_figures": ans["unsupported_figures"].mean(),
            "cites_a_gold_page": ans["cited_gold_page"].mean() if mode != "closed" else None,
            "abstains_when_unanswerable": una["abstained"].mean(),
            "invents_figure_when_unanswerable": una["invented_figure"].mean(),
            "median_seconds": d["seconds"].median(),
            "cost_per_question_usd": d["cost_usd"].mean(),
        })
    return pd.DataFrame(out).round(4)


def main(modes: list[str]) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    pages = json.loads((PROCESSED / "pages.json").read_text(encoding="utf-8"))
    page_text = {p["page"]: p["text"] for p in pages}
    assistant = Assistant()
    frames = []
    for mode in modes:
        print(f"[{mode}] answering")
        answers = generate(mode, questions, assistant)
        print(f"[{mode}] grading")
        frames.append(grade(mode, questions, answers, assistant.client, page_text))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RESULTS / "answers_per_question.csv", index=False)
    summary = summarize(df)
    summary.to_csv(RESULTS / "answers_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"total spent: ${df['cost_usd'].sum() + df['judge_cost_usd'].sum():.2f}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODES))
