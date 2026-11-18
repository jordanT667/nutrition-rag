"""Answer a question from the official guidelines with Claude, citing the pages used.

Three modes, to measure what retrieval adds:
    rag     the top-k retrieved passages, given as citable search results
    full    every passage of the corpus, cached (the whole document fits in context)
    closed  no source at all (what the model says from its own knowledge)

Usage (needs ANTHROPIC_API_KEY):
    python -m src.nutrirag.generate "Combien de poisson par semaine ?"
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import anthropic

from .ingest import ROOT, SOURCES


def load_env(path=ROOT / ".env") -> None:
    """Read KEY=value lines from a local .env file (git-ignored) into the environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()
from .retrieve import Retriever, load_passages

MODEL = "claude-opus-5-5"
EFFORT = "medium"
TOP_K = 5
RETRIEVER = "bm25"

SYSTEM_GROUNDED = """Tu es un assistant qui répond aux questions sur les recommandations nutritionnelles \
officielles françaises (Santé publique France). Réponds en français, en 2 à 5 phrases.

Règles :
- Appuie chaque affirmation sur les sources fournies et cite-les. N'ajoute aucune information \
qui n'y figure pas, même si tu la connais.
- Si les sources ne permettent pas de répondre, dis-le clairement en une phrase \
(« Les recommandations fournies ne répondent pas à cette question. »), puis indique \
si elles contiennent une information proche, sans inventer de chiffre.
- Ces recommandations s'adressent aux adultes en général. Ne donne pas de conseil médical \
personnalisé ; pour une situation particulière, renvoie vers un professionnel de santé."""

SYSTEM_CLOSED = """Tu es un assistant qui répond aux questions sur les recommandations nutritionnelles \
officielles françaises pour les adultes. Réponds en français, en 2 à 5 phrases. \
Si tu ne connais pas la recommandation officielle, dis-le. Ne donne pas de conseil médical personnalisé."""


@dataclass
class Answer:
    mode: str
    text: str
    cited_pages: list[int] = field(default_factory=list)
    retrieved_pages: list[int] = field(default_factory=list)
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    stop_reason: str = ""


def search_result(passage: dict) -> dict:
    src = SOURCES[passage["doc"]]
    return {
        "type": "search_result",
        "source": f"{src['url']}#page={passage['page']}",
        "title": f"{src['title']}, page {passage['page']}",
        "content": [{"type": "text", "text": passage["text"]}],
        "citations": {"enabled": True},
    }


def page_of(source: str) -> int | None:
    return int(source.rsplit("#page=", 1)[1]) if "#page=" in source else None


class Assistant:
    def __init__(self, client: anthropic.Anthropic | None = None):
        self.client = client or anthropic.Anthropic()
        self.passages = load_passages()
        self.retriever = Retriever(self.passages)

    def _call(self, system: str, content: list[dict] | str):
        return self.client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=system,
            output_config={"effort": EFFORT},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": content}],
        )

    def ask(self, question: str, mode: str = "rag") -> Answer:
        retrieved: list[dict] = []
        if mode == "closed":
            system, content = SYSTEM_CLOSED, question
        else:
            if mode == "rag":
                retrieved = self.retriever.search(question, k=TOP_K, method=RETRIEVER)
                blocks = [search_result(p) for p in retrieved]
            elif mode == "full":
                blocks = [search_result(p) for p in self.passages]
                blocks[-1]["cache_control"] = {"type": "ephemeral"}  # same prefix for every question
            else:
                raise ValueError(mode)
            system, content = SYSTEM_GROUNDED, blocks + [{"type": "text", "text": question}]

        start = time.perf_counter()
        response = self._call(system, content)
        seconds = time.perf_counter() - start

        text, cited = [], []
        for block in response.content:
            if block.type != "text":
                continue
            text.append(block.text)
            for c in block.citations or []:
                page = page_of(getattr(c, "source", "") or "")
                if page is not None:
                    cited.append(page)
        usage = response.usage
        return Answer(
            mode=mode,
            text="".join(text).strip(),
            cited_pages=sorted(set(cited)),
            retrieved_pages=[p["page"] for p in retrieved],
            input_tokens=usage.input_tokens,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
            output_tokens=usage.output_tokens,
            seconds=round(seconds, 2),
            stop_reason=response.stop_reason or "",
        )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    answer = Assistant().ask(" ".join(sys.argv[1:]) or "Combien de poisson faut-il manger par semaine ?")
    print(answer.text)
    print(f"\npages cited: {answer.cited_pages}")
