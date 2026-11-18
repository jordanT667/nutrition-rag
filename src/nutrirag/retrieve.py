"""Lexical retrievers over the passages: BM25, character TF-IDF, and their fusion.

All local, no model download. Usage:
    python -m src.nutrirag.retrieve "combien de poisson par semaine ?"
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from .ingest import PROCESSED

STOPWORDS = set(
    """a au aux avec ce ces cet cette dans de des du elle en est et il ils je la le les leur lui
    ma mais me mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te
    tes toi ton tu un une vos votre vous y d l s c j n m t qu est sont faut peut combien quel quelle
    quels quelles comment pourquoi quand faire il elle plus""".split()
)


def fold(text: str) -> str:
    text = text.lower().replace("œ", "oe").replace("æ", "ae")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", fold(text))
    out = []
    for w in words:
        if w in STOPWORDS or len(w) < 2:
            continue
        if len(w) > 4 and w[-1] in "sx":  # light plural stemming
            w = w[:-1]
        out.append(w)
    return out


def load_passages() -> list[dict]:
    return json.loads((PROCESSED / "passages.json").read_text(encoding="utf-8"))


class Retriever:
    def __init__(self, passages: list[dict]):
        self.passages = passages
        texts = [p["text"] for p in passages]
        self.bm25 = BM25Okapi([tokens(t) for t in texts])
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), preprocessor=fold, sublinear_tf=True)
        self.char_matrix = self.char.fit_transform(texts)

    def scores(self, query: str, method: str) -> np.ndarray:
        if method == "bm25":
            return self.bm25.get_scores(tokens(query))
        if method == "char_tfidf":
            return (self.char_matrix @ self.char.transform([query]).T).toarray().ravel()
        raise ValueError(method)

    def search(self, query: str, k: int = 5, method: str = "hybrid", rrf_k: int = 60) -> list[dict]:
        if method == "hybrid":  # reciprocal rank fusion of the two lexical views
            fused = np.zeros(len(self.passages))
            for m in ("bm25", "char_tfidf"):
                ranks = np.argsort(np.argsort(-self.scores(query, m)))
                fused += 1.0 / (rrf_k + ranks + 1)
            order = np.argsort(-fused)
        else:
            order = np.argsort(-self.scores(query, method))
        return [self.passages[i] for i in order[:k]]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    r = Retriever(load_passages())
    for p in r.search(" ".join(sys.argv[1:]) or "poisson gras", k=3):
        print(f"[{p['id']}] {p['text'][:200]}\n")
