from types import SimpleNamespace

import pytest

from src.nutrirag import generate
from src.nutrirag.eval_answers import cost
from src.nutrirag.ingest import SOURCES, clean_page, split_page
from src.nutrirag.retrieve import Retriever, fold, tokens

PASSAGES = [
    {"id": "spf2019-p34-0", "doc": "spf2019", "page": 34,
     "text": "Deux fois par semaine du poisson, dont un poisson gras (sardines, maquereau, hareng, saumon)."},
    {"id": "spf2019-p36-0", "doc": "spf2019", "page": 36,
     "text": "Limiter la charcuterie à 150 g par semaine, environ 3 tranches de jambon blanc."},
    {"id": "spf2019-p35-0", "doc": "spf2019", "page": 35,
     "text": "L'eau est la seule boisson recommandée. Les sodas, même light, doivent être limités."},
]


def test_fold_keeps_ligatures_and_drops_accents():
    assert fold("Œufs, céréales") == "oeufs, cereales"


def test_tokens_remove_stopwords_and_plural():
    assert tokens("Les poissons gras de la semaine") == ["poisson", "gras", "semaine"]


def test_header_is_removed_and_paragraphs_kept():
    raw = ("Santé publique France / Recommandations relatives à l'alimentation,\n"
           "à l'activité physique et à la sédentarité pour les adultes / p. 33\n\nPremier.\n\nSecond.")
    text = clean_page(raw, SOURCES["spf2019"]["header"])
    assert "p. 33" not in text and "Premier." in text and "Second." in text


def test_split_page_respects_target_size():
    text = "\n\n".join(["mot " * 100] * 10)  # 10 paragraphs of 400 chars
    chunks = split_page(text, target=1000)
    assert len(chunks) > 1 and all(len(c) <= 2000 for c in chunks)


@pytest.mark.parametrize("method", ["bm25", "char_tfidf", "hybrid"])
def test_retrievers_find_the_right_page(method):
    r = Retriever(PASSAGES)
    assert r.search("combien de tranches de jambon ?", k=1, method=method)[0]["page"] == 36
    assert r.search("poisson gras sardines", k=1, method=method)[0]["page"] == 34


def test_search_result_block_carries_the_page():
    block = generate.search_result(PASSAGES[0])
    assert block["type"] == "search_result" and block["citations"] == {"enabled": True}
    assert generate.page_of(block["source"]) == 34


class FakeClient:
    """Stands in for anthropic.Anthropic: records the request, returns a cited answer."""

    def __init__(self):
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        cite = SimpleNamespace(source=kwargs["messages"][0]["content"][0]["source"]
                               if isinstance(kwargs["messages"][0]["content"], list) else "")
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Deux fois par semaine.", citations=[cite])],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0),
            stop_reason="end_turn",
        )


@pytest.fixture
def assistant(monkeypatch):
    monkeypatch.setattr(generate, "load_passages", lambda: PASSAGES)
    return generate.Assistant(client=FakeClient())


def test_rag_mode_sends_only_top_k_and_reads_citations(assistant, monkeypatch):
    monkeypatch.setattr(generate, "TOP_K", 2)
    answer = assistant.ask("poisson gras", mode="rag")
    sent = assistant.client.requests[-1]["messages"][0]["content"]
    assert sum(b["type"] == "search_result" for b in sent) == 2
    assert answer.cited_pages == [34] and answer.retrieved_pages[0] == 34


def test_full_mode_sends_everything_and_caches_it(assistant):
    assistant.ask("poisson", mode="full")
    sent = assistant.client.requests[-1]["messages"][0]["content"]
    assert sum(b["type"] == "search_result" for b in sent) == len(PASSAGES)
    assert sent[len(PASSAGES) - 1]["cache_control"] == {"type": "ephemeral"}


def test_closed_mode_sends_no_source(assistant):
    answer = assistant.ask("poisson", mode="closed")
    assert isinstance(assistant.client.requests[-1]["messages"][0]["content"], str)
    assert answer.cited_pages == []


def test_cost_uses_cache_prices():
    row = {"input_tokens": 1_000_000, "output_tokens": 0, "cache_read_tokens": 1_000_000, "cache_write_tokens": 0}
    assert cost(row) == pytest.approx(4.20)


@pytest.mark.parametrize("method", ["bm25", "char_tfidf", "hybrid"])
def test_search_returns_k_distinct_passages(method):
    r = Retriever(PASSAGES)
    hits = r.search("poisson et eau", k=2, method=method)
    assert len(hits) == 2 and len({h["id"] for h in hits}) == 2
    assert len(r.search("poisson", k=10, method=method)) == len(PASSAGES)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        Retriever(PASSAGES).search("poisson", method="dense")


def test_load_env_reads_values_without_overriding(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nNUTRIRAG_A="quoted"\nNUTRIRAG_B=plain\nnot a pair\n', encoding="utf-8")
    monkeypatch.setenv("NUTRIRAG_A", "")
    monkeypatch.delenv("NUTRIRAG_A")  # removed for the test, restored afterwards
    monkeypatch.setenv("NUTRIRAG_B", "already set")
    generate.load_env(env)
    assert generate.os.environ["NUTRIRAG_A"] == "quoted"
    assert generate.os.environ["NUTRIRAG_B"] == "already set"


def test_load_env_without_file_is_a_no_op(tmp_path):
    generate.load_env(tmp_path / "missing.env")

