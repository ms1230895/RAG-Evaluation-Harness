# tests/conftest.py
# Shared fixtures for the whole test suite.
import os
import re

# Backstop, set BEFORE importing anything from rag/ or llm/:
# hide API keys and force Hugging Face libraries offline, so an accidental
# real call fails fast instead of spending credits or downloading a model.
# This is a safety net, not a guarantee (code that calls load_dotenv() can
# reload keys), so tests must still never reach a real client.
for _name in list(os.environ):
    if _name.endswith("_API_KEY"):
        del os.environ[_name]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pytest


@pytest.fixture
def db_path(tmp_path):
    """Path for a throwaway SQLite database, never the real results database."""
    return tmp_path / "test_results.db"


@pytest.fixture
def llm_config(tmp_path):
    """Config dict for llm.client with the cache in tmp_path and zero retry waits.

    Keys match configs/llm.yaml. Tests may change values (e.g. max_attempts)
    on the returned dict before calling generate().
    """
    return {
        "base_url": "http://fake-llm.test/v1",
        "timeout_s": 5,
        "max_attempts": 5,
        "retry_wait_min_s": 0,
        "retry_wait_max_s": 0,
        "cache_path": tmp_path / "llm_cache_test.sqlite",
    }


class FakeApi:
    """Stand-in for llm.client._call_api: returns a canned ApiReply and records calls.

    Set `reply` to change what is returned, or `error` to make every call raise.
    Like the real retry loop it adds 1 to stats.attempts per call, even when it raises.
    """

    def __init__(self, reply):
        self.reply = reply
        self.error = None
        self.calls = []

    def __call__(self, prompt, model, temperature, schema, config, stats,
                 seed=None, reasoning_effort=None):
        stats.attempts += 1
        self.calls.append(
            {"prompt": prompt, "model": model, "temperature": temperature,
             "schema": schema, "config": config, "seed": seed,
             "reasoning_effort": reasoning_effort}
        )
        if self.error is not None:
            raise self.error
        return self.reply


@pytest.fixture
def fake_api(monkeypatch):
    """Replace llm.client._call_api with a FakeApi (no HTTP, no retries)."""
    from llm import client

    api = FakeApi(client.ApiReply(
        text="hello", prompt_tokens=10, completion_tokens=5,
        finish_reason="stop", model_latency_ms=1234.0,
    ))
    monkeypatch.setattr(client, "_call_api", api)
    return api


# ---------------------------------------------------------------- Step 3: RAG pipeline

class FakeEmbed:
    """Offline stand-in for rag.embedder._embed_texts: bag-of-words vectors, records every call.

    Vector = count of each VOCAB word in the text, plus a constant 1.0 at the
    end (so no vector is all zeros). Counts are NOT normalised: the code under
    test must normalise. Words outside VOCAB are ignored.
    """

    VOCAB = ["leave", "salary", "payroll", "laptop", "intern", "holiday", "policy", "onboarding"]

    def __init__(self):
        """Start with no recorded calls."""
        self.calls = []   # one dict per call: {"texts": [...], "model": "..."}

    def __call__(self, texts, model):
        """Record the call and return one raw count vector per text."""
        self.calls.append({"texts": list(texts), "model": model})
        rows = []
        for text in texts:
            words = re.findall(r"[a-z]+", text.lower())
            rows.append([float(words.count(w)) for w in self.VOCAB] + [1.0])
        return rows

    @property
    def texts(self):
        """Every text embedded so far, in call order (prefixes included)."""
        return [t for call in self.calls for t in call["texts"]]


@pytest.fixture
def fake_embed(monkeypatch):
    """Replace rag.embedder._embed_texts (the one seam to llm.client.embed) with a FakeEmbed."""
    from rag import embedder

    fake = FakeEmbed()
    monkeypatch.setattr(embedder, "_embed_texts", fake)
    return fake


def _tiny_doc(title, topic_words, total_words=20):
    """Build a doc text of exactly `total_words` words: '# Title', topic words, then filler x0, x1, ..."""
    head = ["#"] + title.split()
    filler = [f"x{i}" for i in range(total_words - len(head) - len(topic_words))]
    return "\n\n".join([" ".join(head), " ".join(topic_words + filler)]) + "\n"


@pytest.fixture
def pipeline_config(tmp_path, monkeypatch):
    """Pipeline config dict for a tiny 3-doc corpus in tmp_path; resets rag.index._LOADED.

    Corpus: 3 docs of exactly 20 words each. With chunk_size 8 and overlap 2
    (step 6) each doc gives 3 chunks (word starts 0, 6, 12), so 9 chunks.
    index.dir and corpus.docs_dir are absolute tmp_path strings. A fresh dict
    per test, so tests may change values freely.
    """
    from rag import index

    monkeypatch.setattr(index, "_LOADED", {})
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    corpus = {
        "leave_policy": ("Leave Policy", ["leave", "leave", "leave", "interns"]),
        "payroll_policy": ("Payroll Policy", ["salary", "salary", "salary", "monthly"]),
        "it_policy": ("IT Policy", ["laptop", "laptop", "laptop", "onboarding"]),
    }
    for doc_id, (title, topic_words) in corpus.items():
        (docs_dir / f"{doc_id}.md").write_text(_tiny_doc(title, topic_words), encoding="utf-8")
    return {
        "name": "test",
        "corpus": {"docs_dir": str(docs_dir)},
        "chunking": {"chunk_size": 8, "chunk_overlap": 2},
        "embedding": {"model": "fake-embed", "doc_prefix": "search_document: ",
                      "query_prefix": "search_query: ", "batch_size": 4,
                      "max_input_words": 1000},
        "index": {"dir": str(tmp_path / "indexes")},
        "retrieval": {"method": "vector", "top_k": 3},
        "generation": {
            "model": "fake-gen:cloud", "model_digest": None, "temperature": 0.0, "seed": 42,
            "reasoning_effort": None, "max_prompt_tokens": 1000,
            "context_separator": "\n\n---\n\n",
            "prompt_template": "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        },
    }
