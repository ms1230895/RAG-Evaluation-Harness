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


# ---------------------------------------------------------------- Step 4: test set

class ScriptedInput:
    """Fake for the review tool's `input_fn`: returns scripted answers in order.

    Raises AssertionError if the tool asks for more input than was scripted,
    so a test never hangs or silently invents answers. `.answers` holds what
    is left; `.prompts` records every prompt shown.
    """

    def __init__(self, answers):
        """Store the answers to hand out, first to last."""
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt=""):
        """Return the next scripted answer, or fail if there is none."""
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f"review asked for more input than scripted: {prompt!r}")
        return self.answers.pop(0)


@pytest.fixture
def scripted_input():
    """Factory: scripted_input(["a", "s"]) gives a ScriptedInput for `review(input_fn=...)`."""
    return ScriptedInput


@pytest.fixture
def make_query():
    """Factory returning a valid lookup Query as a plain dict; keyword arguments override fields."""

    def _make(**overrides):
        row = {
            "id": "q001",
            "question": "How many casual leaves does an employee get?",
            "type": "lookup",
            "reference_answer": "8 casual leaves per year.",
            "source_doc_ids": ["leave_policy"],
            "split": None,
            "reviewed": False,
        }
        row.update(overrides)
        return row

    return _make


@pytest.fixture
def testset_docs_dir(tmp_path):
    """Tiny docs directory in tmp_path: leave_policy, payroll_policy, it_policy (.md).

    Line facts used by tests: leave_policy has the number 8 and a line with the
    whole word "Casual"; it_policy has "Casualties" (contains "casual" but is
    not that whole word).
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "leave_policy.md").write_text(
        "# Leave Policy\n\nEmployees get 8 Casual leaves per year.\nCarry forward is not allowed.\n",
        encoding="utf-8")
    (docs / "payroll_policy.md").write_text(
        "# Payroll Policy\n\nSalary is paid on the last working day.\nPayslips are on the portal.\n",
        encoding="utf-8")
    (docs / "it_policy.md").write_text(
        "# IT Policy\n\nEvery intern receives a laptop.\nCasualties are not a laptop topic.\n",
        encoding="utf-8")
    return docs


@pytest.fixture
def testset_config(tmp_path, testset_docs_dir):
    """Test-set config dict for the 8-query `ready_queries` set (same keys as configs/test_set.yaml).

    expected_counts: lookup 2 dev + 1 heldout, multi_chunk 1 + 1,
    paraphrased 1 + 0, unanswerable 1 + 1 (8 queries). Paths are absolute tmp_path strings.
    """
    return {
        "path": str(tmp_path / "test_set.jsonl"),
        "docs_dir": str(testset_docs_dir),
        "split_seed": 7,
        "expected_counts": {
            "lookup": {"dev": 2, "heldout": 1},
            "multi_chunk": {"dev": 1, "heldout": 1},
            "paraphrased": {"dev": 1, "heldout": 0},
            "unanswerable": {"dev": 1, "heldout": 1},
        },
    }


@pytest.fixture
def ready_queries(make_query):
    """The 8 correct, reviewed, split queries that match `testset_config` (check_ready gives [])."""
    from evalharness.schemas import Query

    rows = [
        make_query(id="q001", question="Question one about leave?", split="dev", reviewed=True,
                   source_doc_ids=["leave_policy"]),
        make_query(id="q002", question="Question two about pay?", split="dev", reviewed=True,
                   reference_answer="Paid on the last working day.", source_doc_ids=["payroll_policy"]),
        make_query(id="q003", question="Question three about laptops?", split="heldout", reviewed=True,
                   reference_answer="Every intern gets a laptop.", source_doc_ids=["it_policy"]),
        make_query(id="q004", question="Question four across docs?", type="multi_chunk", split="dev",
                   reviewed=True, reference_answer="Leave and pay.",
                   source_doc_ids=["leave_policy", "payroll_policy"]),
        make_query(id="q005", question="Question five across docs?", type="multi_chunk", split="heldout",
                   reviewed=True, reference_answer="Laptop and leave.",
                   source_doc_ids=["it_policy", "leave_policy"]),
        make_query(id="q006", question="Question six, vague?", type="paraphrased", split="dev",
                   reviewed=True, reference_answer="Paid at month end.", source_doc_ids=["payroll_policy"]),
        make_query(id="q007", question="Question seven, not covered?", type="unanswerable", split="dev",
                   reviewed=True, reference_answer="The Kestrel Labs HR policies do not cover this.",
                   source_doc_ids=[]),
        make_query(id="q008", question="Question eight, not covered?", type="unanswerable", split="heldout",
                   reviewed=True, reference_answer="The Kestrel Labs HR policies do not cover this.",
                   source_doc_ids=[]),
    ]
    return [Query.model_validate(row) for row in rows]
