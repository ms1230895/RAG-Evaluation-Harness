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


# ---------------------------------------------------------------- Step 5: runner and store

class ScriptedAnswer:
    """Fake `answer(question, config)` for the runner: valid contract dicts, or scripted failures.

    By default every question gets a valid contract dict built from the question text.
    `script(question, outcome)` replaces that for one question: a dict is returned
    as it is, an exception (instance) is raised. `fail_all(exc)` makes every call raise.
    `.calls` records {"question", "config"} for every call, in call order.
    """

    def __init__(self):
        """Start with no script and no recorded calls."""
        self.calls = []
        self._outcomes = {}
        self._fail_all = None

    @staticmethod
    def valid_output(question):
        """A contract-valid answer() result whose fields all depend on the question."""
        return {
            "answer": f"Answer to: {question}",
            "chunks": [{"chunk_id": "leave_policy#000", "doc_id": "leave_policy",
                        "text": f"Chunk text for: {question}", "score": 0.9}],
            "latency_ms": 12.5,
        }

    def script(self, question, outcome):
        """For this question return `outcome` (dict) or raise it (exception instance)."""
        self._outcomes[question] = outcome

    def fail_all(self, error):
        """Make every call raise `error`."""
        self._fail_all = error

    def __call__(self, question, config):
        """Record the call, then return or raise the scripted outcome."""
        self.calls.append({"question": question, "config": config})
        if self._fail_all is not None:
            raise self._fail_all
        outcome = self._outcomes.get(question, None)
        if outcome is None:
            return self.valid_output(question)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeDescribe:
    """Fake `describe(config)` hook: returns `.result` and records the configs it was given."""

    def __init__(self):
        """Default result: a small dict with an index key and a model digest."""
        self.result = {"index": {"index_key": "key-1"}, "generator_model": "fake-gen",
                       "generator_model_digest": "digest-1"}
        self.calls = []

    def __call__(self, config):
        """Record the call and return a copy of `.result`."""
        self.calls.append(config)
        return dict(self.result)


@pytest.fixture
def fake_system():
    """A runner.System with a ScriptedAnswer, a load_config that echoes the path, and a FakeDescribe.

    Use `fake_system.answer.script(...)` / `.fail_all(...)` / `.calls` and
    `fake_system.describe.result` / `.calls`.
    """
    from evalharness.runner import System

    return System(
        answer=ScriptedAnswer(),
        load_config=lambda path: {"name": "fake", "path": str(path)},
        describe=FakeDescribe(),
    )


@pytest.fixture
def results_db_path(tmp_path):
    """Path of a throwaway results database in tmp_path (never data/results.sqlite)."""
    return tmp_path / "results_test.sqlite"


@pytest.fixture
def results_db(results_db_path):
    """An open evalharness.store connection to a fresh throwaway results DB; closed after the test."""
    from contextlib import closing

    from evalharness import store

    with closing(store.connect(results_db_path)) as conn:
        yield conn


# ---------------------------------------------------------------- Step 6: checks and retrieval metrics

SCORED_RUN_ID = "run-scored-001"


def _chunk(doc_id, number):
    """One contract-valid chunk dict, e.g. _chunk("leave_policy", 0) -> chunk_id "leave_policy#000"."""
    return {"chunk_id": f"{doc_id}#{number:03d}", "doc_id": doc_id,
            "text": f"Text of {doc_id} part {number}.", "score": 0.5}


@pytest.fixture
def checks_config():
    """Small checks config: hit_ks [1, 2], four refusal phrases, a fixed version string.

    Same shape as evalharness.checks.load_checks_config() returns, but built by hand
    so tests of score_output / score_run do not depend on the config loader.
    """
    return {
        "hit_ks": [1, 2],
        "report_confidence": 0.95,
        "refusal_phrases": ["no mention of", "does not mention", "doesn't mention", "cannot find"],
        "version": "checks-test-v1",
    }


@pytest.fixture
def scored_run_db(tmp_path, ready_queries):
    """Tmp results DB with the 8 `ready_queries` and one completed run `SCORED_RUN_ID` (split dev).

    Despite the name nothing is scored yet; tests call score_run themselves.
    Returns a namespace with `.conn` (open, closed after the test), `.path` and `.run_id`.
    Hand-made outputs (query: case; latency; chunk doc ids in rank order):
      q001 lookup, source leave_policy: hit at rank 1; 100 ms; [leave, payroll, it]
      q002 lookup, source payroll_policy: hit at rank 3; 200 ms; [it, leave, payroll]
      q003 lookup, source it_policy: miss + false refusal ("doesn’t mention", curly); 300 ms;
           [leave, payroll, payroll]
      q004 multi_chunk, sources leave + payroll: partial hit (leave only, rank 1); 400 ms;
           [leave, leave, it]
      q005 multi_chunk, sources it + leave: FAILED output "ContextLimitError: prompt too long"
      q006 paraphrased, source payroll_policy: empty answer "  \\n"; 50 ms; only one chunk [payroll]
      q007 unanswerable: correct refusal ("There is no mention of ..."); 150 ms; [leave, payroll, it]
      q008 unanswerable: invented answer, no refusal phrase; 250 ms; [leave, payroll, it]
    Run llm_calls_json: one model, 7 real calls, 0 cache hits, 0 errors.
    """
    from contextlib import closing
    from types import SimpleNamespace

    from evalharness import store

    path = tmp_path / "scored_run.sqlite"
    three = [_chunk("leave_policy", 0), _chunk("payroll_policy", 0), _chunk("it_policy", 0)]
    outputs = {
        "q001": ("Employees get 8 casual leaves per year.", three, 100.0, None),
        "q002": ("Salary is paid on the last working day.",
                 [_chunk("it_policy", 0), _chunk("leave_policy", 1), _chunk("payroll_policy", 2)],
                 200.0, None),
        "q003": ("The context doesn’t mention laptops.",
                 [_chunk("leave_policy", 0), _chunk("payroll_policy", 0), _chunk("payroll_policy", 1)],
                 300.0, None),
        "q004": ("Leave is 8 days and salary is paid monthly.",
                 [_chunk("leave_policy", 0), _chunk("leave_policy", 1), _chunk("it_policy", 0)],
                 400.0, None),
        "q005": (None, None, None, "ContextLimitError: prompt too long"),
        "q006": ("  \n", [_chunk("payroll_policy", 0)], 50.0, None),
        "q007": ("There is no mention of pets in the policies.", three, 150.0, None),
        "q008": ("Interns can use the office gym for free.", three, 250.0, None),
    }
    conn = store.connect(path)
    with closing(conn):
        store.sync_queries(conn, ready_queries)
        store.start_run(conn, run_id=SCORED_RUN_ID, split="dev", config={"name": "fake"},
                        config_path="configs/fake.yaml", git_commit="abc1234", git_dirty=False,
                        test_set_sha256="0" * 64, system_info={}, n_queries=len(outputs),
                        llm_calls_start_id=0)
        for query_id, (answer, chunks, latency, error) in outputs.items():
            store.write_output(conn, run_id=SCORED_RUN_ID, query_id=query_id, answer=answer,
                               chunks=chunks, latency_ms=latency, error=error)
        store.finish_run(conn, SCORED_RUN_ID, status="completed", n_ok=7, n_failed=1,
                         llm_calls_summary={"gen-model": {"calls": 7, "hits": 0, "misses": 7,
                                                          "errors": 0}})
        yield SimpleNamespace(conn=conn, path=path, run_id=SCORED_RUN_ID)
