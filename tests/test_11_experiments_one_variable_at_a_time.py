"""Tests for Step 11: experiments, one variable at a time, written from the spec.

Spec: .claude/specs/11-experiments-one-variable-at-a-time.md. Everything is offline: fake
embedder, fake generator (patched where it is used), a fake rerank worker object with
`.ask(request, timeout_s)` and `.close()`, tmp DBs from the `experiment_db` fixture (conftest).
No real LLM, no real cross-encoder, never data/results.sqlite or data/llm_cache.sqlite.

Read-only reads of real project files (each test docstring says so): configs/v0.yaml,
configs/v1_*.yaml, configs/harness.yaml, configs/judges.yaml and the header of
experiments/log.md.

The spec does not name the keys of the dicts `paired_flips` / `compare_runs` return (beyond
the five flip group names), so number checks search the row by value or shape instead of
reading fixed paths. Hand values are in the `experiment_db` docstring.
"""

import ast
import copy
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evalharness import experiments
from evalharness.checks import ScoringRefused
from evalharness.experiments import (
    ExperimentRefused,
    check_experiment,
    compare_runs,
    config_diff,
    flatten,
    format_markdown,
    format_table,
    load_experiments_config,
    log_row,
    mcnemar_exact,
    paired_bootstrap_ci,
    paired_flips,
)
from evalharness.metrics import wilson_interval
from rag import reranker
from rag.bm25 import bm25_search, rrf_fuse, tokenize
from rag.index import get_index
from rag.pipeline import (
    SUPPORTED_METHODS,
    answer,
    build_prompt,
    describe_system,
    load_pipeline_config,
)
from rag.reranker import RerankError, rerank
from rag.retriever import retrieve, retrieve_with_method
from rag.rewriter import rewrite_query

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = PROJECT_ROOT / "configs"
REAL_V0 = CONFIGS / "v0.yaml"
REAL_HARNESS = CONFIGS / "harness.yaml"
REAL_JUDGES = CONFIGS / "judges.yaml"
REAL_LOG = PROJECT_ROOT / "experiments" / "log.md"
IGNORE = ("name", "experiment")
BOOT_CFG = {"n_bootstrap": 200, "bootstrap_seed": 7, "confidence": 0.95,
            "report_confidence": 0.95}
REFUSED = (ExperimentRefused, ScoringRefused)   # see "Spec ambiguities" in the report

V1_CONFIGS = [
    "v1_seed43", "v1_chunk75", "v1_chunk300", "v1_overlap0", "v1_topk2", "v1_topk6",
    "v1_hybrid", "v1_rerank", "v1_rewrite", "v1_prompt_grounded", "v1_prompt_citations",
    "v1_prompt_refusal",
]


# ---------------------------------------------------------------- helpers

def walk(obj):
    """Yield obj and every dict / list nested inside it."""
    yield obj
    if isinstance(obj, dict):
        for value in obj.values():
            yield from walk(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from walk(value)


def numbers(obj, prefix=""):
    """Yield (key, value) for every int / float (not bool) anywhere in obj."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from numbers(value, str(key))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from numbers(value, prefix)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, obj


def rate_dicts(obj, successes, n):
    """Every dict with `successes` and `n` equal to the given counts (metrics.rate_summary shape)."""
    return [d for d in walk(obj) if isinstance(d, dict)
            and d.get("successes") == successes and d.get("n") == n]


def bounds(d):
    """The interval of a dict: ci_low/ci_high, lo/hi or low/high."""
    for low_key, high_key in (("ci_low", "ci_high"), ("lo", "hi"), ("low", "high")):
        if low_key in d and high_key in d:
            return d[low_key], d[high_key]
    raise AssertionError(f"no interval keys in {d}")


def flip_groups(row):
    """The dict holding the improved / regressed / ... id lists inside a compare_runs row."""
    for d in walk(row):
        if (isinstance(d, dict) and isinstance(d.get("improved"), list)
                and isinstance(d.get("regressed"), list)):
            return d
    raise AssertionError("no dict with improved / regressed lists in the row")


def flip_counts(row):
    """The dict of int counts per flip group inside a compare_runs row."""
    for d in walk(row):
        if (isinstance(d, dict) and isinstance(d.get("improved"), int)
                and isinstance(d.get("regressed"), int)
                and not isinstance(d.get("improved"), bool)):
            return d
    raise AssertionError("no dict with improved / regressed counts in the row")


def ids_of(items):
    """Query ids from a list of ids or of dicts with `query_id`."""
    return sorted(i["query_id"] if isinstance(i, dict) else i for i in items)


def p_values(row):
    """Every number in the row stored under a p-value-like key."""
    return [v for k, v in numbers(row)
            if k in ("p", "p_value", "mcnemar_p") or "mcnemar" in k]


def compare(db, exp_run_id=None, **kwargs):
    """compare_runs on the fixture DB: baseline vs the fixture experiment run."""
    return compare_runs(db.conn, db.run_id, exp_run_id or db.exp_run_id, db.versions, BOOT_CFG,
                        **kwargs)


def sql_one(path, sql, params=()):
    """One row from a SQL query on the fixture DB (own connection, closed after)."""
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute(sql, params).fetchone()


def write_harness(tmp_path, db_path, *, experiments_block="default"):
    """Tmp harness YAML: tmp db_path, valid checks block, the real judges.yaml read-only."""
    config = {
        "db_path": str(db_path),
        "test_set_config": str(tmp_path / "test_set.yaml"),
        "max_consecutive_failures": 5,
        "error_max_chars": 500,
        "judges_config": str(REAL_JUDGES),
        "system": {"answer": "somewhere:answer", "load_config": "somewhere:load", "describe": None},
        "checks": {"hit_ks": [1, 4], "report_confidence": 0.95,
                   "refusal_phrases": ["no mention of", "does not mention", "cannot find"]},
        "report": {"primary_metric": "ungrounded", "include_bad_answer_rate": True,
                   "allow_incomplete": True},
    }
    if experiments_block == "default":
        config["experiments"] = {
            "base_config": str(REAL_V0), "ignore_keys": ["name", "experiment"],
            "n_bootstrap": 200, "bootstrap_seed": 7, "table_path": str(tmp_path / "table.md")}
    elif experiments_block is not None:
        config["experiments"] = experiments_block
    path = tmp_path / "harness_experiments_test.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


def run_main(db, tmp_path, monkeypatch, *argv):
    """experiments.main against the fixture DB, pinning the fixture's three scorer versions."""
    monkeypatch.setattr(experiments, "current_versions", lambda harness: dict(db.versions))
    harness = write_harness(tmp_path, db.path)
    return experiments.main(["--harness-config", harness, *argv])


def table_rows(path):
    """(run_id, kind) of every run_reports row in the fixture DB."""
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute("SELECT run_id, kind FROM run_reports ORDER BY id").fetchall()


def count_rows(path, table):
    """Row count of one table (table name is a literal from the tests, not user input)."""
    return sql_one(path, f"SELECT COUNT(*) FROM {table}")[0]


def imported_modules(path):
    """Every module name imported anywhere in a Python file."""
    names = []
    for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


# ---------------------------------------------------------------- fakes for rag/

class FakeGenerate:
    """Stand-in for a `generate` name (rag.pipeline.generate or rag.rewriter.generate)."""

    def __init__(self, text="the answer", prompt_tokens=100, model_latency_ms=300.0):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.model_latency_ms = model_latency_ms
        self.error = None
        self.calls = []

    def __call__(self, prompt, model, temperature, schema=None, *, config=None,
                 model_digest=None, seed=None, reasoning_effort=None):
        self.calls.append({"prompt": prompt, "model": model, "temperature": temperature,
                           "seed": seed, "model_digest": model_digest,
                           "reasoning_effort": reasoning_effort})
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.text, prompt_tokens=self.prompt_tokens,
                               model_latency_ms=self.model_latency_ms)


class FakeWorker:
    """Stand-in for rag.reranker.WorkerProcess: records requests, scores texts by a function."""

    def __init__(self, scores_fn=None, error=None, reply=None):
        self.scores_fn = scores_fn or (lambda texts: [float(i) for i in range(len(texts))])
        self.error = error
        self.reply = reply
        self.requests = []
        self.timeouts = []
        self.closed = False

    def ask(self, request, timeout_s):
        self.requests.append(request)
        self.timeouts.append(timeout_s)
        if self.error is not None:
            raise self.error
        if self.reply is not None:
            return self.reply
        return {"scores": self.scores_fn(request["texts"])}

    def close(self):
        self.closed = True


class WorkerFactory:
    """Replaces rag.reranker._start_worker; hands out queued workers, then plain FakeWorkers."""

    def __init__(self):
        self.queue = []
        self.workers = []
        self.configs = []

    def __call__(self, rerank_config):
        self.configs.append(rerank_config)
        worker = self.queue.pop(0) if self.queue else FakeWorker()
        self.workers.append(worker)
        return worker


@pytest.fixture(autouse=True)
def block_real_llm_client(monkeypatch):
    """Safety net: any real client creation fails the test instead of reaching a server."""
    from llm import client

    def boom(*args, **kwargs):
        raise AssertionError("a test tried to create a real LLM client")

    monkeypatch.setattr(client, "_get_client", boom)


@pytest.fixture
def fake_generate(monkeypatch):
    """Replace rag.pipeline.generate (the generator call) with a FakeGenerate."""
    from rag import pipeline

    fake = FakeGenerate()
    monkeypatch.setattr(pipeline, "generate", fake)
    return fake


@pytest.fixture
def fake_rewrite(monkeypatch):
    """Replace rag.rewriter.generate (the rewrite call) with a FakeGenerate."""
    from rag import rewriter

    fake = FakeGenerate(text="rewritten search query", model_latency_ms=700.0)
    monkeypatch.setattr(rewriter, "generate", fake)
    return fake


@pytest.fixture
def worker_factory(monkeypatch):
    """Replace rag.reranker._start_worker with a WorkerFactory and start with no workers."""
    factory = WorkerFactory()
    monkeypatch.setattr(reranker, "_WORKERS", {})
    monkeypatch.setattr(reranker, "_start_worker", factory)
    return factory


@pytest.fixture
def no_llm(monkeypatch):
    """Make any LLM call from the shared client or the judges fail loudly."""
    import llm.client
    from evalharness import judges

    def boom(*args, **kwargs):
        raise AssertionError("the comparison must not call an LLM")

    monkeypatch.setattr(llm.client, "generate", boom)
    monkeypatch.setattr(llm.client, "_call_api", boom)
    monkeypatch.setattr(judges, "generate", boom)


def write_pipeline_yaml(tmp_path, config, name="pipeline_step11.yaml"):
    """Write a pipeline config dict to a YAML file and return its path."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def set_path(config, dotted, value):
    """Set a nested key given as 'a.b.c'."""
    parts = dotted.split(".")
    node = config
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def del_path(config, dotted):
    """Delete a nested key given as 'a.b.c'."""
    parts = dotted.split(".")
    node = config
    for part in parts[:-1]:
        node = node[part]
    del node[parts[-1]]


# ================================================================ v0 unchanged

class TestV0Unchanged:
    def test_real_v0_config_loads_with_every_new_knob_off(self):
        """Spec: new configs/v0.yaml loads and every new key is off or inert (read-only real file)."""
        config = load_pipeline_config(REAL_V0)
        assert config["retrieval"]["method"] == "vector"
        assert config["retrieval"]["rerank"]["enabled"] is False
        assert config["query_rewrite"]["enabled"] is False
        assert config["generation"]["context_chunk_template"] == "{text}"

    def test_real_v0_config_has_the_spec_values_for_the_new_keys(self):
        """Spec Config changes: bm25 1.5 / 0.75, hybrid rrf 60 pool 10, rerank pool 10 (read-only)."""
        retrieval = load_pipeline_config(REAL_V0)["retrieval"]
        assert retrieval["top_k"] == 4
        assert retrieval["bm25"] == {"k1": 1.5, "b": 0.75}
        assert retrieval["hybrid"] == {"fusion": "rrf", "rrf_k": 60, "pool_k": 10}
        assert retrieval["rerank"]["pool_k"] == 10
        assert retrieval["rerank"]["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert retrieval["rerank"]["worker_timeout_s"] == 60

    def test_supported_methods_are_vector_bm25_hybrid(self):
        """Spec: SUPPORTED_METHODS = ("vector", "bm25", "hybrid")."""
        assert tuple(SUPPORTED_METHODS) == ("vector", "bm25", "hybrid")

    def test_build_prompt_default_chunk_template_gives_golden_v0_string(self):
        """Spec: build_prompt with the v0 template and "{text}" gives the pre-Step-11 string."""
        template = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        chunks = [{"text": "A text"}, {"text": "B text"}]
        prompt = build_prompt("Q?", chunks, template, "\n\n---\n\n")
        golden = "Context:\nA text\n\n---\n\nB text\n\nQuestion: Q?\n\nAnswer:"
        assert prompt == golden

    def test_build_prompt_explicit_text_template_equals_default(self):
        """Spec: chunk_template "{text}" (the v0 value) renders the same prompt as the default."""
        template = "{context} || {question}"
        chunks = [{"text": "one"}, {"text": "two"}]
        assert (build_prompt("Q", chunks, template, "|", "{text}")
                == build_prompt("Q", chunks, template, "|"))

    def test_build_prompt_real_v0_template_gives_golden_string(self):
        """Spec: the real v0 prompt template renders to the hand-written golden string (read-only)."""
        generation = load_pipeline_config(REAL_V0)["generation"]
        prompt = build_prompt("Q?", [{"text": "C1"}, {"text": "C2"}], generation["prompt_template"],
                              generation["context_separator"], generation["context_chunk_template"])
        golden = ("Use the following context to answer the question.\n\nContext:\nC1\n\n---\n\nC2"
                  "\n\nQuestion: Q?\n\nAnswer:\n")
        assert prompt == golden

    def test_answer_all_switches_off_returns_contract_keys_and_v0_prompt(
            self, pipeline_config, fake_embed, fake_generate, fake_rewrite, worker_factory):
        """Spec: with every switch off answer() has the contract keys, v0's prompt, no extra calls."""
        question = "How is salary paid each month?"
        out = answer(question, pipeline_config)

        assert set(out) == {"answer", "chunks", "latency_ms"}, "contract keys changed"
        assert out["answer"] == "the answer"
        assert isinstance(out["latency_ms"], float) and out["latency_ms"] >= 0
        assert len(out["chunks"]) == 3, "top_k chunks expected"
        for chunk in out["chunks"]:
            assert set(chunk) == {"chunk_id", "doc_id", "text", "score"}
        scores = [c["score"] for c in out["chunks"]]
        assert scores == sorted(scores, reverse=True), "chunks must be in rank order"

        expected_prompt = ("Context:\n" + "\n\n---\n\n".join(c["text"] for c in out["chunks"])
                           + "\n\nQuestion: " + question + "\n\nAnswer:")
        assert len(fake_generate.calls) == 1
        assert fake_generate.calls[0]["prompt"] == expected_prompt
        assert fake_rewrite.calls == [], "rewriter must not run when query_rewrite is off"
        assert worker_factory.workers == [], "reranker worker must not start when rerank is off"

    def test_answer_all_switches_off_chunks_equal_v0_retrieve(
            self, pipeline_config, fake_embed, fake_generate):
        """Spec: answer() with every switch off returns the same chunks the v0 retrieve() gives."""
        question = "laptop onboarding for a new intern"
        index, chunks = get_index(pipeline_config)
        expected = retrieve(question, index, chunks, 3, pipeline_config["embedding"])
        out = answer(question, pipeline_config)
        assert out["chunks"] == expected


# ================================================================ bm25

class TestTokenize:
    @pytest.mark.parametrize("text, expected", [
        ("B3: 8 CL/yr", ["b3", "8", "cl", "yr"]),
        ("Hello, World!", ["hello", "world"]),
        ("KestrelHub", ["kestrelhub"]),
        ("a_b-c", ["a", "b", "c"]),
        ("", []),
        ("   ...  ", []),
        ("Café Zürich 2026", ["café", "zürich", "2026"]),
    ])
    def test_tokenize_cases(self, text, expected):
        """Spec: tokenize lowercases and splits on runs of non-alphanumeric characters."""
        assert tokenize(text) == expected


def corpus_rows(texts):
    """Chunk dicts (chunk_id, doc_id, text) for BM25 tests."""
    return [{"chunk_id": f"d{i}#000", "doc_id": f"d{i}", "text": t, "score": 0.0}
            for i, t in enumerate(texts)]


BM25_CFG = {"k1": 1.5, "b": 0.75}


class TestBm25Search:
    def test_single_unique_term_returns_only_the_matching_chunk(self):
        """Spec: chunks with score <= 0 are never returned; one match gives one result."""
        chunks = corpus_rows(["kestrel leave policy interns", "salary is paid monthly",
                              "laptop issued at onboarding", "holiday calendar published yearly"])
        result = bm25_search("laptop onboarding", chunks, 4, BM25_CFG)
        assert [row for row, _ in result] == [2]
        assert result[0][1] > 0

    def test_query_with_no_shared_term_returns_empty_list(self):
        """Spec: a query with no shared term gives [] and not an arbitrary list."""
        chunks = corpus_rows(["alpha beta", "gamma delta", "epsilon zeta", "eta theta"])
        assert bm25_search("unrelated words only", chunks, 3, BM25_CFG) == []

    def test_higher_term_frequency_ranks_first_even_at_a_later_row(self):
        """Spec: BM25 rewards repeats; row 1 (laptop x3) beats row 0 (laptop x1), equal length."""
        chunks = corpus_rows(["laptop issued today now", "laptop laptop laptop issued",
                              "alpha beta gamma delta", "epsilon zeta eta theta",
                              "iota kappa lambda mu", "nu xi omicron pi"])
        result = bm25_search("laptop", chunks, 5, BM25_CFG)
        assert [row for row, _ in result] == [1, 0]
        assert result[0][1] > result[1][1]

    def test_term_frequency_ratio_matches_spec_formula(self):
        """Spec: f=2 vs f=1 at average length, k1 1.5 b 0.75: ratio (2*2.5/3.5) / (1*2.5/2.5) = 1.4286."""
        # All six chunks have 4 tokens, so |d| / avgdl = 1 for both matching chunks and the idf
        # (same term) cancels: ratio = (2 * (1.5 + 1) / (2 + 1.5)) / (1 * 2.5 / (1 + 1.5))
        #   = (5 / 3.5) / 1.0 = 1.428571
        chunks = corpus_rows(["laptop laptop xa xb", "laptop xc xd xe", "p1 p2 p3 p4",
                              "q1 q2 q3 q4", "r1 r2 r3 r4", "s1 s2 s3 s4"])
        result = dict(bm25_search("laptop", chunks, 6, BM25_CFG))
        assert result[0] / result[1] == pytest.approx(5 / 3.5, rel=1e-6)

    def test_equal_scores_break_ties_on_lower_row(self):
        """Spec: ties break on the lower chunk row, so the result is deterministic."""
        chunks = corpus_rows(["alpha beta gamma delta", "salary is paid monthly",
                              "laptop issued at onboarding", "holiday list posted yearly"])
        result = bm25_search("monthly laptop", chunks, 4, BM25_CFG)
        assert [row for row, _ in result] == [1, 2]
        assert result[0][1] == pytest.approx(result[1][1])

    def test_k_larger_than_matching_chunks_returns_only_matches(self):
        """Spec: k above the number of scoring chunks returns just the scoring ones."""
        chunks = corpus_rows(["laptop one", "laptop two", "gamma delta", "epsilon zeta",
                              "eta theta", "iota kappa"])
        result = bm25_search("laptop", chunks, 50, BM25_CFG)
        assert sorted(row for row, _ in result) == [0, 1]

    def test_k_cuts_the_result(self):
        """Spec: bm25_search returns the top-k only."""
        chunks = corpus_rows(["laptop one", "laptop laptop two", "gamma delta", "epsilon zeta",
                              "eta theta", "iota kappa"])
        result = bm25_search("laptop", chunks, 1, BM25_CFG)
        assert [row for row, _ in result] == [1]

    def test_same_inputs_give_same_result(self):
        """Spec: the result is deterministic."""
        chunks = corpus_rows(["laptop one", "laptop two", "gamma delta", "epsilon zeta",
                              "eta theta", "iota kappa"])
        assert (bm25_search("laptop", chunks, 5, BM25_CFG)
                == bm25_search("laptop", chunks, 5, BM25_CFG))


class TestRrfFuse:
    def test_worked_example_from_spec(self):
        """Spec: rrf_fuse([[7, 2], [2, 9]], 60) -> rows 2, 7, 9 with 1/61+1/62, 1/61, 1/62."""
        # row 2: rank 2 in list 1 (1/62) + rank 1 in list 2 (1/61) = 0.016129 + 0.016393 = 0.032522
        # row 7: rank 1 in list 1 only = 1/61 = 0.016393
        # row 9: rank 2 in list 2 only = 1/62 = 0.016129
        fused = rrf_fuse([[7, 2], [2, 9]], 60)
        assert [row for row, _ in fused] == [2, 7, 9]
        assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
        assert fused[1][1] == pytest.approx(1 / 61)
        assert fused[2][1] == pytest.approx(1 / 62)

    def test_single_ranking_keeps_order_with_reciprocal_scores(self):
        """Spec: a chunk in one ranking only scores 1 / (rrf_k + rank)."""
        fused = rrf_fuse([[5, 6, 7]], 60)
        assert [row for row, _ in fused] == [5, 6, 7]
        assert [s for _, s in fused] == pytest.approx([1 / 61, 1 / 62, 1 / 63])

    def test_spec_example_chunk_in_both_lists_beats_chunk_in_one(self):
        """Spec: A = vector rank 1 + BM25 rank 3 -> 1/61 + 1/63 = 0.03227; B = rank 2 only -> 0.01613."""
        fused = dict(rrf_fuse([["A", "B", "C"], ["X", "Y", "A"]], 60))
        assert fused["A"] == pytest.approx(1 / 61 + 1 / 63)
        assert fused["B"] == pytest.approx(1 / 62)

    def test_tie_breaks_on_best_rank_in_first_ranking(self):
        """Spec: equal fused scores go to the row with the better rank in the first ranking."""
        # rows 1 and 2 both score 1/61 + 1/62; row 1 is rank 1 in the first ranking
        assert [r for r, _ in rrf_fuse([[1, 2], [2, 1]], 60)] == [1, 2]
        assert [r for r, _ in rrf_fuse([[2, 1], [1, 2]], 60)] == [2, 1]

    def test_row_missing_from_first_ranking_loses_a_tie(self):
        """Spec: a tie where one row is absent from the first ranking goes to the row in it."""
        # rows 3 and 4 both score 1/61 (rank 1 in one list each)
        assert [r for r, _ in rrf_fuse([[3], [4]], 60)] == [3, 4]
        assert [r for r, _ in rrf_fuse([[4], [3]], 60)] == [4, 3]

    def test_remaining_ties_break_on_lower_row(self):
        """Spec: after the first-ranking rule, ties break on the lower row."""
        # rows 1, 8, 5 all score 1/61; row 1 is in the first ranking, 8 and 5 are not
        assert [r for r, _ in rrf_fuse([[1], [8], [5]], 60)] == [1, 5, 8]

    def test_rrf_k_changes_order_only_where_formula_says(self):
        """Spec: with rrf_k 1 a lone rank-1 row wins; with rrf_k 60 two rank-4 hits win."""
        # row 10: rank 1 in list 1 only -> 1 / (k + 1)
        # row 20: rank 4 in both lists   -> 2 / (k + 4)
        # k = 60: 1/61 = 0.0164 < 2/64 = 0.03125 -> row 20 first
        # k = 1:  1/2  = 0.5    > 2/5  = 0.4     -> row 10 first
        lists = [[10, 11, 12, 20], [30, 31, 32, 20]]
        assert rrf_fuse(lists, 60)[0][0] == 20
        top_k1 = [row for row, _ in rrf_fuse(lists, 1)]
        assert top_k1[0] == 10
        assert top_k1[1] == 30, "tie at 1/2 goes to the row ranked in the first list"
        assert top_k1[2] == 20


# ================================================================ retrieve_with_method

class TestRetrieveWithMethod:
    def test_bm25_ignores_the_embedder(self, pipeline_config, fake_embed):
        """Spec: method bm25 never calls the embedder; only chunks sharing the word come back."""
        pipeline_config["retrieval"]["method"] = "bm25"
        index, chunks = get_index(pipeline_config)
        fake_embed.calls.clear()

        hits = retrieve_with_method("monthly", index, chunks, pipeline_config)

        assert fake_embed.calls == [], "bm25 must not embed the query"
        assert 1 <= len(hits) <= 2, "only chunks that contain 'monthly' score above zero"
        assert {h["doc_id"] for h in hits} == {"payroll_policy"}
        assert hits[0]["chunk_id"] == "payroll_policy#000"
        assert all(set(h) == {"chunk_id", "doc_id", "text", "score"} for h in hits)
        assert all(h["score"] > 0 for h in hits)

    def test_vector_control_does_not_find_the_bm25_only_chunk(self, pipeline_config, fake_embed):
        """Spec control: the vocabulary fake cannot see 'monthly', so vector search misses payroll#000."""
        pipeline_config["retrieval"]["method"] = "vector"
        index, chunks = get_index(pipeline_config)
        hits = retrieve_with_method("monthly", index, chunks, pipeline_config)
        assert "payroll_policy#000" not in [h["chunk_id"] for h in hits]

    def test_hybrid_finds_chunk_only_bm25_can_see_and_returns_top_k(
            self, pipeline_config, fake_embed):
        """Spec: a chunk found by BM25 only (exact word) appears in hybrid; result length = top_k."""
        pipeline_config["retrieval"]["method"] = "hybrid"
        index, chunks = get_index(pipeline_config)
        fake_embed.calls.clear()

        hits = retrieve_with_method("monthly", index, chunks, pipeline_config)

        ids = [h["chunk_id"] for h in hits]
        assert "payroll_policy#000" in ids, f"BM25-only chunk missing from hybrid result {ids}"
        assert len(hits) == 3, "hybrid result must be cut to top_k"
        assert len(fake_embed.calls) == 1, "hybrid embeds the query once"
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True), "fused scores must be in rank order"

    def test_vector_method_equals_v0_retrieve(self, pipeline_config, fake_embed):
        """Spec: method vector is exactly the v0 retrieve()."""
        index, chunks = get_index(pipeline_config)
        question = "salary payroll question"
        expected = retrieve(question, index, chunks, 3, pipeline_config["embedding"])
        assert retrieve_with_method(question, index, chunks, pipeline_config) == expected

    @pytest.mark.parametrize("method", ["vector", "bm25", "hybrid"])
    def test_top_k_above_chunk_count_raises(self, pipeline_config, fake_embed, method):
        """Spec: top_k larger than the number of chunks raises (9 chunks, top_k 10)."""
        pipeline_config["retrieval"]["method"] = method
        index, chunks = get_index(pipeline_config)
        assert len(chunks) == 9
        pipeline_config["retrieval"]["top_k"] = 10
        pipeline_config["retrieval"]["hybrid"]["pool_k"] = 10
        with pytest.raises(ValueError):
            retrieve_with_method("salary", index, chunks, pipeline_config)

    @pytest.mark.parametrize("method", ["vector", "hybrid"])
    def test_rerank_on_returns_the_rerank_pool_size(self, pipeline_config, fake_embed, method):
        """Spec: with reranking on, retrieval returns rerank.pool_k chunks (5), not top_k (3)."""
        pipeline_config["retrieval"]["method"] = method
        pipeline_config["retrieval"]["rerank"]["enabled"] = True
        index, chunks = get_index(pipeline_config)
        hits = retrieve_with_method("salary payroll", index, chunks, pipeline_config)
        assert len(hits) == 5

    def test_rerank_pool_larger_than_chunk_count_is_capped(self, pipeline_config, fake_embed):
        """Spec: a pool larger than the chunk count is capped at the chunk count (9)."""
        pipeline_config["retrieval"]["rerank"]["enabled"] = True
        pipeline_config["retrieval"]["rerank"]["pool_k"] = 50
        index, chunks = get_index(pipeline_config)
        hits = retrieve_with_method("salary payroll", index, chunks, pipeline_config)
        assert len(hits) == 9


# ================================================================ rerank

def pool_of(n=5):
    """n candidate chunks with descending first-stage scores."""
    return [{"chunk_id": f"c{i}", "doc_id": f"d{i}", "text": f"text {i}", "score": 0.9 - 0.1 * i}
            for i in range(n)]


RERANK_CFG = {"enabled": True, "pool_k": 5, "model": "fake-rerank", "revision": None,
              "worker_timeout_s": 7}


class TestRerank:
    def test_candidates_are_reordered_by_worker_score_and_cut_to_top_k(self, worker_factory):
        """Spec: top_k candidates re-ordered by cross-encoder score, score field replaced."""
        # scores per candidate [c0..c4] = [0.1, 0.9, 0.5, 0.7, 0.3]
        # descending: c1 0.9, c3 0.7, c2 0.5, c4 0.3, c0 0.1 -> top 3 = c1, c3, c2
        worker_factory.queue = [FakeWorker(scores_fn=lambda texts: [0.1, 0.9, 0.5, 0.7, 0.3])]
        out = rerank("What is the leave policy?", pool_of(), 3, RERANK_CFG)
        assert [c["chunk_id"] for c in out] == ["c1", "c3", "c2"]
        assert [c["score"] for c in out] == pytest.approx([0.9, 0.7, 0.5])
        assert [c["text"] for c in out] == ["text 1", "text 3", "text 2"]
        assert all(set(c) == {"chunk_id", "doc_id", "text", "score"} for c in out)

    def test_worker_gets_question_texts_in_pool_order_and_the_timeout(self, worker_factory):
        """Spec: the request is {question, texts}; the timeout comes from worker_timeout_s."""
        rerank("Original question?", pool_of(), 3, RERANK_CFG)
        worker = worker_factory.workers[0]
        assert worker.requests == [{"question": "Original question?",
                                    "texts": [f"text {i}" for i in range(5)]}]
        assert worker.timeouts == [7]

    def test_worker_is_started_once_for_two_calls(self, worker_factory):
        """Spec: the worker loads the model once and is reused."""
        rerank("q1?", pool_of(), 3, RERANK_CFG)
        rerank("q2?", pool_of(), 3, RERANK_CFG)
        assert len(worker_factory.workers) == 1
        assert len(worker_factory.workers[0].requests) == 2

    @pytest.mark.parametrize("scores", [[0.1, 0.2], [0.1] * 6, []])
    def test_wrong_score_count_raises_rerank_error(self, worker_factory, scores):
        """Spec: a wrong-length reply raises RerankError, never a silent fallback."""
        worker_factory.queue = [FakeWorker(reply={"scores": scores})]
        with pytest.raises(RerankError):
            rerank("q?", pool_of(), 3, RERANK_CFG)

    @pytest.mark.parametrize("message", ["rerank worker exited during a request",
                                         "rerank worker gave no reply within 7s"])
    def test_worker_exit_or_timeout_raises_rerank_error(self, worker_factory, message):
        """Spec: a worker that dies or times out raises RerankError (a failed item)."""
        worker_factory.queue = [FakeWorker(error=RerankError(message))]
        with pytest.raises(RerankError, match="rerank worker"):
            rerank("q?", pool_of(), 3, RERANK_CFG)

    def test_failed_worker_is_dropped_and_a_new_one_started(self, worker_factory):
        """Spec: no silent fallback; after a failure the next call uses a fresh worker."""
        worker_factory.queue = [FakeWorker(error=RerankError("rerank worker exited")),
                                FakeWorker()]
        with pytest.raises(RerankError):
            rerank("q?", pool_of(), 3, RERANK_CFG)
        out = rerank("q?", pool_of(), 3, RERANK_CFG)
        assert len(worker_factory.workers) == 2, "a second worker should have been started"
        assert len(worker_factory.workers[1].requests) == 1
        assert len(out) == 3

    def test_answer_sends_original_question_and_whole_pool_to_worker(
            self, pipeline_config, fake_embed, fake_generate, fake_rewrite, worker_factory):
        """Spec: rerank always sees the ORIGINAL question, even when rewriting is on; pool = pool_k."""
        pipeline_config["retrieval"]["rerank"]["enabled"] = True
        pipeline_config["query_rewrite"]["enabled"] = True
        question = "When do employees get paid?"
        answer(question, pipeline_config)
        request = worker_factory.workers[0].requests[0]
        assert request["question"] == question
        assert len(request["texts"]) == 5, "pool_k texts expected"

    def test_answer_pool_is_capped_at_chunk_count(
            self, pipeline_config, fake_embed, fake_generate, worker_factory):
        """Spec: a rerank pool of 50 on a 9-chunk corpus sends 9 candidates."""
        pipeline_config["retrieval"]["rerank"]["enabled"] = True
        pipeline_config["retrieval"]["rerank"]["pool_k"] = 50
        answer("salary payroll?", pipeline_config)
        assert len(worker_factory.workers[0].requests[0]["texts"]) == 9

    def test_answer_returns_top_k_reranked_chunks_and_prompt_uses_them(
            self, pipeline_config, fake_embed, fake_generate, worker_factory):
        """Spec: answer() keeps top_k of the pool in worker-score order and builds the prompt from them."""
        pipeline_config["retrieval"]["rerank"]["enabled"] = True
        question = "salary payroll policy"
        index, chunks = get_index(pipeline_config)
        pool = retrieve(question, index, chunks, 5, pipeline_config["embedding"])
        # worker scores 0, 1, 2, 3, 4 for the pool in order: best = last candidate
        out = answer(question, pipeline_config)
        assert [c["chunk_id"] for c in out["chunks"]] == [pool[4]["chunk_id"], pool[3]["chunk_id"],
                                                          pool[2]["chunk_id"]]
        assert [c["score"] for c in out["chunks"]] == pytest.approx([4.0, 3.0, 2.0])
        expected_context = "\n\n---\n\n".join(c["text"] for c in out["chunks"])
        assert fake_generate.calls[0]["prompt"] == (
            "Context:\n" + expected_context + "\n\nQuestion: " + question + "\n\nAnswer:")

    def test_answer_rerank_failure_propagates_and_generator_is_not_called(
            self, pipeline_config, fake_embed, fake_generate, worker_factory):
        """Spec: a failed rerank is a failed item, never a fallback to the first-stage order."""
        pipeline_config["retrieval"]["rerank"]["enabled"] = True
        worker_factory.queue = [FakeWorker(error=RerankError("rerank worker exited"))]
        with pytest.raises(RerankError):
            answer("salary?", pipeline_config)
        assert fake_generate.calls == []

    def test_describe_system_names_reranker_only_when_enabled(
            self, pipeline_config, monkeypatch, fake_embed):
        """Spec: describe_system returns reranker_model and reranker_revision only when reranking is on."""
        from rag import pipeline

        pipeline_config["generation"]["model_digest"] = "digest-x"
        monkeypatch.setattr(pipeline, "get_model_digest", lambda model: "digest-x")

        off = describe_system(pipeline_config)
        assert "reranker_model" not in off and "reranker_revision" not in off

        pipeline_config["retrieval"]["rerank"]["enabled"] = True
        pipeline_config["retrieval"]["rerank"]["revision"] = "rev123"
        on = describe_system(pipeline_config)
        assert on["reranker_model"] == "fake-rerank"
        assert on["reranker_revision"] == "rev123"


# ================================================================ rewrite

class TestRewrite:
    def test_prompt_contains_the_question_and_generation_settings_are_used(
            self, pipeline_config, fake_rewrite):
        """Spec: the rewrite prompt holds the question; model, temperature and seed come from generation."""
        query, latency = rewrite_query("How many leaves do interns get?", pipeline_config)
        call = fake_rewrite.calls[0]
        assert "How many leaves do interns get?" in call["prompt"]
        assert call["model"] == "fake-gen:cloud"
        assert call["temperature"] == 0.0
        assert call["seed"] == 42
        assert query == "rewritten search query"
        assert latency == pytest.approx(700.0)

    @pytest.mark.parametrize("reply, expected", [
        ('"monthly salary date"', "monthly salary date"),
        ("'monthly salary date'", "monthly salary date"),
        ('  "monthly salary date"  \n', "monthly salary date"),
        ("  plain query  ", "plain query"),
        ('""nested""', '"nested"'),
        ('say "hi" now', 'say "hi" now'),
    ])
    def test_reply_is_stripped_of_whitespace_and_one_pair_of_quotes(
            self, pipeline_config, fake_rewrite, reply, expected):
        """Spec: the reply is stripped of whitespace and one pair of surrounding quotes."""
        fake_rewrite.text = reply
        query, _ = rewrite_query("q?", pipeline_config)
        assert query == expected

    @pytest.mark.parametrize("reply", ["", "   ", '""', "\n\t"])
    def test_blank_reply_raises(self, pipeline_config, fake_rewrite, reply):
        """Spec: a blank reply raises; there is no fallback to the original question."""
        fake_rewrite.text = reply
        with pytest.raises(Exception):
            rewrite_query("q?", pipeline_config)

    def test_answer_searches_with_rewrite_but_generator_gets_original_question(
            self, pipeline_config, fake_embed, fake_generate, fake_rewrite):
        """Spec: only retrieval sees the rewrite; the generator prompt holds the original question."""
        fake_rewrite.text = "RWQUERY monthly salary"
        pipeline_config["query_rewrite"]["enabled"] = True
        question = "When do I get my money?"
        answer(question, pipeline_config)

        assert "search_query: RWQUERY monthly salary" in fake_embed.texts
        assert "search_query: " + question not in fake_embed.texts
        prompt = fake_generate.calls[0]["prompt"]
        assert question in prompt
        assert "RWQUERY" not in prompt

    def test_answer_latency_includes_rewrite_model_latency(
            self, pipeline_config, fake_embed, fake_generate, fake_rewrite):
        """Spec: latency_ms = retrieval time + rewrite model_latency_ms + generator model_latency_ms."""
        # rewrite 700.0 + generator 300.0 = 1000.0, plus a small non-negative retrieval time
        pipeline_config["query_rewrite"]["enabled"] = True
        out = answer("When is salary paid?", pipeline_config)
        assert 1000.0 <= out["latency_ms"] < 1000.0 + 5000.0

    def test_answer_blank_rewrite_fails_and_generator_is_not_called(
            self, pipeline_config, fake_embed, fake_generate, fake_rewrite):
        """Spec: a blank rewrite is a failed item; no silent fallback to the original question."""
        pipeline_config["query_rewrite"]["enabled"] = True
        fake_rewrite.text = "   "
        with pytest.raises(Exception):
            answer("When is salary paid?", pipeline_config)
        assert fake_generate.calls == []

    def test_answer_rewrite_call_error_propagates(
            self, pipeline_config, fake_embed, fake_generate, fake_rewrite):
        """Spec: a failed rewrite call surfaces as an error; the generator is never called."""
        pipeline_config["query_rewrite"]["enabled"] = True
        fake_rewrite.error = RuntimeError("rewrite model down")
        with pytest.raises(RuntimeError, match="rewrite model down"):
            answer("When is salary paid?", pipeline_config)
        assert fake_generate.calls == []


# ================================================================ chunk template

class TestChunkTemplate:
    def test_rank_is_one_based_and_text_is_inserted(self):
        """Spec: build_prompt renders each chunk with {rank} (1-based) and {text} before joining."""
        chunks = [{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}]
        prompt = build_prompt("Q?", chunks, "{context}\nQ: {question}", "\n", "[{rank}] {text}")
        assert prompt == "[1] alpha\n[2] beta\n[3] gamma\nQ: Q?"

    def test_braces_inside_chunk_text_are_not_parsed(self):
        """Spec: chunk text with braces is inserted as a value, not parsed as a template."""
        chunks = [{"text": "use {curly} and {0}"}]
        prompt = build_prompt("Q?", chunks, "{context}|{question}", "|", "[{rank}] {text}")
        assert prompt == "[1] use {curly} and {0}|Q?"

    def test_answer_uses_context_chunk_template_in_the_generator_prompt(
            self, pipeline_config, fake_embed, fake_generate):
        """Spec: answer() passes generation.context_chunk_template into the prompt."""
        pipeline_config["generation"]["context_chunk_template"] = "[{rank}] {text}"
        out = answer("laptop onboarding?", pipeline_config)
        context = "\n\n---\n\n".join(f"[{rank}] {c['text']}"
                                     for rank, c in enumerate(out["chunks"], start=1))
        assert fake_generate.calls[0]["prompt"] == (
            "Context:\n" + context + "\n\nQuestion: laptop onboarding?\n\nAnswer:")


# ================================================================ config validation

NEW_KEYS = [
    "retrieval.bm25", "retrieval.bm25.k1", "retrieval.bm25.b",
    "retrieval.hybrid", "retrieval.hybrid.fusion", "retrieval.hybrid.rrf_k",
    "retrieval.hybrid.pool_k",
    "retrieval.rerank", "retrieval.rerank.enabled", "retrieval.rerank.pool_k",
    "retrieval.rerank.model", "retrieval.rerank.revision", "retrieval.rerank.worker_timeout_s",
    "query_rewrite", "query_rewrite.enabled", "query_rewrite.prompt_template",
    "generation.context_chunk_template",
]


class TestConfigValidation:
    def test_valid_config_with_all_new_keys_loads(self, pipeline_config, tmp_path):
        """Spec: a config holding every new key loads (control for the failure tests)."""
        config = load_pipeline_config(write_pipeline_yaml(tmp_path, pipeline_config))
        assert config["retrieval"]["hybrid"]["rrf_k"] == 60
        assert config["retrieval"]["rerank"]["enabled"] is False

    @pytest.mark.parametrize("dotted", NEW_KEYS)
    def test_missing_new_key_names_the_key(self, pipeline_config, tmp_path, dotted):
        """Spec: every new key removed in turn raises a ValueError that names the key."""
        del_path(pipeline_config, dotted)
        with pytest.raises(ValueError, match=re.escape(dotted)):
            load_pipeline_config(write_pipeline_yaml(tmp_path, pipeline_config))

    @pytest.mark.parametrize("dotted, value", [
        ("retrieval.bm25.k1", 0),
        ("retrieval.bm25.k1", -1.5),
        ("retrieval.bm25.k1", "1.5"),
        ("retrieval.bm25.b", 1.5),
        ("retrieval.bm25.b", -0.1),
        ("retrieval.hybrid.fusion", "weighted"),
        ("retrieval.hybrid.rrf_k", 0),
        ("retrieval.hybrid.rrf_k", 60.5),
        ("retrieval.hybrid.pool_k", 2),
        ("retrieval.rerank.pool_k", 2),
        ("retrieval.rerank.enabled", "yes"),
        ("retrieval.rerank.enabled", 1),
        ("retrieval.rerank.worker_timeout_s", 0),
        ("query_rewrite.enabled", "true"),
        ("query_rewrite.enabled", 1),
        ("query_rewrite.prompt_template", "no placeholder at all"),
        ("query_rewrite.prompt_template", "Rewrite {question} and {}"),
        ("query_rewrite.prompt_template", "Rewrite {question} and {other}"),
        ("generation.context_chunk_template", "no text field"),
        ("generation.context_chunk_template", "{rank} only"),
        ("generation.context_chunk_template", "{text} and {}"),
        ("generation.context_chunk_template", "{text} and {other}"),
        ("retrieval.method", "semantic"),
    ])
    def test_bad_value_raises_value_error_naming_the_key(
            self, pipeline_config, tmp_path, dotted, value):
        """Spec: bad values (k1 0, b 1.5, pool_k < top_k, non-bool enabled, stray braces) fail loudly."""
        set_path(pipeline_config, dotted, value)
        with pytest.raises(ValueError, match=re.escape(dotted)):
            load_pipeline_config(write_pipeline_yaml(tmp_path, pipeline_config))

    @pytest.mark.parametrize("dotted, value", [
        ("retrieval.bm25.k1", 0.1),
        ("retrieval.bm25.b", 0),
        ("retrieval.bm25.b", 1),
        ("retrieval.hybrid.rrf_k", 1),
        ("retrieval.hybrid.pool_k", 3),
        ("retrieval.rerank.pool_k", 3),
        ("retrieval.rerank.worker_timeout_s", 1),
        ("retrieval.method", "bm25"),
        ("retrieval.method", "hybrid"),
        ("generation.context_chunk_template", "[{rank}] {text}"),
        ("query_rewrite.prompt_template", "Rewrite {{literal}} {question}"),
    ])
    def test_boundary_and_valid_values_load(self, pipeline_config, tmp_path, dotted, value):
        """Spec: values at the stated boundaries (pool_k == top_k, b 0 and 1, rrf_k 1) are accepted."""
        set_path(pipeline_config, dotted, value)
        config = load_pipeline_config(write_pipeline_yaml(tmp_path, pipeline_config))
        assert config is not None

    @pytest.mark.parametrize("name", V1_CONFIGS)
    def test_every_real_v1_config_loads(self, name):
        """Spec: every real configs/v1_*.yaml loads with load_pipeline_config (read-only)."""
        config = load_pipeline_config(CONFIGS / f"{name}.yaml")
        assert config["name"] == name


# ================================================================ experiments config block

class TestExperimentsConfig:
    def test_real_harness_experiments_block_has_the_spec_values(self):
        """Spec Config changes: the experiments block of configs/harness.yaml (read-only)."""
        harness = yaml.safe_load(REAL_HARNESS.read_text(encoding="utf-8"))
        cfg = load_experiments_config(harness)
        assert cfg["base_config"] == "configs/v0.yaml"
        assert list(cfg["ignore_keys"]) == ["name", "experiment"]
        assert cfg["n_bootstrap"] == 2000
        assert cfg["bootstrap_seed"] == 2029
        assert cfg["table_path"] == "experiments/experiment_table.md"
        assert (PROJECT_ROOT / cfg["base_config"]).exists()

    def _block(self):
        harness = yaml.safe_load(REAL_HARNESS.read_text(encoding="utf-8"))
        return {"experiments": copy.deepcopy(harness["experiments"])}

    @pytest.mark.parametrize("key", ["base_config", "ignore_keys", "n_bootstrap",
                                     "bootstrap_seed", "table_path"])
    def test_missing_key_raises_value_error_naming_it(self, key):
        """Spec: each experiments key is required; the error names the key."""
        harness = self._block()
        del harness["experiments"][key]
        with pytest.raises(ValueError, match=key):
            load_experiments_config(harness)

    def test_missing_block_raises(self):
        """Spec: a harness config without an experiments block is refused."""
        with pytest.raises(ValueError, match="experiments"):
            load_experiments_config({})

    @pytest.mark.parametrize("key, value", [
        ("n_bootstrap", 0),
        ("n_bootstrap", "2000"),
        ("bootstrap_seed", "2029"),
        ("ignore_keys", "name"),
        ("base_config", ""),
        ("table_path", 5),
    ])
    def test_bad_value_raises_value_error(self, key, value):
        """Spec: bad experiments values (zero resamples, string list, empty path) are refused."""
        harness = self._block()
        harness["experiments"][key] = value
        with pytest.raises(ValueError, match=key):
            load_experiments_config(harness)


# ================================================================ flatten / config_diff

BASE_CONFIG = {
    "name": "base",
    "chunking": {"chunk_size": 150, "chunk_overlap": 30},
    "retrieval": {"method": "vector", "top_k": 4, "bm25": {"k1": 1.5, "b": 0.75}},
    "phrases": ["a", "b"],
}


class TestFlattenAndDiff:
    def test_flatten_nested_dict_gives_dotted_keys_and_lists_stay_leaves(self):
        """Spec: flatten maps nested dicts to dotted keys; lists are leaves."""
        flat = flatten(BASE_CONFIG)
        assert flat["chunking.chunk_size"] == 150
        assert flat["retrieval.bm25.k1"] == 1.5
        assert flat["phrases"] == ["a", "b"]
        assert flat["name"] == "base"
        assert "retrieval" not in flat and "retrieval.bm25" not in flat

    def test_identical_configs_have_empty_diff(self):
        """Spec: identical configs give []."""
        assert config_diff(BASE_CONFIG, copy.deepcopy(BASE_CONFIG), IGNORE) == []

    def test_name_and_experiment_are_ignored(self):
        """Spec: top-level `name` and `experiment` never count as a change."""
        exp = copy.deepcopy(BASE_CONFIG)
        exp["name"] = "other"
        exp["experiment"] = {"variable": "x"}
        assert config_diff(BASE_CONFIG, exp, IGNORE) == []

    def test_one_changed_value_is_reported_by_dotted_key(self):
        """Spec: a changed value is reported as its dotted key."""
        exp = copy.deepcopy(BASE_CONFIG)
        exp["chunking"]["chunk_size"] = 75
        assert config_diff(BASE_CONFIG, exp, IGNORE) == ["chunking.chunk_size"]

    def test_key_on_one_side_only_is_reported_from_either_side(self):
        """Spec: a key present on one side only counts as a difference."""
        exp = copy.deepcopy(BASE_CONFIG)
        exp["retrieval"]["extra"] = 1
        assert config_diff(BASE_CONFIG, exp, IGNORE) == ["retrieval.extra"]
        assert config_diff(exp, BASE_CONFIG, IGNORE) == ["retrieval.extra"]

    def test_diff_is_sorted(self):
        """Spec: config_diff returns sorted dotted keys."""
        exp = copy.deepcopy(BASE_CONFIG)
        exp["retrieval"]["top_k"] = 2
        exp["chunking"]["chunk_overlap"] = 0
        assert config_diff(BASE_CONFIG, exp, IGNORE) == ["chunking.chunk_overlap",
                                                         "retrieval.top_k"]

    def test_changed_list_value_is_one_leaf_difference(self):
        """Spec: lists are leaves, so a changed list is reported once under its key."""
        exp = copy.deepcopy(BASE_CONFIG)
        exp["phrases"] = ["a", "b", "c"]
        assert config_diff(BASE_CONFIG, exp, IGNORE) == ["phrases"]

    def test_ignore_applies_to_top_level_keys_only(self):
        """Spec: `ignore` are top-level keys; a nested key called `name` still counts."""
        base = {"name": "x", "corpus": {"name": "docs"}}
        exp = {"name": "y", "corpus": {"name": "other"}}
        assert config_diff(base, exp, IGNORE) == ["corpus.name"]


# ================================================================ check_experiment

def make_pair(tmp_path, *, changes=None, changed_keys=None, block=None, drop_block=False,
              base_ref=None):
    """Write a base YAML and an experiment YAML (base + changes + experiment block) in tmp_path."""
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(BASE_CONFIG), encoding="utf-8")
    exp = copy.deepcopy(BASE_CONFIG)
    exp["name"] = "exp"
    for dotted, value in (changes or {}).items():
        set_path(exp, dotted, value)
    if not drop_block:
        exp["experiment"] = {"base": base_ref or str(base_path), "variable": "chunk size",
                             "changed_keys": changed_keys or [], "hypothesis": "smaller is better"}
        exp["experiment"].update(block or {})
    exp_path = tmp_path / "exp.yaml"
    exp_path.write_text(yaml.safe_dump(exp), encoding="utf-8")
    return base_path, exp_path


class TestCheckExperiment:
    def test_declared_change_equal_to_real_diff_passes(self, tmp_path):
        """Spec: config_diff equal to changed_keys and a matching base gives []."""
        base, exp = make_pair(tmp_path, changes={"chunking.chunk_size": 75},
                              changed_keys=["chunking.chunk_size"])
        assert check_experiment(base, exp, IGNORE) == []

    def test_undeclared_extra_change_is_a_problem(self, tmp_path):
        """Spec: a changed key missing from changed_keys is reported."""
        base, exp = make_pair(tmp_path, changes={"chunking.chunk_size": 75, "retrieval.top_k": 2},
                              changed_keys=["chunking.chunk_size"])
        problems = check_experiment(base, exp, IGNORE)
        assert problems, "an undeclared change must be reported"
        assert all(isinstance(p, str) for p in problems)

    def test_declared_key_that_did_not_change_is_a_problem(self, tmp_path):
        """Spec: declared keys must equal the real diff exactly."""
        base, exp = make_pair(tmp_path, changes={"chunking.chunk_size": 75},
                              changed_keys=["chunking.chunk_size", "retrieval.top_k"])
        assert check_experiment(base, exp, IGNORE)

    def test_identical_config_with_declared_change_is_a_problem(self, tmp_path):
        """Spec: an experiment equal to the base but declaring a change fails."""
        base, exp = make_pair(tmp_path, changed_keys=["chunking.chunk_size"])
        assert check_experiment(base, exp, IGNORE)

    def test_wrong_base_is_a_problem(self, tmp_path):
        """Spec: experiment.base must be the file passed in."""
        base, exp = make_pair(tmp_path, changes={"chunking.chunk_size": 75},
                              changed_keys=["chunking.chunk_size"],
                              base_ref=str(tmp_path / "another_base.yaml"))
        assert check_experiment(base, exp, IGNORE)

    def test_missing_experiment_block_is_a_problem(self, tmp_path):
        """Spec: the experiment YAML must carry an experiment block."""
        base, exp = make_pair(tmp_path, changes={"chunking.chunk_size": 75}, drop_block=True)
        assert check_experiment(base, exp, IGNORE)

    @pytest.mark.parametrize("missing", ["variable", "changed_keys", "hypothesis", "base"])
    def test_block_missing_a_field_is_a_problem(self, tmp_path, missing):
        """Spec: the block needs base, variable, changed_keys and hypothesis."""
        base, exp = make_pair(tmp_path, changes={"chunking.chunk_size": 75},
                              changed_keys=["chunking.chunk_size"])
        data = yaml.safe_load(exp.read_text(encoding="utf-8"))
        del data["experiment"][missing]
        exp.write_text(yaml.safe_dump(data), encoding="utf-8")
        assert check_experiment(base, exp, IGNORE)

    def test_two_changes_under_one_variable_pass_only_when_both_listed(self, tmp_path):
        """Spec: one feature spread over two keys passes only if both keys are listed."""
        changes = {"chunking.chunk_size": 75, "retrieval.top_k": 2}
        base, both = make_pair(tmp_path, changes=changes,
                               changed_keys=["chunking.chunk_size", "retrieval.top_k"])
        assert check_experiment(base, both, IGNORE) == []
        base, one = make_pair(tmp_path, changes=changes, changed_keys=["chunking.chunk_size"])
        assert check_experiment(base, one, IGNORE)

    def test_key_added_only_in_experiment_must_be_declared(self, tmp_path):
        """Spec: a key present on one side only is a change and must be declared."""
        base, declared = make_pair(tmp_path, changes={"retrieval.extra": 1},
                                   changed_keys=["retrieval.extra"])
        assert check_experiment(base, declared, IGNORE) == []
        base, undeclared = make_pair(tmp_path, changes={"retrieval.extra": 1},
                                     changed_keys=["chunking.chunk_size"])
        assert check_experiment(base, undeclared, IGNORE)


class TestRealExperimentConfigs:
    def test_the_twelve_v1_configs_exist(self):
        """Spec: the 12 experiment configs listed in the table exist (read-only)."""
        found = sorted(p.stem for p in CONFIGS.glob("v1_*.yaml"))
        assert found == sorted(V1_CONFIGS)

    @pytest.mark.parametrize("name", V1_CONFIGS)
    def test_each_v1_config_passes_check_against_v0(self, name):
        """Spec: every real configs/v1_*.yaml is exactly its declared diff against v0 (read-only)."""
        problems = check_experiment(REAL_V0, CONFIGS / f"{name}.yaml", IGNORE)
        assert problems == [], f"{name}: {problems}"

    @pytest.mark.parametrize("name, expected", [
        ("v1_seed43", {"generation.seed": 43}),
        ("v1_chunk75", {"chunking.chunk_size": 75}),
        ("v1_chunk300", {"chunking.chunk_size": 300}),
        ("v1_overlap0", {"chunking.chunk_overlap": 0}),
        ("v1_topk2", {"retrieval.top_k": 2}),
        ("v1_topk6", {"retrieval.top_k": 6}),
        ("v1_hybrid", {"retrieval.method": "hybrid"}),
        ("v1_rerank", {"retrieval.rerank.enabled": True}),
        ("v1_rewrite", {"query_rewrite.enabled": True}),
    ])
    def test_scalar_experiments_change_exactly_the_spec_key_and_value(self, name, expected):
        """Spec table: each scalar experiment changes exactly one key to the listed value (read-only)."""
        base = yaml.safe_load(REAL_V0.read_text(encoding="utf-8"))
        exp = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))
        assert config_diff(base, exp, IGNORE) == sorted(expected)
        flat = flatten(exp)
        for key, value in expected.items():
            assert flat[key] == value

    @pytest.mark.parametrize("name, keys", [
        ("v1_prompt_grounded", ["generation.prompt_template"]),
        ("v1_prompt_citations", ["generation.context_chunk_template",
                                 "generation.prompt_template"]),
        ("v1_prompt_refusal", ["generation.prompt_template"]),
    ])
    def test_prompt_experiments_change_only_the_prompt_keys(self, name, keys):
        """Spec table: prompt experiments change prompt_template (and chunk template for citations)."""
        base = yaml.safe_load(REAL_V0.read_text(encoding="utf-8"))
        exp = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))
        assert config_diff(base, exp, IGNORE) == keys

    def test_citation_config_numbers_the_context_chunks(self):
        """Spec Decision 5: the citation run uses context_chunk_template "[{rank}] {text}" (read-only)."""
        exp = yaml.safe_load((CONFIGS / "v1_prompt_citations.yaml").read_text(encoding="utf-8"))
        assert exp["generation"]["context_chunk_template"] == "[{rank}] {text}"

    def test_refusal_prompt_contains_a_phrase_the_refusal_check_recognises(self):
        """Spec: the refusal sentence contains "does not contain" (in the Step 6 phrase list)."""
        exp = yaml.safe_load((CONFIGS / "v1_prompt_refusal.yaml").read_text(encoding="utf-8"))
        assert "does not contain" in exp["generation"]["prompt_template"]

    @pytest.mark.parametrize("name", V1_CONFIGS)
    def test_each_v1_config_declares_variable_and_hypothesis(self, name):
        """Spec: the hypothesis is written with the config; variable and hypothesis are non-empty."""
        exp = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))
        block = exp["experiment"]
        assert block["variable"].strip()
        assert block["hypothesis"].strip()

    @pytest.mark.parametrize("name", V1_CONFIGS)
    def test_v1_configs_name_no_query_id(self, name):
        """Spec: experiment configs never name a held-out query id (no q### at all; read-only)."""
        text = (CONFIGS / f"{name}.yaml").read_text(encoding="utf-8")
        assert re.search(r"\bq\d{3}\b", text) is None


# ================================================================ statistics

class TestMcNemarExact:
    @pytest.mark.parametrize("improved, regressed, expected", [
        # n = 5, k = 4: P(X >= 4) = (5 + 1) / 32 = 0.1875; p = 0.375
        (4, 1, 0.375),
        # n = 5, k = 5: P(X >= 5) = 1/32; p = 2/32 = 0.0625
        (5, 0, 0.0625),
        # n = 6, k = 6: 1/64; p = 2/64 = 0.03125
        (6, 0, 0.03125),
        # n = 8, k = 7: P(X >= 7) = (8 + 1) / 256; p = 18/256 = 0.0703125
        (7, 1, 0.0703125),
        # n = 7, k = 7: 1/128; p = 2/128 = 0.015625
        (7, 0, 0.015625),
        # n = 6, k = 3: tied split, p capped at 1
        (3, 3, 1.0),
        # direction does not matter: 1 improved, 4 regressed = same as 4 / 1
        (1, 4, 0.375),
    ])
    def test_two_sided_exact_p_values(self, improved, regressed, expected):
        """Spec: two-sided exact McNemar p = 2 * P(X >= max(k, n - k)), X ~ Binomial(n, 0.5)."""
        assert mcnemar_exact(improved, regressed) == pytest.approx(expected)

    def test_no_flips_returns_none(self):
        """Spec: None when there are no flips."""
        assert mcnemar_exact(0, 0) is None


class TestPairedBootstrap:
    def test_constant_difference_gives_degenerate_interval(self):
        """Spec: a constant difference gives lo = hi = mean."""
        out = paired_bootstrap_ci([0.25] * 6, 200, 1, 0.95)
        low, high = bounds(out)
        assert out["n"] == 6
        assert out["mean"] == pytest.approx(0.25)
        assert low == pytest.approx(0.25)
        assert high == pytest.approx(0.25)

    def test_same_seed_gives_same_interval(self):
        """Spec: seeded resampling; the same seed gives the same interval."""
        diffs = [0.5, 0.4, -0.5, 0.25, 0, 0, 0, 0]
        first = paired_bootstrap_ci(diffs, 500, 2029, 0.95)
        second = paired_bootstrap_ci(diffs, 500, 2029, 0.95)
        assert first == second

    def test_mean_matches_hand_value_and_interval_is_ordered_within_data_range(self):
        """Spec: mean of the differences; interval ordered and inside [min, max] of the data."""
        # (0.5 + 0.4 - 0.5 + 0.25 + 0 + 0 + 0 + 0) / 8 = 0.65 / 8 = 0.08125
        diffs = [0.5, 0.4, -0.5, 0.25, 0, 0, 0, 0]
        out = paired_bootstrap_ci(diffs, 2000, 2029, 0.95)
        low, high = bounds(out)
        assert out["n"] == 8
        assert out["mean"] == pytest.approx(0.08125)
        assert low <= out["mean"] <= high
        assert -0.5 <= low and high <= 0.5

    def test_lower_confidence_gives_a_narrower_interval(self):
        """Spec: the confidence argument sets the percentile cut; 50% is no wider than 95%."""
        diffs = [0.5, 0.4, -0.5, 0.25, 0, 0, 0, 0]
        low95, high95 = bounds(paired_bootstrap_ci(diffs, 2000, 5, 0.95))
        low50, high50 = bounds(paired_bootstrap_ci(diffs, 2000, 5, 0.50))
        assert (high50 - low50) <= (high95 - low95) + 1e-12

    def test_zero_pairs_gives_none(self):
        """Spec: None when n == 0 (a None result or a dict whose mean and bounds are None)."""
        out = paired_bootstrap_ci([], 200, 1, 0.95)
        if out is not None:
            assert out["n"] == 0
            assert out["mean"] is None
            low, high = bounds(out)
            assert low is None and high is None


class TestPairedFlips:
    def make_items(self, flags):
        """item_table-like rows: query_id and ungrounded (None = judge failed / output missing)."""
        return [{"query_id": f"q{i}", "ungrounded": v} for i, v in enumerate(flags, start=1)]

    def test_each_pair_goes_to_the_right_group(self):
        """Spec: improved / regressed / both_ungrounded / both_grounded / not_paired, with counts."""
        # baseline  : 1     1     0     0     None  1
        # experiment: 0     1     1     0     0     None
        # group     : impr  both_u regr both_g not_p not_p
        base = self.make_items([1, 1, 0, 0, None, 1])
        exp = self.make_items([0, 1, 1, 0, 0, None])
        out = paired_flips(base, exp)
        assert ids_of(out["improved"]) == ["q1"]
        assert ids_of(out["both_ungrounded"]) == ["q2"]
        assert ids_of(out["regressed"]) == ["q3"]
        assert ids_of(out["both_grounded"]) == ["q4"]
        assert ids_of(out["not_paired"]) == ["q5", "q6"]
        counts = flip_counts(out)
        assert (counts["improved"], counts["regressed"], counts["both_ungrounded"],
                counts["both_grounded"], counts["not_paired"]) == (1, 1, 1, 1, 2)

    def test_pairs_are_matched_by_query_id_not_position(self):
        """Spec: pairing is per query_id, so a different row order gives the same groups."""
        base = self.make_items([1, 0])
        exp = list(reversed(self.make_items([0, 1])))
        out = paired_flips(base, exp)
        assert ids_of(out["improved"]) == ["q1"]
        assert ids_of(out["regressed"]) == ["q2"]


# ================================================================ compare_runs

class TestCompareRuns:
    def test_flip_lists_and_counts_match_hand_values(self, experiment_db):
        """Spec: flip lists and counts (3 improved, 1 regressed, 4 both grounded, 2 not paired)."""
        row = compare(experiment_db)
        flips = flip_groups(row)
        assert ids_of(flips["improved"]) == ["q002", "q003", "q006"]
        assert ids_of(flips["regressed"]) == ["q005"]
        assert ids_of(flips["both_grounded"]) == ["q001", "q004", "q007", "q009"]
        assert ids_of(flips["both_ungrounded"]) == []
        assert ids_of(flips["not_paired"]) == ["q008", "q010"]
        counts = flip_counts(row)
        assert (counts["improved"], counts["regressed"], counts["both_grounded"],
                counts["both_ungrounded"], counts["not_paired"]) == (3, 1, 4, 0, 2)

    def test_mcnemar_p_value_matches_hand_value(self, experiment_db):
        """Spec: 3 improved / 1 regressed: n = 4, k = 3, P(X >= 3) = 5/16, p = 0.625."""
        row = compare(experiment_db)
        assert any(v == pytest.approx(0.625) for v in p_values(row)), (
            f"no p-value 0.625 in the row, found {p_values(row)}")

    def test_ungrounded_rates_and_wilson_intervals_match_hand_values(self, experiment_db):
        """Spec: baseline 4/9 and experiment 1/9 with Wilson CIs; judge-failed pairs are not in n."""
        # Wilson 95%, 4/9: centre 0.4611, margin 0.2723 -> [0.1888, 0.7334]
        # Wilson 95%, 1/9: centre 0.2275, margin 0.2076 -> [0.0199, 0.4351]
        row = compare(experiment_db)
        base = rate_dicts(row, 4, 9)
        exp = rate_dicts(row, 1, 9)
        assert base, "no 4/9 ungrounded rate for the baseline"
        assert exp, "no 1/9 ungrounded rate for the experiment"
        assert bounds(base[0]) == pytest.approx((0.1888, 0.7334), abs=2e-3)
        assert bounds(exp[0]) == pytest.approx((0.0199, 0.4351), abs=2e-3)
        assert bounds(base[0]) == pytest.approx(wilson_interval(4, 9, 0.95))

    def test_answerable_ungrounded_rates_match_hand_values(self, experiment_db):
        """Spec: answerable ungrounded: baseline 3/7 (q008 judge failed), experiment 1/8."""
        row = compare(experiment_db)
        assert rate_dicts(row, 3, 7), "no 3/7 answerable ungrounded rate"
        assert rate_dicts(row, 1, 8), "no 1/8 answerable ungrounded rate"

    def test_change_in_points_matches_hand_value(self, experiment_db):
        """Spec: delta in points = (1/9 - 4/9) * 100 = -33.33."""
        row = compare(experiment_db)
        deltas = [v for k, v in numbers(row) if "delta" in k or "change" in k]
        assert any(v == pytest.approx(-33.3333, abs=0.01) or v == pytest.approx(-0.33333, abs=1e-4)
                   for v in deltas), f"no delta of -33.33 points among {deltas}"

    def test_mean_faithfulness_difference_with_bootstrap_interval(self, experiment_db):
        """Spec: paired mean faithfulness difference 0.65 / 8 = 0.08125 over the 8 fully judged pairs."""
        row = compare(experiment_db)
        found = [d for d in walk(row) if isinstance(d, dict) and d.get("n") == 8
                 and isinstance(d.get("mean"), float)
                 and d["mean"] == pytest.approx(0.08125)]
        assert found, "no paired faithfulness difference (n=8, mean 0.08125) in the row"
        low, high = bounds(found[0])
        assert low <= 0.08125 <= high

    def test_bad_answer_rates_match_hand_values(self, experiment_db):
        """Spec: bad = ungrounded or irrelevant: baseline 5/9 (q002 q003 q004 q006 q010), experiment 2/9 (q004 q005)."""
        row = compare(experiment_db)
        assert rate_dicts(row, 5, 9), "no 5/9 baseline bad-answer rate"
        assert rate_dicts(row, 2, 9), "no 2/9 experiment bad-answer rate"

    def test_false_refusals_are_listed_for_both_runs(self, experiment_db):
        """Spec: false refusals (answerable, refused, irrelevant): q004 in baseline and experiment."""
        row = compare(experiment_db)
        lists = [d for d in walk(row) if isinstance(d, list) and d == ["q004"]]
        assert len(lists) >= 2, f"expected ['q004'] for both runs, found {len(lists)}"

    def test_correct_refusals_on_unanswerable_match_hand_values(self, experiment_db):
        """Spec: correct refusals on unanswerable: baseline 1/2 (q009), experiment 2/2 (q009 q010)."""
        row = compare(experiment_db)
        assert rate_dicts(row, 1, 2), "no 1/2 baseline correct-refusal rate"
        assert rate_dicts(row, 2, 2), "no 2/2 experiment correct-refusal rate"

    def test_retrieval_metrics_match_hand_values(self, experiment_db):
        """Spec: in_context 6/8 -> 7/8 and MRR 0.53125 -> 0.65625 over the 8 answerable queries."""
        # baseline reciprocal ranks 1, .5, 0, 0, 1, .25, .5, 1 = 4.25 / 8 = 0.53125
        # experiment 1, .5, 1, 0, 1, .25, .5, 1 = 5.25 / 8 = 0.65625
        row = compare(experiment_db)
        assert rate_dicts(row, 6, 8), "no 6/8 baseline in_context rate"
        assert rate_dicts(row, 7, 8), "no 7/8 experiment in_context rate"
        values = [v for _, v in numbers(row)]
        assert any(v == pytest.approx(0.53125) for v in values), "baseline MRR missing"
        assert any(v == pytest.approx(0.65625) for v in values), "experiment MRR missing"

    def test_latency_percentiles_for_the_experiment_run(self, experiment_db):
        """Spec: latency p50 / p95 from stored latencies 100..1000 ms: p50 550, p95 955."""
        # numpy linear percentile over 100, 200, ..., 1000: p50 = 550, p95 = 900 + 0.55 * 100 = 955
        row = compare(experiment_db)
        found = [d for d in walk(row) if isinstance(d, dict) and "p50" in d and "p95" in d
                 and d.get("p50") is not None]
        assert found, "no latency summary with a p50 in the row"
        assert found[0]["p50"] == pytest.approx(550.0)
        assert found[0]["p95"] == pytest.approx(955.0)

    def test_flipped_answers_carry_question_answers_and_judge_reasoning(self, experiment_db):
        """Spec: flip lists include the judge reasoning of each flipped answer (plus question and answers)."""
        experiment_db.set_score(experiment_db.versions["faithfulness"], "q002", "ungrounded", 1.0,
                                reasoning="REASON-BASELINE-MARKER-q002")
        row = compare(experiment_db)
        text = json.dumps(row)
        assert "REASON-BASELINE-MARKER-q002" in text, "baseline judge reasoning missing"
        assert "ungrounded reasoning run-exp-001 q002" in text, "experiment judge reasoning missing"
        assert "Baseline question number 2?" in text
        assert "Answer text q002." in text
        assert "Answer run-exp-001 q002." in text

    def test_row_names_variable_hypothesis_changed_keys_and_index_key(self, experiment_db):
        """Spec: the row holds the config diff and the index_key (and the declared variable)."""
        text = json.dumps(compare(experiment_db))
        assert "generation.seed" in text
        assert "INDEX-KEY-FIXTURE" in text
        assert "generation.seed (fixture)" in text
        assert "HYPOTHESIS-FIXTURE-TEXT" in text
        assert experiment_db.exp_run_id in text and experiment_db.run_id in text

    def test_row_is_labelled_uncalibrated_judge(self, experiment_db):
        """Spec: every judge-based number keeps the "uncalibrated judge" label (no calibration report)."""
        assert "uncalibrated judge" in json.dumps(compare(experiment_db))

    def test_rates_equal_the_step_10_report_for_the_same_run(self, experiment_db):
        """Spec: baseline rates and CIs equal Step 10's baseline report for the same run."""
        from evalharness.report import build_baseline_report

        cfg = {"primary_metric": "ungrounded", "include_bad_answer_rate": True,
               "allow_incomplete": True}
        report = build_baseline_report(experiment_db.conn, experiment_db.run_id,
                                       experiment_db.versions, 0.95, cfg)
        step10 = rate_dicts(report, 4, 9)
        step11 = rate_dicts(compare(experiment_db), 4, 9)
        assert step10 and step11
        assert step11[0]["rate"] == pytest.approx(step10[0]["rate"])
        assert bounds(step11[0]) == pytest.approx(bounds(step10[0]))

    def test_numbers_equal_a_direct_sql_recomputation(self, experiment_db):
        """Spec: every number is computed from stored scores; the ungrounded counts match plain SQL."""
        row = compare(experiment_db)
        sql = ("SELECT COALESCE(SUM(value), 0), COUNT(value) FROM scores"
               " WHERE run_id = ? AND version = ? AND metric = 'ungrounded'")
        for run_id in (experiment_db.run_id, experiment_db.exp_run_id):
            successes, n = sql_one(experiment_db.path, sql,
                                   (run_id, experiment_db.versions["faithfulness"]))
            assert rate_dicts(row, int(successes), n), (
                f"SQL says {int(successes)}/{n} ungrounded for {run_id}, not found in the row")

    def test_changing_a_stored_score_changes_the_table(self, experiment_db):
        """Spec: DB-tamper test; making q005 grounded in the experiment removes the regression."""
        before = compare(experiment_db)
        assert ids_of(flip_groups(before)["regressed"]) == ["q005"]

        experiment_db.set_exp_score(experiment_db.versions["faithfulness"], "q005",
                                    "ungrounded", 0.0)
        after = compare(experiment_db)

        # improved 3, regressed 0: n = 3, k = 3, p = 2 * (1/8) = 0.25
        assert ids_of(flip_groups(after)["regressed"]) == []
        assert flip_counts(after)["regressed"] == 0
        assert any(v == pytest.approx(0.25) for v in p_values(after))
        assert rate_dicts(after, 0, 9), "experiment ungrounded rate should now be 0/9"

    def test_compare_runs_writes_nothing_and_calls_no_llm(self, experiment_db, no_llm):
        """Spec: no LLM call, no re-scoring; compare_runs leaves every table unchanged."""
        tables = ("scores", "outputs", "runs", "run_reports")
        before = {t: count_rows(experiment_db.path, t) for t in tables}
        compare(experiment_db)
        after = {t: count_rows(experiment_db.path, t) for t in tables}
        assert after == before

    def test_two_experiment_runs_can_be_compared_with_the_same_baseline(self, experiment_db):
        """Spec: one baseline, many experiment runs; each comparison is independent."""
        experiment_db.add_simple_run("run-exp-002", ["q001"])
        first = compare(experiment_db)
        second = compare(experiment_db, "run-exp-002")
        assert ids_of(flip_groups(first)["improved"]) == ["q002", "q003", "q006"]
        # run-exp-002: only q001 ungrounded; baseline ungrounded are q002 q003 q006 q010
        # -> improved q002 q003 q006 q010, regressed q001, q008 not paired (baseline judge failed)
        assert ids_of(flip_groups(second)["improved"]) == ["q002", "q003", "q006", "q010"]
        assert ids_of(flip_groups(second)["regressed"]) == ["q001"]
        assert ids_of(flip_groups(second)["not_paired"]) == ["q008"]


class TestCompareRunsRefusals:
    def test_same_run_twice_is_refused(self, experiment_db):
        """Spec: the runs must differ."""
        with pytest.raises(ExperimentRefused):
            compare(experiment_db, experiment_db.run_id)

    def test_different_split_is_refused(self, experiment_db):
        """Spec: both runs must use the same split."""
        experiment_db.add_run("run-exp-heldout", split="heldout")
        with pytest.raises(ExperimentRefused):
            compare(experiment_db, "run-exp-heldout", allow_heldout=True)

    def test_different_test_set_sha_is_refused(self, experiment_db):
        """Spec: both runs must have the same test_set_sha256."""
        experiment_db.add_run("run-exp-sha", sha="cd" * 32)
        with pytest.raises(ExperimentRefused):
            compare(experiment_db, "run-exp-sha")

    def test_missing_relevance_version_is_refused(self, experiment_db):
        """Spec: both runs need rows for all three scorer versions (relevance missing)."""
        experiment_db.add_run("run-exp-norel", scorers=("checks", "faithfulness"))
        with pytest.raises(REFUSED):
            compare(experiment_db, "run-exp-norel")

    def test_missing_faithfulness_version_is_refused(self, experiment_db):
        """Spec: both runs need rows for all three scorer versions (faithfulness missing)."""
        experiment_db.add_run("run-exp-nofaith", scorers=("checks", "relevance"))
        with pytest.raises(REFUSED):
            compare(experiment_db, "run-exp-nofaith")

    def test_runs_scored_by_a_different_version_are_refused(self, experiment_db):
        """Spec: runs scored by different versions are not compared; the baseline's versions must exist for both."""
        versions = dict(experiment_db.versions, faithfulness="faith-never-written")
        with pytest.raises(REFUSED):
            compare_runs(experiment_db.conn, experiment_db.run_id, experiment_db.exp_run_id,
                         versions, BOOT_CFG)

    @pytest.mark.parametrize("status", ["aborted", "running"])
    def test_run_not_completed_is_refused(self, experiment_db, status):
        """Spec: both runs must be completed."""
        experiment_db.add_run("run-exp-open", status=status)
        with pytest.raises(REFUSED):
            compare(experiment_db, "run-exp-open")

    def test_heldout_runs_are_refused_without_the_flag(self, experiment_db):
        """Spec: held-out runs are refused unless allow_heldout is passed."""
        experiment_db.add_simple_run("run-held-a", [], split="heldout")
        experiment_db.add_simple_run("run-held-b", ["q001"], split="heldout")
        with pytest.raises(REFUSED):
            compare_runs(experiment_db.conn, "run-held-a", "run-held-b", experiment_db.versions,
                         BOOT_CFG)

    def test_heldout_runs_compare_with_the_flag(self, experiment_db):
        """Spec: allow_heldout=True lets two held-out runs be compared."""
        experiment_db.add_simple_run("run-held-a", [], split="heldout")
        experiment_db.add_simple_run("run-held-b", ["q001"], split="heldout")
        row = compare_runs(experiment_db.conn, "run-held-a", "run-held-b", experiment_db.versions,
                           BOOT_CFG, allow_heldout=True)
        assert ids_of(flip_groups(row)["regressed"]) == ["q001"]

    def test_unknown_run_is_refused(self, experiment_db):
        """Spec: an experiment run that is not in the DB cannot be compared."""
        with pytest.raises(REFUSED):
            compare(experiment_db, "run-does-not-exist")


# ================================================================ outputs

class TestOutputs:
    def significance_rows(self, db, ungrounded_ids):
        """compare_runs row: the all-grounded run `run-good` vs a run with the given ungrounded ids."""
        db.add_simple_run("run-sig-base", ungrounded_ids)
        db.add_simple_run("run-sig-exp", [])
        return compare_runs(db.conn, "run-sig-base", "run-sig-exp", db.versions, BOOT_CFG)

    def test_markdown_labels_uncalibrated_judge_and_prints_n_next_to_rates(self, experiment_db):
        """Spec: format_markdown marks "uncalibrated judge" and prints n next to every rate."""
        md = format_markdown([compare(experiment_db)])
        assert "uncalibrated judge" in md
        assert re.search(r"\b4/9\b", md), "baseline ungrounded count 4/9 not shown"
        assert re.search(r"\b1/9\b", md), "experiment ungrounded count 1/9 not shown"

    def test_markdown_not_significant_row_never_says_significant(self, experiment_db):
        """Spec: p = 0.625 is not below 0.05: no "significant", and "not statistically distinguishable"."""
        md = format_markdown([compare(experiment_db)])
        assert "significant" not in md.lower()
        assert "not statistically distinguishable" in md

    def test_markdown_five_fixed_zero_regressed_is_still_not_significant(self, experiment_db):
        """Spec: 5 improved / 0 regressed gives p = 0.0625 >= 0.05, so the word is not used."""
        row = self.significance_rows(experiment_db, ["q001", "q002", "q003", "q004", "q005"])
        assert any(v == pytest.approx(0.0625) for v in p_values(row))
        md = format_markdown([row])
        assert "significant" not in md.lower()
        assert "not statistically distinguishable" in md

    def test_markdown_six_fixed_zero_regressed_is_significant(self, experiment_db):
        """Spec: 6 improved / 0 regressed gives p = 0.03125 < 0.05, so "significant" may appear."""
        row = self.significance_rows(experiment_db,
                                     ["q001", "q002", "q003", "q004", "q005", "q006"])
        assert any(v == pytest.approx(0.03125) for v in p_values(row))
        md = format_markdown([row])
        assert "significant" in md.lower()
        assert re.search(r"0\.031", md), "the exact p-value should be printed"

    def test_markdown_with_no_flips_does_not_claim_significance(self, experiment_db):
        """Spec: p is None when nothing flipped; the output must not say "significant"."""
        experiment_db.add_simple_run("run-sig-base", ["q001", "q002"])
        experiment_db.add_simple_run("run-sig-twin", ["q001", "q002"])
        row = compare_runs(experiment_db.conn, "run-sig-base", "run-sig-twin",
                           experiment_db.versions, BOOT_CFG)
        assert flip_counts(row)["improved"] == 0 and flip_counts(row)["regressed"] == 0
        assert "significant" not in format_markdown([row]).lower()

    def test_markdown_shows_flipped_answers_with_question_answer_and_reasoning(
            self, experiment_db):
        """Spec: for every flipped answer the markdown shows question, answer and judge reasoning."""
        md = format_markdown([compare(experiment_db)])
        for qid in ("q002", "q003", "q006", "q005"):
            assert qid in md, f"flipped query {qid} missing"
        assert "Baseline question number 2?" in md
        assert "Answer run-exp-001 q002." in md
        assert "ungrounded reasoning run-exp-001 q002" in md

    def test_markdown_lists_not_paired_queries(self, experiment_db):
        """Spec: pairs with a failure on either side are listed, never dropped."""
        md = format_markdown([compare(experiment_db)])
        assert "q008" in md and "q010" in md

    def test_markdown_names_both_run_ids(self, experiment_db):
        """Spec: the table identifies the baseline and each experiment run."""
        md = format_markdown([compare(experiment_db)])
        assert experiment_db.run_id in md
        assert experiment_db.exp_run_id in md

    def test_terminal_table_names_run_and_judge_label(self, experiment_db):
        """Spec: format_table gives terminal text with the run id and the judge label."""
        text = format_table([compare(experiment_db)])
        assert isinstance(text, str)
        assert experiment_db.exp_run_id in text
        assert "uncalibrated judge" in text

    def test_log_row_columns_match_experiments_log_header(self, experiment_db):
        """Spec: log_row has the same columns as experiments/log.md (header read-only)."""
        header = next(line for line in REAL_LOG.read_text(encoding="utf-8").splitlines()
                      if line.startswith("| Date"))
        header_cells = [c.strip() for c in header.strip().strip("|").split("|")]
        line = log_row(compare(experiment_db), "2026-10-16T10:00:00+00:00")
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        assert "\n" not in line
        assert len(cells) == len(header_cells) == 6
        assert cells[0] == "2026-10-16"
        assert f"`{experiment_db.exp_run_id}`" in line
        assert "1/9" in line
        assert "uncalibrated judge" in line


# ================================================================ main (CLI)

class TestMainCheck:
    def test_check_exits_zero_for_a_valid_experiment(self, tmp_path, capsys):
        """Spec: `check` exits 0 when the config differs from the base only in its declared keys."""
        base, exp = make_pair(tmp_path, changes={"chunking.chunk_size": 75},
                              changed_keys=["chunking.chunk_size"])
        harness = write_harness(tmp_path, tmp_path / "unused.sqlite")
        code = experiments.main(["--harness-config", harness, "check", "--config", str(exp),
                                 "--base", str(base)])
        assert code == 0

    def test_check_exits_one_and_prints_problems(self, tmp_path, capsys):
        """Spec: `check` exits 1 and prints the problems."""
        base, exp = make_pair(tmp_path, changes={"chunking.chunk_size": 75, "retrieval.top_k": 2},
                              changed_keys=["chunking.chunk_size"])
        harness = write_harness(tmp_path, tmp_path / "unused.sqlite")
        code = experiments.main(["--harness-config", harness, "check", "--config", str(exp),
                                 "--base", str(base)])
        captured = capsys.readouterr()
        assert code == 1
        assert (captured.out + captured.err).strip(), "problems should be printed"

    @pytest.mark.parametrize("name", V1_CONFIGS)
    def test_check_exits_zero_for_every_real_v1_config(self, name, capsys):
        """Spec Definition of done: `check` exits 0 for all 12 real configs (read-only)."""
        code = experiments.main(["--harness-config", str(REAL_HARNESS), "check", "--config",
                                 str(CONFIGS / f"{name}.yaml")])
        assert code == 0, capsys.readouterr()

    def test_check_with_a_harness_config_without_experiments_block_exits_one(
            self, tmp_path, capsys):
        """Spec: a harness config without the experiments block is refused with exit 1."""
        base, exp = make_pair(tmp_path, changes={"chunking.chunk_size": 75},
                              changed_keys=["chunking.chunk_size"])
        harness = write_harness(tmp_path, tmp_path / "unused.sqlite", experiments_block=None)
        code = experiments.main(["--harness-config", harness, "check", "--config", str(exp),
                                 "--base", str(base)])
        assert code == 1


class TestMainCompare:
    def test_compare_prints_table_and_stores_one_experiment_report(
            self, experiment_db, tmp_path, monkeypatch, capsys, no_llm):
        """Spec: compare prints the table, stores one run_reports row of kind "experiment", makes no LLM call."""
        code = run_main(experiment_db, tmp_path, monkeypatch,
                        "compare", "--baseline", experiment_db.run_id,
                        "--run", experiment_db.exp_run_id)
        out = capsys.readouterr().out
        assert code == 0
        assert experiment_db.exp_run_id in out
        assert table_rows(experiment_db.path) == [(experiment_db.exp_run_id, "experiment")]
        report_json = sql_one(experiment_db.path,
                              "SELECT report_json FROM run_reports WHERE run_id = ?",
                              (experiment_db.exp_run_id,))[0]
        assert ids_of(flip_groups(json.loads(report_json))["improved"]) == ["q002", "q003", "q006"]

    def test_compare_two_runs_stores_one_report_per_run(
            self, experiment_db, tmp_path, monkeypatch):
        """Spec: one run_reports row (kind experiment) per experiment run."""
        experiment_db.add_simple_run("run-exp-002", ["q001"])
        code = run_main(experiment_db, tmp_path, monkeypatch,
                        "compare", "--baseline", experiment_db.run_id,
                        "--run", experiment_db.exp_run_id, "--run", "run-exp-002")
        assert code == 0
        assert sorted(table_rows(experiment_db.path)) == [
            ("run-exp-001", "experiment"), ("run-exp-002", "experiment")]

    def test_compare_does_not_change_scores_or_outputs(
            self, experiment_db, tmp_path, monkeypatch):
        """Spec: no re-run and no re-scoring; scores and outputs row counts are unchanged."""
        before = (count_rows(experiment_db.path, "scores"),
                  count_rows(experiment_db.path, "outputs"))
        run_main(experiment_db, tmp_path, monkeypatch, "compare", "--baseline",
                 experiment_db.run_id, "--run", experiment_db.exp_run_id)
        after = (count_rows(experiment_db.path, "scores"),
                 count_rows(experiment_db.path, "outputs"))
        assert after == before

    def test_compare_writes_markdown_and_refuses_overwrite_without_force(
            self, experiment_db, tmp_path, monkeypatch):
        """Spec: --markdown writes the table; an existing file needs --force (exit 1, unchanged)."""
        md_path = tmp_path / "experiment_table.md"
        argv = ["compare", "--baseline", experiment_db.run_id, "--run", experiment_db.exp_run_id,
                "--markdown", str(md_path)]
        assert run_main(experiment_db, tmp_path, monkeypatch, *argv) == 0
        written = md_path.read_text(encoding="utf-8")
        assert experiment_db.exp_run_id in written
        assert "uncalibrated judge" in written

        md_path.write_text("SENTINEL", encoding="utf-8")
        assert run_main(experiment_db, tmp_path, monkeypatch, *argv) == 1
        assert md_path.read_text(encoding="utf-8") == "SENTINEL"
        assert len(table_rows(experiment_db.path)) == 1, "a refused call must store nothing"

        assert run_main(experiment_db, tmp_path, monkeypatch, *argv, "--force") == 0
        assert experiment_db.exp_run_id in md_path.read_text(encoding="utf-8")

    def test_compare_appends_log_row_and_refuses_a_duplicate_run_id(
            self, experiment_db, tmp_path, monkeypatch):
        """Spec: --log appends one row per run; a run id already in the log is refused, nothing written."""
        log_path = tmp_path / "log.md"
        header = ("# Experiment log\n\n| Date | Config | Variable changed | Run ID | "
                  "Ungrounded rate (95% CI) | Notes |\n|---|---|---|---|---|---|\n")
        log_path.write_text(header, encoding="utf-8")
        argv = ["compare", "--baseline", experiment_db.run_id, "--run", experiment_db.exp_run_id,
                "--log", str(log_path)]

        assert run_main(experiment_db, tmp_path, monkeypatch, *argv) == 0
        text = log_path.read_text(encoding="utf-8")
        assert text.startswith(header)
        assert text.count(f"`{experiment_db.exp_run_id}`") == 1
        assert len(table_rows(experiment_db.path)) == 1

        assert run_main(experiment_db, tmp_path, monkeypatch, *argv) == 1
        assert log_path.read_text(encoding="utf-8") == text, "log must be unchanged"
        assert len(table_rows(experiment_db.path)) == 1, "nothing may be stored on refusal"

    def test_compare_unknown_baseline_exits_one_and_writes_nothing(
            self, experiment_db, tmp_path, monkeypatch):
        """Spec: exit 1 when refused, with nothing written."""
        code = run_main(experiment_db, tmp_path, monkeypatch, "compare", "--baseline",
                        "run-does-not-exist", "--run", experiment_db.exp_run_id)
        assert code == 1
        assert table_rows(experiment_db.path) == []

    def test_compare_same_run_as_baseline_exits_one(
            self, experiment_db, tmp_path, monkeypatch):
        """Spec: the same run twice is refused with exit 1."""
        code = run_main(experiment_db, tmp_path, monkeypatch, "compare", "--baseline",
                        experiment_db.run_id, "--run", experiment_db.run_id)
        assert code == 1
        assert table_rows(experiment_db.path) == []

    def test_compare_one_bad_run_of_two_stores_nothing(
            self, experiment_db, tmp_path, monkeypatch):
        """Spec: refused means nothing written, even for the valid run listed first."""
        experiment_db.add_run("run-exp-sha", sha="cd" * 32)
        code = run_main(experiment_db, tmp_path, monkeypatch, "compare", "--baseline",
                        experiment_db.run_id, "--run", experiment_db.exp_run_id,
                        "--run", "run-exp-sha")
        assert code == 1
        assert table_rows(experiment_db.path) == []

    def test_compare_heldout_runs_exit_one_and_write_nothing(
            self, experiment_db, tmp_path, monkeypatch):
        """Spec: compare keeps refusing the held-out split; this step never passes the flag."""
        experiment_db.add_simple_run("run-held-a", [], split="heldout")
        experiment_db.add_simple_run("run-held-b", ["q001"], split="heldout")
        code = run_main(experiment_db, tmp_path, monkeypatch, "compare", "--baseline",
                        "run-held-a", "--run", "run-held-b")
        assert code == 1
        assert table_rows(experiment_db.path) == []


# ================================================================ layering

class TestLayering:
    STEP11_RAG_FILES = ["bm25.py", "reranker.py", "rerank_worker.py", "rewriter.py",
                        "retriever.py", "pipeline.py"]

    @pytest.mark.parametrize("filename", STEP11_RAG_FILES)
    def test_rag_step11_files_never_import_evalharness(self, filename):
        """CLAUDE.md rule: rag/ must never import from evalharness/ (Step 11 files)."""
        names = imported_modules(PROJECT_ROOT / "rag" / filename)
        assert not any(n.split(".")[0] == "evalharness" for n in names), (
            f"rag/{filename} imports from evalharness")

    @pytest.mark.parametrize("forbidden", ["rag", "llm"])
    def test_experiments_module_imports_nothing_from_rag_or_llm(self, forbidden):
        """Spec: evalharness/experiments.py imports nothing from rag/ or llm/."""
        names = imported_modules(PROJECT_ROOT / "evalharness" / "experiments.py")
        assert not any(n.split(".")[0] == forbidden for n in names), (
            f"experiments.py imports {forbidden}")

    @pytest.mark.parametrize("heavy", ["torch", "sentence_transformers"])
    def test_reranker_module_does_not_import_torch_or_sentence_transformers(self, heavy):
        """Spec: rag/reranker.py never imports torch or sentence_transformers (only the worker does)."""
        names = imported_modules(PROJECT_ROOT / "rag" / "reranker.py")
        assert not any(n.split(".")[0] == heavy for n in names)

    def test_only_the_worker_imports_sentence_transformers_or_torch(self):
        """Spec: the parent process never imports torch; the heavy import lives in rerank_worker.py."""
        for path in (PROJECT_ROOT / "rag").glob("*.py"):
            names = imported_modules(path)
            heavy = [n for n in names if n.split(".")[0] in ("torch", "sentence_transformers")]
            if path.name == "rerank_worker.py":
                assert heavy, "the worker should import sentence_transformers"
            else:
                assert not heavy, f"{path.name} imports {heavy}"

    def test_kmp_duplicate_lib_ok_is_never_set(self):
        """Spec: KMP_DUPLICATE_LIB_OK is never set by rag/ or evalharness/ code."""
        for folder in ("rag", "evalharness"):
            for path in (PROJECT_ROOT / folder).glob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                strings = [n.value for n in ast.walk(tree)
                           if isinstance(n, ast.Constant) and isinstance(n.value, str)]
                assert "KMP_DUPLICATE_LIB_OK" not in strings, f"{path} sets KMP_DUPLICATE_LIB_OK"
