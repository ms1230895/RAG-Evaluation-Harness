"""Tests for Step 05: the results store (evalharness/store.py) and the runner (evalharness/runner.py).

Written from .claude/specs/05-runner-and-store.md. Everything is offline: a fake
`answer`, throwaway databases in tmp_path, a fake `urlopen`. The real results DB
(data/results.sqlite) and the real LLM cache (data/llm_cache.sqlite) are never touched.
"""

import ast
import copy
import hashlib
import io
import json
import re
import sqlite3
import urllib.error
import urllib.request
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from evalharness import runner, store
from evalharness.runner import ContractError, System
from evalharness.testset import filter_split, save_test_set
from llm import client
from rag.pipeline import ModelDigestError

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The digest the spec pins in configs/v0.yaml (Decision 3).
V0_MODEL_DIGEST = "ef09f235533c96cd75e8deed88c628335cb69e2b3ce96275d0d7a67fe9887aba"


# ---------------------------------------------------------------- importable fakes for load_system / main
# harness.yaml names the system as "module:function", so main() needs real importable functions.

def fake_answer(question, config):
    """Importable fake answer(): a valid contract dict that depends on the question."""
    return {
        "answer": f"Answer to: {question}",
        "chunks": [{"chunk_id": "leave_policy#000", "doc_id": "leave_policy",
                    "text": "chunk text", "score": 0.9}],
        "latency_ms": 12.5,
    }


def fake_answer_always_fails(question, config):
    """Importable fake answer() that always raises (a systematic failure, like Ollama being down)."""
    raise RuntimeError("model unreachable")


LOG_CONFIG = {}    # set by the main_env fixture: the llm config whose cache DB fake_answer_logs_call writes to


def fake_answer_logs_call(question, config):
    """Importable fake answer() that logs one llm_calls row (a cache miss), like a real generate() call."""
    insert_llm_call(LOG_CONFIG["llm_config"], "gen-x", 0)
    return fake_answer(question, config)


def fake_load_config(path):
    """Importable fake load_config(): reads the YAML file it is given (so --config is passed through)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fake_describe(config):
    """Importable fake describe(): fixed run metadata."""
    return {"index": {"index_key": "key-1"}, "generator_model_digest": "digest-1"}


def fake_describe_refuses(config):
    """Importable fake describe() that refuses the run, like a model digest mismatch."""
    raise ModelDigestError("digest mismatch")


class _FlippingDescribe:
    """describe() whose index_key differs from the second call on (corpus changed during the run)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, config):
        self.calls += 1
        return {"index": {"index_key": "key-1" if self.calls == 1 else "key-2"}}


fake_describe_flips = _FlippingDescribe()


# ---------------------------------------------------------------- helpers

def count_rows(conn, sql):
    """Run a `SELECT COUNT(*) ...` query without parameters and return the number."""
    return conn.execute(sql).fetchone()[0]


def query_by_id(queries, query_id):
    """The Query with this id from a list."""
    return next(q for q in queries if q.id == query_id)


def chunks_of(row):
    """The parsed chunk list of a get_outputs() row.

    The spec says only "chunks parsed", not the key name, so accept the likely names.
    """
    for key in ("chunks", "retrieved_chunks", "retrieved_chunks_json"):
        if key in row:
            return row[key]
    raise AssertionError(f"no chunks key in get_outputs row: {sorted(row)}")


def make_run_meta(run_id="run-1", split="dev", config=None, system_info=None):
    """The start_run fields that run_split needs in `run_meta`."""
    return {
        "run_id": run_id,
        "split": split,
        "config": config if config is not None else {"name": "fake", "top_k": 3},
        "config_path": "configs/fake.yaml",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "test_set_sha256": "b" * 64,
        "system_info": system_info if system_info is not None else {
            "index": {"index_key": "key-1"}, "generator_model": "fake-gen",
            "generator_model_digest": "digest-1"},
    }


def start_kwargs(**overrides):
    """Keyword arguments for store.start_run with simple, checkable values."""
    kwargs = {
        "run_id": "run-1", "split": "dev", "config": {"name": "v0", "retrieval": {"top_k": 4}},
        "config_path": "configs/v0.yaml", "git_commit": "c" * 40, "git_dirty": True,
        "test_set_sha256": "d" * 64, "system_info": {"index": {"index_key": "k"}, "digest": "x"},
        "n_queries": 5, "llm_calls_start_id": 7,
    }
    kwargs.update(overrides)
    return kwargs


def new_run(conn, queries, run_id="run-1"):
    """Sync queries and start a run, so write_output has a run and queries to point at."""
    store.sync_queries(conn, queries)
    store.start_run(conn, **start_kwargs(run_id=run_id))


def run_dev(system, conn, ready_queries, llm_config, **overrides):
    """Sync the 8 queries, then run_split on the 5 dev queries with default limits; returns the run_id."""
    store.sync_queries(conn, ready_queries)
    options = {"max_consecutive_failures": 5, "error_max_chars": 500}
    options.update(overrides.pop("options", {}))
    meta = overrides.pop("run_meta", None) or make_run_meta()
    return runner.run_split(system, meta["config"], filter_split(ready_queries, "dev"), conn,
                            run_meta=meta, llm_config=llm_config, **options)


def insert_llm_call(llm_config, model, cached, error=None):
    """Append one row to the (tmp_path) llm_calls table, creating the cache DB if needed."""
    with closing(client._connect(Path(llm_config["cache_path"]))) as conn:
        conn.execute("INSERT INTO llm_calls (model, cached, error) VALUES (?, ?, ?)",
                     (model, cached, error))
        conn.commit()


# ================================================================ store

class TestConnect:
    TABLE_COLUMNS = {
        "queries": {"id", "question", "type", "reference_answer", "source_doc_ids", "split",
                    "reviewed"},
        "runs": {"run_id", "timestamp", "git_commit", "config_json", "judge_version", "split",
                 "finished_at", "status", "note", "config_path", "git_dirty", "test_set_sha256",
                 "system_json", "n_queries", "n_ok", "n_failed", "llm_calls_start_id",
                 "llm_calls_json"},
        "outputs": {"run_id", "query_id", "answer", "retrieved_chunks_json", "latency_ms", "error"},
        "scores": {"run_id", "query_id", "metric", "value", "reasoning"},
        "reviews": {"run_id", "query_id", "flag_reasons", "human_label", "notes"},
    }

    def test_connect_new_db_creates_all_five_tables(self, results_db):
        """Spec: connect creates the five tables queries, runs, outputs, scores, reviews."""
        names = {r[0] for r in results_db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert set(self.TABLE_COLUMNS) <= names, f"missing tables: {set(self.TABLE_COLUMNS) - names}"

    @pytest.mark.parametrize("table", ["queries", "runs", "outputs", "scores", "reviews"])
    def test_connect_table_has_exactly_the_spec_columns(self, results_db, table):
        """Spec Data and schema: each table has exactly the listed columns."""
        columns = {r[1] for r in results_db.execute(f"PRAGMA table_info({table})")}
        assert columns == self.TABLE_COLUMNS[table], f"{table} columns differ from the spec"

    def test_connect_sets_foreign_keys_on(self, results_db):
        """Spec: PRAGMA foreign_keys = ON on the connection."""
        assert results_db.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_connect_uses_row_factory_sqlite_row(self, results_db):
        """Spec: row_factory = sqlite3.Row."""
        assert results_db.row_factory is sqlite3.Row

    def test_connect_second_call_is_idempotent_and_keeps_data(self, results_db_path, ready_queries):
        """Spec: connect is idempotent; a second connect keeps existing tables and rows."""
        with closing(store.connect(results_db_path)) as conn:
            store.sync_queries(conn, ready_queries)
        with closing(store.connect(results_db_path)) as conn:
            assert count_rows(conn, "SELECT COUNT(*) FROM queries") == 8, "rows lost on reconnect"
            assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_connect_accepts_string_path(self, tmp_path):
        """Spec Interfaces: db_path is str | Path."""
        with closing(store.connect(str(tmp_path / "as_string.sqlite"))) as conn:
            assert count_rows(conn, "SELECT COUNT(*) FROM runs") == 0

    def test_run_statuses_lists_the_five_spec_statuses(self):
        """Spec: RUN_STATUSES = running, completed, aborted, interrupted, invalid."""
        assert set(store.RUN_STATUSES) == {"running", "completed", "aborted", "interrupted",
                                           "invalid"}


class TestSyncQueries:
    def test_sync_queries_fresh_db_inserts_eight_then_zero(self, results_db, ready_queries):
        """Spec: sync_queries inserts 8 on a fresh DB and 0 on repeat."""
        assert store.sync_queries(results_db, ready_queries) == 8
        assert store.sync_queries(results_db, ready_queries) == 0
        assert count_rows(results_db, "SELECT COUNT(*) FROM queries") == 8

    def test_sync_queries_only_new_ones_are_counted(self, results_db, ready_queries):
        """Spec: returns how many were new (3 stored, 8 given -> 5)."""
        store.sync_queries(results_db, ready_queries[:3])
        assert store.sync_queries(results_db, ready_queries) == 5

    def test_sync_queries_changed_content_raises_value_error_naming_the_id(
            self, results_db, ready_queries):
        """Spec: a query stored with different content raises ValueError naming its id."""
        store.sync_queries(results_db, ready_queries)   # control: same content is fine
        changed = [q.model_copy(update={"question": "A different question?"}) if q.id == "q003" else q
                   for q in ready_queries]
        with pytest.raises(ValueError, match="q003"):
            store.sync_queries(results_db, changed)

    def test_sync_queries_stored_row_matches_the_query(self, results_db, ready_queries):
        """Spec: queries table keeps question, type, reference, split; reviewed is 0/1."""
        store.sync_queries(results_db, ready_queries)
        row = results_db.execute("SELECT * FROM queries WHERE id = ?", ("q004",)).fetchone()
        source = query_by_id(ready_queries, "q004")
        assert row["question"] == source.question
        assert row["type"] == "multi_chunk"
        assert row["reference_answer"] == source.reference_answer
        assert row["split"] == "dev"
        assert row["reviewed"] == 1

    def test_sync_queries_source_doc_ids_round_trip_as_json(self, results_db, ready_queries):
        """Spec: source_doc_ids is stored as a JSON list ([] for unanswerable)."""
        store.sync_queries(results_db, ready_queries)
        rows = {r["id"]: r["source_doc_ids"] for r in results_db.execute(
            "SELECT id, source_doc_ids FROM queries")}
        assert json.loads(rows["q004"]) == ["leave_policy", "payroll_policy"]
        assert json.loads(rows["q007"]) == [], "unanswerable query must store an empty JSON list"


class TestStartAndFinishRun:
    def test_start_run_round_trips_every_column(self, results_db):
        """Spec: start_run inserts the runs row with status 'running' and all given metadata."""
        store.start_run(results_db, **start_kwargs())
        run = store.get_run(results_db, "run-1")
        assert run["run_id"] == "run-1"
        assert run["split"] == "dev"
        assert run["status"] == "running"
        assert run["git_commit"] == "c" * 40
        assert run["git_dirty"] == 1
        assert run["config_path"] == "configs/v0.yaml"
        assert run["test_set_sha256"] == "d" * 64
        assert run["n_queries"] == 5
        assert run["llm_calls_start_id"] == 7
        assert json.loads(run["system_json"]) == {"index": {"index_key": "k"}, "digest": "x"}
        # not finished yet
        assert run["finished_at"] is None
        assert run["n_ok"] is None
        assert run["n_failed"] is None
        assert run["llm_calls_json"] is None
        assert run["note"] is None
        # no judges in Step 5
        assert run["judge_version"] is None

    def test_start_run_config_json_is_sorted_json_of_the_same_dict(self, results_db):
        """Spec: config_json = json.dumps(config, sort_keys=True) and parses back to the same dict."""
        config = {"zeta": 1, "alpha": {"top_k": 4, "b": [1, 2]}, "name": "v0"}
        store.start_run(results_db, **start_kwargs(config=config))
        run = store.get_run(results_db, "run-1")
        assert json.loads(run["config_json"]) == config
        assert run["config_json"] == json.dumps(config, sort_keys=True)

    def test_start_run_clean_tree_stores_git_dirty_zero(self, results_db):
        """Spec: git_dirty is stored as 0/1."""
        store.start_run(results_db, **start_kwargs(git_dirty=False))
        assert store.get_run(results_db, "run-1")["git_dirty"] == 0

    def test_start_run_timestamp_is_an_iso_datetime(self, results_db):
        """Spec: timestamp = now (UTC ISO)."""
        store.start_run(results_db, **start_kwargs())
        stamp = store.get_run(results_db, "run-1")["timestamp"]
        assert isinstance(datetime.fromisoformat(stamp), datetime), f"not ISO: {stamp!r}"

    def test_start_run_duplicate_run_id_raises_integrity_error(self, results_db):
        """Spec: runs.run_id is the primary key."""
        store.start_run(results_db, **start_kwargs())
        with pytest.raises(sqlite3.IntegrityError):
            store.start_run(results_db, **start_kwargs())

    def test_start_run_is_committed_at_once(self, results_db, results_db_path):
        """Spec: a run row is visible to other connections right after start_run (crash safety)."""
        store.start_run(results_db, **start_kwargs())
        with closing(sqlite3.connect(results_db_path)) as other:
            assert count_rows(other, "SELECT COUNT(*) FROM runs") == 1

    def test_finish_run_round_trips_counts_summary_and_note(self, results_db):
        """Spec: finish_run sets finished_at, status, counts, per-model call summary and note."""
        store.start_run(results_db, **start_kwargs())
        summary = {"gemma4:31b-cloud": {"calls": 5, "hits": 1, "misses": 4, "errors": 0}}
        store.finish_run(results_db, "run-1", status="aborted", n_ok=3, n_failed=2,
                         llm_calls_summary=summary, note="stopped early")
        run = store.get_run(results_db, "run-1")
        assert run["status"] == "aborted"
        assert run["n_ok"] == 3
        assert run["n_failed"] == 2
        assert run["finished_at"] is not None
        assert run["note"] == "stopped early"
        assert json.loads(run["llm_calls_json"]) == summary

    def test_finish_run_without_note_leaves_note_null(self, results_db):
        """Spec: note is optional (None by default)."""
        store.start_run(results_db, **start_kwargs())
        store.finish_run(results_db, "run-1", status="completed", n_ok=5, n_failed=0,
                         llm_calls_summary={})
        run = store.get_run(results_db, "run-1")
        assert run["note"] is None
        assert json.loads(run["llm_calls_json"]) == {}

    @pytest.mark.parametrize("status", ["completed", "aborted", "interrupted", "invalid"])
    def test_finish_run_stores_each_final_status(self, results_db, status):
        """Spec: final statuses are completed | aborted | interrupted | invalid."""
        store.start_run(results_db, **start_kwargs())
        store.finish_run(results_db, "run-1", status=status, n_ok=1, n_failed=0,
                         llm_calls_summary={})
        assert store.get_run(results_db, "run-1")["status"] == status

    def test_get_run_unknown_run_id_returns_none(self, results_db):
        """Spec: get_run returns dict | None."""
        assert store.get_run(results_db, "no-such-run") is None

    def test_get_run_returns_a_dict(self, results_db):
        """Spec: get_run returns a dict."""
        store.start_run(results_db, **start_kwargs())
        assert isinstance(store.get_run(results_db, "run-1"), dict)


class TestWriteOutput:
    def test_write_output_success_row_is_stored(self, results_db, ready_queries):
        """Spec: success -> answer, chunks, latency_ms set and error NULL."""
        new_run(results_db, ready_queries)
        chunks = [{"chunk_id": "leave_policy#000", "doc_id": "leave_policy", "text": "t",
                   "score": 0.5}]
        store.write_output(results_db, run_id="run-1", query_id="q001", answer="8 days",
                           chunks=chunks, latency_ms=123.456, error=None)
        row = results_db.execute("SELECT * FROM outputs WHERE run_id = ? AND query_id = ?",
                                 ("run-1", "q001")).fetchone()
        assert row["answer"] == "8 days"
        assert json.loads(row["retrieved_chunks_json"]) == chunks
        assert row["latency_ms"] == pytest.approx(123.456)
        assert row["error"] is None

    def test_write_output_failure_row_has_only_the_error(self, results_db, ready_queries):
        """Spec: failure -> error set; answer, chunks, latency_ms NULL."""
        new_run(results_db, ready_queries)
        store.write_output(results_db, run_id="run-1", query_id="q001", answer=None, chunks=None,
                           latency_ms=None, error="ContextLimitError: too long")
        row = results_db.execute("SELECT * FROM outputs WHERE query_id = ?", ("q001",)).fetchone()
        assert row["error"] == "ContextLimitError: too long"
        assert row["answer"] is None
        assert row["retrieved_chunks_json"] is None
        assert row["latency_ms"] is None

    def test_write_output_empty_answer_is_stored_as_empty_string_not_null(
            self, results_db, ready_queries):
        """Spec Notes: an empty-string answer is stored as '' (not NULL); NULL means failure."""
        new_run(results_db, ready_queries)
        store.write_output(results_db, run_id="run-1", query_id="q001", answer="", chunks=[],
                           latency_ms=0.0, error=None)
        row = results_db.execute("SELECT answer, error FROM outputs WHERE query_id = ?",
                                 ("q001",)).fetchone()
        assert row["answer"] == ""
        assert row["error"] is None

    def test_write_output_both_answer_and_error_null_is_rejected(self, results_db, ready_queries):
        """Spec: CHECK rejects a row with neither answer nor error; nothing is stored."""
        new_run(results_db, ready_queries)
        with pytest.raises((sqlite3.IntegrityError, ValueError)):
            store.write_output(results_db, run_id="run-1", query_id="q001", answer=None,
                               chunks=None, latency_ms=None, error=None)
        assert count_rows(results_db, "SELECT COUNT(*) FROM outputs") == 0

    def test_write_output_both_answer_and_error_set_is_rejected(self, results_db, ready_queries):
        """Spec: CHECK rejects a row with both answer and error; nothing is stored."""
        new_run(results_db, ready_queries)
        with pytest.raises((sqlite3.IntegrityError, ValueError)):
            store.write_output(results_db, run_id="run-1", query_id="q001", answer="a",
                               chunks=[], latency_ms=1.0, error="boom")
        assert count_rows(results_db, "SELECT COUNT(*) FROM outputs") == 0

    def test_write_output_duplicate_run_and_query_raises_integrity_error(
            self, results_db, ready_queries):
        """Spec: (run_id, query_id) is the primary key of outputs."""
        new_run(results_db, ready_queries)
        kwargs = dict(run_id="run-1", query_id="q001", answer="a", chunks=[], latency_ms=1.0,
                      error=None)
        store.write_output(results_db, **kwargs)
        with pytest.raises(sqlite3.IntegrityError):
            store.write_output(results_db, **kwargs)
        assert count_rows(results_db, "SELECT COUNT(*) FROM outputs") == 1

    def test_write_output_same_query_in_two_runs_is_allowed(self, results_db, ready_queries):
        """Spec: the key is (run_id, query_id), so one query may appear once per run."""
        new_run(results_db, ready_queries, run_id="run-1")
        store.start_run(results_db, **start_kwargs(run_id="run-2"))
        for run_id in ("run-1", "run-2"):
            store.write_output(results_db, run_id=run_id, query_id="q001", answer="a", chunks=[],
                               latency_ms=1.0, error=None)
        assert count_rows(results_db, "SELECT COUNT(*) FROM outputs") == 2

    def test_write_output_unknown_run_id_raises_foreign_key_error(self, results_db, ready_queries):
        """Spec: outputs.run_id references runs(run_id); foreign keys are enforced."""
        store.sync_queries(results_db, ready_queries)
        with pytest.raises(sqlite3.IntegrityError):
            store.write_output(results_db, run_id="no-such-run", query_id="q001", answer="a",
                               chunks=[], latency_ms=1.0, error=None)

    def test_write_output_unknown_query_id_raises_foreign_key_error(self, results_db, ready_queries):
        """Spec: outputs.query_id references queries(id)."""
        new_run(results_db, ready_queries)
        with pytest.raises(sqlite3.IntegrityError):
            store.write_output(results_db, run_id="run-1", query_id="q999", answer="a",
                               chunks=[], latency_ms=1.0, error=None)

    def test_write_output_hostile_string_is_stored_verbatim_and_runs_table_survives(
            self, results_db, ready_queries):
        """Spec: parameterised SQL; an injection string is just text."""
        new_run(results_db, ready_queries)
        hostile = "'); DROP TABLE runs; --"
        store.write_output(results_db, run_id="run-1", query_id="q001", answer=hostile,
                           chunks=[{"chunk_id": "c", "doc_id": "d", "text": hostile, "score": 1.0}],
                           latency_ms=1.0, error=None)
        stored = results_db.execute("SELECT answer FROM outputs WHERE query_id = ?",
                                    ("q001",)).fetchone()[0]
        assert stored == hostile
        assert count_rows(results_db, "SELECT COUNT(*) FROM runs") == 1, "runs table was damaged"

    def test_write_output_hostile_error_text_is_stored_verbatim(self, results_db, ready_queries):
        """Spec: parameterised SQL also for the error column."""
        new_run(results_db, ready_queries)
        hostile = "RuntimeError: '); DROP TABLE runs; --"
        store.write_output(results_db, run_id="run-1", query_id="q001", answer=None, chunks=None,
                           latency_ms=None, error=hostile)
        stored = results_db.execute("SELECT error FROM outputs").fetchone()[0]
        assert stored == hostile

    def test_write_output_commits_immediately(self, results_db, results_db_path, ready_queries):
        """Spec: each output row is committed as soon as it is written."""
        new_run(results_db, ready_queries)
        store.write_output(results_db, run_id="run-1", query_id="q001", answer="a", chunks=[],
                           latency_ms=1.0, error=None)
        with closing(sqlite3.connect(results_db_path)) as other:
            assert count_rows(other, "SELECT COUNT(*) FROM outputs") == 1, \
                "row not visible to a second connection, so it was not committed"


class TestGetOutputs:
    def test_get_outputs_returns_rows_in_query_id_order(self, results_db, ready_queries):
        """Spec: get_outputs returns rows in query_id order."""
        new_run(results_db, ready_queries)
        for query_id in ["q004", "q001", "q002"]:    # written out of order on purpose
            store.write_output(results_db, run_id="run-1", query_id=query_id,
                               answer=f"answer {query_id}", chunks=[], latency_ms=1.0, error=None)
        rows = store.get_outputs(results_db, "run-1")
        assert [r["query_id"] for r in rows] == ["q001", "q002", "q004"]

    def test_get_outputs_chunks_are_parsed_back_to_dicts(self, results_db, ready_queries):
        """Spec: get_outputs returns chunks parsed (list of dicts, not a JSON string)."""
        new_run(results_db, ready_queries)
        chunks = [{"chunk_id": "a#000", "doc_id": "a", "text": "café – ₹5000", "score": 0.25},
                  {"chunk_id": "b#001", "doc_id": "b", "text": "second", "score": 0.125}]
        store.write_output(results_db, run_id="run-1", query_id="q001", answer="x", chunks=chunks,
                           latency_ms=2.0, error=None)
        row = store.get_outputs(results_db, "run-1")[0]
        assert chunks_of(row) == chunks, "chunks (incl. non-ASCII text) must round-trip exactly"

    def test_get_outputs_returns_plain_dicts_with_answer_latency_error(self, results_db, ready_queries):
        """Spec: get_outputs returns list[dict] holding the stored answer, latency and error."""
        new_run(results_db, ready_queries)
        store.write_output(results_db, run_id="run-1", query_id="q001", answer="ok", chunks=[],
                           latency_ms=3.5, error=None)
        store.write_output(results_db, run_id="run-1", query_id="q002", answer=None, chunks=None,
                           latency_ms=None, error="Boom: x")
        ok, failed = store.get_outputs(results_db, "run-1")
        assert isinstance(ok, dict)
        assert (ok["answer"], ok["latency_ms"], ok["error"]) == ("ok", 3.5, None)
        assert (failed["answer"], failed["latency_ms"], failed["error"]) == (None, None, "Boom: x")

    def test_get_outputs_only_returns_rows_of_the_given_run(self, results_db, ready_queries):
        """Spec: get_outputs(conn, run_id) is per run."""
        new_run(results_db, ready_queries, run_id="run-1")
        store.start_run(results_db, **start_kwargs(run_id="run-2"))
        store.write_output(results_db, run_id="run-1", query_id="q001", answer="a", chunks=[],
                           latency_ms=1.0, error=None)
        store.write_output(results_db, run_id="run-2", query_id="q002", answer="b", chunks=[],
                           latency_ms=1.0, error=None)
        assert [r["query_id"] for r in store.get_outputs(results_db, "run-1")] == ["q001"]

    def test_get_outputs_unknown_run_returns_empty_list(self, results_db):
        """Spec: get_outputs returns list[dict]; no rows -> []."""
        assert store.get_outputs(results_db, "no-such-run") == []


# ================================================================ runner: contract

def valid_output():
    """A fresh contract-valid answer() result."""
    return {"answer": "Interns get 12 days.",
            "chunks": [{"chunk_id": "intern_leave_policy#000", "doc_id": "intern_leave_policy",
                        "text": "Interns earn 1 day per month.", "score": 0.78}],
            "latency_ms": 1966.4}


def without(mapping, key):
    """A copy of `mapping` without `key`."""
    return {k: v for k, v in mapping.items() if k != key}


class TestCheckContract:
    def test_check_contract_valid_output_passes(self):
        """Spec: check_contract accepts a valid contract dict."""
        assert runner.check_contract(valid_output()) is None

    @pytest.mark.parametrize("make", [
        lambda: {**valid_output(), "chunks": []},
        lambda: {**valid_output(), "answer": ""},
        lambda: {**valid_output(), "latency_ms": 0},
        lambda: {**valid_output(), "latency_ms": 0.0},
        lambda: {**valid_output(), "latency_ms": 5},
        lambda: {**valid_output(), "chunks": [{**valid_output()["chunks"][0], "score": 1}]},
        lambda: {**valid_output(), "chunks": [{**valid_output()["chunks"][0], "score": -0.2}]},
    ], ids=["no-chunks", "empty-answer", "latency-int-zero", "latency-float-zero", "latency-int",
            "score-int", "score-negative"])
    def test_check_contract_edge_but_valid_outputs_pass(self, make):
        """Spec: empty chunks/answer, int scores/latency and latency exactly 0 are valid."""
        runner.check_contract(make())

    @pytest.mark.parametrize("key", ["answer", "chunks", "latency_ms"])
    def test_check_contract_missing_top_level_key_raises(self, key):
        """Spec: exactly the keys answer, chunks, latency_ms; a missing one is rejected."""
        with pytest.raises(ContractError):
            runner.check_contract(without(valid_output(), key))

    def test_check_contract_extra_top_level_key_raises(self):
        """Spec: 'exactly the keys': an extra top-level key is rejected."""
        with pytest.raises(ContractError):
            runner.check_contract({**valid_output(), "debug": 1})

    @pytest.mark.parametrize("bad", [None, 5, ["a"], ("a",), b"text"])
    def test_check_contract_answer_not_a_string_raises(self, bad):
        """Spec: answer must be str."""
        with pytest.raises(ContractError):
            runner.check_contract({**valid_output(), "answer": bad})

    @pytest.mark.parametrize("bad", [None, "chunks", {}, 3])
    def test_check_contract_chunks_not_a_list_raises(self, bad):
        """Spec: chunks must be a list."""
        with pytest.raises(ContractError):
            runner.check_contract({**valid_output(), "chunks": bad})

    @pytest.mark.parametrize("bad", ["chunk", 1, None, ["a"]])
    def test_check_contract_chunk_not_a_dict_raises(self, bad):
        """Spec: every chunk must be a dict."""
        with pytest.raises(ContractError):
            runner.check_contract({**valid_output(), "chunks": [bad]})

    @pytest.mark.parametrize("key", ["chunk_id", "doc_id", "text", "score"])
    def test_check_contract_chunk_missing_key_raises(self, key):
        """Spec: a chunk has exactly chunk_id, doc_id, text, score; a missing key is rejected."""
        chunk = without(valid_output()["chunks"][0], key)
        with pytest.raises(ContractError):
            runner.check_contract({**valid_output(), "chunks": [chunk]})

    def test_check_contract_chunk_extra_key_raises(self):
        """Spec: 'exactly' the chunk keys: an extra chunk key is rejected."""
        chunk = {**valid_output()["chunks"][0], "rank": 1}
        with pytest.raises(ContractError):
            runner.check_contract({**valid_output(), "chunks": [chunk]})

    def test_check_contract_second_bad_chunk_is_still_found(self):
        """Spec: every chunk is checked, not only the first."""
        good = valid_output()["chunks"][0]
        with pytest.raises(ContractError):
            runner.check_contract({**valid_output(), "chunks": [good, without(good, "text")]})

    @pytest.mark.parametrize("field", ["chunk_id", "doc_id", "text"])
    @pytest.mark.parametrize("bad", [None, 7])
    def test_check_contract_chunk_string_field_wrong_type_raises(self, field, bad):
        """Spec: chunk_id, doc_id and text must be str."""
        chunk = {**valid_output()["chunks"][0], field: bad}
        with pytest.raises(ContractError):
            runner.check_contract({**valid_output(), "chunks": [chunk]})

    @pytest.mark.parametrize("bad", [True, False, "0.5", None])
    def test_check_contract_chunk_score_not_a_number_raises(self, bad):
        """Spec: score is int/float, not bool."""
        chunk = {**valid_output()["chunks"][0], "score": bad}
        with pytest.raises(ContractError):
            runner.check_contract({**valid_output(), "chunks": [chunk]})

    @pytest.mark.parametrize("bad", [True, False, "12.5", None])
    def test_check_contract_latency_not_a_number_raises(self, bad):
        """Spec: latency_ms is int/float, not bool."""
        with pytest.raises(ContractError):
            runner.check_contract({**valid_output(), "latency_ms": bad})

    @pytest.mark.parametrize("bad", [-0.001, -1, -1000.0])
    def test_check_contract_negative_latency_raises(self, bad):
        """Spec: latency_ms must be >= 0 (just below 0 is rejected)."""
        with pytest.raises(ContractError):
            runner.check_contract({**valid_output(), "latency_ms": bad})

    @pytest.mark.parametrize("bad", [None, "answer", ["answer", "chunks", "latency_ms"], 5, ()])
    def test_check_contract_non_dict_raises(self, bad):
        """Spec: a non-dict return is rejected."""
        with pytest.raises(ContractError):
            runner.check_contract(bad)


# ================================================================ runner: helpers

class TestGitInfo:
    def test_git_info_real_repo_returns_40_hex_commit_and_bool(self):
        """Spec: git_info returns (HEAD commit, dirty?). Read-only call in the repo."""
        commit, dirty = runner.git_info()
        assert re.fullmatch(r"[0-9a-f]{40}", commit), f"not a 40-hex commit: {commit!r}"
        assert isinstance(dirty, bool)


class TestFileSha256:
    def test_file_sha256_known_content_matches_spec_prefix(self, tmp_path):
        """Spec: sha256("a") starts ca978112."""
        path = tmp_path / "a.txt"
        path.write_bytes(b"a")
        digest = runner.file_sha256(path)
        assert digest.startswith("ca978112"), digest
        assert len(digest) == 64

    def test_file_sha256_other_content_matches_spec_prefix(self, tmp_path):
        """Spec: sha256("b") starts 3e23e816 (one character changes the whole hash)."""
        path = tmp_path / "b.txt"
        path.write_bytes(b"b")
        assert runner.file_sha256(path).startswith("3e23e816")

    def test_file_sha256_empty_file_is_the_known_empty_hash(self, tmp_path):
        """Spec: SHA-256 of the file's bytes (empty file -> e3b0c442...)."""
        path = tmp_path / "empty.txt"
        path.write_bytes(b"")
        assert runner.file_sha256(path).startswith("e3b0c44298fc1c14")

    def test_file_sha256_hashes_bytes_not_text_so_non_ascii_matches_hashlib(self, tmp_path):
        """Spec: SHA-256 of a file's bytes, including non-ASCII UTF-8."""
        content = "café ₹5000\n".encode("utf-8")
        path = tmp_path / "u.txt"
        path.write_bytes(content)
        assert runner.file_sha256(path) == hashlib.sha256(content).hexdigest()

    def test_file_sha256_accepts_str_path(self, tmp_path):
        """Spec Interfaces: path is str | Path."""
        path = tmp_path / "s.txt"
        path.write_bytes(b"a")
        assert runner.file_sha256(str(path)).startswith("ca978112")


class TestMakeRunId:
    def test_make_run_id_matches_spec_format(self):
        """Spec: run id looks like 20261015T151200Z-v0-3fa1c27e (UTC time + name + 8 hex)."""
        assert re.fullmatch(r"\d{8}T\d{6}Z-v0-[0-9a-f]{8}", runner.make_run_id("v0"))

    def test_make_run_id_embeds_the_config_name(self):
        """Spec: run id contains the config name."""
        assert re.fullmatch(r"\d{8}T\d{6}Z-v1_chunk256-[0-9a-f]{8}", runner.make_run_id("v1_chunk256"))

    def test_make_run_id_repeated_calls_are_unique(self):
        """Spec: run_ids unique over 100 calls (8 hex chars: P(collision) = 4950 / 2**32, about 1e-6)."""
        ids = {runner.make_run_id("v0") for _ in range(100)}
        assert len(ids) == 100, "run ids repeated"


class TestLlmCallsLog:
    def test_llm_calls_max_id_missing_cache_file_is_zero(self, llm_config):
        """Spec: llm_calls_max_id is 0 if the table is missing (no cache file at all)."""
        assert not Path(llm_config["cache_path"]).exists()
        assert runner.llm_calls_max_id(llm_config) == 0

    def test_llm_calls_max_id_file_without_table_is_zero(self, llm_config):
        """Spec: llm_calls_max_id is 0 for a missing table (DB file exists, no llm_calls)."""
        with closing(sqlite3.connect(llm_config["cache_path"])) as conn:
            conn.execute("CREATE TABLE other (x INTEGER)")
            conn.commit()
        assert runner.llm_calls_max_id(llm_config) == 0

    def test_llm_calls_max_id_empty_table_is_zero(self, llm_config):
        """Spec: llm_calls_max_id is 0 if the table is empty."""
        with closing(client._connect(Path(llm_config["cache_path"]))):
            pass
        assert runner.llm_calls_max_id(llm_config) == 0

    def test_llm_calls_max_id_returns_highest_id(self, llm_config):
        """Spec: llm_calls_max_id is the highest llm_calls.id."""
        for _ in range(3):
            insert_llm_call(llm_config, "gemma4:31b-cloud", 0)
        assert runner.llm_calls_max_id(llm_config) == 3

    def _six_rows(self, llm_config):
        """ids 1..6: gemma hit, miss, miss, error; judge miss; gemma hit."""
        insert_llm_call(llm_config, "gemma", 1)              # id 1: hit
        insert_llm_call(llm_config, "gemma", 0)              # id 2: miss
        insert_llm_call(llm_config, "gemma", 0)              # id 3: miss
        insert_llm_call(llm_config, "gemma", 0, "Timeout")   # id 4: error
        insert_llm_call(llm_config, "judge", 0)              # id 5: miss
        insert_llm_call(llm_config, "gemma", 1)              # id 6: hit

    def test_llm_calls_summary_counts_hits_misses_errors_per_model_in_id_range(self, llm_config):
        """Spec: counts per model only for ids in (after_id, upto_id]."""
        self._six_rows(llm_config)
        # ids 2..5 -> gemma: ids 2,3 miss + id 4 error = 3 calls, 0 hits, 2 misses, 1 error
        #             judge: id 5 miss = 1 call, 1 miss
        summary = runner.llm_calls_summary(llm_config, 1, 5)
        assert summary == {
            "gemma": {"calls": 3, "hits": 0, "misses": 2, "errors": 1},
            "judge": {"calls": 1, "hits": 0, "misses": 1, "errors": 0},
        }

    def test_llm_calls_summary_after_id_is_exclusive_and_upto_id_inclusive(self, llm_config):
        """Spec: range is (after_id, upto_id]."""
        self._six_rows(llm_config)
        # (0, 2] -> ids 1 and 2: gemma 1 hit + 1 miss
        assert runner.llm_calls_summary(llm_config, 0, 2) == {
            "gemma": {"calls": 2, "hits": 1, "misses": 1, "errors": 0}}
        # (2, 2] -> empty range
        assert runner.llm_calls_summary(llm_config, 2, 2) == {}

    def test_llm_calls_summary_all_rows_totals(self, llm_config):
        """Spec: summary example {calls, hits, misses, errors} per model over the whole log."""
        self._six_rows(llm_config)
        # gemma ids 1,2,3,4,6: 5 calls, 2 hits (1,6), 2 misses (2,3), 1 error (4)
        summary = runner.llm_calls_summary(llm_config, 0, 6)
        assert summary["gemma"] == {"calls": 5, "hits": 2, "misses": 2, "errors": 1}

    def test_llm_calls_summary_missing_file_is_empty(self, llm_config):
        """Spec: no llm_calls log yet -> no calls to report."""
        assert runner.llm_calls_summary(llm_config, 0, 10) == {}


# ================================================================ runner: load_system and config files

class TestLoadSystem:
    def test_load_system_resolves_module_function_strings(self):
        """Spec: load_system imports the functions named "module:function"."""
        system = runner.load_system({
            "answer": f"{__name__}:fake_answer",
            "load_config": f"{__name__}:fake_load_config",
            "describe": f"{__name__}:fake_describe",
        })
        assert isinstance(system, System)
        assert system.answer is fake_answer
        assert system.load_config is fake_load_config
        assert system.describe is fake_describe

    def test_load_system_describe_null_gives_none(self):
        """Spec: describe is optional; null = no system metadata."""
        system = runner.load_system({"answer": f"{__name__}:fake_answer",
                                     "load_config": f"{__name__}:fake_load_config",
                                     "describe": None})
        assert system.describe is None

    def test_load_system_default_harness_yaml_names_resolve_to_callables(self):
        """Spec Config changes: configs/harness.yaml system block imports (read-only)."""
        config = yaml.safe_load((PROJECT_ROOT / "configs" / "harness.yaml").read_text(encoding="utf-8"))
        system = runner.load_system(config["system"])
        assert callable(system.answer) and callable(system.load_config) and callable(system.describe)

    @pytest.mark.parametrize("field", ["answer", "load_config"])
    @pytest.mark.parametrize("bad", [
        "no_colon_here",
        "nonexistent_module_xyz:fn",
        f"{__name__}:no_such_function",
    ])
    def test_load_system_bad_name_raises_value_error_naming_it(self, field, bad):
        """Spec: a bad "module:function" name raises ValueError naming it."""
        good = {"answer": f"{__name__}:fake_answer", "load_config": f"{__name__}:fake_load_config",
                "describe": None}
        runner.load_system(good)    # control: the good config loads
        with pytest.raises(ValueError, match=re.escape(bad)):
            runner.load_system({**good, field: bad})

    def test_load_system_bad_describe_name_raises_value_error_naming_it(self):
        """Spec: a bad describe name also raises ValueError naming it."""
        with pytest.raises(ValueError, match="no_such_describe"):
            runner.load_system({"answer": f"{__name__}:fake_answer",
                                "load_config": f"{__name__}:fake_load_config",
                                "describe": f"{__name__}:no_such_describe"})


class TestRealConfigFiles:
    def test_load_harness_config_real_file_has_spec_values(self):
        """Spec Config changes: configs/harness.yaml starting defaults (read-only)."""
        config = runner.load_harness_config(PROJECT_ROOT / "configs" / "harness.yaml")
        assert config["db_path"] == "data/results.sqlite"
        assert config["test_set_config"] == "configs/test_set.yaml"
        assert config["max_consecutive_failures"] == 5
        assert config["error_max_chars"] == 500
        assert config["system"] == {
            "answer": "rag.pipeline:answer",
            "load_config": "rag.pipeline:load_pipeline_config",
            "describe": "rag.pipeline:describe_system",
        }

    def test_v0_config_pins_the_generator_model_digest(self):
        """Spec Decision 3: configs/v0.yaml generation.model_digest is the /api/tags digest (read-only)."""
        config = yaml.safe_load((PROJECT_ROOT / "configs" / "v0.yaml").read_text(encoding="utf-8"))
        assert config["generation"]["model_digest"] == V0_MODEL_DIGEST


# ================================================================ runner: run_split

class TestRunSplit:
    def test_run_split_writes_one_output_row_per_query_and_completes(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: one outputs row per query, n_ok + n_failed == n_queries, status completed."""
        run_id = run_dev(fake_system, results_db, ready_queries, llm_config)
        run = store.get_run(results_db, run_id)
        # dev queries in the fixture set: q001, q002, q004, q006, q007 -> 5
        assert run_id == "run-1"
        assert run["status"] == "completed"
        assert run["n_queries"] == 5
        assert run["n_ok"] == 5 and run["n_failed"] == 0
        assert run["n_ok"] + run["n_failed"] == run["n_queries"]
        assert run["finished_at"] is not None
        assert len(store.get_outputs(results_db, run_id)) == 5

    def test_run_split_stores_answers_chunks_and_latency_exactly_as_returned(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: answers and chunks stored exactly as returned; latency_ms not changed by the runner."""
        for number, query in enumerate(filter_split(ready_queries, "dev")):
            fake_system.answer.script(query.question, {
                "answer": f"Special answer {query.id} ✓",
                "chunks": [{"chunk_id": f"{query.id}#000", "doc_id": "leave_policy",
                            "text": f"café {query.id}", "score": 0.5 + number / 100}],
                "latency_ms": 100.25 + number,
            })
        run_id = run_dev(fake_system, results_db, ready_queries, llm_config)
        rows = {r["query_id"]: r for r in store.get_outputs(results_db, run_id)}
        for number, query in enumerate(filter_split(ready_queries, "dev")):
            row = rows[query.id]
            assert row["answer"] == f"Special answer {query.id} ✓"
            assert chunks_of(row) == [{"chunk_id": f"{query.id}#000", "doc_id": "leave_policy",
                                       "text": f"café {query.id}", "score": 0.5 + number / 100}]
            # latency is stored untouched: 100.25 + number
            assert row["latency_ms"] == pytest.approx(100.25 + number)
            assert row["error"] is None

    def test_run_split_only_runs_the_given_queries(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: a dev run has outputs only for dev queries (no heldout query is run)."""
        run_id = run_dev(fake_system, results_db, ready_queries, llm_config)
        stored_ids = {r["query_id"] for r in store.get_outputs(results_db, run_id)}
        assert stored_ids == {"q001", "q002", "q004", "q006", "q007"}

    def test_run_split_calls_answer_in_id_order_with_exact_question_and_same_config(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: queries called in id order, exact question text, the same config dict."""
        meta = make_run_meta()
        dev = filter_split(ready_queries, "dev")
        store.sync_queries(results_db, ready_queries)
        runner.run_split(fake_system, meta["config"], list(reversed(dev)), results_db,
                         run_meta=meta, max_consecutive_failures=5, error_max_chars=500,
                         llm_config=llm_config)
        calls = fake_system.answer.calls
        assert [c["question"] for c in calls] == [q.question for q in sorted(dev, key=lambda q: q.id)]
        assert all(c["config"] is meta["config"] for c in calls), "config dict was not passed through"

    def test_run_split_never_edits_the_config(self, fake_system, results_db, ready_queries, llm_config):
        """Spec: the runner never edits the config; stored config_json is that exact dict."""
        meta = make_run_meta(config={"name": "v0", "z": 1, "retrieval": {"top_k": 4}})
        before = copy.deepcopy(meta["config"])
        run_id = run_dev(fake_system, results_db, ready_queries, llm_config, run_meta=meta)
        assert meta["config"] == before
        assert store.get_run(results_db, run_id)["config_json"] == json.dumps(before, sort_keys=True)

    def test_run_split_stores_run_metadata_from_run_meta(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: the run row stores split, config path, git commit/dirty, test-set hash, system info."""
        meta = make_run_meta(split="dev")
        meta.update(git_commit="e" * 40, git_dirty=True, test_set_sha256="f" * 64)
        run_id = run_dev(fake_system, results_db, ready_queries, llm_config, run_meta=meta)
        run = store.get_run(results_db, run_id)
        assert run["split"] == "dev"
        assert run["config_path"] == "configs/fake.yaml"
        assert run["git_commit"] == "e" * 40
        assert run["git_dirty"] == 1
        assert run["test_set_sha256"] == "f" * 64
        assert json.loads(run["system_json"]) == meta["system_info"]

    def test_run_split_leaves_scores_and_reviews_empty(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: no metric or judge in this step; scores and reviews stay empty."""
        run_dev(fake_system, results_db, ready_queries, llm_config)
        assert count_rows(results_db, "SELECT COUNT(*) FROM scores") == 0
        assert count_rows(results_db, "SELECT COUNT(*) FROM reviews") == 0

    def test_run_split_stores_the_run_row_before_outputs_so_status_is_running_during_the_run(
            self, fake_system, results_db, results_db_path, ready_queries, llm_config):
        """Spec: the run is 'running' while queries are processed (a crash never looks complete)."""
        seen = []

        def answer(question, config):
            with closing(sqlite3.connect(results_db_path)) as other:
                seen.append(other.execute("SELECT status FROM runs").fetchone()[0])
            return fake_system.answer.valid_output(question)

        system = System(answer=answer, load_config=fake_system.load_config, describe=None)
        run_dev(system, results_db, ready_queries, llm_config)
        assert seen and set(seen) == {"running"}

    def test_run_split_counts_only_llm_calls_made_during_the_run(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: llm_calls ids between start and end are this run's calls; summary stored per model."""
        insert_llm_call(llm_config, "gen-x", 0)    # id 1: before the run, must not be counted
        insert_llm_call(llm_config, "gen-x", 0)    # id 2: before the run, must not be counted
        state = {"n": 0}

        def answer(question, config):
            # calls 0..4 -> cached 0,1,0,1,0 : 3 misses, 2 hits
            insert_llm_call(llm_config, "gen-x", state["n"] % 2)
            state["n"] += 1
            return fake_system.answer.valid_output(question)

        system = System(answer=answer, load_config=fake_system.load_config, describe=None)
        run_id = run_dev(system, results_db, ready_queries, llm_config)
        run = store.get_run(results_db, run_id)
        assert run["llm_calls_start_id"] == 2
        assert json.loads(run["llm_calls_json"]) == {
            "gen-x": {"calls": 5, "hits": 2, "misses": 3, "errors": 0}}

    def test_run_split_without_describe_hook_still_completes(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: describe is optional (None = no system metadata)."""
        system = System(answer=fake_system.answer, load_config=fake_system.load_config, describe=None)
        run_id = run_dev(system, results_db, ready_queries, llm_config)
        assert store.get_run(results_db, run_id)["status"] == "completed"


class TestRunSplitFailures:
    class ContextLimitError(Exception):
        """Local stand-in for rag.pipeline.ContextLimitError (the name is what the runner records)."""

    def test_run_split_answer_raises_is_stored_as_failure_and_run_completes(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: an exception from answer() becomes a failure row with the error type; run still completed."""
        bad = query_by_id(ready_queries, "q004")
        fake_system.answer.script(bad.question, self.ContextLimitError("prompt too long: 4000"))
        run_id = run_dev(fake_system, results_db, ready_queries, llm_config)
        rows = {r["query_id"]: r for r in store.get_outputs(results_db, run_id)}
        run = store.get_run(results_db, run_id)
        assert len(rows) == 5, "failed query must still have a row"
        assert "ContextLimitError" in rows["q004"]["error"]
        assert "prompt too long: 4000" in rows["q004"]["error"]
        assert rows["q004"]["answer"] is None
        assert rows["q004"]["latency_ms"] is None
        assert chunks_of(rows["q004"]) is None
        assert rows["q001"]["error"] is None and rows["q007"]["answer"] is not None
        assert run["status"] == "completed"
        assert (run["n_ok"], run["n_failed"]) == (4, 1)

    def test_run_split_error_text_is_cut_to_error_max_chars(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: the stored error text is cut to error_max_chars."""
        bad = query_by_id(ready_queries, "q001")
        fake_system.answer.script(bad.question, self.ContextLimitError("x" * 1000))
        run_id = run_dev(fake_system, results_db, ready_queries, llm_config,
                         options={"error_max_chars": 50})
        error = store.get_outputs(results_db, run_id)[0]["error"]
        assert len(error) <= 50, f"error not cut: {len(error)} chars"
        assert error.startswith("ContextLimitError"), "error type must survive the cut"

    def test_run_split_short_error_is_not_padded_or_changed_by_the_limit(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: only errors longer than error_max_chars are cut."""
        bad = query_by_id(ready_queries, "q001")
        fake_system.answer.script(bad.question, ValueError("short"))
        run_id = run_dev(fake_system, results_db, ready_queries, llm_config,
                         options={"error_max_chars": 500})
        error = store.get_outputs(results_db, run_id)[0]["error"]
        assert error.startswith("ValueError") and error.endswith("short") and len(error) < 100

    def test_run_split_contract_breaking_return_is_stored_as_contract_error(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: a contract-breaking return becomes a failure row with ContractError."""
        bad = query_by_id(ready_queries, "q002")
        fake_system.answer.script(bad.question, {"answer": "no chunks or latency"})
        run_id = run_dev(fake_system, results_db, ready_queries, llm_config)
        rows = {r["query_id"]: r for r in store.get_outputs(results_db, run_id)}
        assert "ContractError" in rows["q002"]["error"]
        assert rows["q002"]["answer"] is None, "malformed answer must not be stored as an answer"
        assert store.get_run(results_db, run_id)["n_failed"] == 1

    def test_run_split_failures_stay_in_the_denominator(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: failures are stored and counted, never dropped (rows == n_queries)."""
        fake_system.answer.script(query_by_id(ready_queries, "q001").question, RuntimeError("a"))
        fake_system.answer.script(query_by_id(ready_queries, "q004").question, RuntimeError("b"))
        run_id = run_dev(fake_system, results_db, ready_queries, llm_config)
        run = store.get_run(results_db, run_id)
        assert count_rows(results_db, "SELECT COUNT(*) FROM outputs") == run["n_queries"] == 5
        assert run["n_ok"] + run["n_failed"] == 5
        assert count_rows(results_db, "SELECT COUNT(*) FROM outputs WHERE error IS NOT NULL") == 2

    def test_run_split_all_calls_fail_aborts_after_max_consecutive_failures(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: max_consecutive_failures=2 and every call failing -> aborted after exactly 2 rows."""
        fake_system.answer.fail_all(RuntimeError("model unreachable"))
        run_id = run_dev(fake_system, results_db, ready_queries, llm_config,
                         options={"max_consecutive_failures": 2})
        run = store.get_run(results_db, run_id)
        assert run["status"] == "aborted"
        assert len(store.get_outputs(results_db, run_id)) == 2
        assert len(fake_system.answer.calls) == 2, "run must stop calling answer() after aborting"
        assert (run["n_ok"], run["n_failed"]) == (0, 2)
        assert run["note"], "an aborted run needs a note saying why"
        assert run["finished_at"] is not None

    def test_run_split_ok_then_two_failures_aborts_after_three_rows(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: abort counts failures in a row; an earlier success does not postpone it."""
        # sorted dev ids: q001 ok, q002 fail, q004 fail -> abort (limit 2)
        for query_id in ("q002", "q004"):
            fake_system.answer.script(query_by_id(ready_queries, query_id).question, RuntimeError("x"))
        run_id = run_dev(fake_system, results_db, ready_queries, llm_config,
                         options={"max_consecutive_failures": 2})
        run = store.get_run(results_db, run_id)
        assert run["status"] == "aborted"
        assert [r["query_id"] for r in store.get_outputs(results_db, run_id)] == ["q001", "q002", "q004"]
        assert (run["n_ok"], run["n_failed"]) == (1, 2)

    def test_run_split_success_between_failures_resets_the_counter(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: a success between failures resets the consecutive-failure counter."""
        # sorted dev ids: q001 fail, q002 ok, q004 fail, q006 ok, q007 fail -> never 2 in a row
        for query_id in ("q001", "q004", "q007"):
            fake_system.answer.script(query_by_id(ready_queries, query_id).question, RuntimeError("x"))
        run_id = run_dev(fake_system, results_db, ready_queries, llm_config,
                         options={"max_consecutive_failures": 2})
        run = store.get_run(results_db, run_id)
        assert run["status"] == "completed"
        assert (run["n_ok"], run["n_failed"]) == (2, 3)
        assert len(store.get_outputs(results_db, run_id)) == 5

    def test_run_split_failures_below_the_limit_do_not_abort(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec boundary: limit - 1 failures in a row (1 of 2) do not abort."""
        fake_system.answer.script(query_by_id(ready_queries, "q007").question, RuntimeError("x"))
        run_id = run_dev(fake_system, results_db, ready_queries, llm_config,
                         options={"max_consecutive_failures": 2})
        assert store.get_run(results_db, run_id)["status"] == "completed"

    def test_run_split_keyboard_interrupt_marks_interrupted_keeps_rows_and_reraises(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: KeyboardInterrupt -> status 'interrupted', re-raised, earlier rows kept."""
        # sorted dev ids: q001, q002 stored; q004 raises KeyboardInterrupt
        fake_system.answer.script(query_by_id(ready_queries, "q004").question, KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            run_dev(fake_system, results_db, ready_queries, llm_config)
        run = store.get_run(results_db, "run-1")
        assert run["status"] == "interrupted"
        assert run["finished_at"] is not None
        assert [r["query_id"] for r in store.get_outputs(results_db, "run-1")] == ["q001", "q002"]
        assert run["n_ok"] == 2

    def test_run_split_keyboard_interrupt_is_not_stored_as_a_failure_row(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: KeyboardInterrupt is never caught per item; no failure row for it."""
        fake_system.answer.script(query_by_id(ready_queries, "q001").question, KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            run_dev(fake_system, results_db, ready_queries, llm_config)
        assert count_rows(results_db, "SELECT COUNT(*) FROM outputs") == 0

    def test_run_split_describe_index_key_changed_at_end_marks_run_invalid_with_note(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: describe() returning a different index_key at the end -> status invalid with a note."""
        fake_system.describe.result = {"index": {"index_key": "key-CHANGED"},
                                       "generator_model": "fake-gen",
                                       "generator_model_digest": "digest-1"}
        run_id = run_dev(fake_system, results_db, ready_queries, llm_config)
        run = store.get_run(results_db, run_id)
        assert run["status"] == "invalid"
        assert run["note"], "an invalid run needs a note saying why"
        assert len(store.get_outputs(results_db, run_id)) == 5, "answers are still stored"

    def test_run_split_describe_unchanged_at_end_stays_completed(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec control: identical describe() output at the end leaves the run completed."""
        meta = make_run_meta()
        fake_system.describe.result = dict(meta["system_info"])
        run_id = run_dev(fake_system, results_db, ready_queries, llm_config, run_meta=meta)
        assert store.get_run(results_db, run_id)["status"] == "completed"
        assert fake_system.describe.calls, "describe must be called again at the end of the run"

    def test_run_split_describe_is_called_with_the_run_config(
            self, fake_system, results_db, ready_queries, llm_config):
        """Spec: the end-of-run describe check uses the same config as the run."""
        meta = make_run_meta()
        fake_system.describe.result = dict(meta["system_info"])
        run_dev(fake_system, results_db, ready_queries, llm_config, run_meta=meta)
        assert all(c == meta["config"] for c in fake_system.describe.calls)


# ================================================================ runner: describe_system and get_model_digest

class FakeTagsResponse(io.BytesIO):
    """A urlopen() result: JSON bytes that works as a context manager (BytesIO already does)."""


class FakeUrlopen:
    """Fake urllib.request.urlopen: serves a fixed /api/tags JSON and records the requested URL."""

    def __init__(self, models=None, error=None):
        self.models = models if models is not None else []
        self.error = error
        self.urls = []

    def __call__(self, url, *args, **kwargs):
        self.urls.append(url if isinstance(url, str) else url.full_url)
        if self.error is not None:
            raise self.error
        return FakeTagsResponse(json.dumps({"models": self.models}).encode("utf-8"))


@pytest.fixture
def fake_urlopen(monkeypatch):
    """Replace urllib.request.urlopen with a FakeUrlopen (no network)."""
    fake = FakeUrlopen(models=[
        {"name": "gemma4:31b-cloud", "digest": "digest-gemma"},
        {"name": "gpt-oss:120b-cloud", "digest": "digest-gpt"},
    ])
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


class TestGetModelDigest:
    def test_get_model_digest_returns_digest_of_listed_tag(self, fake_urlopen, llm_config):
        """Spec: parses the /api/tags JSON and returns the digest of the model tag."""
        assert client.get_model_digest("gemma4:31b-cloud", config=llm_config) == "digest-gemma"
        assert client.get_model_digest("gpt-oss:120b-cloud", config=llm_config) == "digest-gpt"

    def test_get_model_digest_unknown_tag_returns_none(self, fake_urlopen, llm_config):
        """Spec: returns None if the tag is not listed."""
        assert client.get_model_digest("no-such-model:1b", config=llm_config) is None

    def test_get_model_digest_strips_v1_from_base_url(self, fake_urlopen, llm_config):
        """Spec: the native endpoint is base_url with the trailing /v1 removed, plus /api/tags."""
        llm_config["base_url"] = "http://localhost:11434/v1"
        client.get_model_digest("gemma4:31b-cloud", config=llm_config)
        assert fake_urlopen.urls == ["http://localhost:11434/api/tags"]

    def test_get_model_digest_trailing_slash_after_v1_is_also_removed(self, fake_urlopen, llm_config):
        """Spec: /v1 is removed from base_url (also with a trailing slash)."""
        llm_config["base_url"] = "http://localhost:11434/v1/"
        client.get_model_digest("gemma4:31b-cloud", config=llm_config)
        assert fake_urlopen.urls == ["http://localhost:11434/api/tags"]

    def test_get_model_digest_connection_error_is_raised_not_swallowed(self, monkeypatch, llm_config):
        """Spec: raises on connection errors (a run must not start against an unreachable server)."""
        fake = FakeUrlopen(error=urllib.error.URLError("connection refused"))
        monkeypatch.setattr(urllib.request, "urlopen", fake)
        with pytest.raises(urllib.error.URLError):
            client.get_model_digest("gemma4:31b-cloud", config=llm_config)

    def test_get_model_digest_empty_model_list_returns_none(self, monkeypatch, llm_config):
        """Spec: no listed models -> None."""
        monkeypatch.setattr(urllib.request, "urlopen", FakeUrlopen(models=[]))
        assert client.get_model_digest("gemma4:31b-cloud", config=llm_config) is None


@pytest.fixture
def digests(monkeypatch):
    """Patch rag.pipeline.get_model_digest (where describe_system uses it) with a fixed table."""
    table = {"fake-gen:cloud": "gen-digest", "fake-embed": "emb-digest"}

    def fake_digest(model, *, config=None):
        return table.get(model)

    monkeypatch.setattr("rag.pipeline.get_model_digest", fake_digest)
    return table


class TestDescribeSystem:
    def test_describe_system_matching_digest_returns_facts(self, pipeline_config, fake_embed, digests):
        """Spec: matching digest -> dict with index, models and digests."""
        from rag.pipeline import describe_system

        pipeline_config["generation"]["model_digest"] = "gen-digest"
        info = describe_system(pipeline_config)
        assert re.fullmatch(r"[0-9a-f]{64}", info["index"]["index_key"])
        assert info["generator_model"] == "fake-gen:cloud"
        assert info["generator_model_digest"] == "gen-digest"
        assert info["embedding_model"] == "fake-embed"
        assert info["embedding_model_digest"] == "emb-digest"

    def test_describe_system_null_digest_raises_model_digest_error(
            self, pipeline_config, fake_embed, digests):
        """Spec: generation.model_digest null -> ModelDigestError."""
        from rag.pipeline import describe_system

        assert pipeline_config["generation"]["model_digest"] is None
        with pytest.raises(ModelDigestError):
            describe_system(pipeline_config)

    def test_describe_system_digest_mismatch_raises_model_digest_error(
            self, pipeline_config, fake_embed, digests):
        """Spec: configured digest differs from the server's -> ModelDigestError."""
        from rag.pipeline import describe_system

        pipeline_config["generation"]["model_digest"] = "stale-digest"
        with pytest.raises(ModelDigestError):
            describe_system(pipeline_config)

    def test_describe_system_model_not_listed_by_server_raises_model_digest_error(
            self, pipeline_config, fake_embed, digests):
        """Spec: the server reports no digest (None) for the model, so it cannot equal the config."""
        from rag.pipeline import describe_system

        del digests["fake-gen:cloud"]
        pipeline_config["generation"]["model_digest"] = "gen-digest"
        with pytest.raises(ModelDigestError):
            describe_system(pipeline_config)

    def test_describe_system_embedding_digest_is_recorded_only_never_refused(
            self, pipeline_config, fake_embed, digests):
        """Spec Decision 2: the embedding digest is metadata only (unknown embedding model -> None, no error)."""
        from rag.pipeline import describe_system

        del digests["fake-embed"]
        pipeline_config["generation"]["model_digest"] = "gen-digest"
        assert describe_system(pipeline_config)["embedding_model_digest"] is None

    def test_describe_system_index_key_is_stable_for_the_same_config(
            self, pipeline_config, fake_embed, digests):
        """Spec: describe is used to compare index keys at run start and end; same config -> same key."""
        from rag.pipeline import describe_system

        pipeline_config["generation"]["model_digest"] = "gen-digest"
        first = describe_system(pipeline_config)
        second = describe_system(pipeline_config)
        assert first["index"]["index_key"] == second["index"]["index_key"]


# ================================================================ runner: main (CLI)

class MainEnv:
    """A throwaway harness setup in tmp_path for runner.main(): config files, test set, results DB."""

    def __init__(self, tmp_path, testset_config, llm_config):
        self.tmp_path = tmp_path
        self.testset_config = testset_config
        self.llm_config = llm_config
        LOG_CONFIG["llm_config"] = llm_config
        self.db_path = tmp_path / "main_results.sqlite"
        self.harness_path = tmp_path / "harness_test.yaml"
        self.pipeline_path = tmp_path / "fake_pipeline.yaml"
        self.pipeline_path.write_text(yaml.safe_dump({"name": "fakecfg", "top_k": 3}), encoding="utf-8")
        (tmp_path / "test_set_test.yaml").write_text(yaml.safe_dump(testset_config), encoding="utf-8")
        self.write_harness()

    def write_harness(self, answer="fake_answer", describe="fake_describe",
                      max_consecutive_failures=2, error_max_chars=100):
        """(Re)write the tmp harness YAML; `describe=None` writes null."""
        prefix = f"{__name__}:"
        config = {
            "db_path": str(self.db_path),
            "test_set_config": str(self.tmp_path / "test_set_test.yaml"),
            "max_consecutive_failures": max_consecutive_failures,
            "error_max_chars": error_max_chars,
            "system": {
                "answer": prefix + answer,
                "load_config": prefix + "fake_load_config",
                "describe": (prefix + describe) if describe is not None else None,
            },
        }
        self.harness_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def argv(self, split="dev", *flags):
        """Command-line arguments for main()."""
        return ["--config", str(self.pipeline_path), "--split", split,
                "--harness-config", str(self.harness_path), *flags]

    def run_rows(self):
        """All rows of the runs table as dicts; [] if the DB file was never created."""
        if not self.db_path.exists():
            return []
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM runs")]

    def output_rows(self):
        """All rows of the outputs table as dicts; [] if the DB file was never created."""
        if not self.db_path.exists():
            return []
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM outputs")]


@pytest.fixture
def main_env(tmp_path, testset_config, ready_queries, llm_config, monkeypatch):
    """Everything main() needs, all inside tmp_path; git is patched to a clean tree, LLM config to tmp."""
    save_test_set(ready_queries, testset_config["path"])
    monkeypatch.setattr(runner, "git_info", lambda: ("a" * 40, False))
    monkeypatch.setattr(runner, "load_llm_config", lambda *args, **kwargs: llm_config)
    fake_describe_flips.calls = 0
    return MainEnv(tmp_path, testset_config, llm_config)


class TestMain:
    def test_main_dev_run_completes_with_exit_zero_and_stores_the_run(self, main_env, capsys):
        """Spec: exit 0 on completed; run row and one output per dev query."""
        assert runner.main(main_env.argv("dev")) == 0
        runs = main_env.run_rows()
        assert len(runs) == 1
        run = runs[0]
        assert run["split"] == "dev"
        assert run["status"] == "completed"
        assert (run["n_queries"], run["n_ok"], run["n_failed"]) == (5, 5, 0)
        assert len(main_env.output_rows()) == 5
        out = capsys.readouterr().out
        assert run["run_id"] in out, "the run_id must be printed"
        assert f"run {run['run_id']}: completed" in out
        assert "5 queries, 5 ok, 0 failed" in out

    def test_main_stores_metadata_git_hash_config_path_and_system_info(self, main_env):
        """Spec: run stores git commit, dirty flag, config path, test-set SHA-256, system info, full config."""
        runner.main(main_env.argv("dev"))
        run = main_env.run_rows()[0]
        test_set_bytes = Path(main_env.testset_config["path"]).read_bytes()
        assert run["git_commit"] == "a" * 40
        assert run["git_dirty"] == 0
        assert run["config_path"] == str(main_env.pipeline_path)
        assert run["test_set_sha256"] == hashlib.sha256(test_set_bytes).hexdigest()
        assert json.loads(run["system_json"]) == fake_describe(None)
        assert json.loads(run["config_json"]) == {"name": "fakecfg", "top_k": 3}

    def test_main_syncs_the_whole_test_set_into_queries_table(self, main_env):
        """Spec: queries table is created from the test set (all 8, not only the split)."""
        runner.main(main_env.argv("dev"))
        with closing(sqlite3.connect(main_env.db_path)) as conn:
            assert count_rows(conn, "SELECT COUNT(*) FROM queries") == 8

    def test_main_dev_run_outputs_only_belong_to_dev_queries(self, main_env):
        """Spec: every output of a dev run belongs to a dev query."""
        runner.main(main_env.argv("dev"))
        with closing(sqlite3.connect(main_env.db_path)) as conn:
            splits = conn.execute(
                "SELECT q.split, COUNT(*) FROM outputs o JOIN queries q ON q.id = o.query_id"
                " GROUP BY q.split").fetchall()
        assert [tuple(r) for r in splits] == [("dev", 5)]

    def test_main_two_runs_get_different_run_ids(self, main_env):
        """Spec: run_ids are unique; a rerun adds a second run row."""
        runner.main(main_env.argv("dev"))
        runner.main(main_env.argv("dev"))
        ids = [r["run_id"] for r in main_env.run_rows()]
        assert len(ids) == 2 and len(set(ids)) == 2

    def test_main_heldout_without_flag_exits_one_and_writes_no_run(self, main_env, capsys):
        """Spec: --split heldout without --allow-heldout is refused (exit 1, no runs row)."""
        assert runner.main(main_env.argv("heldout")) == 1
        assert main_env.run_rows() == []
        assert main_env.output_rows() == []
        assert capsys.readouterr().err, "a refusal must print its reason"

    def test_main_heldout_refusal_message_names_step_12(self, main_env, capsys):
        """Spec: the held-out refusal message names Step 12."""
        runner.main(main_env.argv("heldout"))
        assert "12" in capsys.readouterr().err

    def test_main_heldout_with_allow_flag_runs_the_three_heldout_queries(self, main_env):
        """Spec control: --allow-heldout lets the heldout split run (3 heldout queries in the fixture set)."""
        assert runner.main(main_env.argv("heldout", "--allow-heldout")) == 0
        run = main_env.run_rows()[0]
        assert (run["split"], run["n_queries"]) == ("heldout", 3)

    def test_main_dirty_tree_without_flag_exits_one_and_writes_no_run(self, main_env, monkeypatch):
        """Spec: a dirty working tree is refused unless --allow-dirty (exit 1, no runs row)."""
        monkeypatch.setattr(runner, "git_info", lambda: ("b" * 40, True))
        assert runner.main(main_env.argv("dev")) == 1
        assert main_env.run_rows() == []

    def test_main_dirty_tree_with_allow_dirty_runs_and_stores_git_dirty_one(self, main_env, monkeypatch):
        """Spec control: --allow-dirty runs and the run row records git_dirty = 1."""
        monkeypatch.setattr(runner, "git_info", lambda: ("b" * 40, True))
        assert runner.main(main_env.argv("dev", "--allow-dirty")) == 0
        run = main_env.run_rows()[0]
        assert run["git_dirty"] == 1
        assert run["git_commit"] == "b" * 40

    def test_main_test_set_not_ready_exits_one_and_writes_no_run(self, main_env, ready_queries):
        """Spec: a test set that is not ready (an unreviewed query) is refused: exit 1, no runs row."""
        unreviewed = [q.model_copy(update={"reviewed": False}) if q.id == "q002" else q
                      for q in ready_queries]
        save_test_set(unreviewed, main_env.testset_config["path"])
        assert runner.main(main_env.argv("dev")) == 1
        assert main_env.run_rows() == []

    def test_main_missing_split_assignment_is_not_ready_and_refused(self, main_env, ready_queries):
        """Spec: check_ready() must be [] before a run (a query without a split is refused)."""
        no_split = [q.model_copy(update={"split": None}) if q.id == "q001" else q for q in ready_queries]
        save_test_set(no_split, main_env.testset_config["path"])
        assert runner.main(main_env.argv("dev")) == 1
        assert main_env.run_rows() == []

    def test_main_model_digest_error_from_describe_exits_one_and_writes_no_run(self, main_env):
        """Spec: ModelDigestError from describe -> refused before starting (exit 1, no runs row)."""
        main_env.write_harness(describe="fake_describe_refuses")
        assert runner.main(main_env.argv("dev")) == 1
        assert main_env.run_rows() == []

    def test_main_without_describe_hook_still_completes(self, main_env):
        """Spec: describe may be null in harness.yaml."""
        main_env.write_harness(describe=None)
        assert runner.main(main_env.argv("dev")) == 0
        assert main_env.run_rows()[0]["status"] == "completed"

    def test_main_every_answer_failing_exits_two_with_aborted_status(self, main_env):
        """Spec: exit code 2 when the run finishes as aborted."""
        main_env.write_harness(answer="fake_answer_always_fails", max_consecutive_failures=2)
        assert runner.main(main_env.argv("dev")) == 2
        run = main_env.run_rows()[0]
        assert run["status"] == "aborted"
        assert len(main_env.output_rows()) == 2
        assert all(r["error"] and "RuntimeError" in r["error"] for r in main_env.output_rows())

    def test_main_index_key_changing_during_run_exits_two_with_invalid_status(self, main_env):
        """Spec: exit code 2 when the run finishes as invalid (describe differs at the end)."""
        main_env.write_harness(describe="fake_describe_flips")
        assert runner.main(main_env.argv("dev")) == 2
        run = main_env.run_rows()[0]
        assert run["status"] == "invalid"
        assert run["note"]

    def test_main_passes_the_config_loaded_from_the_config_argument_to_answer(self, main_env):
        """Spec: main loads the pipeline config named by --config via system.load_config."""
        main_env.pipeline_path.write_text(yaml.safe_dump({"name": "other", "top_k": 9}), encoding="utf-8")
        runner.main(main_env.argv("dev"))
        run = main_env.run_rows()[0]
        assert json.loads(run["config_json"]) == {"name": "other", "top_k": 9}

    def test_main_stores_and_prints_this_runs_llm_call_summary(self, main_env, capsys):
        """Spec: the run stores a per-model call summary and the final line prints it."""
        main_env.write_harness(answer="fake_answer_logs_call")
        insert_llm_call(main_env.llm_config, "gen-x", 0)    # id 1: before the run, not counted
        assert runner.main(main_env.argv("dev")) == 0
        run = main_env.run_rows()[0]
        # 5 dev queries -> 5 logged calls, all misses (cached = 0); the earlier row is excluded
        assert json.loads(run["llm_calls_json"]) == {
            "gen-x": {"calls": 5, "hits": 0, "misses": 5, "errors": 0}}
        assert run["llm_calls_start_id"] == 1
        assert "gen-x: 5 calls" in capsys.readouterr().out


# ================================================================ layering

class TestLayering:
    @pytest.mark.parametrize("name", ["store.py", "runner.py"])
    def test_evalharness_store_and_runner_have_no_static_import_of_rag(self, name):
        """Spec: evalharness/store.py and runner.py have no `import rag` / `from rag`."""
        path = PROJECT_ROOT / "evalharness" / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(n.split(".")[0] == "rag" for n in names), f"{path} imports rag"

    def test_rag_package_never_mentions_evalharness(self):
        """Spec: `grep -rn "evalharness" rag/` finds nothing."""
        for path in (PROJECT_ROOT / "rag").rglob("*.py"):
            assert "evalharness" not in path.read_text(encoding="utf-8"), f"{path} mentions evalharness"
