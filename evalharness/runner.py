"""Run one split of the test set through the system under test and store every output (Step 5).

The runner knows the system only through the contract:
    answer(question, config) -> {"answer": str,
                                 "chunks": [{"chunk_id", "doc_id", "text", "score"}],
                                 "latency_ms": float}
The functions are named as "module:function" in configs/harness.yaml and
imported by name, so evalharness/ has no import of rag/ and another RAG system
can be evaluated by editing that file. An optional `describe(config)` hook
returns run metadata the config alone does not pin down (index key, model
digests).

Each run stores: the full config, git commit (and whether the tree was dirty),
the test set's SHA-256, the describe output, one outputs row per query
(answer or error) and a per-model summary of this run's LLM calls (from the
llm_calls log). Failed queries are stored as failures, never dropped. No
metric is computed here.

CLI:
    python -m evalharness.runner --config configs/v0.yaml --split dev
Exit codes: 0 completed; 1 refused before starting (nothing written to runs);
2 the run finished as aborted or invalid.
"""

import argparse
import hashlib
import importlib
import json
import logging
import math
import secrets
import sqlite3
import subprocess
import sys
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml
from tqdm import tqdm

from evalharness import store
from evalharness.schemas import SPLITS, Query
from evalharness.testset import (check_ready, filter_split, load_test_set, load_testset_config,
                                 resolve_path)
from llm.client import load_config as load_llm_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HARNESS_CONFIG = "configs/harness.yaml"
REQUIRED_CONFIG_KEYS = ("db_path", "test_set_config", "max_consecutive_failures",
                        "error_max_chars", "system")

# The contract's exact key sets (CLAUDE.md "THE CONTRACT").
CONTRACT_KEYS = {"answer", "chunks", "latency_ms"}
CHUNK_KEYS = {"chunk_id", "doc_id", "text", "score"}


class ContractError(Exception):
    """answer() returned something that does not match the contract."""


@dataclass
class System:
    """The system under test, as three plain functions."""
    answer: Callable[[str, dict], dict]
    load_config: Callable[[str], dict]
    describe: Callable[[dict], dict] | None


# ---------------------------------------------------------------- config and system

def _is_int(value) -> bool:
    """True for a real int (bool is a subclass of int in Python, so exclude it)."""
    return isinstance(value, int) and not isinstance(value, bool)


def load_harness_config(path: str | Path = DEFAULT_HARNESS_CONFIG) -> dict:
    """Read configs/harness.yaml and check every key; ValueError names the problem."""
    path = resolve_path(path)
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    for key in REQUIRED_CONFIG_KEYS:
        if key not in config:
            raise ValueError(f"{path}: missing key '{key}'")
    for key in ("db_path", "test_set_config"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{path}: '{key}' must be a non-empty string")
    for key in ("max_consecutive_failures", "error_max_chars"):
        if not _is_int(config[key]) or config[key] < 1:
            raise ValueError(f"{path}: '{key}' must be an integer >= 1")
    system = config["system"]
    if not isinstance(system, dict):
        raise ValueError(f"{path}: 'system' must be a mapping")
    for key in ("answer", "load_config"):
        if not isinstance(system.get(key), str):
            raise ValueError(f"{path}: 'system.{key}' must be a \"module:function\" string")
    if "describe" not in system or not (system["describe"] is None
                                        or isinstance(system["describe"], str)):
        raise ValueError(f"{path}: 'system.describe' must be a \"module:function\" string or null")
    return config


def _import_function(name: str) -> Callable:
    """Import "package.module:function" and return the function."""
    module_name, sep, function_name = name.partition(":")
    if not sep or not module_name or not function_name:
        raise ValueError(f"{name!r} is not in \"module:function\" form")
    try:
        function = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"cannot import {name!r}: {error}") from error
    if not callable(function):
        raise ValueError(f"{name!r} is not a function")
    return function


def load_system(system_config: dict) -> System:
    """Build a System from the harness config's `system` block."""
    describe = system_config.get("describe")
    return System(
        answer=_import_function(system_config["answer"]),
        load_config=_import_function(system_config["load_config"]),
        describe=_import_function(describe) if describe is not None else None,
    )


# ---------------------------------------------------------------- contract check

def _is_number(value) -> bool:
    """A finite int or float, not bool (NaN or inf would not survive JSON storage)."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def check_contract(out) -> None:
    """Raise ContractError unless `out` matches the answer() contract exactly."""
    if not isinstance(out, dict):
        raise ContractError(f"answer() returned {type(out).__name__}, expected dict")
    if set(out) != CONTRACT_KEYS:
        raise ContractError(f"answer() keys are {sorted(out)}, expected {sorted(CONTRACT_KEYS)}")
    if not isinstance(out["answer"], str):
        raise ContractError("'answer' must be a str")
    if not _is_number(out["latency_ms"]) or out["latency_ms"] < 0:
        raise ContractError("'latency_ms' must be a finite number >= 0")
    if not isinstance(out["chunks"], list):
        raise ContractError("'chunks' must be a list")
    for rank, chunk in enumerate(out["chunks"], start=1):
        if not isinstance(chunk, dict) or set(chunk) != CHUNK_KEYS:
            raise ContractError(f"chunk {rank} must be a dict with keys {sorted(CHUNK_KEYS)}")
        for key in ("chunk_id", "doc_id", "text"):
            if not isinstance(chunk[key], str):
                raise ContractError(f"chunk {rank} '{key}' must be a str")
        if not _is_number(chunk["score"]):
            raise ContractError(f"chunk {rank} 'score' must be a finite number")


# ---------------------------------------------------------------- run metadata

def git_info() -> tuple[str, bool]:
    """(HEAD commit hash, True if the working tree has uncommitted or untracked changes)."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout
    return git("rev-parse", "HEAD").strip(), bool(git("status", "--porcelain").strip())


def file_sha256(path: str | Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(resolve_path(path).read_bytes()).hexdigest()


def make_run_id(config_name: str) -> str:
    """UTC start time + config name + 8 random hex, e.g. "20261015T151200Z-v0-3fa1c27e"."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{config_name}-{secrets.token_hex(4)}"


def _open_llm_log(llm_config: dict) -> sqlite3.Connection | None:
    """Read-only connection to the LLM cache DB, or None if the file does not exist yet."""
    path = Path(llm_config["cache_path"])
    if not path.exists():
        return None
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)


def llm_calls_max_id(llm_config: dict) -> int:
    """Highest llm_calls.id so far; 0 if the cache file or table does not exist yet."""
    conn = _open_llm_log(llm_config)
    if conn is None:
        return 0
    with closing(conn):
        try:
            return conn.execute("SELECT COALESCE(MAX(id), 0) FROM llm_calls").fetchone()[0]
        except sqlite3.OperationalError:   # no llm_calls table yet
            return 0


def llm_calls_summary(llm_config: dict, after_id: int, upto_id: int) -> dict:
    """Per-model counts of the llm_calls rows with after_id < id <= upto_id (this run's calls).

    {"gemma4:31b-cloud": {"calls": 80, "hits": 1, "misses": 79, "errors": 0}}.
    A miss is a real API call that succeeded; an error is any failed call.
    Valid only while no other process uses the same cache at the same time.
    """
    conn = _open_llm_log(llm_config)
    if conn is None:
        return {}
    with closing(conn):
        try:
            rows = conn.execute(
                "SELECT model, COUNT(*),"
                " SUM(cached = 1 AND error IS NULL),"
                " SUM(cached = 0 AND error IS NULL),"
                " SUM(error IS NOT NULL)"
                " FROM llm_calls WHERE id > ? AND id <= ? GROUP BY model ORDER BY model",
                (after_id, upto_id)).fetchall()
        except sqlite3.OperationalError:   # no llm_calls table yet
            return {}
    return {model: {"calls": calls, "hits": hits, "misses": misses, "errors": errors}
            for model, calls, hits, misses, errors in rows}


# ---------------------------------------------------------------- the run loop

def run_split(system: System, config: dict, queries: list[Query], conn: sqlite3.Connection, *,
              run_meta: dict, max_consecutive_failures: int, error_max_chars: int,
              llm_config: dict) -> str:
    """Run every query (in id order) through system.answer and store each output or failure.

    run_meta holds the start_run fields: run_id, split, config, config_path,
    git_commit, git_dirty, test_set_sha256, system_info. Final status:
    - completed:   every query has an outputs row (some may be failures),
    - aborted:     max_consecutive_failures failures in a row; the rest not run,
    - invalid:     describe() gave different facts (index, digest) at the end,
    - interrupted: Ctrl-C; the KeyboardInterrupt is re-raised after closing the run.
    Returns the run_id.
    """
    run_id = run_meta["run_id"]
    start_id = llm_calls_max_id(llm_config)
    store.start_run(conn, **run_meta, n_queries=len(queries), llm_calls_start_id=start_id)

    n_ok = n_failed = streak = 0
    status, note = "completed", None

    def finish(final_status: str, final_note: str | None) -> None:
        store.finish_run(conn, run_id, status=final_status, n_ok=n_ok, n_failed=n_failed,
                         llm_calls_summary=llm_calls_summary(llm_config, start_id,
                                                             llm_calls_max_id(llm_config)),
                         note=final_note)

    try:
        for query in tqdm(sorted(queries, key=lambda q: q.id), desc=run_id, unit="query"):
            try:
                out = system.answer(query.question, config)
                check_contract(out)
            except Exception as error:
                # Any failure (timeout, ContextLimitError, broken contract) is stored
                # as this query's result and counted; it is never dropped or retried here.
                message = f"{type(error).__name__}: {error}"[:error_max_chars]
                store.write_output(conn, run_id=run_id, query_id=query.id, answer=None,
                                   chunks=None, latency_ms=None, error=message)
                n_failed += 1
                streak += 1
                if streak >= max_consecutive_failures:
                    status = "aborted"
                    note = f"{streak} consecutive failures; last: {message}"
                    break
                continue
            store.write_output(conn, run_id=run_id, query_id=query.id, answer=out["answer"],
                               chunks=out["chunks"], latency_ms=out["latency_ms"], error=None)
            n_ok += 1
            streak = 0
    except KeyboardInterrupt:
        finish("interrupted", f"interrupted after {n_ok + n_failed} of {len(queries)} queries")
        raise

    # The corpus is re-read on every answer() call, so check nothing changed mid-run.
    if status == "completed" and system.describe is not None:
        try:
            end_info = system.describe(config)
        except Exception as error:
            status, note = "invalid", f"describe() failed at the end: {type(error).__name__}: {error}"
        else:
            if end_info != run_meta["system_info"]:
                status, note = "invalid", "system facts changed during the run (index or model digest)"
    finish(status, note)
    return run_id


# ---------------------------------------------------------------- CLI

def _format_calls(summary: dict) -> str:
    """'model: 80 calls, 1 hit, 79 misses, 0 errors' per model, or 'no LLM calls'."""
    if not summary:
        return "no LLM calls"
    return "; ".join(f"{model}: {c['calls']} calls, {c['hits']} hits, {c['misses']} misses, "
                     f"{c['errors']} errors" for model, c in summary.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evalharness.runner",
                                     description="Run one test-set split and store every output.")
    parser.add_argument("--config", required=True, help="pipeline config YAML, e.g. configs/v0.yaml")
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--harness-config", default=DEFAULT_HARNESS_CONFIG)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="run even with uncommitted changes (stored as git_dirty = 1)")
    parser.add_argument("--allow-heldout", action="store_true",
                        help="allow --split heldout (Step 12 only)")
    args = parser.parse_args(argv)
    # Per-call INFO lines from llm.client / rag.index would bury the progress bar;
    # warnings (e.g. retries) still show.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    def refuse(problem: str) -> int:
        print(f"error: {problem}", file=sys.stderr)
        return 1

    # Every check below happens before anything is written to the results DB.
    try:
        harness = load_harness_config(args.harness_config)
        testset_config = load_testset_config(harness["test_set_config"])
        queries = load_test_set(testset_config["path"])
    except ValueError as error:
        return refuse(str(error))
    problems = check_ready(queries, testset_config)
    if problems:
        return refuse("test set is not ready: " + "; ".join(problems))
    if args.split == "heldout" and not args.allow_heldout:
        return refuse("the held-out split is used once, in Step 12; pass --allow-heldout only then")
    git_commit, git_dirty = git_info()
    if git_dirty and not args.allow_dirty:
        return refuse("uncommitted changes: commit first so git_commit is the code that runs "
                      "(or pass --allow-dirty)")
    try:
        system = load_system(harness["system"])
        config = system.load_config(args.config)
    except ValueError as error:
        return refuse(str(error))
    try:
        system_info = system.describe(config) if system.describe is not None else {}
    except Exception as error:
        return refuse(f"describe() failed: {type(error).__name__}: {error}")

    split_queries = filter_split(queries, args.split)
    llm_config = load_llm_config()
    run_id = make_run_id(config.get("name", Path(args.config).stem))
    print(f"run {run_id}: {len(split_queries)} {args.split} queries, config {args.config}")

    with closing(store.connect(harness["db_path"])) as conn:
        store.sync_queries(conn, queries)
        run_split(system, config, split_queries, conn,
                  run_meta={"run_id": run_id, "split": args.split, "config": config,
                            "config_path": args.config, "git_commit": git_commit,
                            "git_dirty": git_dirty,
                            "test_set_sha256": file_sha256(testset_config["path"]),
                            "system_info": system_info},
                  max_consecutive_failures=harness["max_consecutive_failures"],
                  error_max_chars=harness["error_max_chars"], llm_config=llm_config)
        run = store.get_run(conn, run_id)

    print(f"run {run_id}: {run['status']} | {run['n_queries']} queries, {run['n_ok']} ok, "
          f"{run['n_failed']} failed | {_format_calls(json.loads(run['llm_calls_json']))} | "
          f"db {harness['db_path']}")
    if run["note"]:
        print(f"note: {run['note']}")
    return 0 if run["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
