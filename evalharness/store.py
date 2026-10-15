"""The results database (Step 5): data/results.sqlite.

Five tables, from the CLAUDE.md data model plus the columns Gaurav approved
on 2026-10-15 (marked (+) in .claude/specs/05-runner-and-store.md):
- queries: a copy of the test set, so SQL can join outputs to query type/split.
- runs:    one row per run, with everything needed to reproduce it.
- outputs: one row per (run, query): the answer and its chunks, or the error.
- scores:  metric values per (run, query, metric); first written in Step 6.
- reviews: human review per (run, query); first written in Steps 13/15.

Every value goes in through `?` placeholders, never pasted into SQL text.
Foreign keys are only enforced by SQLite when PRAGMA foreign_keys = ON is set
on the connection, which connect() does.

Schema rule: once a real run exists, the database is never deleted to fix the
schema (stored runs cost real API calls). Later steps add columns with
ALTER TABLE ... ADD COLUMN in create_tables().
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from evalharness.schemas import Query
from evalharness.testset import resolve_path

# Allowed values of runs.status. 'running' is set by start_run; finish_run sets one of the others.
RUN_STATUSES = ("running", "completed", "aborted", "interrupted", "invalid")

SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS queries (
        id               TEXT PRIMARY KEY,
        question         TEXT NOT NULL,
        type             TEXT NOT NULL,
        reference_answer TEXT NOT NULL,
        source_doc_ids   TEXT NOT NULL,            -- JSON list, [] for unanswerable
        split            TEXT NOT NULL,            -- dev | heldout
        reviewed         INTEGER NOT NULL          -- 0/1
    )""",
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id             TEXT PRIMARY KEY,
        timestamp          TEXT NOT NULL,          -- UTC ISO start
        git_commit         TEXT NOT NULL,
        config_json        TEXT NOT NULL,          -- json.dumps(config, sort_keys=True)
        judge_version      TEXT,                   -- NULL until judges exist
        split              TEXT NOT NULL,
        finished_at        TEXT,                   -- NULL while running
        status             TEXT NOT NULL,          -- see RUN_STATUSES
        note               TEXT,                   -- why aborted / invalid
        config_path        TEXT NOT NULL,
        git_dirty          INTEGER NOT NULL,       -- 0/1
        test_set_sha256    TEXT NOT NULL,
        system_json        TEXT NOT NULL,          -- describe hook output (index key, digests)
        n_queries          INTEGER NOT NULL,
        n_ok               INTEGER,
        n_failed           INTEGER,
        llm_calls_start_id INTEGER NOT NULL,       -- max llm_calls.id before the run
        llm_calls_json     TEXT                    -- per-model calls / hits / misses / errors
    )""",
    """
    CREATE TABLE IF NOT EXISTS outputs (
        run_id                TEXT NOT NULL REFERENCES runs(run_id),
        query_id              TEXT NOT NULL REFERENCES queries(id),
        answer                TEXT,                -- NULL on failure
        retrieved_chunks_json TEXT,                -- JSON list of chunk dicts; NULL on failure
        latency_ms            REAL,                -- answer()["latency_ms"]; NULL on failure
        error                 TEXT,                -- "ContextLimitError: ..."; NULL on success
        PRIMARY KEY (run_id, query_id),
        CHECK ((answer IS NULL) <> (error IS NULL))
    )""",
    """
    CREATE TABLE IF NOT EXISTS scores (
        run_id    TEXT NOT NULL REFERENCES runs(run_id),
        query_id  TEXT NOT NULL REFERENCES queries(id),
        metric    TEXT NOT NULL,
        value     REAL,
        reasoning TEXT,
        PRIMARY KEY (run_id, query_id, metric)
    )""",
    """
    CREATE TABLE IF NOT EXISTS reviews (
        run_id       TEXT NOT NULL REFERENCES runs(run_id),
        query_id     TEXT NOT NULL REFERENCES queries(id),
        flag_reasons TEXT,                         -- JSON list
        human_label  TEXT,                         -- set only by Gaurav
        notes        TEXT,
        PRIMARY KEY (run_id, query_id)
    )""",
)


def _now() -> str:
    """Current UTC time as ISO-8601 to the second (same format as llm_calls.called_at)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- connection

def create_tables(conn: sqlite3.Connection) -> None:
    """Create all five tables if they do not exist yet."""
    with conn:
        for statement in SCHEMA:
            conn.execute(statement)


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the results DB with foreign keys on and all tables present.

    Rows come back as sqlite3.Row (access by column name). The caller closes
    the connection (use contextlib.closing).
    """
    path = resolve_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        create_tables(conn)
    except sqlite3.Error:
        conn.close()   # do not leak the connection if the file is locked or corrupt
        raise
    return conn


# ---------------------------------------------------------------- queries

def _query_row(query: Query) -> tuple:
    """A Query as the tuple of `queries` columns, in table order."""
    return (query.id, query.question, query.type, query.reference_answer,
            json.dumps(query.source_doc_ids), query.split, int(query.reviewed))


def sync_queries(conn: sqlite3.Connection, queries: list[Query]) -> int:
    """Insert test-set queries not yet stored; return how many were new.

    A query already stored with different content raises ValueError naming its
    id: once runs exist the test set is frozen (Step 4 rule), so a changed
    question must get a new id, never an in-place edit.
    """
    new = 0
    with conn:
        for query in queries:
            row = _query_row(query)
            stored = conn.execute(
                "SELECT id, question, type, reference_answer, source_doc_ids, split, reviewed"
                " FROM queries WHERE id = ?", (query.id,)).fetchone()
            if stored is None:
                conn.execute(
                    "INSERT INTO queries (id, question, type, reference_answer, source_doc_ids,"
                    " split, reviewed) VALUES (?, ?, ?, ?, ?, ?, ?)", row)
                new += 1
            elif tuple(stored) != row:
                raise ValueError(f"query {query.id} differs from the stored copy; the test set "
                                 "is frozen once runs exist (give a changed query a new id)")
    return new


# ---------------------------------------------------------------- runs

def start_run(conn: sqlite3.Connection, *, run_id: str, split: str, config: dict,
              config_path: str, git_commit: str, git_dirty: bool, test_set_sha256: str,
              system_info: dict, n_queries: int, llm_calls_start_id: int) -> None:
    """Insert the runs row with status 'running' and timestamp = now (UTC)."""
    with conn:
        conn.execute(
            "INSERT INTO runs (run_id, timestamp, git_commit, config_json, judge_version, split,"
            " status, config_path, git_dirty, test_set_sha256, system_json, n_queries,"
            " llm_calls_start_id) VALUES (?, ?, ?, ?, NULL, ?, 'running', ?, ?, ?, ?, ?, ?)",
            (run_id, _now(), git_commit, json.dumps(config, sort_keys=True), split, config_path,
             int(git_dirty), test_set_sha256, json.dumps(system_info, sort_keys=True), n_queries,
             llm_calls_start_id),
        )


def finish_run(conn: sqlite3.Connection, run_id: str, *, status: str, n_ok: int, n_failed: int,
               llm_calls_summary: dict, note: str | None = None) -> None:
    """Close a run: finished_at = now, final status, counts, per-model call summary, note."""
    if status not in RUN_STATUSES or status == "running":
        raise ValueError(f"finish_run status must be one of {RUN_STATUSES[1:]}, got {status!r}")
    with conn:
        updated = conn.execute(
            "UPDATE runs SET finished_at = ?, status = ?, n_ok = ?, n_failed = ?,"
            " llm_calls_json = ?, note = ? WHERE run_id = ?",
            (_now(), status, n_ok, n_failed, json.dumps(llm_calls_summary, sort_keys=True),
             note, run_id),
        ).rowcount
    if updated != 1:
        raise ValueError(f"no run {run_id!r} to finish")


def get_run(conn: sqlite3.Connection, run_id: str) -> dict | None:
    """The runs row as a dict, or None if there is no such run."""
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row is not None else None


# ---------------------------------------------------------------- outputs

def write_output(conn: sqlite3.Connection, *, run_id: str, query_id: str, answer: str | None,
                 chunks: list[dict] | None, latency_ms: float | None,
                 error: str | None) -> None:
    """Insert one outputs row and commit it at once (a crash keeps every row so far).

    Success: answer (may be ""), chunks and latency_ms set, error None.
    Failure: error set, the other three None.
    """
    success = error is None
    if success and (answer is None or chunks is None or latency_ms is None):
        raise ValueError("a successful output needs answer, chunks and latency_ms")
    if not success and (answer is not None or chunks is not None or latency_ms is not None):
        raise ValueError("a failed output stores only the error")
    with conn:
        conn.execute(
            "INSERT INTO outputs (run_id, query_id, answer, retrieved_chunks_json, latency_ms,"
            " error) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, query_id, answer, json.dumps(chunks) if success else None, latency_ms,
             error),
        )


def get_outputs(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    """A run's outputs in query_id order; `retrieved_chunks` is the parsed list (None on failure)."""
    rows = conn.execute(
        "SELECT run_id, query_id, answer, retrieved_chunks_json, latency_ms, error"
        " FROM outputs WHERE run_id = ? ORDER BY query_id", (run_id,)).fetchall()
    outputs = []
    for row in rows:
        out = dict(row)
        raw = out.pop("retrieved_chunks_json")
        out["retrieved_chunks"] = json.loads(raw) if raw is not None else None
        outputs.append(out)
    return outputs
