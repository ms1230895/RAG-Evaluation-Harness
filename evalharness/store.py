"""The results database (Step 5): data/results.sqlite.

Five tables, from the CLAUDE.md data model plus the columns Gaurav approved
on 2026-10-15 (marked (+) in .claude/specs/05-runner-and-store.md):
- queries: a copy of the test set, so SQL can join outputs to query type/split.
- runs:    one row per run, with everything needed to reproduce it.
- outputs: one row per (run, query): the answer and its chunks, or the error.
- scores:  metric values per (run, query, metric, version); first written in Step 6.
           `version` names the scorer ("checks-1-<hash>", later judge versions),
           so several versions for one run can be stored side by side.
- reviews: human review per (run, query); first written in Steps 13/15.
- judge_claims: per-claim verdicts of the faithfulness judge (Step 7), one row
           per (run, query, version, claim), so a score can be traced to its claims.
- calibration_labels: Gaurav's own labels for judge calibration (Step 9): the
           fixed tune/check split per answer, grounded / ungrounded / unsure,
           and an optional 0/1/2 relevance label.
- calibration_reports: one row per judge-vs-label comparison (every look is
           logged, including how often the check half was used).
- run_reports: one row per stored run report (Step 10 baseline), with the
           exact scorer versions it was built from.

Every value goes in through `?` placeholders, never pasted into SQL text.
Foreign keys are only enforced by SQLite when PRAGMA foreign_keys = ON is set
on the connection, which connect() does.

Schema rule: once a real run exists, the database is never deleted to fix the
schema (stored runs cost real API calls). Later steps add columns with
ALTER TABLE ... ADD COLUMN in create_tables(). The one exception is Step 6's
change to the `scores` primary key: the old table is dropped and recreated
only while it has 0 rows (see _migrate_scores).
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
        metric    TEXT NOT NULL,                   -- "hit@4", "refusal", ...
        version   TEXT NOT NULL,                   -- "checks-1-3fa1c27e"; judge versions later
        value     REAL,                            -- NULL = not applicable / output failed
        reasoning TEXT NOT NULL,                   -- why this value (or why NULL)
        scored_at TEXT NOT NULL,                   -- UTC ISO
        PRIMARY KEY (run_id, query_id, metric, version)
    )""",
    """
    CREATE TABLE IF NOT EXISTS reviews (
        run_id       TEXT NOT NULL REFERENCES runs(run_id),
        query_id     TEXT NOT NULL REFERENCES queries(id),
        flag_reasons TEXT,                         -- JSON list
        human_label  TEXT,                         -- set only by Gaurav ('ok' | 'bad' from the dashboard)
        notes        TEXT,
        labeler      TEXT,                         -- who gave the label (Step 15); NULL = flag only
        reviewed_at  TEXT,                         -- UTC ISO time of the label (Step 15)
        PRIMARY KEY (run_id, query_id)
    )""",
    """
    CREATE TABLE IF NOT EXISTS judge_claims (
        run_id      TEXT NOT NULL REFERENCES runs(run_id),
        query_id    TEXT NOT NULL REFERENCES queries(id),
        version     TEXT NOT NULL,                 -- same as scores.version
        claim_index INTEGER NOT NULL,              -- 1-based
        claim       TEXT NOT NULL,
        label       TEXT NOT NULL CHECK (label IN ('supported', 'unsupported', 'contradicted')),
        reasoning   TEXT NOT NULL,
        PRIMARY KEY (run_id, query_id, version, claim_index)
    )""",
    """
    CREATE TABLE IF NOT EXISTS calibration_labels (
        run_id          TEXT NOT NULL REFERENCES runs(run_id),
        query_id        TEXT NOT NULL REFERENCES queries(id),
        calib_split     TEXT NOT NULL CHECK (calib_split IN ('tune', 'check')),
        grounded_label  TEXT CHECK (grounded_label IN ('grounded', 'ungrounded', 'unsure')),
        relevance_label INTEGER CHECK (relevance_label IN (0, 1, 2)),
        notes           TEXT,
        labeler         TEXT,                  -- set with the label; only Gaurav labels
        labeled_at      TEXT,                  -- UTC ISO; NULL = not labelled yet
        PRIMARY KEY (run_id, query_id)
    )""",
    """
    CREATE TABLE IF NOT EXISTS calibration_reports (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id        TEXT NOT NULL REFERENCES runs(run_id),
        judge         TEXT NOT NULL,           -- faithfulness | relevance
        judge_version TEXT NOT NULL,
        calib_split   TEXT NOT NULL,           -- tune | check | all
        n             INTEGER NOT NULL,        -- pairs compared
        n_excluded    INTEGER NOT NULL,        -- unsure + unlabelled + not judged
        tp INTEGER NOT NULL, fp INTEGER NOT NULL, fn INTEGER NOT NULL, tn INTEGER NOT NULL,
        report_json   TEXT NOT NULL,
        created_at    TEXT NOT NULL
    )""",
    """
    CREATE TABLE IF NOT EXISTS run_reports (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id        TEXT NOT NULL REFERENCES runs(run_id),
        kind          TEXT NOT NULL,           -- "baseline" (Step 10), "experiment" (11), "heldout" (12), "regression" (13)
        versions_json TEXT NOT NULL,           -- {"checks", "faithfulness", "relevance"}
        report_json   TEXT NOT NULL,
        created_at    TEXT NOT NULL
    )""",
    """
    CREATE TABLE IF NOT EXISTS accepted_runs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      TEXT NOT NULL REFERENCES runs(run_id),
        split       TEXT NOT NULL,             -- dev only (Step 13)
        accepted_at TEXT NOT NULL,             -- UTC ISO
        accepted_by TEXT NOT NULL,             -- a human decision ("gaurav")
        note        TEXT NOT NULL
    )""",
)


def _now() -> str:
    """Current UTC time as ISO-8601 to the second (same format as llm_calls.called_at)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- connection

def _migrate_scores(conn: sqlite3.Connection) -> None:
    """Replace a Step 5 `scores` table (no `version` column) with the Step 6 layout.

    SQLite cannot change a primary key in place, so the old table is dropped
    and recreated by SCHEMA, but only while it is empty. If it holds rows we
    stop: stored scores are never thrown away.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(scores)")}
    if not columns or "version" in columns:
        return   # no table yet, or already the new layout
    n_rows = conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    if n_rows:
        raise RuntimeError(f"scores has {n_rows} rows in the old layout (no version column); "
                           "not dropping stored scores")
    conn.execute("DROP TABLE scores")


def _migrate_reviews(conn: sqlite3.Connection) -> None:
    """Add the Step 15 columns (labeler, reviewed_at) to a `reviews` table made before them.

    ALTER TABLE ADD COLUMN keeps every stored row; new columns are NULL on old rows.
    A fresh database already has the columns from SCHEMA, so nothing happens.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(reviews)")}
    if not columns:
        return   # no table yet: SCHEMA creates it with the new columns
    for column in ("labeler", "reviewed_at"):
        if column not in columns:
            conn.execute(f"ALTER TABLE reviews ADD COLUMN {column} TEXT")


def create_tables(conn: sqlite3.Connection) -> None:
    """Create every table if it does not exist yet (after migrating older layouts)."""
    with conn:
        _migrate_scores(conn)
        _migrate_reviews(conn)
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


def list_runs(conn: sqlite3.Connection, split: str | None = None) -> list[dict]:
    """Run rows for the dashboard, oldest first (timestamp, then run_id); one split or all."""
    columns = ("run_id, timestamp, split, status, config_path, git_commit, git_dirty,"
               " judge_version, n_queries, n_ok, n_failed")
    if split is None:
        rows = conn.execute(f"SELECT {columns} FROM runs ORDER BY timestamp, run_id").fetchall()
    else:
        rows = conn.execute(f"SELECT {columns} FROM runs WHERE split = ?"
                            " ORDER BY timestamp, run_id", (split,)).fetchall()
    return [dict(row) for row in rows]


def runs_with_versions(conn: sqlite3.Connection, versions: list[str]) -> set[str]:
    """Run ids that have score rows for EVERY one of the given scorer versions."""
    wanted = sorted(set(versions))
    if not wanted:
        return set()
    marks = ", ".join("?" for _ in wanted)   # only "?" placeholders go into the SQL text
    rows = conn.execute(
        f"SELECT run_id FROM scores WHERE version IN ({marks})"
        " GROUP BY run_id HAVING COUNT(DISTINCT version) = ?", (*wanted, len(wanted))).fetchall()
    return {row["run_id"] for row in rows}


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


def _parse_output_row(row: sqlite3.Row) -> dict:
    """An outputs row as a dict, with retrieved_chunks_json parsed into `retrieved_chunks`."""
    out = dict(row)
    raw = out.pop("retrieved_chunks_json")
    out["retrieved_chunks"] = json.loads(raw) if raw is not None else None
    return out


def get_outputs(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    """A run's outputs in query_id order; `retrieved_chunks` is the parsed list (None on failure)."""
    rows = conn.execute(
        "SELECT run_id, query_id, answer, retrieved_chunks_json, latency_ms, error"
        " FROM outputs WHERE run_id = ? ORDER BY query_id", (run_id,)).fetchall()
    return [_parse_output_row(row) for row in rows]


def get_outputs_with_queries(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    """get_outputs() rows plus the query's `question`, `type`, `split` and parsed `source_doc_ids`."""
    rows = conn.execute(
        "SELECT o.run_id, o.query_id, o.answer, o.retrieved_chunks_json, o.latency_ms, o.error,"
        " q.question, q.type, q.split, q.source_doc_ids"
        " FROM outputs o JOIN queries q ON q.id = o.query_id"
        " WHERE o.run_id = ? ORDER BY o.query_id", (run_id,)).fetchall()
    outputs = []
    for row in rows:
        out = _parse_output_row(row)
        out["source_doc_ids"] = json.loads(out["source_doc_ids"])
        outputs.append(out)
    return outputs


# ---------------------------------------------------------------- scores

def write_scores(conn: sqlite3.Connection, run_id: str, version: str, rows: list[dict]) -> None:
    """Insert score rows ({"query_id", "metric", "value", "reasoning"}) in one transaction.

    All or nothing: if any row fails (duplicate key, NULL reasoning), none is stored.
    """
    scored_at = _now()
    with conn:
        conn.executemany(
            "INSERT INTO scores (run_id, query_id, metric, version, value, reasoning, scored_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(run_id, row["query_id"], row["metric"], version, row["value"], row["reasoning"],
              scored_at) for row in rows],
        )


def get_scores(conn: sqlite3.Connection, run_id: str, version: str) -> list[dict]:
    """A run's score rows for one version, ordered by query_id, metric."""
    rows = conn.execute(
        "SELECT run_id, query_id, metric, version, value, reasoning, scored_at FROM scores"
        " WHERE run_id = ? AND version = ? ORDER BY query_id, metric", (run_id, version)).fetchall()
    return [dict(row) for row in rows]


# ---------------------------------------------------------------- judge claims (Step 7)

def write_judgement(conn: sqlite3.Connection, run_id: str, version: str, query_id: str,
                    score_rows: list[dict], claim_rows: list[dict]) -> None:
    """Store one output's judge scores and claim verdicts in one transaction.

    score_rows: [{"metric", "value", "reasoning"}]; claim_rows:
    [{"claim_index", "claim", "label", "reasoning"}]. All or nothing, so a
    crash never leaves a score without its claims (or the other way round).
    """
    scored_at = _now()
    with conn:
        conn.executemany(
            "INSERT INTO scores (run_id, query_id, metric, version, value, reasoning, scored_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(run_id, query_id, row["metric"], version, row["value"], row["reasoning"], scored_at)
             for row in score_rows],
        )
        conn.executemany(
            "INSERT INTO judge_claims (run_id, query_id, version, claim_index, claim, label,"
            " reasoning) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(run_id, query_id, version, row["claim_index"], row["claim"], row["label"],
              row["reasoning"]) for row in claim_rows],
        )


def delete_judgement(conn: sqlite3.Connection, run_id: str, version: str, query_id: str) -> None:
    """Remove one output's judge scores and claims for one version (used by --retry-failed)."""
    with conn:
        conn.execute("DELETE FROM scores WHERE run_id = ? AND version = ? AND query_id = ?",
                     (run_id, version, query_id))
        conn.execute("DELETE FROM judge_claims WHERE run_id = ? AND version = ? AND query_id = ?",
                     (run_id, version, query_id))


def get_judge_claims(conn: sqlite3.Connection, run_id: str, version: str) -> list[dict]:
    """A run's claim verdicts for one judge version, ordered by query_id, claim_index."""
    rows = conn.execute(
        "SELECT run_id, query_id, version, claim_index, claim, label, reasoning FROM judge_claims"
        " WHERE run_id = ? AND version = ? ORDER BY query_id, claim_index",
        (run_id, version)).fetchall()
    return [dict(row) for row in rows]


# ---------------------------------------------------------------- calibration (Step 9)

def save_calibration_splits(conn: sqlite3.Connection, run_id: str, splits: dict[str, str]) -> None:
    """Store the tune/check split once (labels NULL). The split never changes afterwards.

    Same splits already stored: nothing happens. Different: ValueError.
    """
    stored = {row["query_id"]: row["calib_split"] for row in conn.execute(
        "SELECT query_id, calib_split FROM calibration_labels WHERE run_id = ?", (run_id,))}
    if stored:
        if stored != splits:
            raise ValueError(f"calibration split for run {run_id} is fixed once assigned "
                             "and differs from the new one")
        return
    with conn:
        conn.executemany(
            "INSERT INTO calibration_labels (run_id, query_id, calib_split) VALUES (?, ?, ?)",
            [(run_id, query_id, split) for query_id, split in sorted(splits.items())])


def get_calibration_labels(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    """A run's calibration rows (split and label, if any), ordered by query_id."""
    rows = conn.execute(
        "SELECT run_id, query_id, calib_split, grounded_label, relevance_label, notes, labeler,"
        " labeled_at FROM calibration_labels WHERE run_id = ? ORDER BY query_id",
        (run_id,)).fetchall()
    return [dict(row) for row in rows]


def write_calibration_label(conn: sqlite3.Connection, run_id: str, query_id: str,
                            grounded_label: str, relevance_label: int | None,
                            notes: str | None, labeler: str) -> None:
    """Store one human label on its split row and commit at once (a quit keeps every label so far)."""
    with conn:
        updated = conn.execute(
            "UPDATE calibration_labels SET grounded_label = ?, relevance_label = ?, notes = ?,"
            " labeler = ?, labeled_at = ? WHERE run_id = ? AND query_id = ?",
            (grounded_label, relevance_label, notes, labeler, _now(), run_id, query_id),
        ).rowcount
    if updated != 1:
        raise ValueError(f"no calibration split row for {run_id} / {query_id}")


def write_calibration_report(conn: sqlite3.Connection, row: dict) -> int:
    """Log one judge-vs-label comparison; returns its id."""
    with conn:
        cursor = conn.execute(
            "INSERT INTO calibration_reports (run_id, judge, judge_version, calib_split, n,"
            " n_excluded, tp, fp, fn, tn, report_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (row["run_id"], row["judge"], row["judge_version"], row["calib_split"], row["n"],
             row["n_excluded"], row["tp"], row["fp"], row["fn"], row["tn"],
             json.dumps(row["report"], sort_keys=True), _now()),
        )
    return cursor.lastrowid


def get_calibration_reports(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    """Every logged comparison for a run, oldest first; `report` is the parsed report_json."""
    rows = conn.execute("SELECT * FROM calibration_reports WHERE run_id = ? ORDER BY id",
                        (run_id,)).fetchall()
    reports = []
    for row in rows:
        report = dict(row)
        report["report"] = json.loads(report.pop("report_json"))
        reports.append(report)
    return reports


# ---------------------------------------------------------------- run reports (Step 10)

def write_run_report(conn: sqlite3.Connection, row: dict) -> int:
    """Store one run report (kind, versions, full report) and return its id."""
    with conn:
        cursor = conn.execute(
            "INSERT INTO run_reports (run_id, kind, versions_json, report_json, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (row["run_id"], row["kind"], json.dumps(row["versions"], sort_keys=True),
             json.dumps(row["report"], sort_keys=True), row.get("created_at") or _now()),
        )
    return cursor.lastrowid


def get_run_reports(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    """A run's stored reports, oldest first; `versions` and `report` are parsed JSON."""
    rows = conn.execute("SELECT * FROM run_reports WHERE run_id = ? ORDER BY id",
                        (run_id,)).fetchall()
    reports = []
    for row in rows:
        item = dict(row)
        item["versions"] = json.loads(item.pop("versions_json"))
        item["report"] = json.loads(item.pop("report_json"))
        reports.append(item)
    return reports


# ---------------------------------------------------------------- accepted runs and flags (Step 13)

def write_accepted_run(conn: sqlite3.Connection, *, run_id: str, split: str, accepted_by: str,
                       note: str) -> int:
    """Record a run as the accepted reference of its split; returns the row id. Rows are never updated."""
    with conn:
        cursor = conn.execute(
            "INSERT INTO accepted_runs (run_id, split, accepted_at, accepted_by, note)"
            " VALUES (?, ?, ?, ?, ?)", (run_id, split, _now(), accepted_by, note))
    return cursor.lastrowid


def get_accepted_runs(conn: sqlite3.Connection, split: str | None = None) -> list[dict]:
    """Accepted-run rows, newest first (id order reversed); one split or all."""
    if split is None:
        rows = conn.execute("SELECT * FROM accepted_runs ORDER BY id DESC").fetchall()
    else:
        rows = conn.execute("SELECT * FROM accepted_runs WHERE split = ? ORDER BY id DESC",
                            (split,)).fetchall()
    return [dict(row) for row in rows]


def write_flag_rows(conn: sqlite3.Connection, run_id: str, flags: dict[str, list[str]]) -> None:
    """Upsert one `reviews` row per flagged query in one transaction.

    Only flag_reasons is written; an existing human_label or notes is left alone.
    """
    with conn:
        conn.executemany(
            "INSERT INTO reviews (run_id, query_id, flag_reasons) VALUES (?, ?, ?)"
            " ON CONFLICT(run_id, query_id) DO UPDATE SET flag_reasons = excluded.flag_reasons",
            [(run_id, query_id, json.dumps(reasons)) for query_id, reasons in sorted(flags.items())])


def get_flag_rows(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    """A run's `reviews` rows ordered by query_id; `flag_reasons` is the parsed JSON list."""
    rows = conn.execute(
        "SELECT run_id, query_id, flag_reasons, human_label, notes, labeler, reviewed_at"
        " FROM reviews WHERE run_id = ? ORDER BY query_id", (run_id,)).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["flag_reasons"] = json.loads(item["flag_reasons"]) if item["flag_reasons"] else []
        out.append(item)
    return out


def delete_unlabelled_flag_rows(conn: sqlite3.Connection, run_id: str) -> int:
    """Delete this run's review rows that have no human_label; returns how many. Labelled rows stay."""
    with conn:
        cursor = conn.execute("DELETE FROM reviews WHERE run_id = ? AND human_label IS NULL",
                              (run_id,))
    return cursor.rowcount


def set_judge_version(conn: sqlite3.Connection, run_id: str, value: str) -> None:
    """Record the judge versions on a run (Step 14): only while runs.judge_version is NULL.

    The same value again is a no-op; a different value raises ValueError, so a run's
    recorded judge version can never silently change.
    """
    row = conn.execute("SELECT judge_version FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"no run {run_id!r}")
    if row["judge_version"] is not None:
        if row["judge_version"] != value:
            raise ValueError(f"run {run_id} already records judge version "
                             f"{row['judge_version']!r}, not {value!r}")
        return
    with conn:
        conn.execute("UPDATE runs SET judge_version = ? WHERE run_id = ? AND judge_version IS NULL",
                     (value, run_id))


# ---------------------------------------------------------------- dashboard (Step 15)

def get_queries(conn: sqlite3.Connection, query_ids: list[str]) -> dict[str, dict]:
    """Test-set rows by id: question, type, reference_answer, source_doc_ids (parsed), split.

    Ids that are not stored are simply absent from the result.
    """
    ids = sorted(set(query_ids))
    if not ids:
        return {}
    marks = ", ".join("?" for _ in ids)   # only "?" placeholders go into the SQL text
    rows = conn.execute(
        "SELECT id, question, type, reference_answer, source_doc_ids, split FROM queries"
        f" WHERE id IN ({marks})", ids).fetchall()
    out = {}
    for row in rows:
        item = dict(row)
        item["source_doc_ids"] = json.loads(item["source_doc_ids"])
        out[item.pop("id")] = item
    return out


def write_review(conn: sqlite3.Connection, run_id: str, query_id: str, human_label: str,
                 notes: str | None, labeler: str, allowed_labels: list[str]) -> None:
    """Store one human review on an EXISTING flagged `reviews` row and commit at once.

    Only human_label, notes, labeler and reviewed_at are written; flag_reasons is
    never touched and no row is created, so a label can only land on an answer the
    flags selected. `allowed_labels` comes from the dashboard config (ok / bad).
    """
    if human_label not in allowed_labels:
        raise ValueError(f"human_label must be one of {list(allowed_labels)}, got {human_label!r}")
    if not labeler or not labeler.strip():
        raise ValueError("a review needs a labeler")
    with conn:
        updated = conn.execute(
            "UPDATE reviews SET human_label = ?, notes = ?, labeler = ?, reviewed_at = ?"
            " WHERE run_id = ? AND query_id = ?",
            (human_label, notes, labeler, _now(), run_id, query_id),
        ).rowcount
    if updated != 1:
        raise ValueError(f"no flagged review row for {run_id} / {query_id}")


def runs_with_reviews(conn: sqlite3.Connection) -> list[str]:
    """Run ids that have at least one `reviews` row (a flagged answer), in id order."""
    rows = conn.execute("SELECT DISTINCT run_id FROM reviews ORDER BY run_id").fetchall()
    return [row["run_id"] for row in rows]


def list_run_reports(conn: sqlite3.Connection, kind: str) -> list[dict]:
    """Every stored run report of one kind, oldest first; `versions` and `report` are parsed JSON."""
    rows = conn.execute("SELECT * FROM run_reports WHERE kind = ? ORDER BY id", (kind,)).fetchall()
    reports = []
    for row in rows:
        item = dict(row)
        item["versions"] = json.loads(item.pop("versions_json"))
        item["report"] = json.loads(item.pop("report_json"))
        reports.append(item)
    return reports
