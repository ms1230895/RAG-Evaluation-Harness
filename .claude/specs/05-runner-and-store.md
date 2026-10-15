# Spec: Step 05 — Runner and SQLite Store

## Overview
This step builds the harness's memory and its main loop. `evalharness/store.py`
creates the results database (`data/results.sqlite`) with the five tables from
the CLAUDE.md data model (`queries`, `runs`, `outputs`, `scores`, `reviews`)
and the functions that write to it. `evalharness/runner.py` loads one split of
the reviewed test set, calls the system under test's `answer()` once per
query, and stores every answer, its retrieved chunks and its latency under a
new `run_id`, together with everything needed to reproduce the run: the full
config, the git commit, the test set's SHA-256, the index key and the
generator's model digest. Failed queries are stored as failures, never dropped.
It comes now because the system under test (Step 3) and the test set (Step 4)
both exist, and every later step (retrieval metrics, judges, calibration,
baseline, experiments, regression) reads its inputs from these stored runs
instead of calling the pipeline again. No metric is computed in this step.

## Concepts to learn
- **A run is an experiment record.** One run = one config × one split × one
  moment in time. A number is only trustworthy if you can say exactly what
  produced it, so each run stores: the full config (not just its file name,
  because files change), the git commit (which code), whether the working tree
  had uncommitted changes, the test set's SHA-256 (which questions), the index
  key (which corpus and chunking were indexed) and the model digest (which model
  build answered). If any of these is missing, two runs cannot be compared
  honestly.
- **Separating "run" from "score".** The runner only collects answers
  (`outputs`). Metrics and judges (Steps 6-8) read `outputs` later and write
  `scores`. This means a judge prompt can be improved (Step 9) and re-applied to
  the same stored answers without calling the generator again, so a change in
  the score comes from the judge, not from a new sample of answers.
- **Relational tables and keys.** A primary key makes a row unique: `runs` by
  `run_id`; `outputs` by `(run_id, query_id)`, so one query cannot be stored
  twice in one run. A foreign key (`outputs.run_id -> runs.run_id`) stops
  orphan rows. SQLite only enforces foreign keys after
  `PRAGMA foreign_keys = ON`, which must be set on every connection.
- **Parameterised SQL.** Values are passed as `?` placeholders
  (`conn.execute("INSERT INTO outputs VALUES (?, ?)", (run_id, qid))`), never
  pasted into the SQL string. An answer containing `'); DROP TABLE runs; --`
  is then just text.
- **Transactions.** `with conn:` commits the block if it succeeds and rolls it
  back if it raises. Each output row is committed as soon as it is written, so a
  crash at query 57 still leaves 56 stored rows and a run marked `running`
  (later set to `interrupted` or left visibly unfinished), never a run that
  looks complete.
- **Failures belong in the denominator.** If 5 of 80 answers fail (timeout,
  `ContextLimitError`) and we silently drop them, the ungrounded rate is computed
  on 75 easier queries. Failures are not random: long, multi-chunk questions hit
  the context guard first. So every failure is stored with its error, counted in
  `n_failed`, and later reported as "scored N, failed M", by query type.
  Example: 80 dev queries, 3 failed → report "77 scored, 3 failed (2
  multi_chunk, 1 lookup)", never a rate over 77 that hides the 3.
- **Fail fast on systematic errors.** If Ollama is down, every call fails. After
  `max_consecutive_failures` failures in a row the run stops and is marked
  `aborted`, instead of recording 80 identical errors.
- **Content hashes as version IDs.** SHA-256 of a file's bytes changes if any
  byte changes. Storing `sha256(data/test_set.jsonl)` with each run proves two
  runs used the same questions. Tiny example: `sha256("a")` starts `ca978112…`,
  `sha256("b")` starts `3e23e816…`; one character changes the whole hash.
- **Model identity.** A model tag like `gemma4:31b-cloud` can point to a new
  build after a re-pull. Ollama reports a digest per tag (`/api/tags`); storing
  it with the run, and putting it in the LLM cache key, means a changed model
  never silently reuses old cached answers. Limitation: for `-cloud` models this
  digest identifies the local manifest, not the remote weights, so a
  provider-side update would not change it.
- **Counting cache hits per run.** `llm_calls` (Step 2) gets one row per
  `generate()` call with an autoincrement `id`. The runner notes the highest
  `id` before and after the run; rows in between are this run's calls, so it can
  store "80 calls: 1 hit, 79 misses, 0 errors" per model. A rerun of the same
  config should be all hits, which proves the run is reproducible from the cache.
- **The contract as a boundary.** The runner only knows `answer(question,
  config) -> {"answer", "chunks", "latency_ms"}`. It checks the returned dict's
  shape (`check_contract`), so a broken or different RAG system fails loudly per
  item instead of storing malformed rows.

## Depends on
- Step 2: `llm/client.py`: `load_config()` (for `cache_path`), the `llm_calls`
  table (`id`, `model`, `cached`, `error`), `generate(..., model_digest=...)`.
- Step 3: `rag/pipeline.py`: `answer(question, config)`, `load_pipeline_config(path)`,
  `ContextLimitError`; `rag/index.py`: `describe_index(config)` (index key and
  saved meta). `configs/v0.yaml` (`generation.model_digest` is `null` today).
- Step 4: `evalharness/schemas.py`: `Query`, `SPLITS`; `evalharness/testset.py`:
  `load_testset_config()`, `load_test_set(path)`, `check_ready(queries, config)`,
  `filter_split(queries, split)`, `resolve_path()`. `configs/test_set.yaml`,
  `data/test_set.jsonl` (100 reviewed queries, 80 dev / 20 heldout).
- Carry-forward items addressed here:
  - Step 2: model identity passed to `generate(model_digest=...)` and recorded
    with the run; per-run cache-hit counts; abort after N consecutive failures;
    per-item failures recorded and reported, never scored as 0 or 1.
  - Step 3: `ContextLimitError` recorded as the item's failure and counted;
    `generation.model_digest` filled; `describe_index(config)` stored with each
    run; index key checked again at the end of the run.
  - Step 4: `queries` table created from `data/test_set.jsonl`; test set SHA-256
    stored with each run (Decision 4); `--split heldout` refused without an
    explicit flag; run refused unless `check_ready()` is `[]`.
- Local setup (checked 2026-10-15): `curl http://localhost:11434/api/tags`
  reports `gemma4:31b-cloud` digest
  `ef09f235533c96cd75e8deed88c628335cb69e2b3ce96275d0d7a67fe9887aba`
  (`remote_host` `https://ollama.com`); `sqlite3` CLI is on PATH.

## Interfaces
This step does **not** change the `answer()` contract. It adds one optional
function next to it, `describe_system(config)`, if Decisions 1-2 are accepted.

`llm/client.py` (new)
```python
def get_model_digest(model: str, *, config: dict | None = None) -> str | None:
    """Digest Ollama reports for `model` on its native /api/tags endpoint
    (base_url with the trailing /v1 removed), via urllib from the standard
    library. Returns None if the tag is not listed. Raises on connection errors
    (a run must not start against an unreachable server)."""
```
Example: `get_model_digest("gemma4:31b-cloud")` → `"ef09f235…aba"`.

`rag/pipeline.py` (new, Decision 2)
```python
class ModelDigestError(Exception):
    """generation.model_digest is null or differs from the digest the server reports."""

def describe_system(config: dict) -> dict:
    """Facts about this pipeline that config_json alone does not pin down, for
    run metadata: {"index": describe_index(config), "generator_model": ...,
    "generator_model_digest": ..., "embedding_model": ...,
    "embedding_model_digest": ...}. Raises ModelDigestError if
    config["generation"]["model_digest"] is null or not equal to
    get_model_digest(model), so a run never starts with a stale digest."""
```

`evalharness/store.py` (new)
```python
def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the results DB: PRAGMA foreign_keys = ON,
    row_factory = sqlite3.Row, all five tables created if missing."""

def sync_queries(conn, queries: list[Query]) -> int:
    """Insert test-set queries not yet in `queries`; return how many were new.
    A query already stored with different content raises ValueError naming its
    id (the test set is frozen once runs exist)."""

def start_run(conn, *, run_id: str, split: str, config: dict, config_path: str,
              git_commit: str, git_dirty: bool, test_set_sha256: str,
              system_info: dict, n_queries: int, llm_calls_start_id: int) -> None:
    """Insert the runs row with status 'running' and timestamp = now (UTC ISO)."""

def write_output(conn, *, run_id: str, query_id: str, answer: str | None,
                 chunks: list[dict] | None, latency_ms: float | None,
                 error: str | None) -> None:
    """Insert one outputs row and commit. Exactly one of (answer, error) is
    set: success → answer, chunks, latency_ms set and error None; failure →
    error set and the other three None."""

def finish_run(conn, run_id: str, *, status: str, n_ok: int, n_failed: int,
               llm_calls_summary: dict, note: str | None = None) -> None:
    """Set finished_at, status ('completed' | 'aborted' | 'interrupted' |
    'invalid'), counts, the per-model call summary and an optional note."""

def get_run(conn, run_id: str) -> dict | None
def get_outputs(conn, run_id: str) -> list[dict]   # in query_id order, chunks parsed
```
Example: `conn = connect("data/results.sqlite"); sync_queries(conn, queries)` → `100` the first time, `0` after.

`evalharness/runner.py` (new)
```python
class ContractError(Exception):
    """answer() returned something that does not match the contract."""

@dataclass
class System:
    answer: Callable[[str, dict], dict]
    load_config: Callable[[str], dict]
    describe: Callable[[dict], dict] | None

def load_harness_config(path: str | Path = "configs/harness.yaml") -> dict
def load_system(system_config: dict) -> System:
    """Import the functions named "module:function" in harness.yaml with
    importlib, so evalharness/ has no static import of rag/."""
def check_contract(out) -> None:
    """Raise ContractError unless out is a dict with exactly the keys answer
    (str), chunks (list of dicts with exactly chunk_id, doc_id, text: str and
    score: int/float, not bool) and latency_ms (int/float, not bool, >= 0)."""
def git_info() -> tuple[str, bool]:      # (HEAD commit hash, working tree dirty?)
def file_sha256(path: str | Path) -> str
def make_run_id(config_name: str) -> str:  # "20261015T151200Z-v0-3fa1" (UTC + name + 4 random hex)
def llm_calls_max_id(llm_config: dict) -> int   # 0 if the table is empty or missing
def llm_calls_summary(llm_config: dict, after_id: int, upto_id: int) -> dict
    # {"gemma4:31b-cloud": {"calls": 80, "hits": 1, "misses": 79, "errors": 0}}
def run_split(system: System, config: dict, queries: list[Query], conn, *,
              run_meta: dict, max_consecutive_failures: int, error_max_chars: int,
              llm_config: dict) -> str:
    """Run every query (id order) through system.answer, store each output or
    failure, finish the run with the right status, return run_id.
    KeyboardInterrupt: finish as 'interrupted', then re-raise."""
def main(argv: list[str] | None = None) -> int
```
CLI:
```
python -m evalharness.runner --config configs/v0.yaml --split dev [--allow-dirty] [--allow-heldout]
```
Prints the run_id, a tqdm progress bar, then `run <id>: completed | 80 queries, 80 ok, 0 failed | gemma4:31b-cloud: 80 calls, 1 hit, 79 misses, 0 errors | db data/results.sqlite`.
Exit codes: 0 completed, 1 refused before starting (not ready, held-out, dirty
tree, digest mismatch), 2 finished as aborted or invalid.

Run order inside `main`:
1. Load harness config, test-set config, test set; refuse if `check_ready()` is not `[]`.
2. Refuse `--split heldout` unless `--allow-heldout` (message names Step 12).
3. `git_info()`; refuse a dirty tree unless `--allow-dirty` (Decision 5).
4. `config = system.load_config(path)`; `system_info = system.describe(config)`
   (builds or loads the index, checks the digest).
5. `sync_queries`, `start_run`, loop, `finish_run`.
6. After the loop, call `describe` again; if `index.index_key` changed, status
   `invalid` with a note.

## Data and schema changes
New database `data/results.sqlite` (gitignored by `*.sqlite`), created by
`store.connect()`. Columns marked **(+)** are additions to the CLAUDE.md data
model and need Gaurav's approval (Decision 4).

```sql
CREATE TABLE IF NOT EXISTS queries (
    id               TEXT PRIMARY KEY,         -- "q001"
    question         TEXT NOT NULL,
    type             TEXT NOT NULL,            -- lookup | multi_chunk | paraphrased | unanswerable
    reference_answer TEXT NOT NULL,
    source_doc_ids   TEXT NOT NULL,            -- JSON list, [] for unanswerable
    split            TEXT NOT NULL,            -- dev | heldout
    reviewed         INTEGER NOT NULL          -- 0/1
);

CREATE TABLE IF NOT EXISTS runs (
    run_id             TEXT PRIMARY KEY,       -- "20261015T151200Z-v0-3fa1"
    timestamp          TEXT NOT NULL,          -- UTC ISO start
    git_commit         TEXT NOT NULL,
    config_json        TEXT NOT NULL,          -- json.dumps(config, sort_keys=True)
    judge_version      TEXT,                   -- NULL in Step 5 (no judges yet)
    split              TEXT NOT NULL,
    finished_at        TEXT,                   -- (+) NULL while running
    status             TEXT NOT NULL,          -- (+) running | completed | aborted | interrupted | invalid
    note               TEXT,                   -- (+) why aborted / invalid
    config_path        TEXT NOT NULL,          -- (+) "configs/v0.yaml"
    git_dirty          INTEGER NOT NULL,       -- (+) 0/1
    test_set_sha256    TEXT NOT NULL,          -- (+) Step 4 carry-forward
    system_json        TEXT NOT NULL,          -- (+) describe_system() output (index key, model digest)
    n_queries          INTEGER NOT NULL,       -- (+) queries in the split
    n_ok               INTEGER,                -- (+) answers stored
    n_failed           INTEGER,                -- (+) failures stored
    llm_calls_start_id INTEGER NOT NULL,       -- (+) max llm_calls.id before the run
    llm_calls_json     TEXT                    -- (+) per-model calls / hits / misses / errors
);

CREATE TABLE IF NOT EXISTS outputs (
    run_id                TEXT NOT NULL REFERENCES runs(run_id),
    query_id              TEXT NOT NULL REFERENCES queries(id),
    answer                TEXT,                -- NULL on failure
    retrieved_chunks_json TEXT,                -- JSON list of the contract's chunk dicts; NULL on failure
    latency_ms            REAL,                -- answer()["latency_ms"]; NULL on failure
    error                 TEXT,                -- (+) "ContextLimitError: ..." ; NULL on success
    PRIMARY KEY (run_id, query_id),
    CHECK ((answer IS NULL) <> (error IS NULL))
);

-- Created now so the schema lives in one place; first written in Steps 6 and 13/15.
CREATE TABLE IF NOT EXISTS scores (
    run_id    TEXT NOT NULL REFERENCES runs(run_id),
    query_id  TEXT NOT NULL REFERENCES queries(id),
    metric    TEXT NOT NULL,
    value     REAL,
    reasoning TEXT,
    PRIMARY KEY (run_id, query_id, metric)
);

CREATE TABLE IF NOT EXISTS reviews (
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    query_id     TEXT NOT NULL REFERENCES queries(id),
    flag_reasons TEXT,                         -- JSON list
    human_label  TEXT,                         -- set only by Gaurav
    notes        TEXT,
    PRIMARY KEY (run_id, query_id)
);
```
Notes:
- `answer` is NULL only on failure; an empty-string answer from the system is
  stored as `""` (Step 6's empty-answer check needs to see it).
- `retrieved_chunks_json` stores chunk text too (Step 7's judge needs the exact
  context the generator saw; the index may be rebuilt later).
- Later steps add columns with `ALTER TABLE ... ADD COLUMN` in `create_tables`,
  never by deleting the DB (stored runs cost real API calls).
- `configs/v0.yaml`: `generation.model_digest` changes from `null` to the
  digest above (Decision 3). This changes the generator cache key, so the one
  Step 3 cached reply is not reused (one extra call).

## Config changes
New file `configs/harness.yaml` (starting defaults, not tuned values):
```yaml
db_path: data/results.sqlite          # relative to the repo root; gitignored by *.sqlite
test_set_config: configs/test_set.yaml
max_consecutive_failures: 5           # stop the run as 'aborted' after this many failures in a row
error_max_chars: 500                  # error text stored per failed item is cut to this length
system:                               # the system under test, as "module:function"
  answer: rag.pipeline:answer
  load_config: rag.pipeline:load_pipeline_config
  describe: rag.pipeline:describe_system   # optional; null = no system metadata
```
`configs/v0.yaml`: `generation.model_digest: ef09f235533c96cd75e8deed88c628335cb69e2b3ce96275d0d7a67fe9887aba`
(comment updated: the Ollama `/api/tags` digest, checked at run start).

## Files to create
- `evalharness/store.py`
- `evalharness/runner.py`
- `configs/harness.yaml`
- `tests/test_05_runner_and_store.py`

## Files to change
- `llm/client.py`: add `get_model_digest()`.
- `rag/pipeline.py`: add `ModelDigestError` and `describe_system()`; `answer()` unchanged.
- `configs/v0.yaml`: fill `generation.model_digest`.
- `tests/conftest.py`: add `fake_system` (a `System` with a scripted fake
  `answer`, success or per-question exceptions) and `results_db` fixtures.
- `tests/README.md`: Step 5 fixtures and test file.
- `CLAUDE.md`: Progress entry.

## New dependencies
No new dependencies (`importlib`, `subprocess`, `hashlib`, `urllib.request`,
`sqlite3` are standard library; `tqdm` is already approved).

## Tests
`tests/test_05_runner_and_store.py`, all offline: fake `answer`, `tmp_path`
databases, `llm_config` fixture for `llm_calls`. No real LLM, embedding or
`/api/tags` call (patch `urllib.request.urlopen`).

Store:
- `connect` creates all five tables with the exact columns above; is idempotent
  on a second call; `PRAGMA foreign_keys` is 1.
- `sync_queries`: inserts 8 on a fresh DB, 0 on repeat; changed content for an
  existing id raises `ValueError` naming the id; `source_doc_ids` round-trips as JSON.
- `start_run` / `finish_run` / `get_run` round-trip every column; `config_json`
  parses back to the same dict.
- `write_output`: success row and failure row; the CHECK rejects both-NULL and
  both-set; duplicate `(run_id, query_id)` raises `IntegrityError`; an
  unknown `run_id` raises (foreign key); a hostile string
  (`"'); DROP TABLE runs; --"`) is stored verbatim and `runs` still exists.
- `get_outputs` returns rows in `query_id` order with chunks parsed.

Runner:
- `check_contract`: accepts a valid dict; rejects missing or extra top-level
  keys, wrong types, chunk with a missing or extra key, bool score or latency,
  negative latency, non-dict return.
- `run_split` with `fake_system` on the 8-query fixture set (dev subset):
  one `outputs` row per query, `n_ok + n_failed == n_queries`, status
  `completed`, answers and chunks stored exactly as returned, queries called
  in id order with the exact question text and the same config dict.
- One query raising `ContextLimitError`-like exception → stored as a failure row
  with the error type in `error` (cut to `error_max_chars`), run still
  `completed`, counted in `n_failed`.
- A contract-breaking return → failure row with `ContractError`.
- `max_consecutive_failures` = 2 and every call failing → status `aborted`
  after exactly 2 rows; a success between failures resets the counter.
- `KeyboardInterrupt` from `answer` → status `interrupted`, re-raised,
  rows before it kept.
- `describe` returning a different `index_key` at the end → status `invalid` with a note.
- `llm_calls_summary`: counts hits, misses and errors per model only for ids in
  `(after_id, upto_id]`; `llm_calls_max_id` is 0 for a missing table.
- `make_run_id` format (regex) and uniqueness over 100 calls; `file_sha256` on a
  known string; `git_info` returns a 40-hex commit (run in the repo, read-only).
- `main` refusals (exit 1, no `runs` row written): test set not ready,
  `--split heldout` without `--allow-heldout`, dirty tree without
  `--allow-dirty` (patch `git_info`), `ModelDigestError` from `describe`.
- `load_system` resolves `"module:function"` strings; a bad name raises
  `ValueError` naming it.
- `describe_system`: null digest → `ModelDigestError`; mismatch →
  `ModelDigestError`; match → dict with `index` and the digest (patched
  `get_model_digest`, fake embed).
- `get_model_digest`: parses a fake `/api/tags` JSON; strips `/v1` from
  `base_url`; returns None for an unknown tag.
- Layering guard (AST): `evalharness/store.py` and `evalharness/runner.py` have
  no `import rag` / `from rag`; `grep -rn "evalharness" rag/` finds nothing.

## Rules for implementation
- Plain Python and only libraries approved in CLAUDE.md
- No LangChain; no RAGAS or DeepEval (except the optional final cross-check)
- `rag/` must never import from `evalharness/`; the harness only calls `answer()`
  (plus the optional `describe` hook from Decision 2)
- All tunable values come from YAML config — no hardcoded magic numbers
- SQLite via the sqlite3 module with parameterised queries only
- All LLM calls go through `llm/client.py` so they are cached (CLAUDE.md's
  `evalharness/llm.py` became the shared `llm/client.py` in Step 2)
- API keys only from `.env`, never in code, logs or commits
- The held-out split is never used before Step 12
- Never invent, estimate or adjust metric numbers; report only what a
  stored run produced
- Human labels (calibration and review) come only from Gaurav
- Code is simple, readable and commented

Step-specific:
- No metric, judge or score is computed in this step. `scores` and `reviews`
  stay empty.
- Every query in the split gets exactly one `outputs` row, success or failure,
  unless the run is aborted or interrupted (then `status` says so). Never skip,
  retry-until-success or drop a failed item; never store a placeholder answer.
- The runner catches `Exception` per item (stores it) but never
  `KeyboardInterrupt` or `SystemExit` beyond marking the run `interrupted`.
- The stored `config_json` is the exact dict passed to `answer()`, serialised
  with `sort_keys=True`. The runner never edits the config.
- `latency_ms` is stored exactly as `answer()` returned it; the runner does not
  add its own timing to it.
- The DB is never deleted or rebuilt to fix a schema mistake once a real run
  exists; add columns instead.
- Do not inspect answers to change `configs/v0.yaml` prompt or retrieval
  settings: v0 is the frozen baseline. The only v0 change is `model_digest`.
- Done-when uses the dev split only. Never run `--split heldout` in this step.
- `evalharness/store.py` and `evalharness/runner.py` have no static import of
  `rag`; the system is loaded by name from `configs/harness.yaml`.

## Decisions for Gaurav
All resolved on 2026-10-15; Gaurav accepted every recommendation:
1. ~~How the harness finds the system under test.~~ **Option A:**
   `configs/harness.yaml` names the functions as `"module:function"`;
   `runner.load_system()` imports them with `importlib`, so `evalharness/` has
   no `import rag`.
2. ~~A `describe_system(config)` hook.~~ **Option A:** optional hook in
   `rag/pipeline.py`; `answer()` unchanged. It also records the embedding
   model's digest (metadata only, no refusal), because `index_key` does not
   cover the embedding build (`rag/index.py` docstring).
3. ~~Model digest source.~~ **Option A:** the `/api/tags` digest is pinned in
   `configs/v0.yaml`; `describe_system` refuses the run if the server reports a
   different one.
4. ~~Extra columns.~~ All (+) columns approved; `scores` and `reviews` are
   created (empty) now.
5. ~~Dirty working tree.~~ **Option A:** refuse unless `--allow-dirty`. So the
   Step 5 code is committed before the real run, and `runs.git_commit` is the
   code that ran.
6. ~~`max_consecutive_failures`.~~ 5.
7. ~~Real API calls.~~ Approved: one dev run (~80 uncached `gemma4:31b-cloud`
   calls) plus one identical rerun.

## Notes found in review
- A `ContextLimitError` item has `error` set in `outputs` but counts as a
  successful call (`errors = 0`) in `llm_calls_json`, because the LLM call
  itself succeeded. Both counts are reported.
- A hard kill (SIGKILL, closed terminal) leaves `status = 'running'` and
  `finished_at` NULL. Later steps treat only `completed` runs as valid.

## Definition of done
- [ ] Done when (CLAUDE.md): one full run produces the expected rows,
      queryable with SQL. Run
      `python -m evalharness.runner --config configs/v0.yaml --split dev`.
      It exits 0 and prints a run_id. Then
      `sqlite3 data/results.sqlite "SELECT run_id, split, status, n_queries, n_ok, n_failed, git_commit, git_dirty FROM runs"`
      shows that run with `split = dev`, `status = completed`,
      `n_queries = 80` and `n_ok + n_failed = 80`.
- [ ] `sqlite3 data/results.sqlite "SELECT COUNT(*), COUNT(DISTINCT query_id), SUM(answer IS NOT NULL), SUM(error IS NOT NULL) FROM outputs WHERE run_id = '<run_id>'"`
      prints `80|80|<n_ok>|<n_failed>`, matching the `runs` row.
- [ ] Every output belongs to a dev query:
      `sqlite3 data/results.sqlite "SELECT q.split, COUNT(*) FROM outputs o JOIN queries q ON q.id = o.query_id WHERE o.run_id = '<run_id>' GROUP BY q.split"`
      prints only `dev|80`. `SELECT COUNT(*) FROM queries` prints `100`.
- [ ] Failures are reported as observed, by type:
      `sqlite3 data/results.sqlite "SELECT q.type, o.error FROM outputs o JOIN queries q ON q.id = o.query_id WHERE o.run_id = '<run_id>' AND o.error IS NOT NULL"`
      (empty is fine; whatever it prints goes into Progress verbatim).
- [ ] Run metadata is complete:
      `sqlite3 data/results.sqlite "SELECT config_path, test_set_sha256, json_extract(system_json, '$.index.index_key'), json_extract(system_json, '$.generator_model_digest'), json_extract(config_json, '$.retrieval.top_k') FROM runs WHERE run_id = '<run_id>'"`
      shows `configs/v0.yaml`, the value of `shasum -a 256 data/test_set.jsonl`,
      a 64-hex index key, the digest from `configs/v0.yaml`, and `4`.
- [ ] Retrieved chunks are stored with the contract's keys:
      `sqlite3 data/results.sqlite "SELECT json_array_length(retrieved_chunks_json), json_extract(retrieved_chunks_json, '$[0].doc_id') FROM outputs WHERE run_id = '<run_id>' AND query_id = (SELECT MIN(query_id) FROM outputs WHERE run_id = '<run_id>')"`
      shows `4` and a doc ID.
- [ ] A rerun of the same command is served from the cache: its
      `json_extract(llm_calls_json, '$."gemma4:31b-cloud"')` shows
      `hits` equal to the first run's `n_ok` and `misses = 0` (failed items,
      if any, are reported as observed).
- [ ] Held-out is refused: `python -m evalharness.runner --config configs/v0.yaml --split heldout`
      exits 1, and `SELECT COUNT(*) FROM runs WHERE split = 'heldout'` prints `0`.
- [ ] Layering: `grep -rnE "^(import|from) rag" evalharness/` prints nothing,
      and `grep -rn "evalharness" rag/` prints nothing.
- [ ] `data/results.sqlite` is not tracked: `git status --porcelain` and
      `git ls-files data/` do not list it.
- [ ] `pytest tests/test_05_runner_and_store.py` passes with no network access.
- [ ] `pytest` passes (whole suite).
- [ ] The Progress section of CLAUDE.md has a Step 5 entry with both run_ids,
      the counts and failures copied from the database, the cache summary, and
      the Step 5 test count.
- [ ] All changes are committed on `feature/05-runner-and-store`.
