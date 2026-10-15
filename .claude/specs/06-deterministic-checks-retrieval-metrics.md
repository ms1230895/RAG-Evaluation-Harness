# Spec: Step 06 — Deterministic Checks and Retrieval Metrics

## Overview
This step scores a stored run without calling any LLM. `evalharness/checks.py`
reads a completed run's `outputs` (Step 5), joins each output to its query's
type and `source_doc_ids`, and writes one row per (query, metric) into the
`scores` table. There are two kinds of metrics. Answer checks: is the answer
empty, does it read as a refusal, and how long did it take. Retrieval metrics:
hit rate@k and MRR, which compare the retrieved chunks' `doc_id`s with the
test set's source docs. It then prints a report, overall and by query type,
with sample sizes and 95% Wilson intervals. `evalharness/metrics.py` holds the
pure formulas (hit@k, reciprocal rank, Wilson interval) so Steps 7-13 can reuse
them. The step comes now because the runner has stored the v0 dev run, and the
retrieval metrics are what later steps need to split failures into
"retrieval miss" (the right doc never reached the prompt) and "generation
error" (Step 10). They are also the first real numbers the harness produces,
so this is where the "scored N, failed M, not applicable K" reporting habit
starts.

## Concepts to learn
- **Deterministic vs judged metrics.** A deterministic check gives the same
  result every time for the same stored output: string tests, numbers and set
  lookups, no model involved. These checks are free to run, exact and easy to
  test. They catch only shallow failures. Whether an answer is *grounded* needs
  the LLM judges (Steps 7-8). Run the cheap checks first; an empty answer does
  not need a judge to tell you it is bad.
- **Scoring is separate from running.** Scores are computed from stored
  `outputs`, never by calling `answer()` again. So scoring the same run twice
  must give identical rows. This step enforces that: re-scoring a run with the
  same settings is a no-op, and different results under the same version are
  refused.
- **Hit rate@k.** For one answerable query: did *any* chunk from a correct
  source doc appear among the first k retrieved chunks? 1 if yes, 0 if no. Hit
  rate@k is the mean over queries. It needs no LLM, because the test set stores
  the correct doc IDs. We match on `doc_id`, not `chunk_id`, because the test
  set names documents (Step 3 carry-forward: two chunks of one doc can take two
  top-k slots).
- **Rank and reciprocal rank (RR).** The rank is the position (1-based) of the
  first chunk whose `doc_id` is a source doc. RR = 1 / rank, or 0 if no source
  chunk was retrieved. **MRR** (mean reciprocal rank) is the mean RR over
  queries. Hit@k only asks "was it there?"; MRR also rewards ranking it high.
  Note that MRR here is MRR@top_k: a source chunk at rank 6 counts as 0 when
  `top_k` is 4. So MRR values are only comparable between configs with the same
  `top_k`.
- **Worked example.** Retrieved chunk doc IDs (rank order):
  `[intern_leave_policy, intern_leave_policy, leave_policy, internship_program_rules]`.
  Source docs `[leave_policy]`. First source rank = 3, so hit@1 = 0, hit@4 = 1,
  RR = 1/3 ≈ 0.333. Over three queries with RR 1, 1/3 and 0:
  MRR = (1 + 0.333 + 0) / 3 ≈ 0.444.
- **Multi-doc queries.** 13 dev queries have 2 or 3 source docs. A "hit" can mean
  *any* source doc retrieved, or *all* of them. Doc recall measures the share:
  retrieved `[A, A, C, D]` with sources `[A, B]` gives hit (any) = 1,
  all-found = 0, doc recall = 1/2 = 0.5 (Decision 1).
- **Not applicable is not zero.** Unanswerable queries have no source docs, so
  hit@k and RR are undefined for them. A failed output (`outputs.error` set)
  has no answer and no chunks. In both cases the row is stored with
  `value = NULL` and a `reasoning` that says why. The report counts these rows
  separately ("scored 68, n/a 12"). They are never stored as 0 or 1 (Step 2
  carry-forward). If unanswerable queries scored 0, hit rate would drop by 15%
  for a reason that has nothing to do with retrieval.
- **Refusal detection by phrase list.** An answer is marked a refusal if, after
  normalising (lowercase, curly apostrophes made straight, whitespace
  collapsed), it contains one of the configured phrases, e.g.
  `"there is no mention of"`. This is fast and transparent, but crude. It
  misses refusals worded differently. It also marks a "hedge then answer" reply
  ("The context doesn't specify X, but generally Y...") as a refusal, even
  though it still states a claim. So it is a cheap signal, not ground truth.
  The Step 8 relevance judge makes the real call. On an unanswerable query a
  refusal is the *correct* behaviour; on an answerable query it is a *false
  refusal*. The report shows the two separately.
- **Wilson interval (95%).** A rate measured on n items is uncertain. The
  textbook interval p ± z·√(p(1−p)/n) breaks for small n or for p near 0 or 1:
  at 12/12 it gives [1, 1], which claims certainty. The Wilson interval does not
  break there:

  ```
  center = (p + z²/(2n)) / (1 + z²/n)
  half   = z · √( p(1−p)/n + z²/(4n²) ) / (1 + z²/n)
  CI     = [center − half, center + half]      z = 1.96 for 95%
  ```
  Worked examples (n = 12, the size of one query type on dev):
  9/12 = 0.75 → Wilson [0.468, 0.911] (normal approx [0.505, 0.995]);
  12/12 → [0.758, 1.000] (normal approx [1, 1]);
  0/12 → [0.000, 0.242]. With 12-16 queries per type the intervals are about
  ±0.2 wide, so per-type differences of a few points mean nothing. The report
  prints n next to every rate for this reason.
- **Latency percentiles.** The mean is pulled up by one slow call, so the
  median (p50) and p95 describe latency better. `latency_ms` is retrieval time
  plus the generator's `model_latency_ms` (Step 3). On a cache-hit rerun it
  repeats the first run's model latency, so latency is only informative for
  runs whose generator calls were real API calls (`runs.llm_calls_json`
  misses > 0). The report says which.

## Depends on
- Step 4: `evalharness/schemas.py` (`QUERY_TYPES`), the `queries` table copy of
  the test set (type, `source_doc_ids` as JSON, split).
- Step 5: `evalharness/store.py` (`connect`, `create_tables`, `get_run`,
  `get_outputs`, the empty `scores` table, `RUN_STATUSES`);
  `evalharness/runner.py` (`load_harness_config`); `configs/harness.yaml`
  (`db_path`). Real data: run `20261015T152638Z-v0-c7c96975` (dev, completed,
  80 ok, 0 failed).
- Step 3: the contract's chunk dicts (`chunk_id`, `doc_id`, `text`, `score`)
  stored in `outputs.retrieved_chunks_json` in rank order; `top_k` 4 in
  `configs/v0.yaml`.
- Carry-forward items addressed here:
  - Step 2: failed items are stored in `scores` (value NULL, reason) and
    reported as "scored N, failed M" by type, never as 0 or 1. Latency figures
    use the stored `latency_ms` (from `model_latency_ms`), and the report says
    when it came from cache hits. Token and cost totals are not reported in
    this step: no LLM calls are made, and generator call counts are already in
    `runs.llm_calls_json`.
  - Step 3: hit@k and rank are defined at doc level (first rank at which any
    source doc's chunk appears).
  - Step 4: retrieval metrics skip unanswerable queries and the report says so;
    "any" vs "all" for multi-doc hit is Decision 1.
  - Step 5: only `completed` runs are scored.
- Observed in the stored v0 dev answers (looked at on 2026-10-15 to choose
  refusal phrases; dev is the tuning split, so this is allowed): refusals are
  worded "Based on the provided context, there is no mention of ..." or "...
  there is no information regarding ...", and some go on to give related facts.
  No answer is empty.

## Interfaces
This step does **not** touch the `answer()` contract, and it does not call
`answer()` or any LLM.

`evalharness/metrics.py` (new; pure functions, no I/O)
```python
def first_source_rank(chunk_doc_ids: list[str], source_doc_ids: list[str]) -> int | None:
    """1-based position of the first chunk whose doc_id is a source doc; None if none.
    Raises ValueError if source_doc_ids is empty (unanswerable: undefined)."""

def hit_at_k(chunk_doc_ids: list[str], source_doc_ids: list[str], k: int) -> int:
    """1 if a source doc appears in the first k chunks, else 0.
    Raises ValueError if k < 1, k > len(chunk_doc_ids) (cannot tell), or no sources."""

def reciprocal_rank(chunk_doc_ids: list[str], source_doc_ids: list[str]) -> float:
    """1 / first_source_rank, or 0.0 if no source doc was retrieved."""

def doc_recall(chunk_doc_ids: list[str], source_doc_ids: list[str]) -> float:
    """Share of distinct source docs that appear anywhere in the retrieved chunks."""

def wilson_interval(successes: int, n: int, confidence: float) -> tuple[float, float]:
    """Wilson score interval; z from scipy.stats.norm.ppf(1 - (1 - confidence) / 2).
    Both bounds are clamped to [0, 1] (float error can give 1.0000000002 at n/n).
    Raises ValueError if n < 1 or successes not in 0..n."""

def rate_summary(values: list[float | None], confidence: float) -> dict:
    """For 0/1 values (None = not applicable): {"n", "n_na", "successes", "rate",
    "ci_low", "ci_high"}; rate and CI are None when n == 0."""

def mean_summary(values: list[float | None]) -> dict:
    """{"n", "n_na", "mean"} (mean None when n == 0). Used for MRR and doc recall."""

def percentile_summary(values: list[float | None]) -> dict:
    """{"n", "n_na", "p50", "p95", "max"} via numpy.percentile (linear)."""
```
Example: `hit_at_k(["a", "a", "b", "c"], ["b"], 1)` → `0`;
`reciprocal_rank(["a", "a", "b", "c"], ["b"])` → `0.3333...`;
`wilson_interval(9, 12, 0.95)` → `(0.4677..., 0.9111...)`.

`evalharness/checks.py` (new)
```python
CHECKS_CODE_VERSION = 1   # bump when the scoring logic (not the settings) changes

class ScoringRefused(Exception):
    """The run cannot be scored (unknown, not completed, held-out, or a version conflict)."""

def load_checks_config(harness_config: dict) -> dict:
    """Check the `checks` block of configs/harness.yaml (keys and value types,
    ValueError names the problem) and return it with an added "version":
    f"checks-{CHECKS_CODE_VERSION}-{sha256(json of hit_ks + refusal_phrases, sort_keys)[:8]}",
    so any change to the phrases or ks gives a new version automatically."""

def normalize_text(text: str) -> str:
    """Lowercase, curly quotes/apostrophes to straight, whitespace runs to one space, strip."""

def is_empty_answer(answer: str) -> bool:          # answer.strip() == ""
def find_refusal_phrase(answer: str, phrases: list[str]) -> str | None:
    """First configured phrase found in normalize_text(answer), or None."""

def score_output(output: dict, query: dict, checks_config: dict) -> list[dict]:
    """All metric rows for one output: [{"metric", "value", "reasoning"}, ...].
    One row per metric in metric_names(checks_config), always, so coverage is
    easy to check. value None + reasoning for failed outputs and for retrieval
    metrics on unanswerable queries."""

def metric_names(checks_config: dict) -> list[str]:
    """["empty_answer", "refusal", "latency_ms", "hit@1", "hit@4", "in_context",
    "reciprocal_rank", "doc_recall"] for ks [1, 4]."""

def score_run(conn, run_id: str, checks_config: dict, *, allow_heldout: bool = False) -> str:
    """Score every output of a completed run and write the rows in one transaction.
    Returns "written" or "unchanged" (same version already stored with identical
    (query_id, metric, value, reasoning); scored_at is ignored). Raises ScoringRefused if the run is unknown, not 'completed', is
    held-out without allow_heldout, or rows for this version exist with different
    values (the scoring changed without a version change)."""

def build_report(conn, run_id: str, version: str, confidence: float) -> dict:
    """Overall and per-type summaries read back from `scores` (never recomputed
    in memory): counts (outputs, ok, failed), empty answers, refusals split into
    correct refusals (unanswerable) and false refusals (answerable), latency
    p50/p95/max, hit@k with Wilson CI, in_context with Wilson CI, MRR, doc
    recall, plus the query ids with in_context = 0 (retrieval misses) and the
    latency source from runs.llm_calls_json ("generator: 80 real calls, 0 cache
    hits"; with hits > 0 it adds that those latencies repeat earlier model
    latencies)."""

def format_report(report: dict) -> str:
    """Plain-text tables; every rate shows n, successes and its CI."""

def main(argv: list[str] | None = None) -> int
```
CLI:
```
python -m evalharness.checks --run-id 20261015T152638Z-v0-c7c96975 [--harness-config configs/harness.yaml] [--allow-heldout]
```
Prints `scored run <id> with checks-1-<hash>: written | 80 outputs x 8 metrics = 640 rows`
(or `unchanged`), then the report. Exit codes: 0 scored (or unchanged) and
reported; 1 refused (unknown run, not completed, held-out without the flag,
version conflict, bad config).

Metric rows written per output (with ks `[1, 4]`, Decision 3):

| metric | value | value NULL when | reasoning example |
|---|---|---|---|
| `empty_answer` | 1 / 0 | output failed | `"answer is blank"` / `"answer has 214 chars"` |
| `refusal` | 1 / 0 | output failed | `"matched 'there is no mention of'"` / `"no refusal phrase"` |
| `latency_ms` | ms | output failed | `"from outputs.latency_ms"` |
| `hit@1`, `hit@4` | 1 / 0 | failed, unanswerable, or fewer than k chunks | `"first source chunk at rank 3 (leave_policy#000)"` |
| `in_context` | 1 / 0 | failed, unanswerable | same; uses all retrieved chunks, whatever `top_k` is |
| `reciprocal_rank` | 0..1 | failed, unanswerable | `"rank 3 -> 1/3"` / `"no source doc retrieved"` |
| `doc_recall` | 0..1 | failed, unanswerable | `"1 of 2 source docs retrieved: leave_policy"` |

Failed output reasoning: `"output failed: ContextLimitError: ..."` (the stored error).
Unanswerable reasoning: `"unanswerable: no source docs"`.

`in_context` is the metric Step 10 uses for "retrieval miss vs generation
error": it asks whether any source doc reached the prompt, under any `top_k`.

`evalharness/store.py` (changed)
```python
def write_scores(conn, run_id: str, version: str, rows: list[dict]) -> None:
    """Insert all rows ({"query_id", "metric", "value", "reasoning"}) in one
    transaction, with scored_at = now; all or nothing."""

def get_scores(conn, run_id: str, version: str) -> list[dict]:
    """Rows for one run and version, ordered by query_id, metric."""

def get_outputs_with_queries(conn, run_id: str) -> list[dict]:
    """get_outputs() rows plus the query's "type", "split" and parsed
    "source_doc_ids", in query_id order."""
```

## Data and schema changes
`scores` gets a `version` column (in the primary key) and a `scored_at` column.
It is the right time: `scores` has 0 rows today, and Step 9 needs to keep the
scores of several judge versions for the same run side by side (Decision 5).

```sql
CREATE TABLE IF NOT EXISTS scores (
    run_id    TEXT NOT NULL REFERENCES runs(run_id),
    query_id  TEXT NOT NULL REFERENCES queries(id),
    metric    TEXT NOT NULL,                 -- "hit@4", "refusal", ...
    version   TEXT NOT NULL,                 -- (+) "checks-1-3fa1c27e"; judge versions from Step 7
    value     REAL,                          -- NULL = not applicable / output failed
    reasoning TEXT NOT NULL,                 -- why this value (or why NULL); now required
    scored_at TEXT NOT NULL,                 -- (+) UTC ISO
    PRIMARY KEY (run_id, query_id, metric, version)
);
```
Migration in `create_tables()`: if a `scores` table exists without the
`version` column, it is dropped and recreated **only if it has 0 rows**. If it
has rows, raise and stop: never drop stored scores. `outputs`, `runs`,
`queries` and `reviews` are unchanged. The DB file is not deleted.

`reasoning` becomes `NOT NULL`: every score explains itself, which the
dashboard (Step 15) and the review queue rely on.

No new data files.

## Config changes
New block in `configs/harness.yaml` (starting values, see Decisions 3-4):
```yaml
# Deterministic checks and retrieval metrics (Step 6). The stored version is
# computed from these settings, so any change gives a new scores.version.
checks:
  hit_ks: [1, 4]               # hit@k reported for each k; NULL when a run retrieved fewer than k chunks
  report_confidence: 0.95      # Wilson interval level (CLAUDE.md: 95%)
  refusal_phrases:             # matched as substrings of the normalised answer (Decision 4)
    - "no mention of"
    - "does not mention"
    - "doesn't mention"
    - "no information"
    - "does not specify"
    - "doesn't specify"
    - "does not provide"
    - "doesn't provide"
    - "not provided in the"
    - "does not contain"
    - "doesn't contain"
    - "cannot find"
    - "can't find"
    - "unable to find"
    - "i don't know"
    - "i do not know"
    - "cannot answer"
    - "can't answer"
    - "unable to answer"
```
`report_confidence` is not part of the version hash (it changes the report,
not the stored values). Phrases must be lowercase and non-empty, `hit_ks` must
be sorted, distinct integers ≥ 1. A hit always means *any* source doc
(Decision 1); `doc_recall == 1` shows "all found", so there is no config key
for the rule.

## Files to create
- `evalharness/metrics.py`
- `evalharness/checks.py`
- `tests/test_06_deterministic_checks_retrieval_metrics.py`

## Files to change
- `evalharness/store.py`: new `scores` schema plus the guarded migration;
  `write_scores`, `get_scores`, `get_outputs_with_queries`; module docstring
  ("first written in Step 6").
- `configs/harness.yaml`: `checks` block.
- `tests/conftest.py`: `scored_run_db` fixture (a tmp results DB with the 8
  test queries and one completed run of hand-made outputs: hits at rank 1 and 3,
  a miss, a multi-doc partial hit, a refusal on an unanswerable query, a false
  refusal, an empty answer, one failed output) and `checks_config` (a small
  phrase list, `hit_ks [1, 2]`).
- `tests/test_05_runner_and_store.py`: the exact-columns test for `scores`
  changes to the new columns.
- `tests/README.md`: Step 6 fixtures, test file and pitfalls.
- `CLAUDE.md`: Progress entry.

## New dependencies
No new dependencies (`numpy` and `scipy` are already approved and installed;
`hashlib`, `json`, `re`, `sqlite3` are standard library).

## Tests
`tests/test_06_deterministic_checks_retrieval_metrics.py`, all offline. No LLM, no
embedding, no network. It uses tmp DBs only and never `data/results.sqlite`.
Every expected number is computed by hand in a comment.

Metrics (`evalharness/metrics.py`):
- `first_source_rank`: rank 1, rank 3 after two non-source chunks, None when
  absent, duplicate source chunks count once (first one), empty sources raise.
- `hit_at_k`: the worked example (hit@1 = 0, hit@3 = 1, hit@4 = 1); k = 0 and
  k > len(chunks) raise; empty sources raise.
- `reciprocal_rank`: 1, 1/3, 0.0; MRR over [1, 1/3, 0] = 0.4444 via
  `mean_summary`.
- `doc_recall`: `[A, A, C, D]` vs `[A, B]` → 0.5; all found → 1.0; none → 0.0.
- `wilson_interval`: 9/12 → (0.4677, 0.9111), 12/12 → (0.7575, 1.0), 0/12 →
  (0.0, 0.2425) to 4 decimals; bounds always within [0, 1]; n = 0 and
  successes > n raise; a higher confidence gives a wider interval.
- `rate_summary` / `mean_summary` / `percentile_summary`: None values counted in
  `n_na` and excluded; all-None gives n = 0 and rate/mean None (no division
  by zero); p50/p95 match `numpy.percentile` on a hand list.

Checks (`evalharness/checks.py`):
- `normalize_text`: case, curly `’` → `'`, newlines and double spaces.
- `is_empty_answer`: `""`, `"  \n"` → True; `"0"` → False.
- `find_refusal_phrase`: matches across a line break and with a curly
  apostrophe ("doesn’t mention"); returns the first phrase in list order; no
  match → None. A control test shows that an answer without any listed phrase
  is not a refusal.
- `score_output`: for each fixture case, the exact rows (metric set equals
  `metric_names`, values, reasoning contains the rank or the reason);
  unanswerable → retrieval rows NULL with `"unanswerable: no source docs"`;
  failed output → every row NULL with the error text; `hit@k` NULL when fewer
  than k chunks; the multi-doc partial case has hit 1 and doc_recall 0.5.
- `load_checks_config`: the real `configs/harness.yaml` loads; missing keys,
  unsorted or duplicate `hit_ks`, `k < 1`, uppercase or empty phrase each raise `ValueError` naming the key (one control
  test first that the valid config passes); version changes when a phrase is
  added and does not change when only `report_confidence` changes; version is
  identical across two loads.
- `score_run`: writes `n_outputs × len(metric_names)` rows; a second call
  returns `"unchanged"` and writes nothing; tampering one stored value then
  re-scoring raises `ScoringRefused`; a run with status `running`, `aborted`
  or `invalid` is refused; an unknown run_id is refused; a held-out run is
  refused without `allow_heldout` and accepted with it (tmp DB, fake run only);
  a write failure mid-way leaves 0 rows (one transaction).
- `build_report`: on `scored_run_db` the overall and per-type counts, hit
  rates, CIs, MRR, refusal split (correct vs false) and the miss list match the
  hand-computed values; failed and n/a counts are shown, not hidden; values are
  read from `scores` (corrupt one stored row in the tmp DB and the report
  changes accordingly).
- `main`: exit 0 and the report printed for the fixture run (tmp harness YAML
  with a tmp `db_path`); exit 1 for an unknown run_id, a non-completed run and
  a held-out run without the flag.

Store:
- New `scores` columns exactly as above; PK allows the same (run, query,
  metric) under two versions; NULL value with reasoning is accepted; NULL
  reasoning is rejected.
- Migration: an old-style empty `scores` table is replaced; an old-style table
  with one row raises, and the row is still there afterwards.
- `get_outputs_with_queries` returns type, split and parsed `source_doc_ids`.
- A hostile reasoning string (`"'); DROP TABLE scores; --"`) is stored
  verbatim.

Layering guard (AST): `evalharness/checks.py` and `evalharness/metrics.py` do
not import `rag` or `llm`; `grep -rn "evalharness" rag/` finds nothing.

## Rules for implementation
- Plain Python and only libraries approved in CLAUDE.md
- No LangChain; no RAGAS or DeepEval (except the optional final cross-check)
- `rag/` must never import from `evalharness/`; the harness only calls `answer()`
- All tunable values come from YAML config — no hardcoded magic numbers
- SQLite via the sqlite3 module with parameterised queries only
- All LLM calls go through `llm/client.py` so they are cached (CLAUDE.md's
  `evalharness/llm.py` became the shared `llm/client.py` in Step 2). This step
  makes no LLM calls.
- API keys only from `.env`, never in code, logs or commits
- The held-out split is never used before Step 12
- Never invent, estimate or adjust metric numbers; report only what a
  stored run produced
- Human labels (calibration and review) come only from Gaurav
- Code is simple, readable and commented

Step-specific:
- Scores are computed only from stored `outputs` and `queries` rows; never call
  `answer()`, the retriever or an LLM.
- Only runs with `status = 'completed'` are scored.
- Every output gets exactly one row per metric. Not applicable or failed means
  `value = NULL` with a reason, never 0 or 1, and never a skipped row.
- The report reads its numbers back from `scores`. Every rate is printed with
  n, successes and its Wilson interval, plus the n/a and failed counts. MRR
  and doc recall are printed with n (no CI in this step).
- The report states that per-type n is 12-16 on dev, so per-type differences
  inside the CIs are not meaningful.
- Retrieval metrics match on `doc_id` and use the stored chunk order; never
  re-sort chunks by score.
- Re-scoring with the same version must be identical; never overwrite stored
  scores. A changed setting gets a new version (hash), and a changed scoring
  logic bumps `CHECKS_CODE_VERSION`.
- The refusal phrase list is a heuristic. The report calls it "refusal
  (phrase match)", and nothing in this step calls an answer correct, grounded
  or relevant.
- Do not change `configs/v0.yaml` or re-run the pipeline. The Done-when uses the
  stored v0 dev run.
- Never score or print anything from a held-out run in this step.

## Decisions for Gaurav
All resolved on 2026-10-15; Gaurav accepted every recommendation:
1. ~~Multi-doc hit rule.~~ **Option A:** hit = **any** source doc retrieved,
   plus `doc_recall` (share of source docs retrieved). No `multi_doc_hit`
   config key: `doc_recall == 1` already shows "all found".
2. ~~Rank unit.~~ **Option A:** chunk position in the retrieved list, so k
   means the same as `top_k`.
3. ~~k values.~~ **Option A:** `hit_ks: [1, 4]` plus `in_context`.
4. ~~Refusal phrase list.~~ **Option A:** the phrase list as drafted, then
   Gaurav looks at the printed correct and false refusal lists and changes a
   phrase only if it is clearly wrong (new version hash, both versions stay
   stored). Follow-up (2026-10-15, trial run on a scratch copy of the DB):
   `"not covered"` marked q013's correct answer "No, outpatient (OPD)
   consultations and medicines are not covered." as a refusal, so Gaurav
   removed it before the real scoring. The list in Config changes now has 19
   phrases. Remaining weak spots: `"does not specify"` can appear in a partly
   answered question, and a hedge-then-answer reply counts as a refusal.
5. ~~Versioning scores.~~ **Option A:** `version` in the `scores` primary key;
   the empty table is recreated (0 rows, nothing lost).
6. ~~Latency check.~~ **Option A:** report-only (p50, p95, max overall and by
   type); no threshold, no `slow` metric.

## Definition of done
- [ ] Done when (CLAUDE.md): these print for a run and are stored in `scores`.
      Run
      `python -m evalharness.checks --run-id 20261015T152638Z-v0-c7c96975`.
      It exits 0. It prints `written`, then the report: empty answers, refusals
      (correct and false), latency p50/p95/max, hit@1, hit@4, in_context
      (each with n and its 95% Wilson CI), MRR and doc recall, overall and for
      each of the 4 query types, plus the retrieval-miss query ids.
- [ ] Every output has every metric, once:
      `sqlite3 data/results.sqlite "SELECT version, COUNT(*), COUNT(DISTINCT query_id), COUNT(DISTINCT metric) FROM scores WHERE run_id = '20261015T152638Z-v0-c7c96975' GROUP BY version"`
      prints one row: `checks-1-<hash>|640|80|8` (with Decisions as
      recommended: 80 outputs × 8 metrics).
- [ ] Not-applicable rows are NULL with a reason, never 0:
      `sqlite3 data/results.sqlite "SELECT q.type, s.metric, COUNT(*) FROM scores s JOIN queries q ON q.id = s.query_id WHERE s.run_id = '20261015T152638Z-v0-c7c96975' AND s.value IS NULL GROUP BY q.type, s.metric"`
      shows only `unanswerable` rows (12 each for `hit@1`, `hit@4`,
      `in_context`, `reciprocal_rank`, `doc_recall`), since the run has 0
      failed outputs; and `SELECT COUNT(*) FROM scores WHERE reasoning IS NULL OR reasoning = ''` prints `0`.
- [ ] The printed numbers match SQL: hit@4 overall equals
      `sqlite3 data/results.sqlite "SELECT SUM(value), COUNT(value) FROM scores WHERE run_id = '20261015T152638Z-v0-c7c96975' AND metric = 'hit@4'"`
      and MRR equals `SELECT AVG(value) ... AND metric = 'reciprocal_rank'`.
- [ ] Re-running the same command prints `unchanged`, exits 0, and the
      row count stays 640.
- [ ] A non-completed or unknown run is refused:
      `python -m evalharness.checks --run-id does-not-exist` exits 1.
- [ ] No held-out data touched: `SELECT COUNT(*) FROM runs WHERE split = 'heldout'`
      prints `0`.
- [ ] Gaurav has looked at the printed refusal lists (correct and false
      refusals) and either accepted the phrase list or changed it (Decision 4).
- [ ] Layering: `grep -rnE "^(import|from) (rag|llm)" evalharness/checks.py evalharness/metrics.py`
      prints nothing, and `grep -rn "evalharness" rag/` prints nothing.
- [ ] `pytest tests/test_06_deterministic_checks_retrieval_metrics.py` passes with no
      network access.
- [ ] `pytest` passes (whole suite).
- [ ] The Progress section of CLAUDE.md has a Step 6 entry with the version,
      the row count, and the overall and per-type numbers copied from the
      report (with n and CIs), the retrieval-miss ids, the refusal counts, and
      the Step 6 test count. Weak or noisy results are stated as such.
- [ ] All changes are committed on `feature/06-deterministic-checks-retrieval-metrics`.
