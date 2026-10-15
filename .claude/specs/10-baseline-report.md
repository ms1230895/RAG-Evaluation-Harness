# Spec: Step 10 — Baseline Report

## Overview
This step turns the scores stored in Steps 6-8 into the project's "before"
number. `evalharness/report.py` reads the v0 dev run
`20261015T152638Z-v0-c7c96975` and joins three kinds of stored scores for
each answer: the deterministic checks (`checks-1-66c5a50a`), the faithfulness
judge (`faith-v2-d130a39f`) and the relevance judge (`rel-v2-c825e4f7`).

From them it reports:
- the **ungrounded rate** with a 95% Wilson interval, overall, for answerable
  queries and for each query type;
- the **failure cause** of each ungrounded answer: a retrieval miss (no source
  document reached the generator) or a generation error (the source was there
  but the answer added or changed facts);
- relevance next to faithfulness, including the two cases CLAUDE.md names: an
  invented answer to an unanswerable query, and a false refusal of an
  answerable one.

No LLM is called and nothing is re-run. Every number is read back from
`data/results.sqlite`. The report is printed, stored as one row in a new
`run_reports` table, written to `experiments/baseline_v0.md` and added as the
first row of `experiments/log.md`.

The step comes now because Step 11 changes one variable at a time and needs a
fixed number to compare against. Step 9 ended **inconclusive**, so the judges
are not calibrated. The baseline is therefore reported as "uncalibrated
judge", with the calibration status printed next to it (Decision 1).

## Concepts to learn
- **Baseline.** The measured score of the unchanged system on a fixed test
  set, under fixed scoring. A later result counts as better only when it is
  measured the same way:
  - same queries (test set SHA-256);
  - same split (dev);
  - same scorers (the same three versions).

  So the report records all of these. A Step 11 run scored by a different
  judge version cannot be compared with this number.
- **Ungrounded rate.** ungrounded answers / judged answers. The judge marks an
  answer ungrounded when faithfulness < 0.8 or any claim is contradicted
  (Step 7). Unanswerable queries stay in the denominator: a refusal is
  grounded, and an invented answer is ungrounded.
- **95% Wilson interval, recap.** It is the range of true rates that fit the
  observed count. For a rate p̂ = k/n with z = 1.96:

      centre = (p̂ + z²/2n) / (1 + z²/n)
      half   = z·sqrt(p̂(1−p̂)/n + z²/4n²) / (1 + z²/n)

  Example: 7/80 gives p̂ = 0.0875 and an interval of about [0.043, 0.170].
  With n = 80, a later result needs to move outside this range before anyone
  should believe it changed. Per type, n is 12-16, so those intervals are much
  wider.
- **Failure cause: where the RAG chain broke.** Each ungrounded answer is
  assigned one cause, using the Step 6 metric `in_context` (did any chunk from
  a source document reach the prompt?):

  | query | `in_context` | cause |
  |---|---|---|
  | answerable | 0 | **retrieval miss**: the generator never saw the source, so it guessed or refused |
  | answerable | 1 | **generation error**: the source was in the prompt, but the answer added or changed facts |
  | unanswerable | n/a (no source exists) | **generation error (invented answer)**: nothing can be retrieved, so any claim is made up |

  This decides which experiment family in Step 11 is worth trying. Retrieval
  misses call for retrieval fixes (chunking, top-k, BM25, reranker).
  Generation errors call for prompt fixes (answer only from context, refusal
  instruction, citations).
- **Shares vs rates.** The report gives each cause two ways:
  - as a rate over all judged answers (k/80, with a Wilson interval): "how
    often does this kind of failure happen";
  - as a share of the ungrounded answers (k/7): "which cause dominates".

  With only 7 ungrounded answers, the share's interval is very wide. That is
  stated, not hidden.
- **Faithfulness and relevance together.** A 2×2 table (ungrounded ×
  irrelevant) separates three failure kinds:
  - unfaithful but on topic;
  - faithful but off target, such as a false refusal;
  - both, such as an invented answer to an unanswerable question.

  A combined **bad answer** rate (ungrounded OR irrelevant) is also given, as
  a secondary number (Decision 2).
- **Reading uncalibrated numbers honestly.** Step 9 compared the judges with
  20 human labels, and none of those answers was flagged by either side. So
  nobody knows yet how often the judges miss ungrounded answers or flag
  grounded ones. The baseline is the judge's measurement, not a verified
  rate. Every line that uses a judge says "uncalibrated judge". A Step 11
  improvement measured with the same judge is still a fair comparison under
  that judge.
- **One sample per query.** Each answer is one sample from a cloud model that
  is not fully deterministic (Step 2). The Step 5 rerun
  (`20261015T152824Z-v0-59c05048`) gave identical answers only because the
  cache served them. A fresh, uncached run could differ by a few answers.
  This is a Step 3 carry-forward, and the report says it.

## Depends on
- Step 4: the `queries` table: `type`, `source_doc_ids`, `split`. Test set
  SHA-256 `0f09379f…11da5c`.
- Step 5:
  - `evalharness/store.py`: `connect`, `get_run` (`config_json`,
    `git_commit`, `git_dirty`, `test_set_sha256`, `system_json`, `split`,
    `status`), `get_outputs_with_queries`.
  - Run `20261015T152638Z-v0-c7c96975` (dev, completed, 80 ok).
- Step 6:
  - `scores` under `checks-1-66c5a50a`: `in_context`, `refusal`,
    `empty_answer`, `hit@1`, `hit@4`, `reciprocal_rank`.
  - `evalharness/checks.py`: `load_checks_config` (`version`,
    `report_confidence`), `get_scorable_run`, `ScoringRefused`,
    `ANSWERABLE_TYPES`.
  - `evalharness/metrics.py`: `wilson_interval`, `rate_summary`,
    `mean_summary`.
- Step 7: `scores` under `faith-v2-d130a39f` (`faithfulness`, `ungrounded`,
  `n_claims`) and `judge_claims`. `evalharness/judges.py`:
  `load_judges_config` (current version), `GROUP_TYPES`, `_coverage_state`.
- Step 8: `scores` under `rel-v2-c825e4f7` (`relevance`, `irrelevant`);
  `load_relevance_config`.
- Step 9: `calibration_reports`, the latest `check` comparison per judge
  (`store.get_calibration_reports`), printed as the calibration status.
- Carry-forward items addressed here:
  - Step 3: reproducibility rests on the gitignored cache, and each answer is
    one sample. Said in the report.
  - Step 4: per-type n is 12-16, so report per-type rates with intervals and
    draw no per-type conclusions.
  - Step 6: `in_context` gives retrieval miss vs generation error.
  - Step 8: cross-tabulate the invented answers (unanswerable, irrelevant and
    ungrounded) and the false refusals (answerable and irrelevant).
  - Step 9: report the ungrounded rate with the judge version used and its
    calibration status. The judges are uncalibrated.

## Interfaces
This step does **not** touch the `answer()` contract, does not call
`answer()`, does not import `rag` or `llm`, and makes **no LLM call**.

`evalharness/report.py` (new)
```python
CAUSES = ("retrieval_miss", "generation_error")

def current_versions(harness: dict) -> dict[str, str]:
    """{"checks": ..., "faithfulness": ..., "relevance": ...}, taken from the CURRENT
    configs (load_checks_config, load_judges_config, load_relevance_config), so the
    baseline and every Step 11 run are scored by the same settings by default."""

def failure_cause(answerable: bool, in_context: float | None) -> str:
    """retrieval_miss if answerable and in_context == 0; generation_error otherwise
    (answerable with the source in context, or unanswerable). Raises ValueError for
    an answerable item with in_context None (the Step 6 scores are missing)."""

def item_table(conn, run_id: str, versions: dict[str, str]) -> list[dict]:
    """One row per output: query_id, type, answerable, in_context, refusal,
    ungrounded, irrelevant, faithfulness, relevance, cause (None unless ungrounded),
    state ("judged" | "judge_failed" | "not_judged" | "missing" for each judge).
    Read from scores only. Refuses (ScoringRefused) if any of the three versions
    has no rows for the run."""

def build_baseline_report(conn, run_id: str, versions: dict[str, str], confidence: float,
                          *, allow_heldout: bool = False) -> dict:
    """Run metadata, coverage, ungrounded rate (all / answerable / each type, Wilson CI),
    failure causes (rate over judged answers and share of ungrounded, each with
    Wilson CI, overall and by type), the ungrounded × irrelevant 2×2 table, the
    bad-answer rate, the invented-answer and false-refusal lists, the retrieval
    summary (in_context, hit@4, MRR) and the calibration status of each judge
    version. Every count comes from item_table()."""

def format_baseline_report(report: dict) -> str          # terminal text
def format_baseline_markdown(report: dict) -> str        # experiments/baseline_v0.md
def log_row(report: dict, config_path: str, note: str) -> str
    """One experiments/log.md table row: date, config, "— (baseline)", run_id,
    "8.75% [4.30%, 16.98%] (7/80), uncalibrated judge" style text, note."""

def main(argv: list[str] | None = None) -> int
```
CLI:
```
python -m evalharness.report --run-id 20261015T152638Z-v0-c7c96975 [--markdown experiments/baseline_v0.md] [--log experiments/log.md] [--checks-version V] [--faithfulness-version V] [--relevance-version V] [--allow-heldout] [--harness-config configs/harness.yaml]
```
- The version flags override `current_versions()`. They exist only to report
  an older scoring.
- `--markdown` writes the file, refusing to overwrite unless `--force` is
  given.
- `--log` appends the row, refusing if a row with the same run_id is already
  there.
- Without either flag, the report is only printed and stored.
- Exit codes: 0 ok; 1 refused (unknown run, run not completed, held-out
  without the flag, a version with no rows, a judge with failed or missing
  items when `allow_incomplete` is false (Decision 3), an existing markdown
  file, a duplicate log row).

`evalharness/store.py` (new functions):
- `write_run_report(conn, row) -> int`
- `get_run_reports(conn, run_id) -> list[dict]`

## Data and schema changes
One new table, created by `store.create_tables()`:
```sql
CREATE TABLE IF NOT EXISTS run_reports (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    kind          TEXT NOT NULL,              -- "baseline" (Step 10); later steps add kinds
    versions_json TEXT NOT NULL,              -- {"checks", "faithfulness", "relevance"}
    report_json   TEXT NOT NULL,              -- full build_baseline_report() output
    created_at    TEXT NOT NULL
);
```
Every report invocation stores a row, so the number quoted in Progress, the
README or a CV can be traced to its stored report.

New committed files:
- `experiments/baseline_v0.md`: the generated report, with every number copied
  by code from the DB.
- One new row in `experiments/log.md`.

Accepting the run as the regression reference ("last accepted run") is left to
Step 13.

## Config changes
New `report` block in `configs/harness.yaml` (starting values; Decisions 2-3):
```yaml
# Baseline / run report (Step 10). Values approved by Gaurav on 2026-10-16.
report:
  primary_metric: ungrounded        # the headline rate (CLAUDE.md)
  include_bad_answer_rate: true     # also report ungrounded OR irrelevant (Decision 2)
  allow_incomplete: false           # refuse when any judge has failed / missing items (Decision 3)
```
The interval level reuses `checks.report_confidence` (0.95).

## Files to create
- `evalharness/report.py`
- `experiments/baseline_v0.md` (generated by the CLI, then committed)
- `tests/test_10_baseline_report.py`

## Files to change
- `evalharness/store.py`: `run_reports` table, `write_run_report`,
  `get_run_reports`.
- `configs/harness.yaml`: `report` block.
- `experiments/log.md`: the baseline row (appended by the CLI).
- `tests/conftest.py`: a `baseline_db` fixture. It is a tmp DB with one
  completed dev run of about 10 outputs, and checks, faithfulness and
  relevance rows that cover every cause and cross-tab cell. Optionally it also
  holds a calibration report.
- `tests/README.md`: Step 10 notes.
- `CLAUDE.md`: Progress entry.

## New dependencies
No new dependencies.

## Tests
`tests/test_10_baseline_report.py`, all offline. Tmp DBs, no LLM, never
`data/results.sqlite`.

`failure_cause`:
- answerable with `in_context` 0 → `retrieval_miss`;
- answerable with `in_context` 1 → `generation_error`;
- unanswerable with `None` → `generation_error`;
- answerable with `None` → `ValueError`.

`item_table`, on a hand fixture of 10 outputs covering:
- answerable, grounded and relevant;
- ungrounded with in_context 1;
- ungrounded with in_context 0;
- a false refusal (grounded, irrelevant, in_context 0);
- an unanswerable correct refusal;
- an unanswerable invented answer (ungrounded and irrelevant);
- a judge-failed item;
- one item per remaining type.

Checks:
- every field matches the hand values;
- the cause is set only for ungrounded rows;
- a version with no rows is refused.

Report numbers (hand arithmetic in the comments):
- ungrounded k/n overall, answerable and per type, with Wilson CIs equal to
  `wilson_interval`;
- the cause counts, the rate over judged answers and the share of ungrounded;
- the 2×2 table, which sums to the judged n;
- the bad-answer rate;
- the invented-answer list and the false-refusal list;
- judge-failed items are excluded from the denominators and counted.

With `allow_incomplete` false, a judge failure refuses; with it true, the
failure is reported and counted.

The numbers come from the DB:
- changing one stored `ungrounded` value changes the report;
- changing `in_context` moves an item between causes.

Versions:
- `current_versions` returns the versions computed from the real configs
  (read-only);
- an override flag is used when given;
- the report records all three versions, the test-set SHA, the git commit and
  the dirty flag.

Calibration status:
- with a stored check report whose positives are 0, the text says
  "inconclusive" and gives the agreement with its n;
- with no report it says "not calibrated";
- every judge-based line contains "uncalibrated judge".

Output:
- the markdown contains the headline rate with its CI, the per-type table, the
  cause table and the metadata;
- an existing file is refused without `--force`;
- the log row has the expected format, and a duplicate run_id is refused;
- one `run_reports` row is written per call.

CLI `main`:
- exit 0 with the printed report;
- exit 1 for an unknown run, a held-out run, a missing version, an existing
  markdown file and a duplicate log row.

Layering (AST): `evalharness/report.py` imports neither `rag` nor `llm`.

## Rules for implementation
- Plain Python and only libraries approved in CLAUDE.md
- No LangChain; no RAGAS or DeepEval (except the optional final cross-check)
- `rag/` must never import from `evalharness/`; the harness only calls `answer()`
- All tunable values come from YAML config — no hardcoded magic numbers
- SQLite via the sqlite3 module with parameterised queries only
- All LLM calls go through `llm/client.py` so they are cached (CLAUDE.md's
  `evalharness/llm.py` became the shared `llm/client.py` in Step 2). This
  step makes none.
- API keys only from `.env`, never in code, logs or commits
- The held-out split is never used before Step 12
- Never invent, estimate or adjust metric numbers; report only what a
  stored run produced
- Human labels (calibration and review) come only from Gaurav
- Code is simple, readable and commented

Step-specific:
- Every number in the report, `experiments/baseline_v0.md`, `experiments/log.md`
  and Progress is computed by code from `data/results.sqlite`. No figure is
  typed by hand, rounded differently or "corrected".
- Every judge-based number is labelled "uncalibrated judge". The calibration
  status line comes from `calibration_reports`, not from memory.
- The baseline names its exact scoring: the three versions, the run_id, the
  test-set SHA and the git commit. A later comparison with different versions
  is refused or marked as not comparable (Step 11 / 13).
- Judge-failed, not-judged and missing items are never counted as grounded or
  as ungrounded. They are excluded from denominators and counted.
- Per-type rates are always shown with n and interval, with the note that
  per-type differences inside the intervals are not meaningful.
- Say plainly where the CV hypothesis (25% → 12%) stands next to the measured
  number. The CV follows the measurement, not the other way round.
- No new run of v0 and no new judging (Decision 4). If a version has no rows,
  the report refuses instead of judging.
- Only the dev run is reported.

## Decisions for Gaurav
All resolved on 2026-10-16; Gaurav accepted every recommendation:
1. ~~Baseline while the judges are uncalibrated.~~ **Option A:** report now;
   every judge number labelled "uncalibrated judge" with the Step 9 status
   printed beside it.
2. ~~Failure-cause rules and secondary metric.~~ **Option A:** the cause table
   above, headline ungrounded rate, plus the "bad answer" rate.
3. ~~Incomplete judging.~~ `allow_incomplete: false`.
4. ~~Which run.~~ **Option A:** reuse `20261015T152638Z-v0-c7c96975`.
5. ~~Where the report lives.~~ **Option A:** printed, `run_reports`,
   `experiments/baseline_v0.md`, row 1 of `experiments/log.md`.

Clarified after the test run (Gaurav, 2026-10-16): percentages use 2 decimals
(k/80 is exact, so 7/80 prints 8.75%, never a float tie at 1 decimal); table
rates use 4 decimals; the markdown group table gives k and n in separate
cells; "uncalibrated judge" is on the headline line and on every judge-based
section heading (per section, not on every table row).

Implementation note: `report.py` imports `evalharness.judges` (for the
version loaders and `_coverage_state`), which itself imports `llm.client`.
The rule is "no direct import of rag or llm and no LLM call"; the check is
that the `llm_calls` max id is the same before and after the report.

## Definition of done
- [ ] Done when (CLAUDE.md): we have a real "before" number.
      `python -m evalharness.report --run-id 20261015T152638Z-v0-c7c96975 --markdown experiments/baseline_v0.md --log experiments/log.md`
      exits 0 and prints the ungrounded rate with its 95% CI, overall, for
      answerable queries and by type, and by failure cause. It also prints the
      relevance cross-tab and the calibration status, all labelled
      "uncalibrated judge".
- [ ] The headline matches SQL:
      `sqlite3 data/results.sqlite "SELECT SUM(value), COUNT(value) FROM scores WHERE run_id = '20261015T152638Z-v0-c7c96975' AND version = 'faith-v2-d130a39f' AND metric = 'ungrounded'"`
      gives the same k and n as the report.
- [ ] The causes match SQL: a join of `ungrounded = 1` with `in_context`
      (`checks-1-66c5a50a`) gives the same retrieval-miss and
      generation-error counts.
- [ ] `run_reports` holds the row:
      `sqlite3 data/results.sqlite "SELECT id, run_id, kind, versions_json FROM run_reports"`.
- [ ] `experiments/baseline_v0.md` exists and `experiments/log.md` has
      exactly one row for the run; a second `--log` call exits 1.
- [ ] No LLM call was made: the `llm_calls` max id is the same before and
      after.
- [ ] Held-out untouched: `SELECT COUNT(*) FROM runs WHERE split = 'heldout'`
      prints `0`.
- [ ] Layering: `grep -rnE "^(import|from) (rag|llm)" evalharness/report.py`
      prints nothing, and `grep -rn "evalharness" rag/` prints nothing.
- [ ] `pytest tests/test_10_baseline_report.py` passes with no network access.
- [ ] `pytest` passes (whole suite).
- [ ] The Progress section of CLAUDE.md has a Step 10 entry. It contains:
      - the run, the three versions and the `run_reports` id;
      - the headline ungrounded rate with n and CI;
      - the per-type rates and the cause counts;
      - the bad-answer rate;
      - the calibration status;
      - the CV hypothesis next to the measured number.

      Everything is copied from the report output and labelled "uncalibrated
      judge".
- [ ] All changes are committed on `feature/10-baseline-report`.

## Carry-forward to later steps
- **Step 11:** each experiment run is judged with the same three versions
  (`current_versions()` defaults), so it is comparable with this baseline. A
  version change, for example a new judge prompt, means re-scoring the
  baseline run under the new version before comparing. The ungrounded
  count here is 7/80, so experiments that fix 1-2 answers stay inside the CI.
  Report differences with their intervals and do not over-claim.
- **Step 12:** held-out confirmation uses `--allow-heldout` here, with the
  same versions.
- **Step 13:** "last accepted run" can start from this `run_reports` row.
- **Step 16:** README quotes `experiments/baseline_v0.md` with the calibration
  caveat.
