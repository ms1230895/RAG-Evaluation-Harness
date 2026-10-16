# Spec: Step 15 — Streamlit Dashboard

## Overview
Steps 5-14 leave a results database (`data/results.sqlite`) with 19 runs (17 dev, 2 held-out), stored judge scores, 13 experiment reports, a regression report and 9 flagged `reviews` rows, all readable only through CLI text. This step adds the Streamlit + Plotly dashboard from the build sequence: trends across runs, run-vs-run comparison (queries that improved / regressed / changed answer), a review queue where Gaurav marks flagged answers OK or bad (saved to `reviews`), and a breakdown by query type with failure causes. The dashboard is a viewer: it computes no metric of its own (every number comes from the same functions the CLI reports use), calls no LLM, starts no run, and writes to exactly one place, the human fields of `reviews`.

## Concepts to learn

**A dashboard is a view, not a second implementation.** The risk with dashboards is that they quietly recompute a metric slightly differently from the CLI, and the README and the screen then disagree. Here every rate, Wilson interval, paired flip and failure cause is produced by an existing function (`item_table`, `rate_summary`, `compare_runs`, `build_baseline_report`, `answer_similarity`). The new code only reshapes those results into DataFrames and figures. Worked example: the dashboard's v0 dev ungrounded rate must print `7/80 = 8.75% [4.30%, 16.98%]`, exactly the stored baseline report; a test pins this.

**Why every bar needs n and an interval.** At n = 80, one answer is 1.25 points; per query type n is 12-16, so most between-type or between-run differences sit inside the 95% Wilson interval. A chart that shows only point estimates makes a 2-answer change look like a trend. Every rate in the app therefore shows `k/n` and its interval, and the y-axis of a rate always starts at 0.

**Streamlit's execution model.** Streamlit re-runs the whole script from top to bottom on every click. Two consequences: (1) expensive reads are wrapped in `st.cache_data(ttl=...)`, keyed by their arguments (run ids, not connections); (2) anything that must survive a click (which review item is current) lives in `st.session_state`. After a review is saved the cache is cleared so the queue reflects the write.

**Anchoring.** Showing the judge's verdict before a human decides pulls the human towards the judge's answer. The Step 9 calibration tool hid it for this reason. The review queue (Step 15) is for triage and must show the reasoning to be useful, so it keeps the reasoning one click away (collapsed) and its labels are NOT calibration labels and are never fed to the judge-agreement figures.

**Review labels measure precision, not recall.** The queue only contains answers the flags selected. The share marked "bad" among them is the flags' precision against Gaurav's review. Nothing about answers that were not flagged is learned, so recall cannot be derived from it.

## Depends on
- Step 5: `evalharness/store.py` (`connect`, `get_run`, `get_outputs_with_queries`, `get_flag_rows`, `get_run_reports`, `get_accepted_runs`), `evalharness/runner.py` (`load_harness_config`, `DEFAULT_HARNESS_CONFIG`), `evalharness/testset.py` (`resolve_path`).
- Step 6: `evalharness/metrics.py` (`rate_summary`, `mean_summary`, `wilson_interval`), `evalharness/checks.py` (`get_scorable_run`, `ScoringRefused`, `load_checks_config`).
- Step 7: `store.get_judge_claims` (per-claim verdicts and reasoning).
- Step 10: `evalharness/report.py` (`item_table`, `current_versions`, `build_baseline_report`, `load_report_config`, `_calibration_status`, `_fmt_rate_pct`).
- Step 11: `evalharness/experiments.py` (`compare_runs`, `load_experiments_config`, `ExperimentRefused`).
- Step 13: `evalharness/regression.py` (`answer_similarity`, `load_regression_config`, `get_accepted_run`, `RegressionRefused`), the `reviews` rows written by `check`.
- Step 14: runs made by `python -m evalharness run` have `runs.judge_version` filled.
- Verified read-only while writing this spec: `dashboard/` is an empty directory; `streamlit==1.66.0` and `plotly==7.1.0` are installed; in 1.66.0 `st.plotly_chart` and `st.dataframe` accept both `width` and the older `use_container_width` (implementation checks which one does not warn); `streamlit.testing.v1.AppTest` exists. The real DB has 17 dev runs (all completed, 0 dirty), 2 held-out runs, 1 accepted run (`20261015T220912Z-v1_prompt_citations-23b6da7c`), 9 `reviews` rows (all `human_label` NULL), 3 `regression` reports.

## Interfaces

**The `answer()` contract is not touched.** `rag/`, the judges, the checks, the flag rules and every pipeline config are not touched. The dashboard imports nothing from `rag/` and nothing from `llm/` directly.

### `evalharness/dashboard_data.py` (new; pure functions, no `streamlit` import)
Each function takes an open `sqlite3.Connection` plus ids and returns plain dicts / lists, so every function is unit-testable and the Streamlit layer only caches and renders.

```python
def eligible_runs(conn, versions: dict[str, str]) -> dict:
    """{"eligible": [run rows], "excluded": [{"run_id", "reason"}]}: completed runs that have score
    rows for ALL THREE current scorer versions. Everything else is listed in "excluded" with the
    reason (not completed / missing scorer version / failed outputs), never silently dropped.
    Each run row: run_id, timestamp, split, config_name (file stem of config_path), config_path,
    git_commit (short), git_dirty, judge_version, n_ok, n_failed."""

def trend_table(conn, run_ids: list[str], versions: dict[str, str], confidence: float) -> list[dict]:
    """One row per run, oldest first. Rates (ungrounded, bad_answer, irrelevant, in_context, hit@4)
    are rate_summary dicts (rate, successes, n, ci_low, ci_high); means (faithfulness, relevance,
    mrr) are mean_summary dicts (mean, n; NO interval). Computed from report.item_table only."""

def breakdown(conn, run_id: str, versions: dict, confidence: float, report_cfg: dict) -> dict:
    """The Step 10 `build_baseline_report` result for one dev run, unchanged: ungrounded by group,
    causes, crosstab, calibration status. Raises ScoringRefused when the run is incomplete."""

def compare_two(conn, base_run_id: str, new_run_id: str, versions: dict, cfg: dict,
                regression_cfg: dict) -> dict:
    """experiments.compare_runs(...) plus "changed_answers": [{"query_id", "similarity",
    "answer_base", "answer_new"}] for queries whose word similarity is below
    regression.answer_similarity_below, plus "similarity_threshold". compare_runs already
    returns "judge_label" per judge. Refusals (same run, different split / test set / queries)
    are raised as ExperimentRefused."""

def review_queue(conn, run_id: str, versions: dict) -> list[dict]:
    """Every `reviews` row of the run, joined with question, type, reference_answer, answer,
    retrieved chunks (doc_id, chunk_id, score, text), the faithfulness reasoning, the per-claim
    verdicts and the relevance reasoning. Ordered: unreviewed first, then by query_id."""

def review_progress(conn, run_id: str) -> dict:
    """{"flagged": N, "reviewed": k, "ok": a, "bad": b} for the run."""

def stored_gate_results(conn, run_id: str) -> list[dict]   # verdicts already stored in run_reports
def judge_labels(conn, run_id, versions) -> dict[str, str]  # "calibrated judge" | "uncalibrated judge"
def scorer_versions(harness: dict) -> dict[str, str]        # report.current_versions
def load_dashboard_config(harness: dict) -> dict            # validates the `dashboard:` block
```

### `dashboard/figures.py` (new; Plotly only, no `streamlit` import)
```python
def make_trend_figure(rows: list[dict], metric: str, confidence: float) -> go.Figure
def make_breakdown_figure(report: dict, group_names: list[str]) -> go.Figure
def make_cause_figure(report: dict, group_names: list[str]) -> go.Figure
def intervals_overlap(rates: list[dict]) -> bool   # True when every pair of Wilson intervals overlaps
```
Rate figures: y-axis fixed to [0, 1] shown as %, error bars from the stored Wilson bounds, `k/n` in hover and under each bar. Metric names reach the figure code only through an allow-list (`TREND_METRICS`), never as free text. Colors: Okabe-Ito palette, one color per metric and per query type, defined once in this file (presentation constants, not experiment values).

### `dashboard/app.py` and `dashboard/views.py` (new; thin Streamlit layer)
`streamlit run dashboard/app.py` (localhost). `app.py` only inserts the repo root into `sys.path`, sets the page config and wires the pages with `st.navigation`. The setup, cached reads and one function per page live in `dashboard/views.py` (a split from the first draft: `AppTest.switch_page` cannot reach callable pages, so tests call a page as `views.page_x()`). Pages:
- `page_trends()`: metric selector (allow-listed); dev runs as points ordered by time, coloured by config name, error bars = 95% Wilson interval; hover shows run_id, config, short git commit, judge versions, `k/n`. Below it, a separate "held-out (Step 12 confirmation)" table (never on the dev chart). A line of text states which scorer versions are shown and lists `excluded` runs with reasons. Caption: points are different configurations and repeats, not one system drifting over time.
- `page_compare()`: two selectboxes (defaults: newest dev run and the accepted run). Shows the ungrounded rate of both with CI, delta in points, paired improved / regressed / both / unpaired counts, the exact paired p, the faithfulness difference with its bootstrap interval, the noise note, then three tables with a text label (`improved`, `regressed`, `answer changed`), never colour alone. Selecting a row shows both answers side by side with each judge's reasoning. Different split or test set: `st.error` with the refusal text. Same run twice: refused.
- `page_review_queue()`: run selector (runs that have `reviews` rows), progress line `3 / 9 reviewed`, then one flagged answer at a time: question and type, reference answer, generated answer, flag reasons, retrieved chunks in an expander (`doc_id`, `chunk_id`, score), judge reasoning in a collapsed expander, notes box, **OK** and **Bad** buttons, Previous / Next. A click calls `store.write_review(...)`, clears the data cache and advances to the next unreviewed item.
- `page_breakdown()`: run selector (dev, complete), grouped bars of the ungrounded rate per type with CI and n, a second figure with the failure-cause split (retrieval_miss / generation_error), the faithfulness x relevance 2x2 table, caption "Difference is within noise at this sample size" whenever `intervals_overlap` is true, and the judge status line ("uncalibrated judge", Step 9 inconclusive).
- Sidebar: DB path, scorer versions in use, accepted run id, link-free caption "Judge numbers come from judges that are not calibrated against human labels" while `_calibration_status` is not `measured`.

### `evalharness/store.py` (new functions; SQL stays here, `?` placeholders only)
```python
def list_runs(conn, split: str | None = None) -> list[dict]
    # runs.run_id, timestamp, split, status, config_path, git_commit, git_dirty, judge_version,
    # n_queries, n_ok, n_failed; oldest first
def runs_with_versions(conn, versions: list[str]) -> set[str]
    # run ids that have score rows for every given version
def get_queries(conn, query_ids: list[str]) -> dict[str, dict]
    # question, type, reference_answer, source_doc_ids, split per id (reference answer is needed
    # by the queue; get_outputs_with_queries does not return it)
def runs_with_reviews(conn) -> list[str]
    # run ids that have at least one reviews row (the review page's run selector)
def write_review(conn, run_id, query_id, human_label: str, notes: str | None, labeler: str,
                 allowed_labels: list[str]) -> None
    # UPDATEs only an EXISTING reviews row (a flagged answer), sets human_label ('ok' | 'bad'),
    # notes, labeler, reviewed_at = now; ValueError for another label or a missing row.
    # Never touches flag_reasons, scores or outputs.
```
`write_review` follows the existing pattern (`with conn:` single transaction, rowcount check as in `write_calibration_label`).

## Data and schema changes
`reviews` gets two columns by `ALTER TABLE ... ADD COLUMN` in `store.create_tables` (the stored-runs-are-never-deleted rule), guarded so it runs once:
- `labeler TEXT`: who gave the label; set only by the dashboard from `regression.labeler` ("gaurav"). NULL on flag-only rows.
- `reviewed_at TEXT`: UTC ISO time of the label; NULL = not reviewed.

`human_label` stays the existing column; allowed values written by the dashboard: `ok`, `bad` (SQLite cannot add a CHECK to an existing column, so `write_review` and a test enforce the vocabulary). The 9 existing rows stay with NULL in the new columns. `outputs`, `scores`, `judge_claims`, `runs` and the calibration tables are not changed. `delete_unlabelled_flag_rows` already keeps rows that have a `human_label`, so re-checking a run never deletes a review.

## Config changes
New `dashboard:` block in `configs/harness.yaml` (starting defaults, for Gaurav to approve):
```yaml
dashboard:
  cache_ttl_seconds: 60        # how long a cached DB read is reused; cleared after every saved review
  show_heldout: true           # show the separate held-out table on the trends page
  excerpt_chars: 400           # display only: answers / chunks longer than this are expandable
  review_labels: [ok, bad]     # the only values the review buttons can write
```
The reviewer name is read from the existing `regression.labeler`. `load_dashboard_config(harness)` in `dashboard_data.py` validates the block (same style as `load_regression_config`). No other config changes; `configs/judges.yaml` and every pipeline config stay byte-identical.

## Files to create
- `evalharness/dashboard_data.py`
- `dashboard/__init__.py` (empty, so tests can import `dashboard.figures`)
- `dashboard/figures.py`
- `dashboard/app.py`
- `dashboard/views.py`
- `tests/test_15_streamlit_dashboard.py`

## Files to change
- `evalharness/store.py`: `list_runs`, `runs_with_versions`, `get_queries`, `write_review`, the two `reviews` columns, docstring/table list.
- `configs/harness.yaml`: `dashboard:` block.
- `tests/conftest.py`, `tests/README.md`: a `dashboard_db` fixture (built on `experiment_db`, plus reviews rows) and Step 15 notes.
- `tests/test_05_runner_and_store.py`: the exact-column assertion for `reviews` gets the two new columns (same kind of update Step 6 made for `scores`).
- `CLAUDE.md`: Progress entry (after the done-when check). `README.md` is Step 16.

## New dependencies
No new dependencies. `streamlit==1.66.0`, `plotly==7.1.0` and `pandas==3.0.6` are already pinned in `requirements.txt`. No Altair, aggrid or extras packages.

## Tests
`tests/test_15_streamlit_dashboard.py`, offline, tmp DBs, `llm.client.generate` patched to raise (target 30-45 tests). Data and figure functions are tested directly; a few `streamlit.testing.v1.AppTest` smoke tests cover the page wiring.
- **Config:** `load_dashboard_config` accepts the real block; missing key, unknown key, non-integer ttl, labels other than `[ok, bad]` raise `ValueError`.
- **Numbers equal the CLI:** `trend_table` and `breakdown` on a fixture run equal `build_baseline_report` / `rate_summary` output field by field; a hand-computed 3/10 gives the Wilson interval of `metrics.wilson_interval`.
- **Eligibility:** a run missing one scorer version, a running / aborted run and a run with failed outputs are in `excluded` with a reason; held-out runs are returned with `split == "heldout"` and never in the dev list.
- **No mixing:** `trend_table` for dev ids never includes a held-out run; asking it for a mix raises.
- **Compare:** improved / regressed lists equal `paired_flips`; `changed_answers` uses the configured threshold (0.59 in, 0.60 out; `[2]` markers ignored); same run twice, different split, different test-set sha256 raise `ExperimentRefused`.
- **Review queue:** unreviewed first; chunks carry `doc_id`, `chunk_id`, score; per-claim verdicts and reasoning present; progress counts correct.
- **`write_review`:** writes `ok` / `bad`, `notes`, `labeler`, `reviewed_at`; refuses another label and a row that is not flagged (no row inserted); never changes `flag_reasons`, `scores`, `outputs` (row counts and checksums before and after); a later `regression check --replace-flags` keeps the labelled row.
- **Schema migration:** a DB created with the old `reviews` layout gets both columns once; rerunning `connect` is a no-op; existing rows keep their values.
- **Figures:** y-axis range is [0, 1] for rate metrics; each bar's customdata holds `k/n`; the color of a query type is the same in the trend and breakdown figures; `intervals_overlap` true / false cases by hand; the metric allow-list rejects an unknown metric.
- **Safety:** answer text containing `<script>` and `<b>` reaches the page through `st.text` / `st.markdown` without `unsafe_allow_html` (grep test over `app.py`); no `unsafe_allow_html=True` anywhere.
- **AppTest smoke:** each page runs on the fixture DB without an exception; the review page after one OK click has a stored `ok` row and shows `1 / N reviewed`.
- **Layering:** `dashboard/` and `evalharness/dashboard_data.py` import nothing from `rag/` or `llm/`; `dashboard_data.py` does not import `streamlit`.
- **Read-only pages:** with the DB file opened read-only (a copy with write permission removed), the three viewing pages still render.

## Rules for implementation
- Plain Python and only libraries approved in CLAUDE.md
- No LangChain; no RAGAS or DeepEval (except the optional final cross-check)
- `rag/` must never import from `evalharness/`; the harness only calls `answer()`
- All tunable values come from YAML config — no hardcoded magic numbers
- SQLite via the sqlite3 module with parameterised queries only
- All LLM calls go through `llm/client.py` so they are cached (the spec template says `evalharness/llm.py`; the real shared client is `llm/client.py`). The dashboard makes no LLM call at all.
- API keys only from `.env`, never in code, logs or commits
- The held-out split is never used before Step 12 (done); in this step held-out runs are only displayed, read-only, in a separate table, and never feed a chart, comparison or review with dev runs
- Never invent, estimate or adjust metric numbers; report only what a stored run produced
- Human labels (calibration and review) come only from Gaurav: the buttons are the only writer, no label is pre-selected, suggested or defaulted
- Code is simple, readable and commented

Step-specific rules:
- No metric maths in `app.py` or `figures.py`: figures take stored summary dicts and draw them. A rate with n = 0 is shown as "n = 0", never 0%.
- Every rate shows `k/n` and its Wilson interval; rate axes start at 0 and end at 100%; means that have no interval are labelled "mean, no interval".
- Every judge-based number is labelled "uncalibrated judge" until a stored calibration is `measured` (reuse `_calibration_status`); the dashboard never says "accuracy" or "real rate".
- Dev and held-out never share a figure. Runs scored under other scorer versions are excluded and listed, never mixed in; a comparison of two runs checks split and test-set sha256 and refuses on a mismatch.
- Improved / regressed / changed are marked with a text label and an icon as well as colour.
- LLM answers, judge reasoning and chunk text are untrusted: rendered with `st.text`, `st.code` or `st.markdown` without `unsafe_allow_html`.
- The page writes only through `store.write_review`; the dashboard never edits `scores`, `outputs`, `runs`, `calibration_labels` or any run's flags.
- It never starts a run, never accepts a run (`regression accept` stays a CLI command for Gaurav) and never touches the held-out guard.
- Reads use short-lived connections inside cached functions keyed by run ids; the DB path comes from `configs/harness.yaml`, never a hardcoded string. A missing DB file shows an error instead of creating an empty one.

## Decisions for Gaurav

1. **Review label vocabulary and what is stored.** *Recommendation:* `ok` / `bad` in the existing `human_label` column, plus two new columns `labeler` and `reviewed_at` on `reviews` (so a label is traceable to a person and a time, as the calibration table already does). "bad" means "this answer should not ship", not specifically "ungrounded". Alternative: no new columns, only `human_label` and `notes`; simpler, but no timestamp and no author on the label.
2. **Judge reasoning in the review queue.** *Recommendation:* collapsed by default (one click to open) so you form your own view first; the labels are triage labels and are never used for judge calibration. Alternative: always visible (faster, more anchoring), or hidden completely (honest but you lose the reason for the flag).
3. **Held-out runs on the dashboard.** *Recommendation:* show the two held-out runs only in a separate, read-only table on the Trends page (labelled "held-out (Step 12 confirmation)"); they are not selectable on the Compare, Breakdown or Review pages, so nothing can be tuned on them. Alternative: hide them completely.
4. **Which scorer versions the dashboard uses.** *Recommendation:* only the current versions (`checks-1-66c5a50a`, `faith-v2-d130a39f`, `rel-v2-c825e4f7`); a run lacking rows for one of them is listed as excluded with the reason (today this leaves out one dev run, `20261015T152824Z-v0-59c05048`, the identical v0 rerun from Step 5, which was never scored; 16 of 17 dev runs and both held-out runs have all three versions). Alternative: a version selector, which risks putting runs judged by different prompts on one line.
5. **Re-labelling a review.** *Recommendation:* allowed (latest label wins, `reviewed_at` updated, shown with a "changed" note); it is a single-user tool and a mis-click should be fixable. Alternative: write-once, change only by SQL.
6. **Step 9 labeling page.** The skill file lists a blind labeling page for calibration, but Step 9 shipped as a CLI and CLAUDE.md's Step 15 lists only four pages. *Recommendation:* do not build it here; the 20 labels you gave are all "grounded", and the honest next step is a careful labelling pass that includes flagged answers, which the CLI already supports blind. Alternative: add `page_label()` that writes `calibration_labels` with the judge output hidden (more code, more room for anchoring bugs).
7. **Unflagged answers in the queue.** *Recommendation:* flagged answers only (what CLAUDE.md asks for), with a caption that this measures flag precision, not recall. Alternative: also show a small random sample of unflagged answers so recall can be estimated later (a new sampling rule and a new reviews-row source).

## Definition of done
- [ ] "Done when" from CLAUDE.md: all pages work on the real database: `streamlit run dashboard/app.py` starts without error; Trends shows the 16 eligible dev runs (and lists the 1 excluded) with intervals and the 2 held-out runs in their own table; Compare shows `20261015T152638Z-v0-c7c96975` against the accepted run with 7/80 vs 2/80, improved 0 / regressed 5, exact p 0.0625 (matching Step 13); Breakdown on the baseline run shows 7/80 = 8.75% [4.30%, 16.98%] and the same per-type counts as Step 10; Review queue lists the stored flags of a checked run and one OK click stores a row. Record the saved row (`SELECT run_id, query_id, flag_reasons, human_label, labeler, reviewed_at FROM reviews WHERE human_label IS NOT NULL`) and say which label Gaurav gave; nothing is labelled on his behalf.
- [ ] `python -c "from evalharness.dashboard_data import trend_table"` and `python -c "import dashboard.figures"` succeed without importing `streamlit` or `rag`.
- [ ] SQL: `SELECT COUNT(*) FROM outputs`, `scores`, `runs`, `judge_claims` unchanged before and after a full click-through of every page (only `reviews` may change).
- [ ] `grep -rn "unsafe_allow_html" dashboard/` finds nothing; `grep -rn "import rag\|from rag\|from llm\|import llm" dashboard evalharness/dashboard_data.py` finds nothing.
- [ ] `python -m pytest tests/test_05_runner_and_store.py tests/test_13_regression_comparator_and_flagging.py -q` pass (the `reviews` column assertion updated, nothing else).
- [ ] `pytest` passes (`tests/test_15_streamlit_dashboard.py` plus the full suite).
- [ ] The Progress section of CLAUDE.md is updated with the real command output and what the dashboard does not show (no recall, judges uncalibrated, per-type n 12-16, held-out only displayed).
- [ ] All changes are committed on `feature/15-streamlit-dashboard`.
