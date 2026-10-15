# Spec: Step 13 — Regression Comparator and Flagging

## Overview
Steps 5-12 produce one scored run at a time. This step adds the regression check that sits on top: compare a new dev run with the last *accepted* run, flag a subset of answers for human review, and print one pass/fail verdict from a configurable gate (default: fail if the ungrounded rate rises by more than 3 percentage points). It is the piece that Step 14 (one-command automation) and Step 15 (review queue) both consume: Step 13 writes the flags into the existing `reviews` table, Step 15 will show them. No LLM call is made: everything is read from stored scores. One caveat shapes the whole step: the only human labels so far are 20 answers, all "grounded", so the flag thresholds cannot honestly be tuned against labels yet. The step ships with fixed starting thresholds, measures them against whatever labels exist, and says "undefined" where there are no positives.

## Concepts to learn

**Regression testing.** A change to the prompt, the retriever or a judge can silently make answers worse. A regression check re-runs the same fixed queries and compares the result with a reference run you already trust. "Accepted run" = the reference run a human has signed off; it is stored, not guessed ("last run" would drift every time someone runs an experiment).

**Gate.** A gate turns numbers into one decision a script can act on. Here: `delta = (ungrounded rate of new run − ungrounded rate of accepted run) × 100`, in percentage points; FAIL if `delta > max_ungrounded_rise_points`. Worked example with 80 queries: accepted 7/80 = 8.75%, new 10/80 = 12.50%, delta = +3.75 points, FAIL at a 3-point gate; new 9/80 = 11.25%, delta = +2.50, PASS. One answer is 1.25 points at n = 80, so a 3-point gate trips on 3 extra ungrounded answers. The Step 11 noise control (seed 43, same config) moved 7 → 5, i.e. 2 answers = 2.50 points, so a gate this tight can fire on sampling noise alone; the report prints the paired flips and the exact p-value (Step 11 tools) beside the verdict so a person can see that.

**Flagging.** A flag is a reason to ask a human to look at one answer. Four reasons, each a plain yes/no test on stored values:
1. `ungrounded`: the stored `ungrounded` flag is 1 (faithfulness below the judge threshold 0.8, or any contradicted claim).
2. `low_relevance`: stored relevance value below `relevance_below` (0.75, so rubric scores 0 and 1 of 0/1/2).
3. `judges_disagree`: two independent signals about the same answer conflict (rules below).
4. `answer_changed`: the answer's word-level similarity to the accepted run's answer for the same query is below `answer_similarity_below`.
The **flag rate** is flagged answers / answers in the run. It measures how much human reviewing the gate creates, not how good the system is.

**Answer similarity.** `difflib.SequenceMatcher` ratio on lowercased word lists: `ratio = 2·M / T`, `M` = words in matching blocks, `T` = total words in both. Example: "meal bills cannot be claimed" vs "meal bills cannot be claimed separately": M = 5, T = 5 + 6 = 11, ratio = 10/11 = 0.909. Citation markers like `[2]` are removed first, otherwise the Step 11 citation prompt looks "changed" everywhere. Measured on stored dev runs while writing this spec (design information, not a reported result): the seed-43 noise control vs baseline had a minimum similarity of 0.67 and 64 of 80 identical answers, so a threshold of 0.6 flags nothing for sampling noise alone, while a real prompt change (citations) puts 13 of 80 below 0.6. This flag is meant for "nothing was supposed to change" checks, not for experiments.

**Judges disagree.** Faithfulness and relevance measure different things, so "ungrounded but relevant" (7 of 80 on the v0 baseline) is normal and is NOT a disagreement. What is a disagreement is a *deterministic check* and a *judge* contradicting each other about whether the answer refused:
- `refusal_vs_relevance_a`: answerable query, the phrase check says "refusal" (Step 6), but the relevance judge scored it fully relevant (1.0): one of the two is wrong.
- `refusal_vs_relevance_b`: unanswerable query, the phrase check says "no refusal", but the relevance judge scored it fully relevant (1.0, i.e. a clean refusal): the phrase list probably missed a wording.
Preview on the stored v0 dev run (design information): 0 answers match either rule.

**Tuning thresholds against human labels.** Flag precision = labelled-ungrounded among flagged ÷ flagged labelled; flag recall = flagged among labelled-ungrounded ÷ labelled-ungrounded. Without at least one human "ungrounded" label both are 0/0 = undefined, never 0 or 1. Threshold tuning means: change a threshold in YAML (new flag version), recompute precision and recall on the labelled `tune` answers, keep the change if it helps, then report once on `check`. Labels must come from Gaurav (CLAUDE.md); Claude never labels.

## Depends on
- Step 5: `evalharness/store.py` (`connect`, `get_run`, `get_outputs_with_queries`, `write_run_report`, `get_run_reports`, the `reviews` table), `evalharness/runner.py` (`load_harness_config`, `DEFAULT_HARNESS_CONFIG`).
- Step 6: `evalharness/checks.py` (`get_scorable_run`, `ScoringRefused`, `load_checks_config`, metric `refusal`, `in_context`), `evalharness/metrics.py` (`rate_summary`, `wilson_interval`).
- Steps 7-8: stored `faithfulness`, `ungrounded`, `relevance`, `irrelevant` rows; judge versions `faith-v2-d130a39f`, `rel-v2-c825e4f7`.
- Step 9: `calibration_labels` (Gaurav's labels), `evalharness/calibration.py` `status` conventions (0/0 printed as "undefined (0/0)").
- Step 10: `evalharness/report.py` (`item_table`, `current_versions`, `_calibration_status`, `_fmt_rate_pct`).
- Step 11: `evalharness/experiments.py` (`paired_flips`, `mcnemar_exact`, `paired_bootstrap_ci`, `compare_runs`).
- Step 12: `evalharness/heldout.py` pattern (pre-registration, refusals before writes); held-out runs are never compared here.
- Verified read-only on `data/results.sqlite` while writing this spec: 15 dev runs and 2 held-out runs, all completed; `reviews` is empty; `calibration_labels` holds 20 labelled rows, all grounded.

## Interfaces

**The `answer()` contract is not touched.** `rag/` is not touched at all.

### `evalharness/regression.py` (new; imports nothing from `rag/` or `llm/` directly)

```python
class RegressionRefused(Exception):
    """The runs cannot be compared, or a command is not allowed."""

def load_regression_config(harness: dict) -> dict:
    """The checked `regression` block of configs/harness.yaml plus "flag_version"
    = "flags-<CODE_VERSION>-<sha256 of the flag settings>[:8]". The gate value is
    NOT part of flag_version (changing the gate does not change which answers are flagged).
    ValueError names the missing or bad key."""

def answer_similarity(old: str, new: str, ignore_pattern: str) -> float:
    """Word-level SequenceMatcher ratio in [0, 1] after lowercasing and removing ignore_pattern.
    Two empty answers -> 1.0."""

def flag_reasons(item: dict, previous: dict | None, similarity: float | None, cfg: dict) -> list[str]:
    """Sorted list drawn from {"ungrounded", "low_relevance", "judges_disagree",
    "answer_changed", "not_judged"}. `item` / `previous` are report.item_table rows.
    A judge or output failure gives ["not_judged"] (never silently unflagged)."""

def accept_run(conn, run_id: str, note: str, harness: dict) -> int:
    """Record run_id as the accepted reference for its split; returns accepted_runs.id.
    Refuses: unknown / not completed / held-out / git_dirty / any judge failure or
    unjudged item / scorer versions without rows / already the current accepted run."""

def get_accepted_run(conn, split: str) -> dict | None:
    """The newest accepted_runs row for the split, or None."""

def build_regression_report(conn, run_id: str, against_run_id: str | None, harness: dict) -> dict:
    """Compare run_id with `against_run_id` (default: the accepted run of its split).
    Refuses (RegressionRefused / ScoringRefused): no accepted run; held-out split; runs not
    completed; different split, test-set sha256 or query ids; scorer rows missing;
    new run has failed outputs or unjudged items (a gate cannot pass on incomplete data).
    Pure read: no write."""

def store_flags(conn, report: dict, *, replace: bool = False) -> str:
    """Write one `reviews` row per flagged answer (flag_reasons = JSON list) and one
    `run_reports` row of kind "regression"; returns "written" or "unchanged".
    Never touches human_label or notes. A second call with the same flag_version is
    "unchanged"; with a different flag_version it raises RegressionRefused unless
    replace=True, which deletes only this run's flag rows whose human_label IS NULL."""

def flag_quality(conn, run_id: str, harness: dict) -> dict:
    """Flag precision / recall of this run's flags against Gaurav's calibration_labels
    (rows with grounded_label in grounded / ungrounded): tp, fp, fn, tn, precision, recall
    with Wilson intervals; undefined (None) when a denominator is 0; counts of unsure and
    unlabelled answers reported separately."""

def format_report(report: dict) -> str: ...        # terminal text, verdict first
def format_markdown(report: dict) -> str: ...      # optional --markdown file
def log_row(report: dict, created_at: str) -> str: # optional --log row
```

CLI (`python -m evalharness.regression ...`; exit codes: **0 gate passed, 1 gate failed, 2 refused or cannot decide**; nothing is written on exit 2):

```
accept     --run-id <id> --note "<why>"
check      --run-id <id> [--against <run id>] [--markdown PATH [--force]] [--log PATH] [--no-store]
flags      --run-id <id>                      # list stored flags of a run, no recompute
thresholds --run-id <id>                      # flag precision / recall vs Gaurav's labels
```

`check` is the Done-when command: it prints the flagged list (query id, type, reasons, question, answer excerpt, judge reasoning), the flag rate with Wilson interval, the gate numbers (accepted rate, new rate, delta in points, gate value, paired improved / regressed, exact p) and `PASS` or `FAIL`. Example:

```
python -m evalharness.regression accept --run-id 20261015T220912Z-v1_prompt_citations-23b6da7c --note "chosen config"
python -m evalharness.regression check  --run-id <new dev run id> --markdown experiments/regression_latest.md
```

## Data and schema changes
- **New table `accepted_runs`** (added in `store.SCHEMA` with `CREATE TABLE IF NOT EXISTS`; no change to existing tables, the database is not recreated):
  ```
  id          INTEGER PRIMARY KEY AUTOINCREMENT
  run_id      TEXT NOT NULL REFERENCES runs(run_id)
  split       TEXT NOT NULL            -- dev only in this step
  accepted_at TEXT NOT NULL            -- UTC ISO
  accepted_by TEXT NOT NULL            -- config labeler ("gaurav"): a human decision
  note        TEXT NOT NULL
  ```
  The accepted run of a split is its newest row. Rows are never updated or deleted.
- **`reviews`** (existing, empty): written by `store_flags`, `flag_reasons` = JSON list, `human_label` and `notes` NULL. Step 15 will set the label. Flag rows are keyed `(run_id, query_id)`; the flag version lives in the matching `run_reports` row.
- **`run_reports.kind`** also takes `"regression"` (free-text column; comment only).
- New store helpers: `write_accepted_run`, `get_accepted_runs`, `write_flag_rows`, `get_flag_rows`, `delete_unlabelled_flag_rows`.

## Config changes
New `regression:` block in `configs/harness.yaml` (starting defaults, not tuned values; Gaurav approves them as Decision 2):
```yaml
regression:
  relevance_below: 0.75               # low_relevance when relevance value < this (rubric 0 or 1 of 0/1/2)
  answer_similarity_below: 0.6        # answer_changed when similarity to the accepted answer < this
  similarity_ignore_pattern: '\[\d+\]'   # removed before comparing (citation markers)
  disagreement_rules:                 # which judges_disagree rules are on
    - refusal_vs_relevance_a
    - refusal_vs_relevance_b
  max_ungrounded_rise_points: 3.0     # gate: FAIL when new - accepted ungrounded rate > this, in points
  labeler: gaurav                     # accepted_by value; labels in flag_quality come only from this labeler
```
`ungrounded` flagging reuses the faithfulness threshold in `configs/judges.yaml` through the stored `ungrounded` flag (not a second threshold). No other config files change.

## Files to create
- `evalharness/regression.py`
- `tests/test_13_regression_comparator_and_flagging.py`
- `experiments/regression_latest.md` (generated by `check --markdown`, committed only if Gaurav wants a record)

## Files to change
- `evalharness/store.py`: `accepted_runs` table and helpers; `run_reports.kind` comment.
- `configs/harness.yaml`: the `regression:` block.
- `experiments/log.md`: rows appended by `check --log` when used.
- `tests/conftest.py` / `tests/README.md`: new fixtures and the Step 13 notes if needed.
- `CLAUDE.md`: Progress entry (after the done-when check).

## New dependencies
No new dependencies (`difflib`, `sqlite3`, `scipy`, `numpy`, `pyyaml` already available).

## Tests
`tests/test_13_regression_comparator_and_flagging.py`, offline, tmp DBs (target 30-50 tests):
- **Config:** valid block loads with a `flag_version`; missing key, negative gate, `relevance_below` outside [0, 1], unknown rule name each raise `ValueError`; changing the gate keeps `flag_version`, changing `relevance_below` or the similarity threshold changes it.
- **Similarity:** identical = 1.0; the worked example (5 vs 6 words) = 10/11; `[2]` markers ignored; case-insensitive; both empty = 1.0; one empty = 0.0.
- **Flag reasons (table-driven):** ungrounded=1 -> `ungrounded`; relevance 0.5 flagged and 1.0 not; answerable + refusal=1 + relevance 1.0 -> `judges_disagree`; unanswerable + refusal=0 + relevance 1.0 -> `judges_disagree`; "ungrounded but relevant" alone is NOT a disagreement; similarity 0.59 flagged, 0.6 not; no previous run -> no `answer_changed`; judge failure -> `not_judged`; multiple reasons sorted and de-duplicated.
- **Accept:** records a row; refuses held-out, dirty, incomplete judging, unknown run, already-accepted run; accepted run of a split is the newest row; rows never overwritten.
- **Report refusals (nothing written):** no accepted run; held-out split; different test-set sha256; different query ids; new run with a failed output or judge failure; missing scorer rows.
- **Gate (table-driven):** delta +2.5 PASS, exactly +3.0 PASS, +3.75 FAIL, negative PASS; custom gate value honoured; exit codes 0 / 1 / 2 from `main`.
- **Numbers:** flag rate k/n with Wilson interval equals `metrics.wilson_interval`; improved / regressed / p equal `experiments.paired_flips` / `mcnemar_exact` on a hand-built case; per-type counts sum to the total.
- **Storage:** `check` writes one `reviews` row per flagged answer and one `run_reports` row (kind `regression`); `human_label` and `notes` of an existing review row survive a re-check; same flag version = `unchanged`; different flag version without `replace` refused; `replace` deletes only unlabelled rows; `--no-store` writes nothing.
- **Flag quality:** zero labelled positives gives precision / recall `None` and the text "undefined (0/0)"; a hand-built labelled set gives the expected tp / fp / fn / tn; unsure and unlabelled are counted, not folded in.
- **Honesty strings:** report always contains "uncalibrated judge" while Step 9 is inconclusive, the gate value, n, and the noise note; every percent in the markdown traces back to the report dict.
- **No LLM:** `llm.client.generate` patched to raise; `llm_calls` unchanged.
- **Layering:** `regression.py` has no direct import of `rag` or `llm`.

## Rules for implementation
- Plain Python and only libraries approved in CLAUDE.md
- No LangChain; no RAGAS or DeepEval (except the optional final cross-check)
- `rag/` must never import from `evalharness/`; the harness only calls `answer()`
- All tunable values come from YAML config — no hardcoded magic numbers
- SQLite via the sqlite3 module with parameterised queries only
- All LLM calls go through `llm/client.py` so they are cached (the spec template says `evalharness/llm.py`; the real shared client is `llm/client.py`); this step makes no LLM call
- API keys only from `.env`, never in code, logs or commits
- The held-out split is never used before Step 12 (done); in this step it is never compared or flagged either: held-out runs are refused by every command
- Never invent, estimate or adjust metric numbers; report only what a stored run produced
- Human labels (calibration and review) come only from Gaurav
- Code is simple, readable and commented

Step-specific rules:
- Flags are written only to `flag_reasons`; `human_label` and `notes` are never written by this step and never overwritten.
- The accepted run is a human decision: `accept` is only ever run by a command from Gaurav; the code never accepts a run on its own, and `check` never auto-accepts a passing run.
- A gate result is only produced for complete data: failed outputs or judge failures in the new run mean exit 2, not PASS.
- The thresholds are fixed before looking at flag precision / recall on `check` labels. Threshold changes go through YAML (new `flag_version`) and are logged in `experiments/log.md`; no per-run threshold flags on the command line.
- Print the judge calibration status next to every judge-based number ("uncalibrated judge" until a `measured` calibration exists).
- A flag rate is reported as a rate with its Wilson interval and n; the CV hypothesis (10-15% flagged) is a hypothesis, not a target, and thresholds are never adjusted to reach it.
- Report the noise caveat of the gate (one answer = 100/n points; Step 11 noise control moved 2 answers) every time a gate verdict is printed.

## Decisions for Gaurav

1. **Which dev run is the first accepted run?** Accepting is your decision. *Recommendation:* the dev run of the chosen config, `20261015T220912Z-v1_prompt_citations-23b6da7c` (2/80 ungrounded), because that is the config the project would ship and future regressions should be measured against it. Alternative: the v0 baseline `20261015T152638Z-v0-c7c96975` (7/80), which makes nearly any later run "pass" the gate.
2. **Starting thresholds and "tune against my human labels".** Your only labels are 20 answers, all grounded, so there is nothing to tune against yet. *Recommendation:* ship the starting values in the config block above, add the read-only `thresholds` command (precision / recall of the flags against your labels, "undefined" with no positives), and you label more answers later with `python -m evalharness.calibration label` (the 8 flagged v0 answers and a sample of the rest) so a threshold change can be checked on real labels. Alternative: pause this step until you have labelled about 50 answers carefully, then tune before shipping.
3. **Definition of "the judges disagree".** *Recommendation:* the two refusal-vs-relevance rules in the spec (a deterministic check against a judge about whether the answer refused); faithfulness-vs-relevance cross-tab mismatches are reported as counts but not flagged because they are normal. Alternative: flag every "ungrounded but fully relevant" answer as a disagreement (7 of 80 on the v0 baseline, all of which are already flagged as `ungrounded`, so the flag list would not grow).
4. **Gate rule.** *Recommendation:* exactly the CLAUDE.md rule: FAIL when the ungrounded rate rises by more than 3.0 points (point estimate), print the paired flips and exact p beside it, and treat sampling noise as a reviewer's job. Alternative: FAIL only when the rise is above the gate AND the exact paired p < 0.05 (fewer false alarms, but at n = 80 a real +3.75 point rise would rarely reach p < 0.05, so real regressions would pass).

## Definition of done
- [ ] "Done when" from CLAUDE.md: one command outputs the flagged list, the flag rate and pass/fail: `python -m evalharness.regression check --run-id <dev run id>` prints the flagged answers, `flag rate k/n [low, high]`, and `PASS` or `FAIL`, with exit code 0 / 1; refusals exit 2.
- [ ] `python -m evalharness.regression accept --run-id <id> --note "..."` records the accepted run (`SELECT run_id, split, accepted_by, note FROM accepted_runs;` returns it), and refuses a held-out run (exit 2, no row).
- [ ] `check` on the accepted run itself (or an identical rerun such as the v0 twin) gives delta +0.00 points, PASS, and 0 `answer_changed` flags.
- [ ] SQL: `SELECT COUNT(*) FROM reviews WHERE run_id = ?;` equals the number of flagged answers printed; every row has a non-empty JSON `flag_reasons` and NULL `human_label`.
- [ ] A rerun of `check` prints `unchanged` and writes no new rows; `SELECT COUNT(*) FROM run_reports WHERE kind = 'regression';` grows by one per distinct (run, flag version) only.
- [ ] `python -m evalharness.regression thresholds --run-id <id>` prints flag precision and recall against the existing labels, with "undefined (0/0)" where there are no positives.
- [ ] `check` on a held-out run id exits 2 and writes nothing; `llm_calls` max id is unchanged by every command of this step.
- [ ] `pytest` passes (`tests/test_13_regression_comparator_and_flagging.py` plus the full suite).
- [ ] The Progress section of CLAUDE.md is updated with the real numbers and what they do and do not show.
- [ ] All changes are committed on `feature/13-regression-comparator-and-flagging`.
