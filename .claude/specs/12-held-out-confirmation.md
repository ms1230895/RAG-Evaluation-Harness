# Spec: Step 12 — Held-Out Confirmation

## Overview
Step 11 picked one config, `configs/v1_prompt_citations.yaml`, by a rule fixed before the results, after looking at 12 experiments on the same 80 dev queries. Because the best of 12 looks better than it is, its dev number (2.50%, 2/80) is an optimistic estimate. This step runs the chosen config **once** on the 20 held-out queries that were never used for tuning, scores it with the same frozen judges, and reports the result with its Wilson CI. The harness already has the `--allow-heldout` switches; what is missing is (a) a hard guard that makes "once" real instead of a promise, (b) a pre-registration file written before the run, and (c) a report that puts held-out numbers next to the dev numbers and says plainly what 20 queries can and cannot show. No pipeline code, judge prompt, threshold or config may change in this step.

## Concepts to learn

**Why a held-out set exists.** Every time you look at dev results and pick the best config, you fit a little to the dev queries' noise. The more things you try, the more the winner's score is luck. Held-out queries were never looked at, so a number measured on them is an honest estimate of how the chosen config behaves on new questions. It only works once: if you run it, dislike the number and tweak, the held-out set becomes a second dev set.

**Winner's curse (regression to the mean).** Roll 12 dice; the highest is probably above 3.5 because you picked it for being high. Re-roll that die and it is probably lower. The dev figure 2/80 was the lowest of 12 runs; on fresh queries expect it to drift back towards the baseline's rate. A held-out result worse than dev is the normal case, not a failure.

**Wilson interval with n = 20.** Same formula as before: for `k` successes in `n`, with `z = 1.96`,
`centre = (p + z²/2n) / (1 + z²/n)`, `half = z·sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n)`.
Illustrative arithmetic (not a measurement): 0/20 gives an upper bound of `z²/(n+z²) = 3.84/23.84 = 0.161`, so even zero ungrounded answers is compatible with a true rate of about 16%. 2/20 = 10% gives about [2.8%, 30.1%]. One answer is 5 percentage points. A 20-query held-out set cannot show a 6-point improvement; it can only fail to contradict one.

**Paired comparison, again.** Run v0 and the chosen config on the same 20 queries. Only queries where the ungrounded flag differs carry information. If `b` improved and `c` regressed, the exact two-sided p-value is `2 × P(X ≤ min(b,c))` for `X ~ Binomial(b+c, 0.5)`, capped at 1. With b = 3, c = 0: p = 0.25. With 20 queries and a baseline near 9%, `b + c` will likely be 0-3, so p will not be below 0.05. The step must say that out loud.

**Pre-registration.** Write down, in a committed file, which config you will run, which judge versions score it, how many times you may run it, and how the result will be worded, *before* you see the result. Then the wording cannot be bent afterwards to fit the number.

**Selection bias vs generalisation gap.** Two different questions: "is the chosen config better than v0 on unseen queries?" (paired comparison, held-out v0 vs held-out chosen) and "does the dev number hold up on unseen queries?" (chosen on dev vs chosen on held-out, unpaired, different queries). The report shows both and tests neither strongly; n is too small.

## Depends on
- Step 4: `data/test_set.jsonl`, `evalharness/testset.py` (`load_test_set`, `filter_split`), 20 held-out queries (lookup 10, multi_chunk 4, paraphrased 3, unanswerable 3); `configs/test_set.yaml`.
- Step 5: `evalharness/runner.py` (`main`, `run_split`, `--allow-heldout`, `git_info`, `load_harness_config`), `evalharness/store.py` (`connect`, `get_run`, `write_run_report`, `get_run_reports`).
- Step 6: `evalharness/checks.py` (`get_scorable_run(..., allow_heldout=True)`, `score_run`, CLI `--allow-heldout`).
- Steps 7-8: `evalharness/judges.py` CLI `faithfulness` / `relevance` with `--allow-heldout`, `--retry-failed`; judge versions `faith-v2-d130a39f`, `rel-v2-c825e4f7`.
- Step 10: `evalharness/report.py` (`build_baseline_report`, `current_versions`, `item_table`, `_calibration_status`).
- Step 11: `evalharness/experiments.py` (`compare_runs`, `paired_flips`, `mcnemar_exact`, `paired_bootstrap_ci`, `load_experiments_config`), `configs/v1_prompt_citations.yaml`, `experiments/experiment_table.md`, `experiments/log.md`. Dev runs: baseline `20261015T152638Z-v0-c7c96975`, chosen `20261015T220912Z-v1_prompt_citations-23b6da7c`.
- Verified now (read-only on `data/results.sqlite`): `runs` holds 15 rows, all `split = dev`, all `completed`; no held-out run exists; `queries` has 80 dev + 20 heldout rows.

## Interfaces

**The `answer()` contract is not touched.** `rag/` is not touched at all.

### `evalharness/heldout.py` (new; imports nothing from `rag/` or `llm/`)

```python
class HeldoutRefused(Exception):
    """A held-out run or report is not allowed under the pre-registration."""

def load_heldout_config(harness: dict) -> dict:
    """Read and validate configs/heldout.yaml (path = harness["heldout_config"]).
    ValueError names the missing or bad key."""

def check_heldout_allowed(conn, config_path: str, config: dict, heldout_cfg: dict) -> dict:
    """Raise HeldoutRefused unless a held-out run of this config is allowed.
    Returns {"used": int, "max": int, "config_sha256": str} for printing.
    Refuses when:
      - config_path is not in heldout_cfg["allowed_configs"];
      - sha256 of the config file bytes != the pinned value for that path;
      - the stored runs table already has max_runs_per_config held-out rows
        for this config (ANY status counts: completed, aborted, invalid,
        interrupted, running);
      - the three current scorer versions != heldout_cfg["pinned_versions"]
        (the judges must be the ones used on dev)."""

def build_confirmation(conn, baseline_run_id: str, run_id: str, heldout_cfg: dict,
                       harness: dict) -> dict:
    """The held-out confirmation report, computed only from stored rows (no LLM call).
    Refuses (HeldoutRefused / ScoringRefused / ExperimentRefused) unless:
      both runs completed, split heldout, same test set sha256 and query ids,
      config paths are the registered baseline and chosen configs,
      all three scorer versions == pinned_versions and have rows for both runs,
      no judge failures or unjudged items (report.allow_incomplete is not used here),
      git_dirty = 0 for both runs."""

def format_confirmation(report: dict) -> str: ...         # terminal text
def format_confirmation_markdown(report: dict) -> str: ... # experiments/heldout_confirmation.md
def log_row(report: dict, created_at: str) -> str: ...     # one row per run for experiments/log.md
```

CLI (all exit 0 ok, 1 refused with nothing written):

```
python -m evalharness.heldout check   --config configs/v1_prompt_citations.yaml
python -m evalharness.heldout confirm --baseline-run <v0 heldout run id> --run <chosen heldout run id> \
        [--markdown experiments/heldout_confirmation.md [--force]] [--log experiments/log.md]
```

- `check` prints the preflight result (allowed / refused and why, runs used vs max, pinned versions, config sha256). It never starts a run.
- `confirm` writes one `run_reports` row of kind `"heldout"` for the chosen run, prints the report, optionally writes the markdown and appends log rows (one for each of the two held-out runs).

### `evalharness/runner.py` (changed)
When `--split heldout`, `main` calls `heldout.check_heldout_allowed` after the existing refusals and before anything is written. Extra rules on held-out only: `--allow-dirty` is refused (the commit must be the code that runs), and a refusal exits 1 with nothing written. The runner imports `evalharness.heldout`, not `rag/`.

Example call (the real run, done last, only after the pre-registration is committed):

```
python -m evalharness.runner --config configs/v1_prompt_citations.yaml --split heldout --allow-heldout
python -m evalharness.runner --config configs/v0.yaml --split heldout --allow-heldout
python -m evalharness.checks  --run-id <id> --allow-heldout            # for each run
python -m evalharness.judges faithfulness --run-id <id> --allow-heldout
python -m evalharness.judges relevance    --run-id <id> --allow-heldout
python -m evalharness.heldout confirm --baseline-run <v0 id> --run <chosen id> \
        --markdown experiments/heldout_confirmation.md --log experiments/log.md
```

### Report content (`build_confirmation` result, also stored in `run_reports.report_json`)
1. **Registration echo:** selection rule text, chosen and baseline config paths, config sha256, pinned scorer versions, runs used / max, judge labels (`uncalibrated judge` unless Step 9 measured, which it did not).
2. **Held-out numbers per run** (reuse `build_baseline_report` pieces): ungrounded all / answerable / per type with Wilson CIs and k/n, bad-answer rate, false refusals, invented answers, hit@4, in_context, MRR, failure cause split, failed outputs.
3. **Paired comparison on the 20 held-out queries** (reuse `compare_runs`): delta in points, improved / regressed lists, exact p, faithfulness difference with seeded paired bootstrap CI, flipped answers with judge reasoning.
4. **Dev vs held-out side by side** for the chosen config and for v0: rate with CI on dev (read from the stored dev runs named in `heldout.yaml`) and on held-out; "held-out rate inside the dev CI: yes/no" as a fact, not a verdict.
5. **Verdict (fixed rule, see Decision 3)** and **limitations** block (n = 20, per-type n 3-10, uncalibrated judge, winner's curse, one sample from a non-deterministic cloud model, the CV hypothesis 25% -> 12% is not a target).
6. Every ungrounded held-out answer listed with question, answer, retrieved chunk ids and judge reasoning, so Gaurav can read them.

## Data and schema changes
No schema changes. `run_reports.kind` also takes `"heldout"` (a free-text column; update the comment in `evalharness/store.py`). The "used once" ledger is the existing `runs` table (`split = 'heldout'`, matched by `config_path`), so it cannot drift from the real runs. New data file: `configs/heldout.yaml` (below).

## Config changes

### New `configs/heldout.yaml` (pre-registration; committed BEFORE any held-out run)
```yaml
# Written 2026-10-16, before any held-out run. Do not edit after the first held-out run
# except to add a dated, explained note under `amendments`.
selection_rule: >
  Lowest ungrounded rate among Step 11 dev runs whose bad-answer rate and false-refusal
  count are not worse than baseline, noise control excluded (fixed before the results).
baseline_config: configs/v0.yaml
chosen_config: configs/v1_prompt_citations.yaml
allowed_configs: [configs/v0.yaml, configs/v1_prompt_citations.yaml]
config_sha256:                       # sha256 of each file's bytes, filled in at pre-registration
  configs/v0.yaml: <64 hex>
  configs/v1_prompt_citations.yaml: <64 hex>
max_runs_per_config: 1               # every held-out row in `runs` counts, whatever its status
pinned_versions:                     # the scorers used on dev; a changed judge refuses the run
  checks: checks-1-66c5a50a
  faithfulness: faith-v2-d130a39f
  relevance: rel-v2-c825e4f7
dev_runs:                            # stored dev runs shown beside the held-out numbers
  baseline: 20261015T152638Z-v0-c7c96975
  chosen: 20261015T220912Z-v1_prompt_citations-23b6da7c
significance_level: 0.05             # exact paired p below this = "supported"
output:
  markdown_path: experiments/heldout_confirmation.md
amendments: []
```

### `configs/harness.yaml` — one new key
`heldout_config: configs/heldout.yaml` (path of the pre-registration file; required when `--split heldout` is used).

No other config changes. `configs/v0.yaml`, `configs/v1_prompt_citations.yaml`, `configs/judges.yaml`, `configs/test_set.yaml` stay byte-identical (pinned by sha256).

## Files to create
- `evalharness/heldout.py`
- `configs/heldout.yaml`
- `tests/test_12_heldout_confirmation.py`
- `experiments/heldout_confirmation.md` (generated by `confirm`, committed after the real run)

## Files to change
- `evalharness/runner.py`: held-out guard call, `--allow-dirty` refused on held-out.
- `evalharness/store.py`: comment only (`kind` may be `"heldout"`).
- `configs/harness.yaml`: `heldout_config` key.
- `experiments/log.md`: two rows appended by `confirm --log`.
- `CLAUDE.md`: Progress entry for Step 12 (only after the done-when check).
- `tests/conftest.py`: small fixture for a tmp DB with dev and held-out runs, only if the existing ones do not cover it.

## New dependencies
No new dependencies (`scipy`, `numpy`, `pyyaml`, `sqlite3` already in use).

## Tests
`tests/test_12_heldout_confirmation.py`, offline, fake system, tmp DBs, fake judge LLM; target 25-40 tests.

- **Config:** valid `heldout.yaml` loads; missing key, `max_runs_per_config < 1`, a baseline or chosen config absent from `allowed_configs`, a missing sha256 entry each raise `ValueError`.
- **Guard:** a config not in `allowed_configs` is refused; a changed config file (sha mismatch) is refused; with 0 prior held-out rows the run is allowed; with 1 prior held-out row of any status (`completed`, `aborted`, `invalid`, `interrupted`, `running`) a second run is refused; a held-out row of a *different* config does not count; a dev row of the same config does not count; a changed pinned scorer version is refused.
- **Runner:** `--split heldout` without `--allow-heldout` still refused; with `--allow-heldout` but `--allow-dirty` refused; refusal writes nothing to `runs`; an allowed run writes one `runs` row with `split = heldout` and exactly the 20 held-out query ids; a second invocation of the same config is refused with exit 1 and no new rows; dev runs are unaffected by the guard.
- **Report refusals:** unknown run; run not completed; dev-split run passed to `confirm`; two runs with different test-set sha256; a judge failure or unjudged item; scorer versions not equal to pinned; `git_dirty = 1`; baseline run config path not the registered baseline.
- **Report numbers:** on a hand-built tiny DB, ungrounded k/n and Wilson CI equal the metrics module's values; improved / regressed lists and exact p match a hand-computed case (b = 3, c = 0 gives p = 0.25; b = 0, c = 0 gives p = 1.0 or None per `mcnemar_exact`); dev-rate-inside-held-out-CI flag correct at both boundaries; per-type counts sum to the overall count.
- **Verdict rule:** table-driven over (b, c, p): `supported_better`, `not_distinguishable`, `supported_worse` labels exactly as pre-registered; a lower point estimate with p >= 0.05 is never labelled better.
- **Honesty:** the markdown always contains "uncalibrated judge" while Step 9 status is inconclusive, the held-out n, the line that one answer equals 5 points, and the CV-hypothesis line; no number in the markdown that is not in the report dict (regex all `k/n` and percent tokens back to the dict).
- **Storage:** `confirm` writes exactly one `run_reports` row of kind `heldout`; re-running `confirm` for the same run is refused when `--log` already has the run id; `--markdown` refuses overwrite without `--force`; `llm_calls` max id unchanged by `confirm` (no LLM call).
- **Layering:** `evalharness/heldout.py` has no import of `rag` or `llm` (AST check, like earlier steps).

## Rules for implementation
- Plain Python and only libraries approved in CLAUDE.md
- No LangChain; no RAGAS or DeepEval (except the optional final cross-check)
- `rag/` must never import from `evalharness/`; the harness only calls `answer()`
- All tunable values come from YAML config — no hardcoded magic numbers
- SQLite via the sqlite3 module with parameterised queries only
- All LLM calls go through `llm/client.py` so they are cached (the spec template said `evalharness/llm.py`; the real shared client is `llm/client.py`)
- API keys only from `.env`, never in code, logs or commits
- The held-out split is never used before Step 12 (this is Step 12; see the specific rules below)
- Never invent, estimate or adjust metric numbers; report only what a stored run produced
- Human labels (calibration and review) come only from Gaurav
- Code is simple, readable and commented

Step-specific rules:
- **Order is fixed:** (1) write the code and tests and get `pytest` green using fakes only; (2) write `configs/heldout.yaml` with real sha256 values; (3) commit; (4) only then run the held-out runs. The implementation must not run `runner`, `checks`, `judges`, `report` or `confirm` on the real database with the held-out split before step 4, and must not read held-out outputs while developing.
- **No tuning, ever.** No change to the pipeline, prompts, configs, judges, thresholds or refusal phrases after the first held-out run starts. If the result is disappointing, report it. If a bug is found, record it, do not re-run silently; Gaurav decides (Decision 2).
- Run each config at most once; a failed, aborted or invalid held-out run still counts as used.
- Generator failures on held-out are counted and reported as failures, never dropped or retried by the runner. Judge failures may be retried with `--retry-failed` (same cached call, same version), because that does not change the generator's output.
- The scorer versions must equal the dev versions. No judge prompt iteration.
- The report states `uncalibrated judge` unless Step 9 produced a `measured` calibration, and says in words that a held-out ungrounded rate is the judge's measurement, not a verified rate.
- Do not claim the held-out result "confirms" improvement unless the pre-registered verdict rule says `supported_better`; otherwise use the rule's wording. No per-type claims: per-type n is 3-10.
- `confirm` is read-only on scores: it computes everything from stored rows and makes no LLM call.
- Both held-out runs use the same cached judge settings; the first run's judge calls are real misses (new queries), which is expected.

## Decisions for Gaurav

1. **Run v0 on held-out as well?** CLAUDE.md says "run the best config once". A before/after claim needs the baseline on the same unseen queries; otherwise "after" has nothing to be compared with except the dev v0 number (different queries). *Recommendation: run both, once each, both pre-registered (40 generator calls, about 120 judge calls).* Alternative: run only the chosen config and compare its held-out rate to the dev v0 rate (unpaired, weaker).
2. **Policy if a held-out run breaks** (Ollama timeout, abort, cache or digest problem). *Recommendation: strict.* Any held-out row counts as used and `max_runs_per_config` stays 1; if a run is void for an infrastructure reason, you decide, and any extra run is recorded as an amendment in `heldout.yaml` with the reason and reported as "run 2 of 2". Alternative: allow one automatic repeat only when the run status is `aborted`, `invalid` or `interrupted` (nothing scored).
3. **Verdict wording rule (fixed now).** *Recommendation:* with the exact paired p at `significance_level` 0.05 — `supported_better` if chosen ungrounded count < v0 count and p < 0.05; `supported_worse` if chosen > v0 and p < 0.05; otherwise `not_distinguishable`, printed with the point difference and its CI context and the sentence "20 queries cannot detect a change of this size." Also a plain-fact line comparing held-out rate with the dev CI. With n = 20 expect `not_distinguishable`; that is an honest outcome, not a failed step. Alternative: add a non-inferiority rule ("not more than 1 answer worse than v0").
4. **Human check of held-out ungrounded answers.** The judge is uncalibrated, so the held-out number rests on an unverified judge. *Recommendation:* after the run, you read the listed ungrounded answers (likely 0-3) and the false refusals in `experiments/heldout_confirmation.md` and note which you agree with, as a human-review note, outside the calibration statistics, no code. Alternative: skip. Hand-labelling a proper sample for calibration stays with Step 13, where flagging thresholds are tuned against your labels.

## Definition of done
- [ ] "Done when" from CLAUDE.md: the final "after" number is confirmed on unseen queries and reported with its CI: `python -m evalharness.heldout confirm --baseline-run <id> --run <id>` prints the chosen config's held-out ungrounded rate k/n with 95% Wilson CI, read from `data/results.sqlite`.
- [ ] `python -m evalharness.heldout check --config configs/v1_prompt_citations.yaml` printed "allowed" before the run and prints "refused: 1 of 1 held-out runs used" after it; a second `runner --split heldout` for the same config exits 1 and writes nothing.
- [ ] SQL: `SELECT config_path, status, n_queries, n_ok, n_failed, git_dirty FROM runs WHERE split='heldout';` returns exactly the registered runs (two if Decision 1 is accepted), each `completed`, `n_queries = 20`, `git_dirty = 0`, `git_commit` is the pre-registration commit or a later commit on this branch (`git merge-base --is-ancestor <pre-registration commit> <git_commit>`), and `git_dirty = 0`.
- [ ] SQL: `SELECT count(*) FROM scores WHERE run_id = ? AND version = ?` gives 160 rows for checks (20 queries x 8 metrics), 20 each of `faithfulness`, `n_claims`, `ungrounded`, `relevance`, `irrelevant`, with 0 NULL values unless a judge failure is reported.
- [ ] `experiments/heldout_confirmation.md` exists, contains "uncalibrated judge", the held-out n, both rates with CIs, the paired flips with p, the dev numbers beside them, the pre-registered verdict label, and the CV-hypothesis line; `experiments/log.md` has the two new rows.
- [ ] `git diff` since the pre-registration commit shows no change to `rag/`, `configs/v0.yaml`, `configs/v1_prompt_citations.yaml`, `configs/judges.yaml`, `configs/test_set.yaml`, or judge/metric code.
- [ ] `llm_calls` shows held-out generator calls only for `gemma4:31b-cloud` (20 per run) and judge calls only for `gpt-oss:120b-cloud`.
- [ ] `pytest` passes (`tests/test_12_heldout_confirmation.py` plus the full suite).
- [ ] The Progress section of CLAUDE.md is updated with the real numbers and what they do and do not show.
- [ ] All changes are committed on `feature/12-held-out-confirmation`.
