# Spec: Step 16 — README and Write-up

## Overview
Steps 0-15 built and measured everything; the repository's `README.md` is still the 15-line stub from Step 0. This step writes the README that explains the project to someone new (architecture, metric definitions, judge calibration result, experiment table, before/after with confidence intervals, limitations) and tells them exactly how to reproduce it, including what cannot be reproduced bit for bit (the results database and the LLM cache are gitignored, and the models are cloud models that are not fully deterministic). Because a README full of hand-typed numbers is the easiest place to break the project's honesty rule, the results part is generated from the stored reports and a `check` command fails when the README no longer matches them. No pipeline, judge, metric, threshold or config of the system changes in this step.

## Concepts to learn

**A README has three readers.** (1) A reviewer or interviewer wants the headline, the method and the honest limits in two minutes. (2) A builder wants to run it. (3) Gaurav wants to explain every choice. The README is therefore ordered: result and what it supports, then architecture and metrics, then how to run, then limits and design decisions. Each section answers one question.

**Reproduce vs repeat.** *Re-running the code* on a new machine is possible: the corpus, test set, configs, prompts, pinned versions and tests are committed. *Getting the same numbers* is not guaranteed: `data/results.sqlite` and `data/llm_cache.sqlite` are gitignored, the generator and judge are Ollama cloud models (the digest pinned in the configs is the local manifest, not the remote weights), and the same prompt at temperature 0 gave three different replies across three cache rebuilds in Step 2. A new run produces new numbers that should land inside the stored confidence intervals, and a claim of "same numbers" would be false. The README says so.

**Numbers have one source.** Every rate in the README must be the same string that a stored report produced. Worked example: the baseline report stored in `run_reports` has `ungrounded.all` = 7 successes out of 80, and `_fmt_rate_pct` renders it as `8.75% [4.30%, 16.98%] (7/80)`. The README's results block is produced by calling that same formatter on the stored report, never typed. `check` regenerates the block and compares it with the file.

**What a result supports.** Wilson intervals show that 2/80 [0.69%, 8.66%] and 7/80 [4.30%, 16.98%] overlap; the exact paired test on the 5 flipped answers gives p = 0.0625; the noise control (same config, seed 42 -> 43) moved 7 -> 5 alone. So "the citation prompt reduced ungrounded answers from 8.75% to 2.50%" is a dev point estimate from an uncalibrated judge, not an established effect. The write-up keeps point estimate, interval, test and judge status together every time.

## Depends on
- Step 0: `README.md` (stub), `requirements.txt`, `.gitignore`, `.github/workflows/tests.yml` (Step 14: offline `pytest` only).
- Step 1-4: corpus `data/docs/` (22 documents), `data/test_set.jsonl` (100 queries, 80 dev / 20 held-out), `configs/test_set.yaml`.
- Step 2-3: `llm/client.py`, `configs/llm.yaml`, `rag/`, `configs/v0.yaml`; the `answer()` contract.
- Step 5-8: `evalharness/runner.py`, `store.py`, `checks.py`, `metrics.py`, `judges.py`, `configs/judges.yaml`.
- Step 9: `experiments/calibration.md`, tables `calibration_labels`, `calibration_reports` (calibration result: inconclusive).
- Step 10-12: `experiments/baseline_v0.md`, `experiment_table.md`, `heldout_confirmation.md`, `log.md`; `run_reports` rows of kind `baseline` (id 1), `experiment` (13 rows: the v0 twin check, run `20261015T215007Z-v0-a9da5e5a`, plus the 12 experiments including the seed-43 noise control), `heldout` (id 15); `configs/heldout.yaml`.
- Step 13-14: `evalharness/regression.py`, `evalharness/cli.py`, `python -m evalharness run | score`.
- Step 15: `dashboard/`, `streamlit run dashboard/app.py`.
- Reused for the generated block: `evalharness/report.py` (`_fmt_rate_pct`, `_fmt_calibration`, `current_versions`), `store.get_run_reports`, `store.get_calibration_reports`, `store.get_accepted_runs`.
- Verified read-only while writing this spec: README.md is 15 lines; `.env.example` does not exist (`.env` is gitignored, so a new user has no template); `data/test_set.jsonl`, `data/docs/*.md` and all configs are committed; `data/*.sqlite` and `indexes/` are gitignored; Ollama here lists `gemma4:31b-cloud`, `gpt-oss:120b-cloud` and `nomic-embed-text`; the `-cloud` models need an Ollama sign-in; the runner refuses when the server's model digest differs from the pinned one (`generation.model_digest`, `judges.yaml` digest); `python -m evalharness run` refuses (exit 2) when no accepted run exists unless `--no-gate`.

## Interfaces

**The `answer()` contract is not touched.** `rag/`, `llm/`, the judges, the checks, the flag rules, the dashboard and every pipeline config are not touched.

### `evalharness/writeup.py` (new; imports nothing from `rag/`, `llm/` or `streamlit`)
Builds the README's results block from stored reports. Read-only on the database; no LLM call.

```python
START_MARKER = "<!-- results:start (generated by python -m evalharness.writeup; do not edit by hand) -->"
END_MARKER = "<!-- results:end -->"

class WriteupRefused(Exception):
    """A required stored report is missing, or the README has no / duplicate markers."""

def load_writeup_config(harness: dict) -> dict:
    """The checked `writeup` block: {"readme_path"}; ValueError names the problem."""

def build_results(conn, harness: dict) -> dict:
    """Everything the block needs, read from run_reports / calibration_reports / accepted_runs:
    baseline (kind baseline, the v0 dev run), experiments (the 12 kind experiment rows whose config is not the baseline config, in run order; the v0 twin check is shown as a one-line reproducibility note, not as an experiment),
    heldout (the kind heldout report), regression (the stored Step 13 checks, one per run / reference / flag version), calibration (per judge, via report._calibration_status),
    accepted run id. Raises WriteupRefused when baseline, experiments or heldout is missing."""

def format_results_markdown(results: dict) -> str:
    """The block body: headline (before -> after, dev and held-out, with CIs and judge status),
    experiment table (variable, ungrounded k/n with CI, change in points, improved/regressed, exact p,
    false refusals), stored regression checks (gate, ungrounded change, flag rate, flag reasons), calibration status table. Every rate goes through report._fmt_rate_pct."""

def replace_block(readme_text: str, block_body: str) -> str:
    """readme_text with the text between the two markers replaced; everything else byte-identical.
    WriteupRefused when a marker is missing or appears more than once."""

def check_readme(readme_text: str, block_body: str) -> list[str]:
    """Problems found: [] when the README block equals block_body, else a one-line description."""
```

CLI (`python -m evalharness.writeup ...`, `--harness-config PATH` as in the other tools):
```
render                      # print the block to the terminal; writes nothing
update  [--readme PATH]     # replace the marked block in the README (default writeup.readme_path)
check   [--readme PATH]     # exit 0 when the README block matches the database, 1 when it differs, 2 when refused
```
Example: `python -m evalharness.writeup check` prints `README results block matches data/results.sqlite (baseline run 20261015T152638Z-v0-c7c96975, 12 experiments, held-out run_reports id 15)`.

### `README.md` (rewritten by hand, except the generated block)
Fixed outline (each heading must exist; a test checks them):
1. **Title, one-paragraph summary, status.** What it is, for whom, "uncalibrated judges" stated up front.
2. **Results at a glance.** One short paragraph in words, then the generated block between the markers: v0 dev baseline, best single change and its held-out confirmation, each as rate + 95% Wilson interval + k/n + exact paired p; the experiment table; calibration status.
3. **What these results support, and what they do not.** Plain list. The CV draft (25% -> 12%, 10-15% flagged) is stated as a hypothesis that the stored runs do not support; the README never presents it as achieved.
4. **Architecture.** The diagram from CLAUDE.md, the `answer()` contract, the layering rule (`rag/` never imports `evalharness/`), the shared `llm/client.py`.
5. **Repository layout.** Tree with one line per directory.
6. **Setup and reproduce.** Prerequisites (Python 3.13, Ollama, `ollama signin`, `ollama pull nomic-embed-text`, the two cloud models); `.env` from `.env.example`; `pip install -r requirements.txt`; offline `pytest`; try one question (`python -m rag.pipeline ...`); the first-time sequence (`run --no-gate`, `regression accept`, later `run` with the gate); `score` to resume; `streamlit run dashboard/app.py`; what is gitignored and why; digest pinning (how to read and update `model_digest`); held-out guard (pre-registered, once); what will differ on your machine.
7. **Metric definitions.** Faithfulness, ungrounded, relevance (0/1/2), retrieval hit@k, MRR, in_context, failure cause, refusal check, Wilson interval, exact paired test, flag reasons, gate; each with formula and a tiny worked example.
8. **Method.** Corpus (synthetic fictional company, confusable pairs, deliberate gaps), test set (100 queries by type and split), generator and judge models, temperatures, seeds, caching.
9. **Judge calibration.** The Step 9 procedure and the result copied from the generated block: n = 20, TP 0 FP 0 FN 0 TN 20, agreement 20/20, precision / recall / kappa undefined; why this is inconclusive.
10. **Experiments.** How to read the table, the noise control, the selection rule fixed before the results, the 12 dev looks (no multiple-comparison correction), the held-out verdict `not_distinguishable`.
11. **Regression check, flagging and gate.** Flag reasons, the 3-point gate and what it means at n = 80.
12. **Dashboard.** Pages and what each shows; honesty rules it follows; review labels are triage labels.
13. **Limitations.** Explicit list (below).
14. **Design decisions (interview notes).** One short answer each: own judges instead of RAGAS / DeepEval, no LangChain, cache everything, judge != generator, held-out used once, one variable per experiment, child-process reranker (libomp clash).
15. **Troubleshooting.** `402 Payment Required` on a model outside the Free plan, digest mismatch, `OMP: Error #15`, Ollama not running, `ContextLimitError`.

Required limitations (copied in substance from the stored reports, not softened): both judges uncalibrated (Step 9 inconclusive: 20 labels, all grounded, zero positives); synthetic corpus of one fictional company; 80 dev / 20 held-out queries, per-type n 12-16 (dev) and 3-10 (held-out); the best config was chosen from 12 dev looks (winner's curse) and one look at n = 20 cannot confirm it; ungrounded rate is the judge's measurement, not a verified rate; refusal detection is a phrase match; citation correctness is not scored; cloud models are not fully deterministic and the digest is a local manifest; results are one sample per answer; flag precision/recall against human labels cannot be measured yet; CI does not run an evaluation.

### `.env.example` (new)
```
LLM_API_KEY=ollama-dummy-key   # Ollama ignores it; the openai SDK needs a non-empty value
```

## Data and schema changes
No data or schema changes. The new command reads `run_reports`, `calibration_reports` and `accepted_runs` and writes only the README file.

## Config changes
New `writeup:` block in `configs/harness.yaml`:
```yaml
writeup:
  readme_path: README.md       # file `update` / `check` read, relative to the repo root
```
No other config changes; `configs/judges.yaml` and every pipeline config stay byte-identical.

## Files to create
- `evalharness/writeup.py`
- `tests/test_16_readme_and_writeup.py`
- `.env.example`

## Files to change
- `README.md`: full rewrite per the outline (generated block via `writeup update`).
- `configs/harness.yaml`: `writeup:` block.
- `tests/conftest.py`, `tests/README.md`: fixture and Step 16 notes.
- `CLAUDE.md`: Progress entry (after the done-when check). Also correct the "Repo layout" tree if it differs from the repository (for example the `dashboard/` module names), without touching any rule.

## New dependencies
No new dependencies. (RAGAS is not added; see Decision 3.)

## Tests
`tests/test_16_readme_and_writeup.py`, offline, tmp DBs, `generate` patched to raise (target 25-35 tests). The generated-block tests build stored reports on the `experiment_db` fixture (real `report.build_baseline_report` / `experiments.compare_runs` output written with `store.write_run_report`), so expected strings come from hand values already pinned in earlier tests.
- **Config:** valid `writeup` block loads; missing / unknown key and non-string path raise `ValueError`.
- **Build:** `build_results` returns the baseline, the 12 experiment rows in run order (the baseline-config twin row excluded), the held-out report and the calibration status; missing baseline, no experiments and missing held-out each raise `WriteupRefused`.
- **Format:** the baseline line equals `_fmt_rate_pct` of the stored summary (hand value 4/9 on the fixture); every rate in the block has k/n and an interval; every judge-based line carries "uncalibrated judge" while no calibration is `measured`; an n = 0 rate prints "n = 0", never 0%; the block contains no number that is absent from the stored reports (extract all `k/n` pairs and check each against the reports).
- **Replace / check:** `replace_block` leaves text outside the markers byte-identical and is idempotent; zero or duplicate markers raise; `check_readme` returns [] for a fresh block and a message after any digit in the block is edited; `check` exits 0 / 1 / 2; `update` then `check` exits 0; `render` and `check` write nothing to the database (row counts and file bytes unchanged).
- **Real README (static):** both markers exist exactly once; every outline heading exists; every `python -m <module>` named in the README is a real module file; every `configs/*.yaml`, `experiments/*.md`, `.env.example` path it names exists; every relative link resolves; the words "uncalibrated" and "not_distinguishable" appear; no `sk-`, `api_key=`, `Bearer` strings; the CV figures "25%" and "12%" appear only inside a sentence that also contains "hypothesis".
- **`.env.example`:** has `LLM_API_KEY`, nothing else, no real-looking secret; `.env` stays gitignored and `.env.example` is not.
- **Commands parse:** for each documented `python -m evalharness...` / `evalharness.<tool>` command, `--help` exits 0 (subprocess; no model, no database).
- **Layering:** `writeup.py` imports nothing from `rag/`, `llm/`, `streamlit`; `rag/` still imports nothing from `evalharness/`.

## Rules for implementation
- Plain Python and only libraries approved in CLAUDE.md
- No LangChain; no RAGAS or DeepEval (except the optional final cross-check)
- `rag/` must never import from `evalharness/`; the harness only calls `answer()`
- All tunable values come from YAML config — no hardcoded magic numbers
- SQLite via the sqlite3 module with parameterised queries only
- All LLM calls go through `llm/client.py` so they are cached (the spec template says `evalharness/llm.py`; the real shared client is `llm/client.py`). This step makes no LLM call.
- API keys only from `.env`, never in code, logs or commits
- The held-out split is never used before Step 12 (done). This step reads the stored Step 12 report; it starts no run and uses no held-out answer for any decision.
- Never invent, estimate or adjust metric numbers; report only what a stored run produced
- Human labels (calibration and review) come only from Gaurav
- Code is simple, readable and commented

Step-specific rules:
- Every number in the README comes from a stored report, through the generated block, or is copied unchanged from a committed `experiments/*.md` file (which are themselves generated). A number is never rounded differently, re-derived by hand or "cleaned up". Prose may describe a number but never restate it with different precision.
- Every judge-based number appears with its judge status ("uncalibrated judge") and its 95% Wilson interval and k/n; the README never uses "accuracy", "true rate" or "verified" for them.
- The README never claims the CV hypothesis (25% -> 12%, 10-15% flagged) as a result. It states the measured values and says the hypothesis is not supported by any stored run.
- The README never says an improvement is "significant" unless the stored exact p < 0.05 and the judge status is given beside it. At the time of writing only `rewrite` (p = 0.0312) qualifies, and it is 1 of 12 looks with 4 false refusals against 1 at baseline.
- Reproduction instructions list only commands that were actually run, or marks them "not run by the author (needs a paid or rate-limited model call)". Commands that need Ollama are labelled as such.
- The README states what is gitignored and therefore absent from a clone (`data/*.sqlite`, `indexes/`, `.env`), and that stored numbers cannot be re-created exactly.
- The generated block is the only machine-written part; `writeup update` never touches text outside the markers.
- No screenshots, badges or licence text are invented (Decision 7).

## Decisions for Gaurav

1. **How the README gets its numbers.** *Recommendation:* the generated block plus `writeup check` described above (about 150 lines of code, tested). It makes "every number comes from a stored run" checkable. Alternative: hand-copy numbers from `experiments/*.md` and add only a test that every `k/n` and percentage in the README also appears in a committed `experiments/*.md` file; less code, weaker guarantee (it cannot catch a wrong pairing).
2. **Calibration status in the write-up.** *Recommendation:* publish the README now with the honest Step 9 result (inconclusive; both judges "uncalibrated"). Alternative: before writing, do the real calibration labelling pass yourself (careful labels on about 50 answers that include the 7 judge-ungrounded ones and a random check sample), then regenerate. That would change the headline status, so it is a separate step you may want first; the README block regenerates from the database, so a later calibration needs only `writeup update`.
3. **RAGAS cross-check.** CLAUDE.md allows one at the very end. *Recommendation:* skip it. It needs the `ragas` package (not on the approved list), extra LLM spend through another judge path, and its disagreement could not be interpreted while our judges are uncalibrated. Alternative: approve the dependency and run it once on the v0 dev run, reported as a cross-check only.
4. **One document or two.** *Recommendation:* README only, with the design-decision notes inside it. Alternative: README for setup and results plus a separate `WRITEUP.md` for the narrative and interview notes.
5. **How far to verify reproduction.** *Recommendation:* in a fresh clone in a scratch folder, follow the README's offline steps (install check, `pytest`, `python -m evalharness.testset summary`, `python -m rag.loader`, `writeup check` expected to refuse because the clone has no database) plus one real uncached question through `rag.pipeline` (about 2 cloud-model calls). Alternative: also run the full `python -m evalharness run --config configs/v0.yaml --split dev --no-gate` in the clone (about 80 generator and 230 judge calls on the Free plan) and report how far the new rate lands from 8.75%; this measures reproducibility honestly but costs time and may hit rate limits.
6. **CV section.** *Recommendation:* include "What these results support, and what they do not", naming the CV draft as an unsupported hypothesis, because the repository is part of what a reader will compare the CV to. Alternative: leave the CV out of the public README and keep that comparison in your own notes.
7. **Extras.** *Recommendation:* no screenshots, badges or licence file. Screenshots can only be taken by you from your running dashboard, and the licence is your choice; say if you want either, and which licence.
8. **`.env.example`.** *Recommendation:* add it (dummy key only), because `.env` is gitignored and a new user would otherwise not know the variable name. Alternative: describe the variable in the README only.

## Definition of done
- [ ] "Done when" from CLAUDE.md: someone new could understand and reproduce the project from the README. Checked by following the README in a fresh `git clone` in a scratch folder as in Decision 5: every documented offline command runs and its output matches what the README says; each command that needs a model is labelled as such; the README states what will differ. Record the exact commands and outputs.
- [ ] `python -m evalharness.writeup check` exits 0 on the real database, and `python -m evalharness.writeup update` run twice leaves `git diff --stat` empty the second time.
- [ ] SQL / file check: `outputs`, `scores`, `runs`, `judge_claims`, `reviews`, `run_reports` row counts and `data/results.sqlite` unchanged before and after (`writeup` is read-only on the database).
- [ ] `grep -nP "(?<![\d.])(25|12)%|10-15%" README.md` shows each occurrence inside a sentence that also says "hypothes" (a plain `grep "25%"` also matches table cells such as 6.25%); `grep -nE "sk-|api_key *=|Bearer " README.md .env.example` finds nothing.
- [ ] Every headline number in the README equals the stored report string: baseline `8.75% [4.30%, 16.98%] (7/80)`, chosen config dev `2.50% [0.69%, 8.66%] (2/80)`, held-out v0 `10.00% [2.79%, 30.10%] (2/20)` and chosen `5.00% [0.89%, 23.61%] (1/20)`, held-out verdict `not_distinguishable` (verify by `writeup render` and by SQL on `run_reports`).
- [ ] `python -m pytest tests/test_16_readme_and_writeup.py -q` and the full `pytest` pass.
- [ ] `git status --porcelain` shows no database, index, cache or `.env` file; `.env.example` is tracked.
- [ ] The Progress section of CLAUDE.md is updated with the real command outputs and what the README does not show (no calibrated judge, no re-run of the numbers on another machine unless Decision 5 chose it).
- [ ] All changes are committed on `feature/16-readme-and-write-up`.
