# Spec: Step 14 — Automation

## Overview
A full evaluation today is five separate commands: `runner`, `checks`, `judges faithfulness`, `judges relevance`, `regression check`, each with its own run id to copy by hand. This step adds one command, `python -m evalharness run --config configs/<name>.yaml --split dev`, that runs those stages in order, passes the run id along, stops at the first stage that cannot continue, and ends with the regression gate's exit code (0 pass, 1 fail, 2 could not decide). It also adds a preflight that checks everything that can be checked without an LLM call before any expensive call starts, and a `score` command that resumes the judging and gate stages for an existing run. An optional GitHub Actions workflow is part of the build sequence; because the LLM models run on Gaurav's machine through Ollama and the results database is not committed, the workflow can only run the offline test suite (Decision 1). Nothing in the pipeline, judges, metrics or thresholds changes; this step only wires existing, tested stages together.

## Concepts to learn

**Orchestration vs logic.** Each stage already has a tested `main(argv)` with a clear exit code. The new `evalharness/cli.py` contains no evaluation logic: it calls those entry points in order and decides what to do with their exit codes. Keeping logic out of the orchestrator means a bug in it cannot change a metric.

**Exit codes as the contract with automation.** A shell script or CI job only sees an integer. The stages already use: 0 ok, 1 refused (runner, checks, judges) or gate FAIL (regression), 2 partial / aborted / cannot decide. Because `1` means two different things, the orchestrator maps them to its own table (below) and prints which stage produced it, so a person never has to guess.

**Fail fast, in the cheap place.** The slowest and most expensive stage is 80 generator calls. Everything that can fail without calling a model (dirty tree, missing config, split not allowed, judge digest drift, no accepted run, generator model equal to judge model) should fail in a preflight before that stage. Worked example: with 80 queries at about 1 s each plus about 3 judge calls per answer, discovering a stale judge digest at stage 3 wastes the whole generation stage.

**Idempotence and resumability.** All LLM calls are cached by their inputs, so running the same command again costs only cache lookups and produces the same answers. A resumed `score` only judges items not yet judged (existing behaviour of `--retry-failed` and "skipped (already judged)"), so it can be run repeatedly after a transient failure.

**What CI can and cannot do here.** Continuous integration runs on a fresh machine. It has the repo but not `data/results.sqlite` (gitignored), not the Ollama server, not the cloud model sign-in. So CI can run offline tests; it cannot run a real evaluation or the regression gate against real runs. Pretending otherwise (for example by committing a results database) would put a stale number behind a green tick.

## Depends on
- Step 5: `evalharness/runner.py` (`main`, `run_split`, `load_harness_config`, `DEFAULT_HARNESS_CONFIG`, exit codes 0 / 1 / 2), `evalharness/store.py`.
- Step 6: `evalharness/checks.py` `main` (score a run; exit 0 / 1).
- Steps 7-8: `evalharness/judges.py` `main` with subcommands `faithfulness` / `relevance` (exit 0 complete, 1 refused, 2 partial), `load_judges_config`, `load_relevance_config`, `get_model_digest` from `llm/client.py`.
- Step 13: `evalharness/regression.py` `main check` (exit 0 PASS, 1 FAIL, 2 refused), `get_accepted_run`, `load_regression_config`.
- Step 12: held-out guard; the pipeline never runs the held-out split.
- Verified read-only: `evalharness/__main__.py` and `evalharness/cli.py` do not exist; there is no `.github/` directory; `runs.judge_version` is NULL on every stored run (judges write the version on each score row instead).

## Interfaces

**The `answer()` contract is not touched.** `rag/` is not touched at all.

### `evalharness/cli.py` (new)

```python
STAGES = ("run", "checks", "faithfulness", "relevance", "gate")

def preflight_run(config_path: str, split: str, harness: dict, auto: dict, *, gate: bool,
                  allow_dirty: bool) -> list[str]:
    """Problems found WITHOUT any LLM call, empty list if none. Checks: split in
    automation.allowed_splits (never heldout); config file loads through the system's
    load_config; git tree clean unless allow_dirty; judge configs load; judge model != the
    config's generator model; judge digest equals the pinned one (get_model_digest);
    an accepted run exists for the split when gate is True."""

def run_pipeline(args) -> int:
    """run -> checks -> faithfulness -> relevance -> gate. Stops at the first stage whose exit
    code is not success; prints one status line per stage; returns the exit code below."""

def score_pipeline(args) -> int:
    """checks -> faithfulness -> relevance -> gate for an EXISTING completed run
    (judge stages use --retry-failed). Same stop rule and exit code."""

def main(argv: list[str] | None = None) -> int: ...
```

CLI (`python -m evalharness ...`, via the new `evalharness/__main__.py`):

```
run   --config configs/v1_prompt_citations.yaml --split dev [--allow-dirty] [--no-gate] [--no-store]
      [--against RUN_ID] [--markdown PATH [--force]] [--harness-config PATH]
score --run-id ID [--no-gate] [--no-store] [--against RUN_ID] [--markdown PATH [--force]]
      [--harness-config PATH]
```

Exit codes of `run` / `score`:

| Code | Meaning |
|---|---|
| 0 | every stage succeeded and the gate PASSED (or `--no-gate`, said in the last line) |
| 1 | every stage succeeded and the gate FAILED |
| 2 | preflight problem, a stage refused or ended partial / aborted, or the gate could not decide; the last status line names the stage and the next command to run |

Example last lines:

```
stage run          ok      run 20261016T101500Z-v1_prompt_citations-1a2b3c4d: completed, 80 ok, 0 failed
stage checks       ok      scored with checks-1-66c5a50a
stage faithfulness ok      80 judged, 0 judge failed
stage relevance    ok      80 judged, 0 judge failed
stage gate         FAIL    ungrounded +6.25 points (gate +3.00); flag rate 20/80; exit 1
```

### `evalharness/runner.py` (small refactor, behaviour unchanged)
Add `run(argv) -> tuple[int, str | None]` returning `(exit_code, run_id_or_None)`; `main(argv)` becomes `return run(argv)[0]`. The existing 186 runner tests and the Step 12 guard tests must pass unchanged.

### `evalharness/store.py`
`set_judge_version(conn, run_id, value)`: sets `runs.judge_version` only while it is NULL (same value again is a no-op; a different value raises `ValueError`). Used after both judges completed (Decision 3).

## Data and schema changes
No schema changes. `runs.judge_version` (existing, always NULL so far) is filled by the pipeline with `faithfulness=<version>;relevance=<version>` once both judges completed for the run, so the CLAUDE.md convention "every run records the judge prompt version" becomes true for runs made by `run`. Older runs keep NULL; their versions remain on each `scores` row. No new data files.

## Config changes
New `automation:` block in `configs/harness.yaml` (starting defaults, approved by Gaurav as part of this spec):
```yaml
automation:
  allowed_splits: [dev]             # `run` refuses any other split; held-out has its own Step 12 commands
  retry_failed_judges_in_score: true   # `score` passes --retry-failed to both judges
```
No other config changes. `configs/judges.yaml`, `configs/v0.yaml` and every experiment config stay byte-identical.

## Files to create
- `evalharness/cli.py`
- `evalharness/__main__.py` (3 lines: `from evalharness.cli import main`; `sys.exit(main())`)
- `tests/test_14_automation.py`
- `.github/workflows/tests.yml` (only if Decision 1 is accepted as recommended)

## Files to change
- `evalharness/runner.py`: `run()` returning `(exit_code, run_id)`; `main` delegates.
- `evalharness/store.py`: `set_judge_version`.
- `configs/harness.yaml`: `automation:` block.
- `tests/README.md`: Step 14 notes.
- `CLAUDE.md`: Progress entry; fix the repo-layout note if `cli.py` wording needs it.

## New dependencies
No new dependencies. (The workflow installs the existing `requirements.txt`; no new package enters the project.)

## Tests
`tests/test_14_automation.py`, offline: the four stage entry points are replaced by fakes recording their argv and returning scripted exit codes; tmp DB and tmp harness YAML (target 25-40 tests).
- **Config:** `automation` block validation (missing key, bad types, `heldout` in allowed splits is refused even if listed).
- **Preflight:** each problem is reported without any stage being called: split not allowed (heldout), dirty tree, unknown config, judge model equals generator model, digest mismatch (fake `get_model_digest`), no accepted run with gate on, no accepted run with `--no-gate` is fine; several problems are all listed.
- **Order and run-id passing:** stages called in the order run, checks, faithfulness, relevance, gate; the run id returned by `runner.run` is the one given to every later stage; `--harness-config` is passed to every stage.
- **Stop rule (table-driven):** a non-success code from each stage stops the later stages, prints the stage name, and returns 2 (gate FAIL returns 1, gate refusal returns 2); runner status aborted / invalid returns 2.
- **Exit code table:** all stages ok + gate 0 -> 0; gate 1 -> 1; `--no-gate` skips the gate stage and returns 0 with the "gate skipped" line.
- **Flags:** `--allow-dirty`, `--no-store`, `--against`, `--markdown`, `--force` reach the right stage and only that stage; `--allow-heldout` does not exist on `run` / `score`.
- **score:** requires an existing completed run; passes `--retry-failed` when configured; refuses an unknown run id and a held-out run id with exit 2.
- **judge_version:** filled once after both judges succeeded; not filled when the second judge is partial; a second fill with another value raises; `store.set_judge_version` on a stored run with a value is a no-op for the same value.
- **runner refactor:** `run()` returns `(0, run_id)` for a completed run and `(1, None)` for a refusal; `main` exit codes unchanged.
- **Module entry:** `python -m evalharness --help` exits 0 (subprocess, no stage runs).
- **No LLM:** `generate` and `get_model_digest` patched; real `llm_calls` untouched.
- **Layering:** `cli.py` imports nothing from `rag/` or `llm/` directly (the system under test is reached through the harness config, as the runner does).

## Rules for implementation
- Plain Python and only libraries approved in CLAUDE.md
- No LangChain; no RAGAS or DeepEval (except the optional final cross-check)
- `rag/` must never import from `evalharness/`; the harness only calls `answer()`
- All tunable values come from YAML config — no hardcoded magic numbers
- SQLite via the sqlite3 module with parameterised queries only
- All LLM calls go through `llm/client.py` so they are cached (the spec template says `evalharness/llm.py`; the real shared client is `llm/client.py`)
- API keys only from `.env`, never in code, logs or commits
- The held-out split is never used before Step 12 (done); the pipeline never runs it and has no `--allow-heldout` flag
- Never invent, estimate or adjust metric numbers; report only what a stored run produced
- Human labels (calibration and review) come only from Gaurav
- Code is simple, readable and commented

Step-specific rules:
- The orchestrator contains no evaluation logic and computes no metric: every number it prints is returned by the stage it called or read back from the database.
- It never accepts a run, never edits a threshold, never changes a judge version and never retries a generator call. Accepting a run stays a separate human command.
- Stage failures are reported, never swallowed; a partial stage ends the pipeline with exit 2 and the exact command to resume (`python -m evalharness score --run-id <id>`).
- `--allow-dirty` is passed through to the runner only when given; it is recorded by the runner as `git_dirty = 1`, and the gate stage then reports it in its output.
- The GitHub workflow must not need or contain any secret, must not call a model and must not claim to run an evaluation.
- Every status line prints the real run id and counts from the stages; no hand-written numbers.

## Decisions for Gaurav

1. **What the GitHub Actions workflow does.** *Recommendation:* `.github/workflows/tests.yml` runs `pytest` on every push and pull request (offline, uses fakes; needs no secret). It cannot run the evaluation or the regression gate because CI has no Ollama server and no results database. Alternatives: (a) a workflow on a self-hosted runner on your own machine that runs `python -m evalharness run` (needs the runner installed and kept online; the first real gate in CI); (b) no workflow in this step.
2. **When no accepted run exists, what does `run` do?** *Recommendation:* the preflight refuses (exit 2, nothing started) unless `--no-gate` is given, because a run whose gate cannot be decided after 80 paid calls is wasted effort; your accepted run already exists (id 1). Alternative: always run and score, then exit 2 at the gate stage.
3. **Fill `runs.judge_version`?** *Recommendation:* yes, with `faithfulness=<version>;relevance=<version>` after both judges complete, so the CLAUDE.md convention holds for pipeline runs. Alternative: leave it NULL, because the version already sits on every `scores` row.
4. **Include the `score` resume command?** *Recommendation:* yes: a transient judge failure (the only failure seen in real runs so far: one relevance judge reply with prose before the JSON) then costs one command, not five. Alternative: only `run`, and resume by hand with the existing `judges ... --retry-failed`.

## Definition of done
- [ ] "Done when" from CLAUDE.md: a full evaluation runs from one command: `python -m evalharness run --config configs/v1_prompt_citations.yaml --split dev` runs the generator, checks, both judges and the regression gate, prints one status line per stage and exits 0 / 1 / 2 as in the table. Real-run check is cache-only: the config and split were already run (all generator and judge calls are cache hits), so no new LLM call is needed; confirm with `llm_calls` (`cached = 1` for every new row, 0 errors).
- [ ] `python -m evalharness run --config configs/v0.yaml --split heldout` exits 2 from the preflight with nothing written (`SELECT COUNT(*) FROM runs` unchanged).
- [ ] `python -m evalharness score --run-id <the new run id>` after the run prints all stages `ok` and every judge as "skipped (already judged)".
- [ ] SQL: the new run has `status = 'completed'`, `n_queries = 80`, `git_dirty = 0`, and `judge_version` equal to `faithfulness=faith-v2-d130a39f;relevance=rel-v2-c825e4f7` (if Decision 3 is accepted).
- [ ] `python -m pytest tests/test_05_runner_and_store.py tests/test_12_held_out_confirmation.py -q` pass unchanged (runner refactor is behaviour-preserving).
- [ ] The workflow file (if Decision 1 is accepted) parses (`python -c "import yaml; yaml.safe_load(open('.github/workflows/tests.yml'))"`), contains no secret, and its test command is `python -m pytest`.
- [ ] `pytest` passes (`tests/test_14_automation.py` plus the full suite).
- [ ] The Progress section of CLAUDE.md is updated with the real command output and what it does not show (CI does not run an evaluation).
- [ ] All changes are committed on `feature/14-automation`.
