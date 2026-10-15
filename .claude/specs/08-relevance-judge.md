# Spec: Step 08 — Relevance Judge

## Overview
This step adds the second LLM judge. It asks one question per answer: **does
the answer actually address the question?** The Step 7 judge cannot answer
this. A grounded answer can still miss the point: it may quote the right
policy but answer a different question, or refuse a question the policies do
cover. For every output of a completed run, `evalharness/judges.py` makes one
cached call to the judge model with the question, the answer and whether the
test set says the question is answerable. The model returns a short reasoning
and a rubric score of 0, 1 or 2. The score is stored as `relevance` (0, 0.5,
1) and as an `irrelevant` flag (score 0) in `scores` under a relevance judge
version. CLAUDE.md's rules for unanswerable queries are built into the rubric.
A correct refusal on an unanswerable query is relevant. An invented answer to
an unanswerable query is irrelevant, and the Step 7 judge already marks it
ungrounded. The step comes now because Step 13's review flags ("relevance is
low", "the judges disagree") and Step 10's report need both judges. Like the
faithfulness numbers, these are the judge's opinion and are labelled
"uncalibrated".

## Concepts to learn
- **Three different questions about one answer.**
  - Faithful (Step 7): is every claim backed by the retrieved context?
  - Relevant (this step): does it address what was asked?
  - Correct: does it match the truth? Correctness is not measured as a judge
    here; the reference answers stay a guide (Step 4 carry-forward).

  The questions are independent. Four examples for "How many casual leaves do
  interns get?":

  | answer | faithful | relevant |
  |---|---|---|
  | "1 per completed month" (context says so) | yes | yes |
  | "Employees get 8 casual leaves a year" (context says so) | yes | **no**: wrong group |
  | "1 per month, plus 5 bonus days" (bonus not in context) | **no** | yes |
  | "The context does not mention this" (it does) | yes (0 claims) | **no** |

  The last row is the q053 case from Step 6: a false refusal. Faithfulness
  scores it as fine. Only relevance catches it.
- **Answerability decides what "relevant" means.** For an answerable question
  a refusal does not address it, so it is irrelevant. For an unanswerable
  question (the policies do not cover it) the correct, relevant reply is a
  refusal. An answer that invents a policy is irrelevant, because it answers a
  question the company never answered. The judge cannot know which case
  applies from the question alone. So it is told, from the test set: a query
  with `source_doc_ids == []` is unanswerable. It is not shown the reference
  answer, so it grades relevance and not correctness (Decision 2).
- **Rubric scoring.** A rubric spells out what each score means, so the judge
  (and a human checking it) applies the same standard to every answer:
  - **Answerable question:**
    - 2 = addresses everything asked.
    - 1 = addresses only part of it, answers a nearby question, or is too
      vague to act on.
    - 0 = does not answer it (refuses, says the information is missing, or
      talks about something else).
  - **Unanswerable question:**
    - 2 = clearly says the policies do not cover it. Related facts may be
      added if no answer is claimed.
    - 1 = says it is not covered but still guesses an answer.
    - 0 = answers as if a policy existed (an invented answer).
- **Why few levels.** LLM judges, like people, agree with themselves and with
  humans more on 3 levels than on 1-10. A score of 7 vs 8 carries little
  information, while 0 / 1 / 2 maps to "no / partly / yes" (Decision 1).
- **Normalising.** `relevance = score / 2`, so 0, 0.5 or 1. A later rubric with
  another maximum stays comparable, and the mean reads as a share. Worked
  example: four answers scored 2, 2, 1, 0 → relevance 1, 1, 0.5, 0 → mean
  2.5 / 4 = 0.625. `irrelevant` = 1 when the score is at or below
  `irrelevant_max_score` (0), so 1 of 4 → irrelevant rate 0.25.
- **Known judge biases.** LLM judges tend to prefer longer answers (verbosity
  bias) and answers written in their own style (self-preference). Two
  defences are used here. The judge model differs from the generator
  (`gpt-oss:120b-cloud` vs `gemma4:31b-cloud`). The rubric says length and
  style do not matter, only whether the question is addressed.
- **Same machinery as Step 7.** The call uses temperature 0, a fixed seed, an
  explicit reasoning effort and a pinned digest. The reply has a strict schema
  with reasoning before the score, untrusted text sits inside tags, and the
  version is `rel-v<prompt_version>-<hash>`. Failures are stored as NULL
  "judge failed" and counted, and judged items are resumed, not repeated.
- **Not calibrated by Step 9.** Step 9 calibrates the grounded / ungrounded
  judgement against Gaurav's labels. The relevance judge stays uncalibrated
  unless Step 9 also collects relevance labels (carry-forward). Its numbers
  are labelled so.

## Depends on
- Step 2: `llm/client.py`: `generate(..., schema, model_digest, seed,
  reasoning_effort)`, `get_model_digest`, `LLMOutputError`, `load_config`.
- Step 4: `queries.source_doc_ids` ([] means unanswerable),
  `queries.type`; Decision 5 (unanswerable reference text) is NOT shown to
  the judge.
- Step 5: `evalharness/runner.py`: `load_harness_config`, `llm_calls_max_id`,
  `llm_calls_summary`, `_format_calls`. Run `20261015T152638Z-v0-c7c96975`.
- Step 6: `evalharness/checks.py`: `get_scorable_run`, `ScoringRefused`,
  `ANSWERABLE_TYPES`, `load_checks_config` (`report_confidence`);
  `evalharness/metrics.py`: `rate_summary`, `mean_summary`;
  `evalharness/store.py`: versioned `scores`, `get_scores`,
  `get_outputs_with_queries` (question, answer, type, `source_doc_ids`, error).
  Step 6 phrase-match result for comparison: false refusal q053; correct
  refusals 12/12 unanswerable.
- Step 7: `evalharness/judges.py`: `_call` (generate + token guard +
  `JudgeError`), `judge_run`'s per-item loop (resume, `--retry-failed`,
  `--query-ids`, consecutive-failure stop, "not judged" / "judge failed" NULL
  rows), `_template_fields`, the CLI with subcommands;
  `store.write_judgement` (works with no claim rows), `delete_judgement`;
  `configs/judges.yaml` (`faithfulness` block). Step 7 results for the same
  run: `faith-v2-d130a39f`, 7/80 ungrounded.
- Carry-forward items addressed here:
  - Step 4: a refusal on an unanswerable query is relevant. The reference
    sentence is not used as a match target; it is not shown at all.
  - Step 3: untrusted text is delimited and declared data.
  - Step 2: the reply schema is strict, with `Literal` scores and no defaults.
    Per-item failures are stored and counted.

## Interfaces
This step does **not** touch the `answer()` contract, does not call
`answer()`, and does not import `rag`.

`evalharness/schemas.py` (new)
```python
class RelevanceVerdict(BaseModel):
    """Relevance judge reply. extra="forbid", strict. Reasoning before score."""
    reasoning: str                 # non-blank
    score: Literal[0, 1, 2]        # no default
```

`evalharness/judges.py` (changed and new)
```python
RELEVANCE_METRICS = ("relevance", "irrelevant")

@dataclass
class RelevanceJudgement:
    score: int            # 0 / 1 / 2 from the rubric
    relevance: float      # score / 2
    irrelevant: int       # 1 if score <= irrelevant_max_score
    reasoning: str        # the judge's reasoning, stored in scores.reasoning

def load_relevance_config(path: str | Path) -> dict:
    """Check the `relevance` block of configs/judges.yaml (same common checks as
    faithfulness: model, digest, temperature 0, seed, effort, token guard,
    failure limit; plus irrelevant_max_score in {0, 1}, the two rubrics
    (answerable_rubric, unanswerable_rubric), and a prompt_template using
    exactly {question}, {answer}, {rubric}). Adds "version": f"rel-v{prompt_version}-{hash8}" over all
    other settings and the RelevanceVerdict schema."""

def rubric_for(source_doc_ids: list[str], cfg: dict) -> str:
    """cfg["answerable_rubric"] if there are source docs, else cfg["unanswerable_rubric"].
    Only this one rubric goes into the prompt (prompt v2)."""

def judge_relevance_output(output: dict, cfg: dict) -> RelevanceJudgement:
    """One call: prompt_template.format(question, answer, rubric) through
    _call(..., RelevanceVerdict, cfg). Raises JudgeError on any failure."""

def relevance_run(conn, run_id: str, cfg: dict, *, allow_heldout: bool = False,
                  retry_failed: bool = False, query_ids: list[str] | None = None,
                  error_max_chars: int = 500) -> dict:
    """Same behaviour as judge_run, for relevance: 2 rows per output
    (relevance, irrelevant), NULL + "not judged: ..." for failed outputs and
    blank answers, NULL + "judge failed: ..." for judge failures, resume,
    retry_failed, query_ids, stop after max_consecutive_failures in a row.
    Returns {"status", "n_outputs", "judged", "judge_failed", "not_judged",
    "skipped_existing"}."""

def build_relevance_report(conn, run_id: str, version: str, confidence: float) -> dict
def format_relevance_report(report: dict) -> str
```
Shared loop refactor: the per-item loop in `judge_run` moves into one private
helper, `_judge_items(conn, run_id, version, outputs, judge_one, null_metrics,
*, retry_failed, query_ids, error_max_chars, max_consecutive_failures)`.
`judge_one(output)` returns `(score_rows, claim_rows)` or raises `JudgeError`.
`judge_run` and `relevance_run` both call it. `judge_run`'s signature and
behaviour do not change: the 266 Step 7 tests must pass unchanged. The common
config checks move into `_check_common(block, path)`, also with no behaviour
change.

CLI (new subcommand; same flags as `faithfulness`):
```
python -m evalharness.judges relevance --run-id 20261015T152638Z-v0-c7c96975 [--query-ids q001,q053,q097] [--retry-failed] [--allow-heldout] [--harness-config configs/harness.yaml]
```
Summary line: `judged run <id> with rel-v1-<hash>: complete | 80 outputs: 80
judged, 0 judge failed, 0 not judged, 0 skipped (already judged) |
gpt-oss:120b-cloud: 80 calls, 0 hits, 80 misses, 0 errors`, then the report.
Exit codes and refusals are as for `faithfulness`: 0 complete; 1 for a bad
config, a digest mismatch, a run that cannot be scored, or a judge model equal
to the generator; 2 for partial or aborted.

Report (`build_relevance_report`, read back from `scores` only). Groups are
`all`, `answerable` and each type.
- **Coverage:** outputs, judged, judge failed, not judged, missing.
- **Score counts:** how many 0 / 1 / 2.
- **Mean relevance** with n.
- **Irrelevant rate:** successes/n with a Wilson CI, labelled "(uncalibrated
  judge)".
- **Lists:** irrelevant ids, the unanswerable ids not scored 2 (anything but a
  clean refusal), and the judge-failed ids.

## Data and schema changes
No new table and no new columns. Two new metric names in `scores` per output,
under the relevance version (`rel-v<prompt_version>-<hash>`):

| metric | value | value NULL when | reasoning |
|---|---|---|---|
| `relevance` | 0, 0.5, 1 | not judged, judge failed | `"score 2/2: <judge reasoning>"` |
| `irrelevant` | 1 / 0 | not judged, judge failed | `"score 0 <= 0"` / `"score 2 > 0"` |

NULL reasons use the Step 7 prefixes `"not judged: ..."` and
`"judge failed: ..."`. No `judge_claims` rows are written for relevance.

## Config changes
New `relevance` block in `configs/judges.yaml`; the `faithfulness` block is
unchanged (`faith-v2-d130a39f`). Keys: `prompt_version`, `model`,
`model_digest`, `temperature` (0.0), `seed` (42), `reasoning_effort` (medium),
`max_prompt_tokens` (3500), `max_consecutive_failures` (5),
`irrelevant_max_score` (0), `answerable_rubric`, `unanswerable_rubric`,
`prompt_template` (placeholders `{question}`, `{answer}`, `{rubric}`; JSON
braces doubled). The file itself holds the full prompt text.

Prompt history (smoke test on q001, q053, q097, 2026-10-15):
- **v1 (`rel-v1-3ae4b6e7`):** one prompt held both rubrics plus an
  "ANSWERABLE / NOT ANSWERABLE" line. The judge ignored the line on q053 (an
  answerable query with a false refusal). It applied the not-answerable rubric
  and scored 2 instead of 0.
- **v2 (`prompt_version: 2`), Gaurav's choice:** the prompt gets only the rubric
  that applies, chosen from the test set. Each rubric states that answerability
  is a fact from the test set. The answerable rubric says that "not available
  / not mentioned" scores 0. The v1 smoke rows stay stored under their own
  version.

## Files to create
- `tests/test_08_relevance_judge.py`

## Files to change
- `evalharness/schemas.py`: `RelevanceVerdict`.
- `evalharness/judges.py`: `_check_common`, `_judge_items` (refactor, no
  behaviour change), the relevance functions, the `relevance` subcommand,
  and the module docstring.
- `configs/judges.yaml`: `relevance` block; header comment updated.
- `tests/conftest.py`: `relevance_config` fixture, plus a fake relevance
  reply in the fake judge LLM (or a small `FakeRelevanceLLM`).
- `tests/README.md`: Step 8 notes.
- `CLAUDE.md`: Progress entry.

## New dependencies
No new dependencies.

## Tests
`tests/test_08_relevance_judge.py`, all offline. A fake replaces
`evalharness.judges.generate`, `get_model_digest` and `load_llm_config` are
patched for `main()`, and the DBs and LLM cache are in `tmp_path`.

Schema:
- `RelevanceVerdict` accepts scores 0, 1 and 2.
- It rejects 3, -1, `"2"` (strict), 1.5, a missing score, blank reasoning and
  extra keys.
- The JSON Schema property order is `reasoning, score`.

Config:
- The real `configs/judges.yaml` relevance block loads (read-only).
- Each missing key raises `ValueError` naming it, after a control.
- Bad values raise: temperature ≠ 0, `irrelevant_max_score` 2 or -1, an
  unknown `reasoning_effort`, a template with missing or extra placeholders.
- Version: matches `rel-v<prompt_version>-[0-9a-f]{8}` (v2 in the real file) and is stable across loads. It changes
  with the template, either rubric, the model, the digest, the
  seed, the effort or `irrelevant_max_score`. It is different from the
  faithfulness version.
- `load_judges_config` (faithfulness) still behaves exactly as in Step 7.

Prompt:
- `rubric_for` gives the answerable rubric for `["leave_policy"]` and the
  unanswerable rubric for `[]`; the prompt contains exactly one of them.
- The prompt has the question and answer inside their tags and only the
  applicable rubric.
- Braces and an injection line in the answer stay verbatim inside `<answer>`.
- The reference answer text never appears in the prompt.
- `generate` gets the model, temperature, seed, effort, digest and
  `RelevanceVerdict` from the config.

Scoring (by hand):
- score 2 → relevance 1.0, irrelevant 0.
- score 1 → 0.5, 0.
- score 0 → 0.0, 1.
- With `irrelevant_max_score` 1, score 1 → irrelevant 1.

`judge_relevance_output`:
- `LLMOutputError` → `JudgeError`.
- `prompt_tokens` `None` or ≥ the limit → `JudgeError`.

`relevance_run`, on a fixture run of 8 outputs:
- The 8 cases: answerable answered (2), partial (1), false refusal on an
  answerable query (0), correct refusal on an unanswerable query (2), invented
  answer to an unanswerable query (0), failed output, blank answer, judge
  failure.
- Exactly 2 rows per output (16), with values and reasoning by hand.
- The failed output and the blank answer get no LLM call.
- The judge failure gets NULL rows and the run continues.
- A rerun makes 0 calls. `retry_failed` re-judges only the failed item.
  `query_ids` gives partial.
- Abort after N failures in a row.
- One transaction per item.
- Refusals: unknown run, non-completed run, held-out without the flag.
- Faithfulness rows of the same run are untouched (the versions are separate).

Report:
- Coverage, the 0/1/2 counts, mean relevance (hand value), irrelevant rate
  with its Wilson CI, and the lists all match.
- The numbers are read from the DB (tamper test).
- The text contains "uncalibrated".

CLI `main relevance`:
- Exit 0 and the summary line.
- Exit 1 for a digest mismatch (nothing written), an unknown run, and a judge
  model equal to the generator.
- Exit 2 with `--query-ids`.
- `main faithfulness` still works (regression).

Layering (AST): `judges.py` does not import `rag`.

## Rules for implementation
- Plain Python and only libraries approved in CLAUDE.md
- No LangChain; no RAGAS or DeepEval (except the optional final cross-check)
- `rag/` must never import from `evalharness/`; the harness only calls `answer()`
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
- The judge sees only the question, the answer and the applicable rubric. It
  never sees the retrieved chunks or the reference answer.
- Answerability comes only from the test set (`source_doc_ids == []`), never
  from the answer or the Step 6 refusal flag.
- The prompt lives in `configs/judges.yaml`; there is no prompt text in Python.
- The refactor into `_judge_items` and `_check_common` must not change Step 7
  behaviour. `tests/test_07_faithfulness_judge.py` passes unchanged, and the
  stored `faith-v2-d130a39f` version string stays the same (its hash inputs
  do not change).
- Every output gets exactly 2 relevance rows for a version, unless the judging
  was partial or aborted. Failures are NULL, counted and listed, never 0 or 1.
- Never overwrite a stored judgement of the same version, except a
  "judge failed" item with `--retry-failed`.
- Every printed relevance rate says "uncalibrated".
- Prompt changes after the smoke test follow Step 7's rule. A fix that makes the
  judge follow an approved decision is allowed, with Gaurav's OK and a bump of
  `prompt_version`. Tuning toward wanted numbers is not.
- Only the dev run is judged.

## Decisions for Gaurav
All resolved on 2026-10-15; Gaurav accepted every recommendation:
1. ~~Rubric scale.~~ **Option A:** 0 / 1 / 2 with the two rubrics above;
   `relevance` = score / 2; `irrelevant` = score 0.
2. ~~What the judge is told.~~ **Option A:** a one-line answerable /
   not-answerable flag from the test set; no reference answer, no chunks.
3. ~~Judge settings.~~ **Option A:** same as faithfulness (`gpt-oss:120b-cloud`,
   pinned digest, temperature 0, seed 42, `reasoning_effort: medium`).
4. ~~Code layout.~~ **Option A:** both judges in `evalharness/judges.py`,
   sharing `_judge_items` and `_check_common` (refactor with no Step 7
   behaviour change).
5. ~~Real API calls.~~ **Approved:** smoke test on q001, q053, q097 (Gaurav
   reads it), then the full run (at most 80 calls), then a rerun with no calls.

## Definition of done
- [ ] Smoke test:
      `python -m evalharness.judges relevance --run-id 20261015T152638Z-v0-c7c96975 --query-ids q001,q053,q097`
      exits 2 (`partial`), and
      `sqlite3 data/results.sqlite "SELECT query_id, metric, value, reasoning FROM scores WHERE version LIKE 'rel-%' ORDER BY query_id, metric"`
      shows a score and reasoning for each of the 3. Gaurav has read them.
- [ ] Done when (CLAUDE.md): every answer has a relevance score and reasoning.
      Run `python -m evalharness.judges relevance --run-id 20261015T152638Z-v0-c7c96975`.
      It exits 0 (`complete`) and prints the summary line and the report.
- [ ] Coverage:
      `sqlite3 data/results.sqlite "SELECT metric, COUNT(*), COUNT(value), SUM(reasoning <> '') FROM scores WHERE run_id = '20261015T152638Z-v0-c7c96975' AND version = '<rel version>' GROUP BY metric"`
      prints 80 rows for each of `relevance` and `irrelevant`, with 80
      non-blank reasonings. Every NULL row's reasoning starts with "not
      judged" or "judge failed" (counts copied into Progress as observed).
- [ ] Consistency: `irrelevant = 1` exactly where `relevance = 0`
      (`irrelevant_max_score` 0); one SQL JOIN, 0 mismatches.
- [ ] A rerun prints `80 skipped (already judged) | no LLM calls`.
- [ ] Step 7 unchanged: `python -m evalharness.judges faithfulness --run-id 20261015T152638Z-v0-c7c96975`
      still prints `faith-v2-d130a39f: complete | ... 80 skipped (already judged) | no LLM calls`.
- [ ] Held-out untouched: `SELECT COUNT(*) FROM runs WHERE split = 'heldout'`
      prints `0`.
- [ ] Layering: `grep -rnE "^(import|from) rag" evalharness/` prints nothing,
      and `grep -rn "evalharness" rag/` prints nothing.
- [ ] `pytest tests/test_08_relevance_judge.py` passes with no network access,
      and `pytest tests/test_07_faithfulness_judge.py` still passes unchanged.
- [ ] `pytest` passes (whole suite).
- [ ] The Progress section of CLAUDE.md has a Step 8 entry. It contains the
      relevance version, the call summary, coverage, the 0/1/2 counts, mean
      relevance, and the uncalibrated irrelevant rate with n and its CI,
      overall and by type, all copied from the report. It also gives a short
      comparison with Step 6's phrase-match refusals: does q053 get 0, and do
      the 12 unanswerable answers get 2? It is labelled "uncalibrated".
- [ ] All changes are committed on `feature/08-relevance-judge`.

## Carry-forward to later steps
- **Step 9:** the calibration labels are grounded / ungrounded only (CLAUDE.md).
  Decide whether Gaurav also labels relevance for the same ~50 answers, so the
  relevance judge gets an agreement figure too. *Decision for Gaurav then.*
- **Step 10:** report relevance next to faithfulness. Cross-tabulate
  unanswerable answers that are both irrelevant (0) and ungrounded (the
  CLAUDE.md "invented answer" case), and answerable answers that are
  irrelevant because of a false refusal.
- **Step 13:** "relevance is low" means `irrelevant = 1`, or `relevance <= 0.5`
  if a stricter flag is wanted; the threshold is tuned against Gaurav's labels
  there. "The judges disagree" can compare `ungrounded` and `irrelevant`.
