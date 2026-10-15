# Spec: Step 09 — Judge Calibration

## Overview
Steps 7-8 produced judge numbers that are only the judge's opinion (7/80
ungrounded, 1/80 irrelevant, both labelled "uncalibrated"). This step measures
how far that opinion can be trusted. A small labelling tool shows Gaurav each
stored answer of the v0 dev run with the exact chunks the generator saw, but
never the judge's verdict. He labels each answer grounded or ungrounded, and
by Decision 5 also gives it a relevance score. The labels go into a new
`calibration_labels` table. `evalharness/calibration.py` then compares a judge
version with his labels. It reports agreement %, precision and recall on
"ungrounded", and Cohen's kappa, each with an interval, plus the confusion
table and the items where the judge and Gaurav disagree. The faithfulness
prompt may be improved from those disagreements, and every change bumps the
judge version. To keep the final figure honest, the labelled answers are split
once into a `tune` half (used to improve the prompt) and a `check` half (used
to report the chosen version). This step comes before the baseline (Step 10)
because the baseline's main number, the ungrounded rate, is only meaningful
once the judge that produces it has a measured agreement with a human.

## Concepts to learn
- **Calibration = measuring a judge against human labels.** Gaurav's label is
  treated as the truth, and the judge is graded like a classifier. "Positive"
  means ungrounded, because ungrounded answers are the thing we want to find.
  Every labelled answer then falls into one cell of a 2×2 table:

  |  | Gaurav: ungrounded | Gaurav: grounded |
  |---|---|---|
  | judge: ungrounded | TP (true positive) | FP (false positive: judge too strict) |
  | judge: grounded | FN (false negative: judge missed it) | TN (true negative) |
- **The four numbers**, with n = TP + FP + FN + TN:
  - agreement % = (TP + TN) / n, the share where both say the same.
  - precision = TP / (TP + FP): when the judge says ungrounded, how often it is right.
  - recall = TP / (TP + FN): of the truly ungrounded answers, how many the judge catches.
  - Cohen's kappa = (p_o − p_e) / (1 − p_e). Here p_o is the agreement, and
    p_e = P(judge+)·P(human+) + P(judge−)·P(human−) is the agreement two
    raters would reach by chance with the same label frequencies.
- **Worked example** (n = 40, TP 3, FP 1, FN 2, TN 34):
  - Agreement: 37/40 = 0.925.
  - Precision: 3/4 = 0.75.
  - Recall: 3/5 = 0.60.
  - The judge says ungrounded for 4/40 = 0.10 and Gaurav for 5/40 = 0.125.
    So p_e = 0.10·0.125 + 0.90·0.875 = 0.0125 + 0.7875 = 0.80.
  - Kappa: (0.925 − 0.80) / (1 − 0.80) = 0.625.

  The lesson: with few ungrounded answers, a judge that always says
  "grounded" would already reach 35/40 = 0.875 agreement. Agreement % looks
  high almost for free. Kappa subtracts that free part, which is why CLAUDE.md
  asks for it. Common reading: below 0.4 weak, 0.4-0.6 moderate, 0.6-0.8
  substantial, above 0.8 near-perfect.
- **Undefined cases are reported, not hidden.** If the judge flags nothing,
  precision is 0/0. If Gaurav labels nothing ungrounded, recall is 0/0. If both
  give one label only, kappa is 0/0. These print as "undefined (0/0)", never
  as 0 or 1.
- **Uncertainty.** Agreement, precision and recall are rates, so each gets a
  Wilson interval (Step 6). Kappa is not a simple rate, so it gets a
  **bootstrap** interval:
  1. Resample the labelled pairs with replacement, as many as there are.
  2. Recompute kappa.
  3. Repeat 2000 times with a fixed seed.
  4. Take the 2.5th and 97.5th percentiles.

  With about 40 items and only a handful of ungrounded ones, these intervals
  will be wide. That is the honest result, not a defect.
- **Blind labelling.** Seeing the judge's verdict pulls a human toward it
  (anchoring), which inflates agreement. So the tool shows the question, the
  answer and the stored chunks. It never shows the judge's claims, labels,
  scores or the reference answer. The reference is about correctness, not
  groundedness. Answers come in a seeded shuffled order, so items of one type
  are not labelled in a row.
- **Some verdicts were already seen.** This session's chat showed judge
  verdicts for q001, q053 and q097 (smoke tests) and for q029, q057, q058, q068,
  q069, q071, q082 and q085 (the Step 7 non-supported claims list). Gaurav's
  labels for those 11 may be anchored. They are forced into the `tune` half,
  so the reported `check` figure uses only answers whose verdicts he has not
  seen.
- **Tune / check split (the held-out idea, applied to the judge).** If the
  prompt is tweaked until it agrees with the labels, then reported on those
  same labels, the figure is optimistic: it measures fitting, not judging. So
  the labelled answers are split once, seeded and stratified by query type,
  into `tune` and `check`. The prompt is iterated on `tune` disagreements only.
  The chosen version is then compared on `check`. Each `check` comparison is
  logged, and the report prints how many times `check` was looked at. A prompt
  change after a `check` look is allowed, but it is recorded, and the figure
  must then be reported with that caveat.
- **One label definition for human and judge.** Gaurav labels with the same
  rule the judge uses (Step 7, Decision 3). An answer is **ungrounded** if any
  factual claim in it is not supported by the shown chunks, or contradicts
  them. Statements about the documents themselves ("the text does not mention
  X") and pure refusals are **grounded**. If he cannot decide, he marks
  **unsure**. Unsure items are counted and left out of the metrics. Known
  mismatch: the judge only flags a single unsupported claim when the answer
  has 4 claims or fewer (threshold 0.8), while the human rule flags any.
  Step 13 tunes that threshold against these labels.

## Depends on
- Step 5: `evalharness/store.py` (`connect`, `create_tables`, `get_run`,
  `get_outputs_with_queries` with the stored chunk text). Run
  `20261015T152638Z-v0-c7c96975` (dev, 80 outputs).
- Step 6: `evalharness/metrics.py`: `wilson_interval`, `rate_summary`.
  `evalharness/checks.py`: `get_scorable_run`, `ScoringRefused`,
  `load_checks_config` (`report_confidence` 0.95).
- Step 7: `scores` rows `ungrounded` (0/1/NULL) under `faith-v2-d130a39f`, all
  80 judged. The CLI `python -m evalharness.judges faithfulness --run-id ...
  --query-ids ...` is used to judge only the `tune` ids for a new prompt
  version. Prompt text is in `configs/judges.yaml` (`prompt_version` 2).
- Step 8: `scores` rows `irrelevant` / `relevance` under `rel-v2-c825e4f7`. The
  Step 8 carry-forward asks whether relevance is labelled too (Decision 5).
- Step 4 / test helpers: the `scripted_input` fixture (`ScriptedInput`) and the
  `input_fn` pattern of `evalharness/testset.py review`, which this tool
  copies so it can be tested without a terminal.
- Carry-forward items addressed here:
  - Step 7: possible judge errors q057 (one "or" split into two contradicted
    "only" claims) and q058 ("should" vs "can"). They go into `tune` and are
    the first candidates for a prompt fix, if Gaurav's labels say so.
  - Step 8: decide whether relevance is labelled too (Decision 5).

## Interfaces
This step does **not** touch the `answer()` contract, does not call
`answer()`, and does not import `rag`. It makes **no LLM call** itself. New
judge versions are produced with the existing `evalharness.judges` CLI.

`evalharness/calibration.py` (new)
```python
GROUNDED_LABELS = ("grounded", "ungrounded", "unsure")

def load_calibration_config(path: str | Path) -> dict:
    """Read and check configs/calibration.yaml (ValueError names the problem)."""

def assign_calibration_splits(query_types: dict[str, str], *, seed: int, check_fraction: float,
                              forced_tune_ids: list[str]) -> dict[str, str]:
    """{query_id: "tune" | "check"}. Stratified by query type: in each type,
    round(check_fraction * eligible) ids go to check, drawn with
    random.Random(seed) from the sorted ids that are not forced into tune.
    Deterministic for the same input."""

def label_session(conn, run_id: str, cfg: dict, *, input_fn=input, print_fn=print,
                  relabel: list[str] | None = None) -> dict:
    """Interactive labelling of the run's outputs that have no label yet, in a
    seeded shuffled order. On the first session it assigns and stores the
    tune/check split for every output. Shows: progress (12/80), query id, type,
    question, answer, and the stored chunks (rank, doc_id, text). Never shows
    judge scores, claims or the reference answer. Keys: g = grounded,
    u = ungrounded, ? = unsure, n = add a note first, s = skip for now,
    q = quit (labels so far are saved; each label is committed at once).
    With relevance labels on (Decision 5), it then asks 0 / 1 / 2 for "does it
    address the question?". relabel: re-ask only these ids (overwrites after a
    y/n confirmation). Failed outputs (outputs.error) are skipped and counted.
    Returns {"labelled": n, "skipped": n, "remaining": n}."""

def confusion(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs of (human_ungrounded, judge_ungrounded) -> {"tp", "fp", "fn", "tn", "n"}."""

def cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float | None:
    """(p_o - p_e) / (1 - p_e); None when 1 - p_e == 0 (undefined)."""

def kappa_bootstrap_ci(pairs: list[tuple[bool, bool]], *, n_boot: int, seed: int,
                       confidence: float) -> tuple[float, float] | None:
    """Percentile bootstrap (numpy, seeded). Resamples where kappa is undefined
    are dropped and counted. None if fewer than half the resamples are defined."""

def agreement_report(pairs: list[tuple[bool, bool]], cfg: dict, confidence: float) -> dict:
    """Confusion counts plus agreement / precision / recall (each successes, n,
    rate, Wilson CI, or "undefined" for 0/0) and kappa with its bootstrap CI."""

def compare(conn, run_id: str, judge: str, judge_version: str, calib_split: str, cfg: dict,
            confidence: float) -> dict:
    """Pairs Gaurav's labels with the judge's stored flags for one split
    ("tune", "check" or "all"): `ungrounded` for judge="faithfulness",
    `irrelevant` for judge="relevance" (needs relevance labels). Unsure labels,
    unlabelled items and items the judge did not judge (NULL / missing) are left
    out and counted. Writes one calibration_reports row (every look is logged)
    and returns the report, including how many earlier "check" looks exist for
    this judge. For tune and all it also lists the disagreements (query id,
    human, judge). For check it lists counts only, unless show_check_items is
    set in config."""

def format_compare(report: dict) -> str
def main(argv: list[str] | None = None) -> int
```
CLI:
```
python -m evalharness.calibration label   --run-id 20261015T152638Z-v0-c7c96975 [--relabel q057,q058]
python -m evalharness.calibration status  --run-id 20261015T152638Z-v0-c7c96975
python -m evalharness.calibration compare --run-id 20261015T152638Z-v0-c7c96975 --judge faithfulness --version faith-v2-d130a39f --split tune
python -m evalharness.calibration compare ... --split check --confirm-check
python -m evalharness.calibration history --run-id 20261015T152638Z-v0-c7c96975
```
- `status` prints labels by split and type: grounded, ungrounded, unsure,
  unlabelled. It never shows a judge result. `status --ids tune|check` prints
  only the comma-separated ids of that split, to pass to
  `python -m evalharness.judges faithfulness --query-ids ...` during iteration.
- Labelling sessions are run by Gaurav in his own terminal (for example the
  VS Code terminal). The tool needs real keyboard input, and labels must come
  only from him.
- With 11 ids forced into `tune`, `check` gets about `round(0.5 × eligible)`
  ids per type, roughly 34-35 in total. The exact count comes from the code
  and is copied into Progress.
- `compare --split check` is refused (exit 1) without `--confirm-check`. This
  makes looking at `check` a deliberate act.
- `history` prints every stored `calibration_reports` row in time order:
  version, split, n, agreement, precision, recall and kappa.
- Exit codes: 0 ok, 1 refused (bad config, unknown or not-completed run,
  held-out run, check without `--confirm-check`, no labels for the split).

`evalharness/store.py` (new functions):
- `save_calibration_splits(conn, run_id, splits)`: inserts the split rows
  once; raises if they already exist with different values.
- `write_calibration_label(conn, run_id, query_id, grounded_label,
  relevance_label, notes, labeler)`: one row, committed at once.
- `get_calibration_labels(conn, run_id)`.
- `write_calibration_report(conn, row)`.
- `get_calibration_reports(conn, run_id)`.

## Data and schema changes
Two new tables, created by `store.create_tables()` (Decision 7). `reviews` is
left for Step 13's flagged-answer queue.
```sql
CREATE TABLE IF NOT EXISTS calibration_labels (
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    query_id        TEXT NOT NULL REFERENCES queries(id),
    calib_split     TEXT NOT NULL CHECK (calib_split IN ('tune', 'check')),
    grounded_label  TEXT CHECK (grounded_label IN ('grounded', 'ungrounded', 'unsure')),  -- NULL = not labelled yet
    relevance_label INTEGER CHECK (relevance_label IN (0, 1, 2)),   -- NULL unless Decision 5
    notes           TEXT,
    labeler         TEXT,                     -- "gaurav"; set with the label
    labeled_at      TEXT,                     -- UTC ISO; set with the label
    PRIMARY KEY (run_id, query_id)
);

CREATE TABLE IF NOT EXISTS calibration_reports (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    judge         TEXT NOT NULL,              -- faithfulness | relevance
    judge_version TEXT NOT NULL,              -- e.g. faith-v2-d130a39f
    calib_split   TEXT NOT NULL,              -- tune | check | all
    n             INTEGER NOT NULL,           -- pairs compared
    n_excluded    INTEGER NOT NULL,           -- unsure + unlabelled + not judged
    tp INTEGER NOT NULL, fp INTEGER NOT NULL, fn INTEGER NOT NULL, tn INTEGER NOT NULL,
    report_json   TEXT NOT NULL,              -- the full agreement_report output
    created_at    TEXT NOT NULL
);
```
No new data files. The labels are local data in the gitignored
`data/results.sqlite`. The Progress entry copies the counts and figures.

## Config changes
New file `configs/calibration.yaml` (starting values; Decisions 2-5):
```yaml
# Judge calibration (Step 9). Values approved by Gaurav on 2026-10-15.
labeler: gaurav
split_seed: 2027                 # random.Random seed for the tune/check split; used once
check_fraction: 0.5              # share of each query type's eligible ids that go to check
# Ids whose judge verdicts were shown in chat before labelling (possible anchoring):
forced_tune_ids: [q001, q029, q053, q057, q058, q068, q069, q071, q082, q085, q097]
order_seed: 11                   # shuffled labelling order
label_relevance: true            # also ask 0/1/2 relevance per answer (Decision 5)
show_check_items: false          # compare --split check prints counts only, not item ids
n_bootstrap: 2000                # kappa bootstrap resamples
bootstrap_seed: 7
labeling_guide: |                # printed at the start of every session
  Label each answer against the CHUNKS shown, not your own knowledge.
  u = ungrounded: at least one factual claim is not supported by the chunks, or contradicts them.
  g = grounded: every factual claim is supported. A pure refusal ("not mentioned") and
      statements about the documents themselves count as grounded.
  ? = unsure: you cannot decide; it is left out of the metrics and counted.
  Relevance (if asked): 2 = addresses everything asked (or, for a question the policies
  do not cover, clearly says so); 1 = partly; 0 = does not address it (including a
  refusal of a question the policies do cover).
```
The interval level comes from `configs/harness.yaml` `checks.report_confidence`
(0.95), so it is not repeated here.

## Files to create
- `evalharness/calibration.py`
- `configs/calibration.yaml`
- `experiments/calibration.md` (a human-readable log: one entry per judge
  version, with the reason for the change and the `history` numbers copied
  from the DB)
- `tests/test_09_judge_calibration.py`

## Files to change
- `evalharness/store.py`: the two tables and their functions.
- `configs/judges.yaml`: only if a prompt iteration happens (bump
  `prompt_version`; the version hash changes automatically).
- `tests/conftest.py`: a `calibration_config` fixture, plus a
  `calibration_db` (a tmp DB with a completed run, 8 outputs with chunks, and
  judge `ungrounded` / `irrelevant` rows for two versions).
- `tests/README.md`: Step 9 notes.
- `CLAUDE.md`: Progress entry.

## New dependencies
No new dependencies. numpy (bootstrap) and scipy (through `wilson_interval`)
are already approved. Cohen's kappa is written by hand, without scikit-learn,
which is not on the approved list.

## Tests
`tests/test_09_judge_calibration.py`, all offline. Tmp DBs only. The labelling
session is driven with `scripted_input`. No LLM call (calibration makes none).

Metrics, by hand:
- `confusion` on a hand list.
- `cohen_kappa` (worked example):
  - TP 3, FP 1, FN 2, TN 34 → 0.625.
  - Perfect agreement → 1.0.
  - Agreement at chance level → 0.0.
  - Both raters give one label only (p_e = 1) → None.
- Agreement / precision / recall:
  - 37/40, 3/4, 3/5 with their Wilson CIs.
  - Precision undefined when the judge flags nothing; recall undefined when
    the human flags nothing. Both print "undefined (0/0)".
- `kappa_bootstrap_ci`:
  - Identical across two calls with the same seed; different with another seed.
  - Lower ≤ point ≤ upper on the worked example.
  - None when almost every resample is undefined.

Splits:
- Deterministic for the same seed.
- Forced ids are always `tune`.
- Per-type check counts equal `round(fraction × eligible)`.
- Every output gets exactly one split.
- `save_calibration_splits` refuses to change a stored split.

`label_session` with `scripted_input`:
- `g`, `u`, `?` and the relevance answers are stored with `labeler` and
  `labeled_at`.
- `q` mid-way keeps the labels so far, and a second session resumes with the
  rest.
- `s` leaves an item unlabelled.
- `n` stores a note.
- An invalid key re-asks.
- Failed outputs are skipped and counted.
- The order is the seeded shuffle.
- The printed text never contains the judge's reasoning, claim text, score
  values or the reference answer. The fixture plants marker strings to prove
  this.
- `relabel` asks for confirmation and overwrites only the named ids.

`compare`:
- Faithfulness `ungrounded` vs labels on `tune`, `check` and `all`. The
  counts match the hand-made fixture.
- Unsure, unlabelled and judge-NULL items are excluded and counted.
- Relevance `irrelevant` vs `relevance_label == 0`.
- A `calibration_reports` row is written per call, and `check` looks are
  counted in the report.
- `--split check` without `--confirm-check` gives exit 1.
- Check output lists no item ids when `show_check_items` is false.
- An unknown version or a split without labels gives exit 1.
- A held-out run is refused.

Store:
- Table columns and CHECK constraints (a bad label is rejected).
- A label write is committed at once.
- A hostile note is stored verbatim.

Layering (AST): `evalharness/calibration.py` imports neither `rag` nor
`llm.client`; no LLM call is possible from it.

## Rules for implementation
- Plain Python and only libraries approved in CLAUDE.md
- No LangChain; no RAGAS or DeepEval (except the optional final cross-check)
- `rag/` must never import from `evalharness/`; the harness only calls `answer()`
- All tunable values come from YAML config — no hardcoded magic numbers
- SQLite via the sqlite3 module with parameterised queries only
- All LLM calls go through `llm/client.py` so they are cached (CLAUDE.md's
  `evalharness/llm.py` became the shared `llm/client.py` in Step 2). The
  calibration module makes none; new judge versions use `evalharness.judges`.
- API keys only from `.env`, never in code, logs or commits
- The held-out split is never used before Step 12
- Never invent, estimate or adjust metric numbers; report only what a
  stored run produced
- Human labels (calibration and review) come only from Gaurav
- Code is simple, readable and commented

Step-specific:
- **Only Gaurav labels.** Claude never fills, suggests, pre-selects or "fixes"
  a label, and never writes to `calibration_labels` except through the tool
  driven by Gaurav's own keystrokes. Tests use scripted input in tmp DBs only.
- **Blind tool.** `label` and `status` never print judge outputs (scores,
  claims, verdict reasoning) or the reference answer. Claude does not show
  judge results for unlabelled items in chat while labelling is in progress.
- **Tune/check discipline.**
  - The split is assigned once and never changed.
  - Prompt changes are justified only by `tune` disagreements, and each one
    bumps `prompt_version`.
  - New versions are judged on the `tune` ids only (`--query-ids`). The
    chosen version is then judged on the remaining ids, and compared once on
    `check` with `--confirm-check`.
  - Every `compare` call is logged in `calibration_reports`. The reported
    figure names the version and how many `check` looks happened.
- Each prompt iteration gets an `experiments/calibration.md` entry:
  - the version;
  - which `tune` disagreements motivated the change;
  - the `tune` numbers, copied from `history`.
- The figures are reported as measured, with n and their intervals, even if
  they are weak. With few ungrounded answers, the CIs will be wide; say so.
  Never relabel to raise agreement. A relabel is allowed only for a clear
  labelling mistake, Gaurav's own call, and it is noted.
- The judge threshold (0.8) is not tuned here (Step 13). The relevance prompt
  is measured, not iterated, unless Gaurav decides otherwise.
- Only the v0 dev run is labelled. No held-out answer is shown or labelled.

## Decisions for Gaurav
All resolved on 2026-10-15; Gaurav accepted every recommendation:
1. ~~Labelling tool.~~ **Option A:** terminal CLI, same style as the Step 4
   `review` tool.
2. ~~How many answers.~~ **Option A:** all 80 v0 dev answers.
3. ~~Honest reporting.~~ **Option A:** tune/check split (`check_fraction` 0.5,
   stratified by type, the 11 already-seen ids forced into `tune`); iterate on
   `tune`, report the chosen version once on `check`.
4. ~~Label definition.~~ Accepted as written in `labeling_guide`.
5. ~~Relevance labels.~~ **Option A:** yes, 0/1/2 per answer; the relevance
   judge is measured only, not iterated.
6. ~~Prompt iteration budget.~~ **Option A:** up to 3 new faithfulness prompt
   versions, each judged on `tune` ids only, then the chosen version on `check`
   ids (at most about 300 calls; 0 if v2 holds up).
7. ~~Where labels live.~~ **Option A:** new `calibration_labels` and
   `calibration_reports` tables.

**Change on 2026-10-15 (Gaurav's choice, after labelling):** the first
labelling pass (63 labels, all "grounded" / relevance 2, saved in about 142
seconds) was judged not to be careful labelling and was cleared with Gaurav's
explicit OK (backup kept outside the repo); the split was kept. Gaurav then
chose a smaller set: label **only the 35 `check` answers**, keep the judge at
`faith-v2-d130a39f` / `rel-v2-c825e4f7`, **no prompt iteration** (Decisions 2
and 6 reduced). `label --split check` queues only that half. The `tune` half
stays unlabelled and is excluded and counted; the figure is reported on
`check` once, with its wide intervals.

**Further reduced on 2026-10-15 (Gaurav's choice):** Gaurav asked Claude to
label the answers; Claude declined (CLAUDE.md: human labels come only from
Gaurav). Gaurav then chose to label **20 of the 35 `check` answers**: a fixed
subset, stratified by type with largest-remainder allocation, seeded by
`subset_seed` (2028): `label --split check --limit 20`. The other 15 `check`
answers stay unlabelled and are excluded and counted. The figure is reported
on n ≤ 20 with very wide intervals, stated plainly.

## Definition of done
- [ ] Labels exist, from Gaurav's own sessions:
      `sqlite3 data/results.sqlite "SELECT calib_split, grounded_label, COUNT(*) FROM calibration_labels WHERE run_id = '20261015T152638Z-v0-c7c96975' GROUP BY calib_split, grounded_label"`
      shows the labelled counts (80 minus failures if Decision 2A, with unsure
      counted). `python -m evalharness.calibration status --run-id ...` shows
      no unlabelled item, or only the ones Gaurav chose to leave.
- [ ] Every `forced_tune_ids` id has `calib_split = 'tune'` (SQL), and the
      split was never changed (`save_calibration_splits` refuses).
- [ ] Tune comparison for the starting judge:
      `python -m evalharness.calibration compare --run-id 20261015T152638Z-v0-c7c96975 --judge faithfulness --version faith-v2-d130a39f --split tune`
      prints the confusion table, agreement, precision, recall and kappa with
      intervals, and the disagreements.
- [ ] Any prompt iteration: new `prompt_version` in `configs/judges.yaml`,
      judged on `tune` ids only, compared on `tune`, with an entry in
      `experiments/calibration.md`. Or a note there that no iteration was
      needed, and why.
- [ ] Done when (CLAUDE.md): we have a measured agreement figure we can report
      honestly. The chosen version is judged on all labelled ids, then
      `python -m evalharness.calibration compare ... --split check --confirm-check`
      prints agreement %, precision and recall on ungrounded, and kappa, each
      with n and an interval. `python -m evalharness.calibration history
      --run-id ...` shows that row and the number of `check` looks.
- [ ] (If Decision 5A) the relevance comparison on `check` for
      `rel-v2-c825e4f7` is printed and stored.
- [ ] `calibration_reports` holds one row per comparison:
      `sqlite3 data/results.sqlite "SELECT judge, judge_version, calib_split, n, tp, fp, fn, tn FROM calibration_reports ORDER BY id"`.
- [ ] Held-out untouched: `SELECT COUNT(*) FROM runs WHERE split = 'heldout'`
      prints `0`.
- [ ] Layering: `grep -rnE "^(import|from) (rag|llm)" evalharness/calibration.py`
      prints nothing, and `grep -rn "evalharness" rag/` prints nothing.
- [ ] `pytest tests/test_09_judge_calibration.py` passes with no network access.
- [ ] `pytest` passes (whole suite).
- [ ] The Progress section of CLAUDE.md has a Step 9 entry:
      - the label counts by split;
      - the starting-judge `tune` figures;
      - each iteration;
      - the final `check` figures (judge version, n, the 2×2 counts, agreement,
        precision, recall, kappa, each with its interval, and the number of
        `check` looks), all copied from `calibration_reports`;
      - plainly, whether they are weak or noisy.
- [ ] All changes are committed on `feature/09-judge-calibration`.

## Carry-forward to later steps
- **Step 10:** report the ungrounded rate with the judge version that was
  calibrated here, and quote its `check` agreement next to it. If the final
  version is not `faith-v2-d130a39f`, the v0 dev run must be judged with it
  first (cached calls only where prompts are unchanged).
- **Step 13:** tune the faithfulness threshold (0.8) and the "relevance is
  low" cut against these labels, using the stored `faithfulness` values (no
  new LLM calls); use `tune` for tuning and `check` for the reported flag
  quality.
- **Step 16:** the README reports the calibration figures with n, intervals,
  the tune/check method, the 11 anchored ids, and that one person labelled.
