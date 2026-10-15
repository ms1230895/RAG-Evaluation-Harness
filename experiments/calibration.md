# Judge calibration log (Step 9)

Gaurav's labels vs the LLM judges, on the v0 dev run `20261015T152638Z-v0-c7c96975`.
Labels: `calibration_labels` in `data/results.sqlite` (only Gaurav labels, blind to judge outputs).
Every comparison is logged in `calibration_reports`; the numbers below are copied from
`python -m evalharness.calibration history --run-id 20261015T152638Z-v0-c7c96975`.

Method: the answers are split once into `tune` and `check` (seed 2027, stratified by type,
11 ids whose verdicts were seen in chat forced into `tune`). Prompt changes are justified only
by `tune` disagreements and bump `prompt_version`; the chosen version is reported once on `check`.

| Date | Judge | Version | Split | Why this version / what changed | n | TP | FP | FN | TN | Agreement | Precision | Recall | Kappa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-15 | faithfulness | faith-v2-d130a39f | check | Starting judge; no prompt iteration (scope reduced, see notes) | 20 | 0 | 0 | 0 | 20 | 1.000 20/20 [0.839, 1.000] | undefined (0/0) | undefined (0/0) | undefined |
| 2026-10-15 | relevance | rel-v2-c825e4f7 | check | Measured only (no iteration planned) | 20 | 0 | 0 | 0 | 20 | 1.000 20/20 [0.839, 1.000] | undefined (0/0) | undefined (0/0) | undefined |

## Notes (2026-10-15): result is inconclusive

- First labelling pass: 63 labels, all "grounded" / relevance 2, saved in about 142 seconds
  (~2.3 s per answer). Gaurav agreed they were not careful labels; they were cleared with his OK
  (the tune/check split was kept). No comparison was run on them.
- Scope then reduced by Gaurav: label only `check`, then only a fixed stratified subset of 20 of the
  35 `check` answers (`label --split check --limit 20`, `subset_seed` 2028). No prompt iteration.
- Second pass: 20 labels, all "grounded" / relevance 2, no notes, saved between 18:30:10 and
  18:31:05 UTC (55 seconds, ~2.8 s per answer).
- On these 20 answers both judges also flag nothing (faithfulness `ungrounded` 0/20, relevance
  `irrelevant` 0/20): every judge-positive answer (7 ungrounded, 1 irrelevant) is in `tune`, forced
  there because its verdict had been shown in chat. So the check comparison has **zero positives on
  both sides**: agreement 20/20 [0.839, 1.000], precision / recall / kappa undefined (0/0; all 2000
  bootstrap resamples undefined).
- What this does and does not show: the judges and the labels agree on 20 answers that both call
  grounded and relevant. It says **nothing** about whether the judges detect ungrounded or
  irrelevant answers (no positive case was compared), and the labelling speed means even the
  agreement is weak evidence. The judges therefore remain **uncalibrated** for reporting purposes.
- Open: a real calibration needs careful labels that include judge-positive answers (for example
  the 7 judge-ungrounded answers, noted as anchored, plus unlabelled `check` answers).
