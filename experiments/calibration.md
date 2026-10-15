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
