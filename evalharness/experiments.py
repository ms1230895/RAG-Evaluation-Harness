"""Experiment tooling for Step 11: one-variable check and paired run comparison.

Two jobs, no LLM call and no re-run, and no import of rag/ or llm/:

1. `check`: an experiment YAML must differ from the base config (v0) in exactly
   the keys it declares in its `experiment:` block. This is how "one variable
   at a time" is enforced instead of trusted.
2. `compare`: each experiment run is compared with the baseline run query by
   query. Both runs answered the same queries, so a query's ungrounded flag is
   either unchanged or it flipped:
       improved:  ungrounded in the baseline, grounded in the experiment
       regressed: grounded in the baseline, ungrounded in the experiment
   Only flips carry information about the change. The exact McNemar test
   (a sign test on the flips) asks how surprising the split would be if the
   change did nothing: with n flips and k in the majority direction,
       p = 2 * P(X >= k), X ~ Binomial(n, 0.5)   (capped at 1)
   Worked example: 4 improved, 1 regressed -> n = 5, k = 4,
   P(X >= 4) = 6/32 = 0.1875, p = 0.375: not distinguishable from no change.

The mean faithfulness difference (experiment - baseline) comes with a seeded
paired bootstrap interval: resample the per-query differences with replacement
and take the 2.5th and 97.5th percentiles of the resampled means.

Every number is read from stored scores (the same item_table the Step 10
report uses). Every judge-based number is labelled with its calibration status.

CLI:
    python -m evalharness.experiments check --config configs/v1_chunk75.yaml
    python -m evalharness.experiments compare --baseline <run_id> --run <run_id> [--run ...] \\
        [--markdown experiments/experiment_table.md [--force]] [--log experiments/log.md]
Exit codes: 0 ok; 1 problems or refused (nothing written).
"""

import argparse
import json
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import binomtest

from evalharness import store
from evalharness.checks import ScoringRefused, get_scorable_run, load_checks_config
from evalharness.metrics import mean_summary, percentile_summary, rate_summary
from evalharness.report import _calibration_status, _fmt_rate_pct, current_versions, item_table
from evalharness.runner import DEFAULT_HARNESS_CONFIG, load_harness_config
from evalharness.testset import resolve_path

EXPERIMENT_KEYS = {"base_config", "ignore_keys", "n_bootstrap", "bootstrap_seed", "table_path"}
BLOCK_KEYS = ("base", "variable", "changed_keys", "hypothesis")   # the `experiment:` block of a config
ALPHA = 0.05   # the p-value below which a flip split is called significant


class ExperimentRefused(Exception):
    """The runs cannot be compared (not comparable, not completed, held-out, ...)."""


# ---------------------------------------------------------------- config

def load_experiments_config(harness: dict) -> dict:
    """The `experiments` block of configs/harness.yaml, checked; ValueError names the problem."""
    cfg = harness.get("experiments")
    if not isinstance(cfg, dict):
        raise ValueError("harness config: 'experiments' must be a mapping")
    missing, extra = EXPERIMENT_KEYS - set(cfg), set(cfg) - EXPERIMENT_KEYS
    if missing or extra:
        raise ValueError(f"harness config: experiments keys missing {sorted(missing)}, "
                         f"unknown {sorted(extra)}")
    for key in ("base_config", "table_path"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"harness config: 'experiments.{key}' must be a non-empty string")
    if (not isinstance(cfg["ignore_keys"], list)
            or not all(isinstance(k, str) and k for k in cfg["ignore_keys"])):
        raise ValueError("harness config: 'experiments.ignore_keys' must be a list of key names")
    for key in ("n_bootstrap", "bootstrap_seed"):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(f"harness config: 'experiments.{key}' must be an integer >= 1")
    return cfg


# ---------------------------------------------------------------- one-variable check

def flatten(config: dict, prefix: str = "") -> dict[str, object]:
    """Nested dict -> {"chunking.chunk_size": 150, ...}. Lists and scalars are leaves."""
    flat: dict[str, object] = {}
    for key, value in config.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, dotted + "."))
        else:
            flat[dotted] = value
    return flat


def config_diff(base: dict, exp: dict, ignore: tuple[str, ...] | list[str]) -> list[str]:
    """Sorted dotted keys whose value differs, or that exist on one side only.

    `ignore` are TOP-LEVEL keys (name, experiment) that never count.
    """
    flat_base = flatten({k: v for k, v in base.items() if k not in ignore})
    flat_exp = flatten({k: v for k, v in exp.items() if k not in ignore})
    return sorted(key for key in set(flat_base) | set(flat_exp)
                  if key not in flat_base or key not in flat_exp or flat_base[key] != flat_exp[key])


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the top level must be a mapping")
    return data


def check_experiment(base_path, exp_path, ignore: tuple[str, ...] | list[str]) -> list[str]:
    """Problems with an experiment config, [] if it is ok.

    The config must carry an `experiment:` block with base, variable,
    changed_keys and hypothesis; `base` must be the file passed as base_path;
    and the real diff against the base must equal changed_keys exactly.
    """
    base_path, exp_path = resolve_path(base_path), resolve_path(exp_path)
    try:
        base, exp = _load_yaml(base_path), _load_yaml(exp_path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        return [str(error)]

    block = exp.get("experiment")
    if not isinstance(block, dict):
        return ["missing the 'experiment' block (base, variable, changed_keys, hypothesis)"]
    problems = []
    for key in BLOCK_KEYS:
        if key not in block:
            problems.append(f"experiment.{key} is missing")
    extra = set(block) - set(BLOCK_KEYS)
    if extra:
        problems.append(f"experiment has unknown keys {sorted(extra)}")
    if problems:
        return problems

    for key in ("variable", "hypothesis"):
        if not isinstance(block[key], str) or not block[key].strip():
            problems.append(f"experiment.{key} must be a non-empty string")
    changed = block["changed_keys"]
    if (not isinstance(changed, list) or not changed
            or not all(isinstance(k, str) and k for k in changed)):
        problems.append("experiment.changed_keys must be a non-empty list of dotted keys")
    if not isinstance(block["base"], str) or resolve_path(block["base"]) != base_path:
        problems.append(f"experiment.base is {block['base']!r}, not the base file {str(base_path)!r}")
    if problems:
        return problems

    actual = config_diff(base, exp, ignore)
    declared = sorted(set(changed))
    if actual != declared:
        undeclared = sorted(set(actual) - set(declared))
        unchanged = sorted(set(declared) - set(actual))
        if undeclared:
            problems.append(f"keys changed but not declared: {undeclared}")
        if unchanged:
            problems.append(f"keys declared but not changed: {unchanged}")
    return problems


# ---------------------------------------------------------------- statistics

def paired_flips(base_items: list[dict], exp_items: list[dict]) -> dict:
    """Split the queries by what happened to their ungrounded flag.

    Items are item_table rows. A query goes to `not_paired` when either side
    has no judged ungrounded value (judge failure, failed output). Returns
    {"improved", "regressed", "both_ungrounded", "both_grounded", "not_paired"}
    (sorted query-id lists) and "counts" with the same keys.
    """
    exp_by_id = {item["query_id"]: item for item in exp_items}
    groups = {"improved": [], "regressed": [], "both_ungrounded": [], "both_grounded": [],
              "not_paired": []}
    for item in base_items:
        b = item["ungrounded"]
        e = exp_by_id[item["query_id"]]["ungrounded"] if item["query_id"] in exp_by_id else None
        if b is None or e is None:
            groups["not_paired"].append(item["query_id"])
        elif b == 1 and e == 0:
            groups["improved"].append(item["query_id"])
        elif b == 0 and e == 1:
            groups["regressed"].append(item["query_id"])
        elif b == 1:
            groups["both_ungrounded"].append(item["query_id"])
        else:
            groups["both_grounded"].append(item["query_id"])
    groups = {name: sorted(ids) for name, ids in groups.items()}
    groups["counts"] = {name: len(ids) for name, ids in groups.items()}
    return groups


def mcnemar_exact(improved: int, regressed: int) -> float | None:
    """Two-sided exact p-value of the flip split; None when nothing flipped.

    mcnemar_exact(4, 1) == 0.375; mcnemar_exact(6, 0) == 0.03125.
    """
    flips = improved + regressed
    if flips == 0:
        return None
    return float(binomtest(improved, flips, 0.5, alternative="two-sided").pvalue)


def paired_bootstrap_ci(diffs: list[float], n_boot: int, seed: int, confidence: float) -> dict:
    """{"n", "mean", "ci_low", "ci_high"} of the mean of `diffs`, by seeded bootstrap.

    ci_low / ci_high are None when there are no differences.
    """
    n = len(diffs)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    values = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return {"n": n, "mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


# ---------------------------------------------------------------- one comparison

def _run_stats(items: list[dict], latency: dict[str, float | None], confidence: float) -> dict:
    """Per-run numbers for one table row, all from item_table rows and stored latency."""
    answerable = [i for i in items if i["answerable"]]
    unanswerable = [i for i in items if not i["answerable"]]
    return {
        "ungrounded": {
            "all": rate_summary([i["ungrounded"] for i in items], confidence),
            "answerable": rate_summary([i["ungrounded"] for i in answerable], confidence),
        },
        "bad_answer": rate_summary([i["bad"] for i in items], confidence),
        "irrelevant": rate_summary([i["irrelevant"] for i in items], confidence),
        "mean_faithfulness": mean_summary([i["faithfulness"] for i in items]),
        "false_refusals": sorted(i["query_id"] for i in answerable
                                 if i["irrelevant"] == 1 and i["refusal"] == 1),
        "correct_refusals": rate_summary([i["refusal"] for i in unanswerable], confidence),
        "in_context": rate_summary([i["in_context"] for i in answerable], confidence),
        "hit@4": rate_summary([i["hit@4"] for i in answerable], confidence),
        "mrr": mean_summary([i["reciprocal_rank"] for i in answerable]),
        "latency_ms": percentile_summary([latency.get(i["query_id"]) for i in items]),
    }


def _latency(conn, run_id: str, checks_version: str) -> dict[str, float | None]:
    return {r["query_id"]: r["value"] for r in store.get_scores(conn, run_id, checks_version)
            if r["metric"] == "latency_ms"}


def _faith_reasoning(conn, run_id: str, version: str) -> dict[str, str]:
    return {r["query_id"]: r["reasoning"] for r in store.get_scores(conn, run_id, version)
            if r["metric"] == "ungrounded"}


def compare_runs(conn, base_run_id: str, exp_run_id: str, versions: dict[str, str], cfg: dict,
                 *, allow_heldout: bool = False) -> dict:
    """One experiment row: `exp_run_id` against `base_run_id`.

    `cfg` holds n_bootstrap, bootstrap_seed (the experiments block) and
    confidence (checks.report_confidence). Raises ExperimentRefused or
    ScoringRefused unless: both runs completed and not held-out (unless
    allow_heldout); the runs differ; same split, test set and query ids; all
    three scorer versions have rows for both runs.
    """
    if base_run_id == exp_run_id:
        raise ExperimentRefused("the baseline and the experiment are the same run")
    base_run = get_scorable_run(conn, base_run_id, allow_heldout=allow_heldout)
    exp_run = get_scorable_run(conn, exp_run_id, allow_heldout=allow_heldout)
    for field, what in (("split", "split"), ("test_set_sha256", "test set (SHA-256)")):
        if base_run[field] != exp_run[field]:
            raise ExperimentRefused(f"different {what}: baseline {base_run[field]!r}, "
                                    f"experiment {exp_run[field]!r}")

    confidence = cfg["confidence"]
    base_items = item_table(conn, base_run_id, versions)
    exp_items = item_table(conn, exp_run_id, versions)
    if {i["query_id"] for i in base_items} != {i["query_id"] for i in exp_items}:
        raise ExperimentRefused("the two runs do not cover the same queries")

    flips = paired_flips(base_items, exp_items)
    base_stats = _run_stats(base_items, _latency(conn, base_run_id, versions["checks"]), confidence)
    exp_stats = _run_stats(exp_items, _latency(conn, exp_run_id, versions["checks"]), confidence)

    exp_by_id = {i["query_id"]: i for i in exp_items}
    diffs = [exp_by_id[i["query_id"]]["faithfulness"] - i["faithfulness"] for i in base_items
             if i["faithfulness"] is not None
             and exp_by_id[i["query_id"]]["faithfulness"] is not None]

    base_rate, exp_rate = base_stats["ungrounded"]["all"]["rate"], exp_stats["ungrounded"]["all"]["rate"]
    delta = None if base_rate is None or exp_rate is None else (exp_rate - base_rate) * 100

    # Flipped answers with the judge's reasoning, so a person can check the flip by eye.
    outputs = {run_id: {o["query_id"]: o for o in store.get_outputs_with_queries(conn, run_id)}
               for run_id in (base_run_id, exp_run_id)}
    reasoning = {run_id: _faith_reasoning(conn, run_id, versions["faithfulness"])
                 for run_id in (base_run_id, exp_run_id)}
    flipped = []
    for direction in ("improved", "regressed"):
        for query_id in flips[direction]:
            flipped.append({
                "direction": direction, "query_id": query_id,
                "type": exp_by_id[query_id]["type"],
                "question": outputs[exp_run_id][query_id]["question"],
                "answer_baseline": outputs[base_run_id][query_id]["answer"],
                "answer_experiment": outputs[exp_run_id][query_id]["answer"],
                "reasoning_baseline": reasoning[base_run_id].get(query_id),
                "reasoning_experiment": reasoning[exp_run_id].get(query_id),
            })

    base_config = json.loads(base_run["config_json"])
    exp_config = json.loads(exp_run["config_json"])
    block = exp_config.get("experiment") if isinstance(exp_config.get("experiment"), dict) else {}
    system = json.loads(exp_run["system_json"]) if exp_run["system_json"] else {}
    labels = {judge: ("calibrated judge"
                      if _calibration_status(conn, base_run_id, judge, versions[judge])["status"]
                      == "measured" else "uncalibrated judge")
              for judge in ("faithfulness", "relevance")}
    return {
        "kind": "experiment",
        "run_id": exp_run_id,
        "baseline_run_id": base_run_id,
        "config_path": exp_run["config_path"],
        "variable": block.get("variable"),
        "hypothesis": block.get("hypothesis"),
        "changed_keys": config_diff(base_config, exp_config, ["name", "experiment"]),
        "split": exp_run["split"],
        "git_commit": exp_run["git_commit"],
        "index_key": (system.get("index") or {}).get("index_key"),
        "n_outputs": len(exp_items), "n_failed": exp_run["n_failed"],
        "baseline_n_failed": base_run["n_failed"],
        "versions": versions, "confidence": confidence, "judge_label": labels,
        "baseline": base_stats, "experiment": exp_stats,
        "delta_points": delta,
        "flips": flips,
        "p_value": mcnemar_exact(flips["counts"]["improved"], flips["counts"]["regressed"]),
        "faithfulness_diff": paired_bootstrap_ci(diffs, cfg["n_bootstrap"], cfg["bootstrap_seed"],
                                                 confidence),
        "flipped_answers": flipped,
    }


# ---------------------------------------------------------------- formatting

def _verdict(row: dict) -> str:
    """Plain words for the p-value; 'significant' only when p < ALPHA."""
    p, c = row["p_value"], row["flips"]["counts"]
    if p is None:
        return "no query flipped"
    split = f"improved {c['improved']}, regressed {c['regressed']}"
    if p < ALPHA:
        return f"{split}; p = {p:.4f}, significant at {ALPHA}"
    return f"{split}; p = {p:.4f}, not statistically distinguishable from no change"


def _delta(row: dict) -> str:
    d = row["delta_points"]
    return "-" if d is None else f"{d:+.2f} pts"


def _faith_diff(row: dict) -> str:
    f = row["faithfulness_diff"]
    if f["mean"] is None:
        return "-"
    return f"{f['mean']:+.3f} [{f['ci_low']:+.3f}, {f['ci_high']:+.3f}] (n={f['n']})"


def _lat(stats: dict) -> str:
    lat = stats["latency_ms"]
    return "-" if lat["p50"] is None else f"p50 {lat['p50']:.0f} / p95 {lat['p95']:.0f} ms"


def _mrr(stats: dict) -> str:
    return "-" if stats["mrr"]["mean"] is None else f"{stats['mrr']['mean']:.3f}"


def format_table(rows: list[dict]) -> str:
    """Terminal text: one block per experiment."""
    lines = []
    for row in rows:
        base, exp = row["baseline"], row["experiment"]
        label = row["judge_label"]["faithfulness"]
        lines += [
            f"{row['variable'] or row['config_path']}   [{row['config_path']}]  run {row['run_id']}",
            f"  ungrounded ({label}):  baseline {_fmt_rate_pct(base['ungrounded']['all'])}",
            f"                        experiment {_fmt_rate_pct(exp['ungrounded']['all'])}  "
            f"({_delta(row)})",
            f"  flips:  {_verdict(row)}",
            f"  mean faithfulness difference (paired bootstrap): {_faith_diff(row)}",
            f"  bad answer:      {_fmt_rate_pct(base['bad_answer'])}  ->  "
            f"{_fmt_rate_pct(exp['bad_answer'])}",
            f"  false refusals:  {len(base['false_refusals'])}  ->  {len(exp['false_refusals'])}"
            f"   correct refusals: {_fmt_rate_pct(base['correct_refusals'])}  ->  "
            f"{_fmt_rate_pct(exp['correct_refusals'])}",
            f"  retrieval:       hit@4 {_fmt_rate_pct(base['hit@4'])}  ->  "
            f"{_fmt_rate_pct(exp['hit@4'])};  MRR {_mrr(base)} -> {_mrr(exp)}",
            f"  latency:         {_lat(base)}  ->  {_lat(exp)}",
            f"  failed outputs:  {row['baseline_n_failed']}  ->  {row['n_failed']};  "
            f"not paired: {row['flips']['counts']['not_paired']}",
            "",
        ]
    return "\n".join(lines)


def format_markdown(rows: list[dict]) -> str:
    """The experiment table (experiments/experiment_table.md)."""
    first = rows[0]
    lines = [
        "# Experiment table",
        "",
        f"Baseline run `{first['baseline_run_id']}`, split `{first['split']}`. Every run is scored "
        f"by the same versions: checks `{first['versions']['checks']}`, faithfulness "
        f"`{first['versions']['faithfulness']}`, relevance `{first['versions']['relevance']}`. "
        f"Judge numbers come from an **{first['judge_label']['faithfulness']}** "
        "(faithfulness) and an "
        f"**{first['judge_label']['relevance']}** (relevance).",
        "",
        "Intervals are 95% Wilson intervals. The p-value is an exact two-sided sign test on the "
        "queries whose ungrounded flag flipped. Each run is one sample of a cloud model; the "
        "`seed43` row shows how much the numbers move with no real change. About "
        f"{len(rows)} runs were compared on the same queries, so the best one looks better than "
        "it is; Step 12 confirms the chosen config once on held-out queries.",
        "",
        "| Variable | Run | Ungrounded (n) | Change | Improved / regressed | p | "
        "Bad answer | False refusals | hit@4 | MRR | Latency | Failed |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    base = first["baseline"]
    lines.append(
        f"| baseline | `{first['baseline_run_id']}` | {_fmt_rate_pct(base['ungrounded']['all'])} "
        f"| - | - | - | {_fmt_rate_pct(base['bad_answer'])} | {len(base['false_refusals'])} "
        f"| {_fmt_rate_pct(base['hit@4'])} | {_mrr(base)} | {_lat(base)} "
        f"| {first['baseline_n_failed']} |")
    for row in rows:
        exp, c = row["experiment"], row["flips"]["counts"]
        p = "-" if row["p_value"] is None else f"{row['p_value']:.4f}"
        lines.append(
            f"| {row['variable'] or row['config_path']} | `{row['run_id']}` "
            f"| {_fmt_rate_pct(exp['ungrounded']['all'])} | {_delta(row)} "
            f"| {c['improved']} / {c['regressed']} | {p} | {_fmt_rate_pct(exp['bad_answer'])} "
            f"| {len(exp['false_refusals'])} | {_fmt_rate_pct(exp['hit@4'])} | {_mrr(exp)} "
            f"| {_lat(exp)} | {row['n_failed']} |")

    lines += ["", "## Flipped answers", ""]
    for row in rows:
        lines += [f"### {row['variable'] or row['config_path']} (`{row['run_id']}`)", "",
                  f"- Hypothesis (written before the run): {row['hypothesis'] or '-'}",
                  f"- Result: {_verdict(row)}",
                  f"- Mean faithfulness difference: {_faith_diff(row)}",
                  f"- Changed keys: {', '.join(row['changed_keys']) or 'none'}",
                  f"- Not paired (a judge or output failed on either side): "
                  f"{', '.join(row['flips']['not_paired']) or 'none'}", ""]
        for item in row["flipped_answers"]:
            lines += [
                f"**{item['query_id']}** ({item['type']}): {item['direction']}",
                f"- Question: {item['question']}",
                f"- Baseline answer: {item['answer_baseline']}",
                f"- Experiment answer: {item['answer_experiment']}",
                f"- Judge reasoning, baseline: {item['reasoning_baseline']}",
                f"- Judge reasoning, experiment: {item['reasoning_experiment']}",
                "",
            ]
    return "\n".join(lines)


def log_row(row: dict, created_at: str) -> str:
    """One experiments/log.md row; every number comes from the row."""
    exp = row["experiment"]["ungrounded"]["all"]
    rate = (f"{_fmt_rate_pct(exp)}, {row['judge_label']['faithfulness']}"
            if exp["rate"] is not None else "-")
    c = row["flips"]["counts"]
    p = "no flips" if row["p_value"] is None else f"p = {row['p_value']:.4f}"
    note = (f"{_delta(row)} vs baseline; improved {c['improved']}, regressed {c['regressed']}, "
            f"{p}; false refusals {len(row['experiment']['false_refusals'])}")
    return (f"| {created_at[:10]} | `{row['config_path']}` | {row['variable'] or '-'} "
            f"| `{row['run_id']}` | {rate} | {note} |")


# ---------------------------------------------------------------- CLI

def _cmd_check(args, harness: dict, cfg: dict) -> int:
    base = args.base or cfg["base_config"]
    problems = check_experiment(base, args.config, cfg["ignore_keys"])
    if problems:
        for problem in problems:
            print(f"error: {args.config}: {problem}", file=sys.stderr)
        return 1
    print(f"ok: {args.config} differs from {base} only in its declared keys")
    return 0


def _cmd_compare(args, harness: dict, cfg: dict) -> int:
    def refuse(problem: str) -> int:
        print(f"error: {problem}", file=sys.stderr)
        return 1

    markdown_path = resolve_path(args.markdown) if args.markdown else None
    log_path = resolve_path(args.log) if args.log else None
    if markdown_path is not None and markdown_path.exists() and not args.force:
        return refuse(f"{markdown_path} exists; pass --force to overwrite it")
    if log_path is not None:
        if not log_path.exists():
            return refuse(f"{log_path} does not exist")
        text = log_path.read_text(encoding="utf-8")
        for run_id in args.run:
            if f"`{run_id}`" in text:
                return refuse(f"{log_path} already has a row for run {run_id}")

    compare_cfg = {"n_bootstrap": cfg["n_bootstrap"], "bootstrap_seed": cfg["bootstrap_seed"],
                   "confidence": load_checks_config(harness)["report_confidence"]}
    versions = current_versions(harness)
    with closing(store.connect(harness["db_path"])) as conn:
        rows = []
        for run_id in args.run:   # every comparison is computed before anything is written
            try:
                rows.append(compare_runs(conn, args.baseline, run_id, versions, compare_cfg,
                                         allow_heldout=args.allow_heldout))
            except (ExperimentRefused, ScoringRefused, ValueError) as error:
                return refuse(str(error))
        created_at = store._now()
        ids = [store.write_run_report(conn, {"run_id": row["run_id"], "kind": "experiment",
                                             "versions": versions, "report": row,
                                             "created_at": created_at}) for row in rows]

    print(format_table(rows))
    print(f"stored as run_reports ids {', '.join(map(str, ids))}")
    if markdown_path is not None:
        markdown_path.write_text(format_markdown(rows), encoding="utf-8")
        print(f"wrote {markdown_path}")
    if log_path is not None:
        text = log_path.read_text(encoding="utf-8")
        new_rows = "\n".join(log_row(row, created_at) for row in rows)
        log_path.write_text(text + ("" if text.endswith("\n") else "\n") + new_rows + "\n",
                            encoding="utf-8")
        print(f"appended {len(rows)} rows to {log_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evalharness.experiments",
                                     description="Check and compare experiment runs.")
    parser.add_argument("--harness-config", default=DEFAULT_HARNESS_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="an experiment config differs from the base only in "
                                         "its declared keys")
    check.add_argument("--config", required=True)
    check.add_argument("--base", help="base config (default: experiments.base_config)")

    compare = sub.add_parser("compare", help="compare experiment runs with the baseline run")
    compare.add_argument("--baseline", required=True, help="baseline run id")
    compare.add_argument("--run", required=True, action="append", help="experiment run id "
                                                                      "(repeat for more)")
    compare.add_argument("--markdown", help="write the table as markdown to this path")
    compare.add_argument("--force", action="store_true", help="overwrite an existing --markdown file")
    compare.add_argument("--log", help="append one row per run to this experiment log")
    compare.add_argument("--allow-heldout", action="store_true",
                         help="allow held-out runs (Step 12 only)")
    args = parser.parse_args(argv)

    try:
        harness = load_harness_config(args.harness_config)
        cfg = load_experiments_config(harness)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return (_cmd_check if args.command == "check" else _cmd_compare)(args, harness, cfg)


if __name__ == "__main__":
    sys.exit(main())
