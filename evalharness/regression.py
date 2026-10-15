"""Regression comparison and flagging for Step 13.

Compares a new dev run with the last ACCEPTED run, flags answers for human
review and prints one PASS / FAIL verdict from a gate. No LLM call: every
number is read from stored scores (Steps 6-8) and answers (Step 5).

Flag reasons (plain yes/no tests on stored values, settings in the
`regression` block of configs/harness.yaml):
    ungrounded       the stored `ungrounded` flag is 1
                     (faithfulness below the judge threshold, or a contradicted claim)
    low_relevance    relevance value below `relevance_below`
    judges_disagree  a deterministic check and a judge contradict each other about
                     whether the answer refused (rules refusal_vs_relevance_a / _b)
    answer_changed   word-level similarity to the accepted run's answer is below
                     `answer_similarity_below`
    not_judged       an output or a judge failed, so nothing can be said (never hidden)

Gate: delta = (ungrounded rate of the new run - of the accepted run) * 100, in
points. FAIL when delta > max_ungrounded_rise_points. One answer is 100 / n
points, so a 3-point gate trips on a few answers; the paired flips and the
exact p-value are printed beside the verdict.

The accepted run is a human decision: `accept` is only run by Gaurav's command,
and `check` never accepts a run by itself.

CLI (python -m evalharness.regression ...):
    accept     --run-id ID --note "why"
    check      --run-id ID [--against ID] [--markdown PATH [--force]] [--log PATH] [--no-store]
    flags      --run-id ID
    thresholds --run-id ID [--against ID]
Exit codes: 0 gate passed; 1 gate failed; 2 refused or cannot decide
(nothing is written on exit 2).
"""

import argparse
import difflib
import hashlib
import json
import re
import sys
from collections import Counter
from contextlib import closing
from fractions import Fraction

from evalharness import store
from evalharness.checks import ScoringRefused, get_scorable_run, load_checks_config
from evalharness.experiments import (_latency, _run_stats, load_experiments_config, mcnemar_exact,
                                     paired_bootstrap_ci, paired_flips)
from evalharness.metrics import rate_summary, wilson_interval
from evalharness.report import _calibration_status, _fmt_rate_pct, current_versions, item_table
from evalharness.runner import DEFAULT_HARNESS_CONFIG, load_harness_config
from evalharness.testset import resolve_path

CODE_VERSION = 1   # bump when the flag rules in flag_reasons change
REGRESSION_KEYS = {"relevance_below", "answer_similarity_below", "similarity_ignore_pattern",
                   "disagreement_rules", "max_ungrounded_rise_points", "labeler"}
DISAGREEMENT_RULES = ("refusal_vs_relevance_a", "refusal_vs_relevance_b")
EXCERPT_CHARS = 300   # display only: how much of an answer the terminal / markdown shows
NOISE_NOTE = ("One answer moves the rate by 100 / n points. The Step 11 noise control "
              "(same config, generation.seed 42 -> 43) shows how far the rate moves with no "
              "real change; see experiments/experiment_table.md. The verdict is a point "
              "estimate: read the paired flips and p beside it.")


class RegressionRefused(Exception):
    """The runs cannot be compared, or a command is not allowed."""


# ---------------------------------------------------------------- config

def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_regression_config(harness: dict) -> dict:
    """The checked `regression` block plus its "flag_version".

    flag_version = "flags-<CODE_VERSION>-<first 8 hex of sha256>" over the settings that
    decide which answers are flagged. The gate value and labeler are NOT part of it:
    changing the gate does not change which answers are flagged. ValueError names the problem.
    """
    cfg = harness.get("regression")
    if not isinstance(cfg, dict):
        raise ValueError("harness config: 'regression' must be a mapping")
    missing, extra = REGRESSION_KEYS - set(cfg), set(cfg) - REGRESSION_KEYS
    if missing or extra:
        raise ValueError(f"harness config: regression keys missing {sorted(missing)}, "
                         f"unknown {sorted(extra)}")
    for key in ("relevance_below", "answer_similarity_below"):
        if not _is_number(cfg[key]) or not 0 <= cfg[key] <= 1:
            raise ValueError(f"harness config: 'regression.{key}' must be a number in [0, 1]")
    if not _is_number(cfg["max_ungrounded_rise_points"]) or cfg["max_ungrounded_rise_points"] < 0:
        raise ValueError("harness config: 'regression.max_ungrounded_rise_points' must be a "
                         "number >= 0")
    if not isinstance(cfg["similarity_ignore_pattern"], str):
        raise ValueError("harness config: 'regression.similarity_ignore_pattern' must be a string")
    try:
        re.compile(cfg["similarity_ignore_pattern"])
    except re.error as error:
        raise ValueError(f"harness config: 'regression.similarity_ignore_pattern': {error}") from error
    rules = cfg["disagreement_rules"]
    if not isinstance(rules, list) or any(r not in DISAGREEMENT_RULES for r in rules):
        raise ValueError(f"harness config: 'regression.disagreement_rules' must be a list of "
                         f"{list(DISAGREEMENT_RULES)}")
    if not isinstance(cfg["labeler"], str) or not cfg["labeler"].strip():
        raise ValueError("harness config: 'regression.labeler' must be a non-empty string")

    settings = {key: cfg[key] for key in ("relevance_below", "answer_similarity_below",
                                          "similarity_ignore_pattern")}
    settings["disagreement_rules"] = sorted(rules)
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]
    return {**cfg, "flag_version": f"flags-{CODE_VERSION}-{digest}"}


# ---------------------------------------------------------------- pure rules

def _words(text: str, ignore_pattern: str) -> list[str]:
    cleaned = re.sub(ignore_pattern, " ", text) if ignore_pattern else text
    return cleaned.lower().split()


def answer_similarity(old: str, new: str, ignore_pattern: str) -> float:
    """Word-level SequenceMatcher ratio in [0, 1]: 2 * matching words / total words.

    Example: "meal bills cannot be claimed" vs "... claimed separately" is 10 / 11.
    Both empty is 1.0, one empty is 0.0.
    """
    a, b = _words(old, ignore_pattern), _words(new, ignore_pattern)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def flag_reasons(item: dict, similarity: float | None, cfg: dict) -> list[str]:
    """Sorted flag reasons for one answer (`item` is a report.item_table row).

    `similarity` is the word similarity to the accepted run's answer, or None when
    there is nothing to compare with (no accepted answer, or this is not a comparison).
    A failed output or judge failure gives ["not_judged"] only.
    """
    if item["faith_state"] != "judged" or item["rel_state"] != "judged":
        return ["not_judged"]
    reasons = []
    if item["ungrounded"] == 1:
        reasons.append("ungrounded")
    relevance = item["relevance"]
    if relevance is not None and relevance < cfg["relevance_below"]:
        reasons.append("low_relevance")
    refusal = item["refusal"]
    rules = cfg["disagreement_rules"]
    if relevance == 1.0 and refusal is not None:
        if "refusal_vs_relevance_a" in rules and item["answerable"] and refusal == 1:
            reasons.append("judges_disagree")
        if "refusal_vs_relevance_b" in rules and not item["answerable"] and refusal == 0:
            reasons.append("judges_disagree")
    if similarity is not None and similarity < cfg["answer_similarity_below"]:
        reasons.append("answer_changed")
    return sorted(set(reasons))


# ---------------------------------------------------------------- accepted run

def _require_complete(items: list[dict], what: str) -> None:
    """Raise RegressionRefused when any answer is failed or not fully judged."""
    bad = sorted(i["query_id"] for i in items
                 if i["faith_state"] != "judged" or i["rel_state"] != "judged")
    if bad:
        raise RegressionRefused(f"{what} has {len(bad)} answers that failed or were not judged "
                                f"({', '.join(bad[:8])}{'...' if len(bad) > 8 else ''}); a gate "
                                "needs complete data")


def get_accepted_run(conn, split: str) -> dict | None:
    """The newest accepted_runs row for the split, or None."""
    rows = store.get_accepted_runs(conn, split)
    return rows[0] if rows else None


def accept_run(conn, run_id: str, note: str, harness: dict) -> int:
    """Record run_id as the accepted reference run of its split; returns the row id."""
    cfg = load_regression_config(harness)
    run = get_scorable_run(conn, run_id)   # unknown / not completed / held-out are refused
    if run["split"] != "dev":
        raise RegressionRefused(f"only dev runs can be accepted, run {run_id} is {run['split']!r}")
    if run["git_dirty"]:
        raise RegressionRefused(f"run {run_id} was made on a dirty working tree")
    if not note.strip():
        raise RegressionRefused("a note is required: why is this run the reference?")
    _require_complete(item_table(conn, run_id, current_versions(harness)), f"run {run_id}")
    current = get_accepted_run(conn, "dev")
    if current is not None and current["run_id"] == run_id:
        raise RegressionRefused(f"run {run_id} is already the accepted dev run")
    return store.write_accepted_run(conn, run_id=run_id, split="dev",
                                    accepted_by=cfg["labeler"], note=note.strip())


# ---------------------------------------------------------------- the report

def _check_comparable(prev_run: dict, new_run: dict, prev_items: list[dict],
                      new_items: list[dict]) -> None:
    for field, what in (("split", "split"), ("test_set_sha256", "test set (SHA-256)")):
        if prev_run[field] != new_run[field]:
            raise RegressionRefused(f"different {what}: accepted {prev_run[field]!r}, "
                                    f"new {new_run[field]!r}")
    if {i["query_id"] for i in prev_items} != {i["query_id"] for i in new_items}:
        raise RegressionRefused("the two runs do not cover the same queries")


def build_regression_report(conn, run_id: str, against_run_id: str | None, harness: dict) -> dict:
    """Compare run_id with against_run_id (default: the accepted dev run). Pure read.

    Raises RegressionRefused or ScoringRefused when the comparison is not possible.
    """
    cfg = load_regression_config(harness)
    exp_cfg = load_experiments_config(harness)
    confidence = load_checks_config(harness)["report_confidence"]
    versions = current_versions(harness)

    new_run = get_scorable_run(conn, run_id)
    if new_run["split"] != "dev":
        raise RegressionRefused(f"only dev runs are compared, run {run_id} is {new_run['split']!r}")
    accepted = get_accepted_run(conn, new_run["split"])
    if against_run_id is None:
        if accepted is None:
            raise RegressionRefused("no accepted run: record one with "
                                    "`python -m evalharness.regression accept --run-id ...`")
        against_run_id = accepted["run_id"]
    prev_run = get_scorable_run(conn, against_run_id)

    new_items = item_table(conn, run_id, versions)
    prev_items = item_table(conn, against_run_id, versions)
    _check_comparable(prev_run, new_run, prev_items, new_items)
    _require_complete(new_items, f"run {run_id}")
    _require_complete(prev_items, f"reference run {against_run_id}")

    new_outputs = {o["query_id"]: o for o in store.get_outputs_with_queries(conn, run_id)}
    prev_outputs = {o["query_id"]: o for o in store.get_outputs_with_queries(conn, against_run_id)}
    reasoning = {r["query_id"]: r["reasoning"]
                 for r in store.get_scores(conn, run_id, versions["faithfulness"])
                 if r["metric"] == "ungrounded"}
    prev_by_id = {i["query_id"]: i for i in prev_items}

    flagged, n_flagged_by_reason, flag_flags = [], Counter(), []
    for item in new_items:
        q = item["query_id"]
        new_answer, old_answer = new_outputs[q]["answer"], prev_outputs[q]["answer"]
        similarity = (None if old_answer is None or new_answer is None else
                      answer_similarity(old_answer, new_answer, cfg["similarity_ignore_pattern"]))
        reasons = flag_reasons(item, similarity, cfg)
        flag_flags.append(1 if reasons else 0)
        if not reasons:
            continue
        n_flagged_by_reason.update(reasons)
        flagged.append({
            "query_id": q, "type": item["type"], "reasons": reasons, "similarity": similarity,
            "question": new_outputs[q]["question"], "answer": new_answer,
            "previous_answer": old_answer,
            "chunk_ids": [c["chunk_id"] for c in new_outputs[q]["retrieved_chunks"] or []],
            "judge_reasoning": reasoning.get(q),
        })

    # Gate: compare with exact fractions so "exactly 3.0 points" is not a float accident.
    new_rate = rate_summary([i["ungrounded"] for i in new_items], confidence)
    prev_rate = rate_summary([i["ungrounded"] for i in prev_items], confidence)
    diff = (Fraction(new_rate["successes"], new_rate["n"])
            - Fraction(prev_rate["successes"], prev_rate["n"]))
    gate_points = Fraction(str(cfg["max_ungrounded_rise_points"]))
    verdict = "FAIL" if diff * 100 > gate_points else "PASS"

    flips = paired_flips(prev_items, new_items)
    new_by_id = {i["query_id"]: i for i in new_items}
    faith_diffs = [new_by_id[q]["faithfulness"] - i["faithfulness"] for q, i in prev_by_id.items()
                   if i["faithfulness"] is not None and new_by_id[q]["faithfulness"] is not None]
    counts = flips["counts"]

    calibration = {}
    for judge in ("faithfulness", "relevance"):
        status = {"status": "not calibrated"}
        for rid in (against_run_id, run_id):
            found = _calibration_status(conn, rid, judge, versions[judge])
            if found["status"] != "not calibrated":
                status = found
                break
        calibration[judge] = status
    judge_label = {j: ("calibrated judge" if c["status"] == "measured" else "uncalibrated judge")
                   for j, c in calibration.items()}

    return {
        "kind": "regression",
        "run_id": run_id, "against_run_id": against_run_id,
        "accepted_run_id": accepted["run_id"] if accepted else None,
        "split": new_run["split"], "git_commit": new_run["git_commit"],
        "config_path": new_run["config_path"],
        "flag_version": cfg["flag_version"], "versions": versions, "confidence": confidence,
        "n": len(new_items),
        "flag_rate": rate_summary(flag_flags, confidence),
        "flag_reason_counts": dict(sorted(n_flagged_by_reason.items())),
        "flag_type_counts": dict(sorted(Counter(f["type"] for f in flagged).items())),
        "flagged": flagged,
        "gate": {
            "verdict": verdict, "max_points": cfg["max_ungrounded_rise_points"],
            "accepted_rate": prev_rate, "new_rate": new_rate,
            "delta_points": float(diff * 100),
            "improved": counts["improved"], "regressed": counts["regressed"],
            "p_value": mcnemar_exact(counts["improved"], counts["regressed"]),
            "faithfulness_diff": paired_bootstrap_ci(faith_diffs, exp_cfg["n_bootstrap"],
                                                     exp_cfg["bootstrap_seed"], confidence),
        },
        "secondary": {
            "accepted": _run_stats(prev_items, _latency(conn, against_run_id, versions["checks"]),
                                   confidence),
            "new": _run_stats(new_items, _latency(conn, run_id, versions["checks"]), confidence),
        },
        "judge_label": judge_label, "calibration": calibration, "noise_note": NOISE_NOTE,
    }


# ---------------------------------------------------------------- storing flags

def store_flags(conn, report: dict, *, replace: bool = False) -> str:
    """Write the `reviews` rows and one `run_reports` row (kind "regression"); "written" or "unchanged".

    Same run, flag version and reference run as a stored report: "unchanged" (the
    flagged ids must match, otherwise the stored data changed under one version and
    we refuse). A different flag version or reference needs replace=True, which first
    deletes this run's review rows that have no human_label. human_label and notes are
    never written here.
    """
    run_id, flags = report["run_id"], {f["query_id"]: f["reasons"] for f in report["flagged"]}
    stored = [r for r in store.get_run_reports(conn, run_id) if r["kind"] == "regression"]
    same = [r for r in stored if r["report"]["flag_version"] == report["flag_version"]
            and r["report"]["against_run_id"] == report["against_run_id"]]
    if same:
        old_flags = {f["query_id"]: f["reasons"] for f in same[-1]["report"]["flagged"]}
        if old_flags != flags:
            raise RegressionRefused("stored flags for this run, flag version and reference run "
                                    "differ from the recomputed ones; refusing to overwrite")
        return "unchanged"
    if stored and not replace:
        raise RegressionRefused("run already has flags under another flag version or reference "
                                "run; pass --replace-flags to replace the unlabelled ones")
    if stored:
        store.delete_unlabelled_flag_rows(conn, run_id)
    store.write_flag_rows(conn, run_id, flags)
    store.write_run_report(conn, {"run_id": run_id, "kind": "regression",
                                  "versions": report["versions"], "report": report})
    return "written"


# ---------------------------------------------------------------- flags vs human labels

def _ratio(successes: int, n: int, confidence: float) -> dict:
    if n == 0:
        return {"successes": 0, "n": 0, "rate": None, "ci_low": None, "ci_high": None}
    low, high = wilson_interval(successes, n, confidence)
    return {"successes": successes, "n": n, "rate": successes / n, "ci_low": low, "ci_high": high}


def flag_quality(conn, run_id: str, harness: dict, against_run_id: str | None = None) -> dict:
    """Flags of this run against Gaurav's calibration labels (positive class: ungrounded).

    Flags are recomputed in memory (`answer_changed` only when against_run_id is given).
    Only labels from the configured labeler count; unsure and unlabelled answers are
    counted, not folded in. precision / recall are None-rate (0/0) when a denominator is 0.
    """
    cfg = load_regression_config(harness)
    confidence = load_checks_config(harness)["report_confidence"]
    versions = current_versions(harness)
    run = get_scorable_run(conn, run_id)
    if run["split"] != "dev":
        raise RegressionRefused(f"only dev runs are evaluated, run {run_id} is {run['split']!r}")
    items = {i["query_id"]: i for i in item_table(conn, run_id, versions)}
    outputs = {o["query_id"]: o["answer"] for o in store.get_outputs_with_queries(conn, run_id)}
    previous = {}
    if against_run_id is not None:
        previous = {o["query_id"]: o["answer"]
                    for o in store.get_outputs_with_queries(conn, against_run_id)}

    tp = fp = fn = tn = unsure = unlabelled = other_labeler = 0
    for row in store.get_calibration_labels(conn, run_id):
        q = row["query_id"]
        if row["grounded_label"] is None:
            unlabelled += 1
            continue
        if row["labeler"] != cfg["labeler"]:
            other_labeler += 1
            continue
        if row["grounded_label"] == "unsure":
            unsure += 1
            continue
        item = items.get(q)
        if item is None:
            continue
        similarity = None
        if q in previous and previous[q] is not None and outputs.get(q) is not None:
            similarity = answer_similarity(previous[q], outputs[q], cfg["similarity_ignore_pattern"])
        flagged = bool(flag_reasons(item, similarity, cfg))
        human_ungrounded = row["grounded_label"] == "ungrounded"
        if flagged and human_ungrounded:
            tp += 1
        elif flagged:
            fp += 1
        elif human_ungrounded:
            fn += 1
        else:
            tn += 1
    return {"run_id": run_id, "flag_version": cfg["flag_version"], "n": tp + fp + fn + tn,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "unsure": unsure, "unlabelled": unlabelled,
            "other_labeler": other_labeler,
            "precision": _ratio(tp, tp + fp, confidence), "recall": _ratio(tp, tp + fn, confidence),
            "confidence": confidence}


# ---------------------------------------------------------------- formatting

def _excerpt(text: str | None) -> str:
    if text is None:
        return "-"
    text = " ".join(text.split())
    return text if len(text) <= EXCERPT_CHARS else text[:EXCERPT_CHARS] + "..."


def _p(value: float | None) -> str:
    return "no query flipped" if value is None else f"exact paired p = {value:.4f}"


def _fmt_ratio(s: dict) -> str:
    if s["rate"] is None:
        return "undefined (0/0)"
    return f"{s['successes']}/{s['n']} = {s['rate']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]"


def format_report(report: dict) -> str:
    """Terminal text: verdict first, then the flagged list."""
    g, label = report["gate"], report["judge_label"]["faithfulness"]
    pct = round(report["confidence"] * 100)
    lines = [
        f"Regression check: run {report['run_id']} against {report['against_run_id']} "
        f"({report['split']}, {report['n']} queries, flag version {report['flag_version']})",
        f"{g['verdict']}  ungrounded rate {g['delta_points']:+.2f} points "
        f"(gate: fail above +{g['max_points']:.2f})   {label}",
        f"  accepted {_fmt_rate_pct(g['accepted_rate'])}",
        f"  new      {_fmt_rate_pct(g['new_rate'])}",
        f"  paired: improved {g['improved']}, regressed {g['regressed']}, {_p(g['p_value'])}",
        "",
        f"flag rate {_fmt_rate_pct(report['flag_rate'])} [{pct}% Wilson CI]",
        "  by reason: " + (", ".join(f"{k} {v}" for k, v in report["flag_reason_counts"].items())
                           or "none"),
        "  by type:   " + (", ".join(f"{k} {v}" for k, v in report["flag_type_counts"].items())
                           or "none"),
        "",
        f"flagged answers ({len(report['flagged'])}):",
    ]
    for f in report["flagged"]:
        sim = "" if f["similarity"] is None else f"  similarity {f['similarity']:.2f}"
        lines += [
            f"  {f['query_id']} ({f['type']}): {', '.join(f['reasons'])}{sim}",
            f"    question: {f['question']}",
            f"    answer:   {_excerpt(f['answer'])}",
            f"    judge:    {f['judge_reasoning']}",
        ]
    if not report["flagged"]:
        lines.append("  none")
    lines += ["", "note: " + report["noise_note"],
              f"judges: faithfulness {report['judge_label']['faithfulness']}, "
              f"relevance {report['judge_label']['relevance']}: the flags are the judges' "
              "measurement, not verified labels"]
    return "\n".join(lines)


def format_markdown(report: dict) -> str:
    """The report as markdown (experiments/regression_latest.md)."""
    g = report["gate"]
    lines = [
        "# Regression check",
        "",
        f"Run `{report['run_id']}` against `{report['against_run_id']}` "
        f"({report['split']}, {report['n']} queries, flag version `{report['flag_version']}`). "
        f"Judge numbers come from an **{report['judge_label']['faithfulness']}** "
        f"(faithfulness) and an **{report['judge_label']['relevance']}** (relevance).",
        "",
        f"## Verdict: {g['verdict']}",
        "",
        f"Ungrounded rate {g['delta_points']:+.2f} points (gate: fail above "
        f"+{g['max_points']:.2f}).",
        "",
        "| | Ungrounded |",
        "|---|---|",
        f"| accepted | {_fmt_rate_pct(g['accepted_rate'])} |",
        f"| new | {_fmt_rate_pct(g['new_rate'])} |",
        "",
        f"Paired: improved {g['improved']}, regressed {g['regressed']}, {_p(g['p_value'])}.",
        "",
        f"## Flag rate: {_fmt_rate_pct(report['flag_rate'])}",
        "",
        "By reason: " + (", ".join(f"{k} {v}" for k, v in report["flag_reason_counts"].items())
                         or "none"),
        "",
        "## Flagged answers",
    ]
    for f in report["flagged"]:
        lines += ["", f"**{f['query_id']}** ({f['type']}): {', '.join(f['reasons'])}",
                  f"- Question: {f['question']}",
                  f"- Answer: {_excerpt(f['answer'])}",
                  f"- Judge reasoning: {f['judge_reasoning']}"]
    if not report["flagged"]:
        lines += ["", "None."]
    lines += ["", "## Note", "", report["noise_note"], ""]
    return "\n".join(lines)


def log_row(report: dict, created_at: str) -> str:
    """One experiments/log.md row; every number comes from the report."""
    g = report["gate"]
    rate = f"{_fmt_rate_pct(g['new_rate'])}, {report['judge_label']['faithfulness']}"
    note = (f"regression check {g['verdict']}: {g['delta_points']:+.2f} pts vs accepted "
            f"`{report['against_run_id']}`; flag rate {_fmt_rate_pct(report['flag_rate'])}")
    return (f"| {created_at[:10]} | `{report['config_path']}` | regression check "
            f"| `{report['run_id']}` | {rate} | {note} |")


def format_flag_quality(result: dict) -> str:
    lines = [
        f"Flag quality for run {result['run_id']} (flag version {result['flag_version']}; "
        "positive class = ungrounded, labels from Gaurav only)",
        f"  labelled answers used: {result['n']} (TP {result['tp']}, FP {result['fp']}, "
        f"FN {result['fn']}, TN {result['tn']})",
        f"  left out: {result['unsure']} unsure, {result['unlabelled']} unlabelled, "
        f"{result['other_labeler']} from another labeler",
        f"  flag precision: {_fmt_ratio(result['precision'])}",
        f"  flag recall:    {_fmt_ratio(result['recall'])}",
    ]
    if result["tp"] + result["fn"] == 0:
        lines.append("  no human 'ungrounded' label exists, so recall is undefined and "
                     "thresholds cannot be tuned yet")
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI

def _refuse(problem: str) -> int:
    print(f"error: {problem}", file=sys.stderr)
    return 2


def _cmd_accept(args, harness: dict) -> int:
    with closing(store.connect(harness["db_path"])) as conn:
        try:
            row_id = accept_run(conn, args.run_id, args.note, harness)
        except (RegressionRefused, ScoringRefused, ValueError) as error:
            return _refuse(str(error))
    print(f"accepted run {args.run_id} as the reference dev run (accepted_runs id {row_id})")
    return 0


def _cmd_check(args, harness: dict) -> int:
    markdown_path = resolve_path(args.markdown) if args.markdown else None
    log_path = resolve_path(args.log) if args.log else None
    if markdown_path is not None and markdown_path.exists() and not args.force:
        return _refuse(f"{markdown_path} exists; pass --force to overwrite it")
    if log_path is not None:
        if not log_path.exists():
            return _refuse(f"{log_path} does not exist")
        if f"`{args.run_id}`" in log_path.read_text(encoding="utf-8"):
            return _refuse(f"{log_path} already has a row for run {args.run_id}")

    with closing(store.connect(harness["db_path"])) as conn:
        try:   # the report is built before anything is written
            report = build_regression_report(conn, args.run_id, args.against, harness)
        except (RegressionRefused, ScoringRefused, ValueError) as error:
            return _refuse(str(error))
        outcome = None
        if not args.no_store:
            try:
                outcome = store_flags(conn, report, replace=args.replace_flags)
            except RegressionRefused as error:
                return _refuse(str(error))
    print(format_report(report))
    if outcome is not None:
        print(f"\nflags {outcome}: {len(report['flagged'])} reviews rows for run {args.run_id}")
    if markdown_path is not None:
        markdown_path.write_text(format_markdown(report), encoding="utf-8")
        print(f"wrote {markdown_path}")
    if log_path is not None:
        text = log_path.read_text(encoding="utf-8")
        log_path.write_text(text + ("" if text.endswith("\n") else "\n")
                            + log_row(report, store._now()) + "\n", encoding="utf-8")
        print(f"appended a row to {log_path}")
    return 0 if report["gate"]["verdict"] == "PASS" else 1


def _cmd_flags(args, harness: dict) -> int:
    with closing(store.connect(harness["db_path"])) as conn:
        rows = store.get_flag_rows(conn, args.run_id)
    if not rows:
        print(f"no stored flags for run {args.run_id}")
        return 0
    for row in rows:
        label = row["human_label"] or "-"
        print(f"{row['query_id']}  {', '.join(row['flag_reasons'])}  human_label: {label}")
    print(f"{len(rows)} flagged answers")
    return 0


def _cmd_thresholds(args, harness: dict) -> int:
    with closing(store.connect(harness["db_path"])) as conn:
        try:
            result = flag_quality(conn, args.run_id, harness, args.against)
        except (RegressionRefused, ScoringRefused, ValueError) as error:
            return _refuse(str(error))
    print(format_flag_quality(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evalharness.regression",
                                     description="Compare a dev run with the accepted run, "
                                                 "flag answers, gate.")
    parser.add_argument("--harness-config", default=DEFAULT_HARNESS_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    accept = sub.add_parser("accept", help="record a dev run as the accepted reference run")
    accept.add_argument("--run-id", required=True)
    accept.add_argument("--note", required=True, help="why this run is the reference")

    check = sub.add_parser("check", help="compare a run with the accepted run; print flags and verdict")
    check.add_argument("--run-id", required=True)
    check.add_argument("--against", help="reference run id (default: the accepted dev run)")
    check.add_argument("--markdown", help="write the report as markdown to this path")
    check.add_argument("--force", action="store_true", help="overwrite an existing --markdown file")
    check.add_argument("--log", help="append one row to this experiment log")
    check.add_argument("--no-store", action="store_true", help="do not write reviews / run_reports")
    check.add_argument("--replace-flags", action="store_true",
                       help="replace flags stored under another flag version (labelled rows stay)")

    flags = sub.add_parser("flags", help="list the stored flags of a run")
    flags.add_argument("--run-id", required=True)

    thresholds = sub.add_parser("thresholds", help="flag precision / recall against Gaurav's labels")
    thresholds.add_argument("--run-id", required=True, help="the run the labels were made on")
    thresholds.add_argument("--against", help="reference run, to include answer_changed flags")
    args = parser.parse_args(argv)

    try:
        harness = load_harness_config(args.harness_config)
        load_regression_config(harness)
    except ValueError as error:
        return _refuse(str(error))
    handler = {"accept": _cmd_accept, "check": _cmd_check, "flags": _cmd_flags,
               "thresholds": _cmd_thresholds}[args.command]
    return handler(args, harness)


if __name__ == "__main__":
    sys.exit(main())
