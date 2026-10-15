"""Judge calibration (Step 9): Gaurav's labels vs the LLM judges.

1. `label`: a blind terminal tool. It shows each stored answer of a run with the
   exact chunks the generator saw (never the judge's verdict, claims or scores,
   and never the reference answer) and stores Gaurav's label: grounded /
   ungrounded / unsure, plus a 0/1/2 relevance label if configured.
   On the first session every answer is assigned once to a fixed `tune` or
   `check` half (seeded, stratified by query type; answers whose verdicts were
   already seen are forced into `tune`).
2. `compare`: grades a judge version against the labels on one half, treating
   "ungrounded" (or "irrelevant") as the positive class:
       agreement = (TP + TN) / n     precision = TP / (TP + FP)
       recall    = TP / (TP + FN)    kappa = (p_o - p_e) / (1 - p_e)
   Rates get Wilson intervals, kappa a seeded bootstrap interval; 0/0 is
   reported as undefined, never as 0 or 1. Every comparison is logged in
   calibration_reports, so the number of looks at `check` is visible.

The prompt is improved on `tune` disagreements only; the chosen version is
reported once on `check`. This module makes no LLM call and never writes a
label by itself: labels come only from Gaurav's keystrokes.

CLI:
    python -m evalharness.calibration label   --run-id <id> [--relabel q057,q058]
    python -m evalharness.calibration status  --run-id <id> [--ids tune|check]
    python -m evalharness.calibration compare --run-id <id> --judge faithfulness --version <v> --split tune
    python -m evalharness.calibration history --run-id <id>
Exit codes: 0 ok; 1 refused.
"""

import argparse
import random
import re
import sys
from collections import Counter
from contextlib import closing
from pathlib import Path

import numpy as np
import yaml

from evalharness import store
from evalharness.checks import ScoringRefused, get_scorable_run, load_checks_config
from evalharness.metrics import wilson_interval
from evalharness.runner import DEFAULT_HARNESS_CONFIG, load_harness_config
from evalharness.schemas import QUERY_TYPES
from evalharness.testset import resolve_path

DEFAULT_CALIBRATION_CONFIG = "configs/calibration.yaml"
GROUNDED_LABELS = ("grounded", "ungrounded", "unsure")
CALIB_SPLITS = ("tune", "check")
CALIBRATION_KEYS = {
    "labeler", "split_seed", "check_fraction", "forced_tune_ids", "order_seed",
    "label_relevance", "show_check_items", "n_bootstrap", "bootstrap_seed", "labeling_guide",
}
# The judge's 0/1 flag compared with the human label, per judge.
JUDGE_METRICS = {"faithfulness": "ungrounded", "relevance": "irrelevant"}
LABEL_KEYS = {"g": "grounded", "u": "ungrounded", "?": "unsure"}
SMALL_N = 50   # below this the report reminds the reader that intervals are wide


# ---------------------------------------------------------------- config

def _is_int(value) -> bool:
    """True for a real int (bool is a subclass of int in Python, so exclude it)."""
    return isinstance(value, int) and not isinstance(value, bool)


def load_calibration_config(path: str | Path = DEFAULT_CALIBRATION_CONFIG) -> dict:
    """Read configs/calibration.yaml and check every key; ValueError names the problem."""
    path = resolve_path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    missing, extra = CALIBRATION_KEYS - set(cfg), set(cfg) - CALIBRATION_KEYS
    if missing or extra:
        raise ValueError(f"{path}: keys missing {sorted(missing)}, unknown {sorted(extra)}")

    def bad(key: str, rule: str) -> ValueError:
        return ValueError(f"{path}: '{key}' {rule}")

    for key in ("labeler", "labeling_guide"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise bad(key, "must be a non-empty string")
    for key in ("split_seed", "order_seed", "bootstrap_seed"):
        if not _is_int(cfg[key]):
            raise bad(key, "must be an integer")
    fraction = cfg["check_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise bad("check_fraction", "must be a number between 0 and 1")
    forced = cfg["forced_tune_ids"]
    if not isinstance(forced, list) or not all(isinstance(q, str) and re.fullmatch(r"q\d{3}", q)
                                               for q in forced):
        raise bad("forced_tune_ids", "must be a list of query ids like q001")
    for key in ("label_relevance", "show_check_items"):
        if not isinstance(cfg[key], bool):
            raise bad(key, "must be true or false")
    if not _is_int(cfg["n_bootstrap"]) or cfg["n_bootstrap"] < 100:
        raise bad("n_bootstrap", "must be an integer >= 100")
    return cfg


# ---------------------------------------------------------------- tune / check split

def assign_calibration_splits(query_types: dict[str, str], *, seed: int, check_fraction: float,
                              forced_tune_ids: list[str]) -> dict[str, str]:
    """{query_id: "tune" | "check"}, stratified by query type and deterministic.

    In each type (sorted), round(check_fraction * eligible) ids go to check,
    drawn with one random.Random(seed) from the sorted ids not forced into tune.
    """
    rng = random.Random(seed)
    forced = set(forced_tune_ids)
    splits = {query_id: "tune" for query_id in query_types}
    for query_type in sorted(set(query_types.values())):
        eligible = sorted(q for q, t in query_types.items() if t == query_type and q not in forced)
        for query_id in rng.sample(eligible, round(check_fraction * len(eligible))):
            splits[query_id] = "check"
    return splits


def _ensure_splits(conn, run_id: str, outputs: list[dict], cfg: dict) -> None:
    """Assign and store the split on first use; afterwards a no-op (the split is fixed)."""
    splits = assign_calibration_splits({o["query_id"]: o["type"] for o in outputs},
                                       seed=cfg["split_seed"], check_fraction=cfg["check_fraction"],
                                       forced_tune_ids=cfg["forced_tune_ids"])
    store.save_calibration_splits(conn, run_id, splits)


# ---------------------------------------------------------------- labelling session

def _print_item(print_fn, index: int, total: int, output: dict) -> None:
    """Show one answer for labelling: question, answer and the stored chunks only.

    It gets the output row alone, so it cannot show judge scores, claims or the
    reference answer (the queries' reference text is not part of this row).
    """
    print_fn("")
    print_fn("=" * 78)
    print_fn(f"[{index}/{total}] {output['query_id']} ({output['type']})")
    print_fn(f"QUESTION: {output['question']}")
    print_fn("")
    print_fn("ANSWER:")
    print_fn(output["answer"])
    print_fn("")
    print_fn("CHUNKS the generator saw (rank order):")
    for rank, chunk in enumerate(output["retrieved_chunks"], start=1):
        print_fn(f"--- [{rank}] {chunk['doc_id']} ({chunk['chunk_id']})")
        print_fn(chunk["text"])


def _ask_relevance(input_fn) -> int:
    """Ask until the answer is 0, 1 or 2."""
    while True:
        raw = input_fn("relevance: does it address the question? [0/1/2] > ").strip()
        if raw in ("0", "1", "2"):
            return int(raw)


def label_session(conn, run_id: str, cfg: dict, *, input_fn=input, print_fn=print,
                  relabel: list[str] | None = None) -> dict:
    """Label the run's unlabelled answers (or re-label the named ids) with Gaurav's keystrokes.

    Keys: g grounded, u ungrounded, ? unsure, n add a note, s skip for now,
    q quit. Each label is committed at once, so quitting keeps everything so far.
    Failed outputs (outputs.error) have nothing to label and are skipped.
    """
    get_scorable_run(conn, run_id)
    outputs = store.get_outputs_with_queries(conn, run_id)
    _ensure_splits(conn, run_id, outputs, cfg)
    labels = {row["query_id"]: row for row in store.get_calibration_labels(conn, run_id)}
    by_id = {o["query_id"]: o for o in outputs}

    if relabel:
        unknown = sorted(set(relabel) - set(by_id))
        if unknown:
            raise ScoringRefused(f"query ids not in run {run_id}: {', '.join(unknown)}")
        todo = sorted(relabel)
    else:
        todo = sorted(q for q, row in labels.items() if row["grounded_label"] is None)
        random.Random(cfg["order_seed"]).shuffle(todo)

    print_fn(cfg["labeling_guide"])
    counts = {"labelled": 0, "skipped": 0, "failed_outputs": 0}
    for index, query_id in enumerate(todo, start=1):
        output = by_id[query_id]
        if output["error"] is not None:
            counts["failed_outputs"] += 1
            continue
        if relabel and labels[query_id]["grounded_label"] is not None:
            answer = input_fn(f"overwrite your label for {query_id}? [y/n] > ").strip().lower()
            if answer != "y":
                counts["skipped"] += 1
                continue
        _print_item(print_fn, index, len(todo), output)
        note = None
        while True:
            key = input_fn("[g]rounded [u]ngrounded [?] unsure [n]ote [s]kip [q]uit > ").strip().lower()
            if key == "q":
                return _finish(conn, run_id, counts)
            if key == "s":
                counts["skipped"] += 1
                break
            if key == "n":
                note = input_fn("note > ").strip() or None
                continue
            if key in LABEL_KEYS:
                relevance = _ask_relevance(input_fn) if cfg["label_relevance"] else None
                store.write_calibration_label(conn, run_id, query_id, LABEL_KEYS[key], relevance,
                                              note, cfg["labeler"])
                counts["labelled"] += 1
                break
            # Anything else: ask again.
    return _finish(conn, run_id, counts)


def _finish(conn, run_id: str, counts: dict) -> dict:
    """Add how many answers still have no label."""
    remaining = sum(row["grounded_label"] is None for row in store.get_calibration_labels(conn, run_id))
    return {**counts, "remaining": remaining}


# ---------------------------------------------------------------- agreement metrics

def confusion(pairs: list[tuple[bool, bool]]) -> dict:
    """(human_positive, judge_positive) pairs -> {"tp", "fp", "fn", "tn", "n"}."""
    tp = sum(h and j for h, j in pairs)
    fp = sum(j and not h for h, j in pairs)
    fn = sum(h and not j for h, j in pairs)
    tn = sum(not h and not j for h, j in pairs)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(pairs)}


def cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float | None:
    """(p_o - p_e) / (1 - p_e): agreement beyond what chance alone would give.

    p_e = P(judge+)P(human+) + P(judge-)P(human-). None when undefined (n = 0,
    or both raters always give the same single label, so 1 - p_e = 0).
    Example: TP 3, FP 1, FN 2, TN 34 -> p_o 0.925, p_e 0.80, kappa 0.625.
    """
    n = tp + fp + fn + tn
    if n == 0:
        return None
    p_o = (tp + tn) / n
    judge_pos, human_pos = (tp + fp) / n, (tp + fn) / n
    p_e = judge_pos * human_pos + (1 - judge_pos) * (1 - human_pos)
    if 1 - p_e == 0:
        return None
    return (p_o - p_e) / (1 - p_e)


def kappa_bootstrap_ci(pairs: list[tuple[bool, bool]], *, n_boot: int, seed: int,
                       confidence: float) -> dict:
    """Percentile bootstrap interval for kappa (resample the pairs with replacement).

    Returns {"ci_low", "ci_high", "n_boot", "n_undefined"}; the bounds are None
    when fewer than half the resamples have a defined kappa.
    """
    if not pairs:
        return {"ci_low": None, "ci_high": None, "n_boot": n_boot, "n_undefined": n_boot}
    rng = np.random.default_rng(seed)
    human = np.array([h for h, _ in pairs], dtype=bool)
    judge = np.array([j for _, j in pairs], dtype=bool)
    kappas = []
    for sample in rng.integers(0, len(pairs), size=(n_boot, len(pairs))):
        h, j = human[sample], judge[sample]
        kappa = cohen_kappa(int(np.sum(h & j)), int(np.sum(j & ~h)),
                            int(np.sum(h & ~j)), int(np.sum(~h & ~j)))
        if kappa is not None:
            kappas.append(kappa)
    n_undefined = n_boot - len(kappas)
    if len(kappas) < n_boot / 2:
        return {"ci_low": None, "ci_high": None, "n_boot": n_boot, "n_undefined": n_undefined}
    alpha = 1 - confidence
    low, high = np.percentile(kappas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ci_low": float(low), "ci_high": float(high), "n_boot": n_boot,
            "n_undefined": n_undefined}


def _rate(successes: int, n: int, confidence: float) -> dict:
    """A rate with its Wilson interval, or "undefined" for 0/0 (never shown as 0 or 1)."""
    if n == 0:
        return {"successes": 0, "n": 0, "undefined": True}
    low, high = wilson_interval(successes, n, confidence)
    return {"successes": successes, "n": n, "rate": successes / n, "ci_low": low, "ci_high": high,
            "undefined": False}


def agreement_report(pairs: list[tuple[bool, bool]], cfg: dict, confidence: float) -> dict:
    """Confusion counts, agreement / precision / recall with Wilson CIs, kappa with a bootstrap CI."""
    c = confusion(pairs)
    return {
        **c,
        "agreement": _rate(c["tp"] + c["tn"], c["n"], confidence),
        "precision": _rate(c["tp"], c["tp"] + c["fp"], confidence),
        "recall": _rate(c["tp"], c["tp"] + c["fn"], confidence),
        "kappa": cohen_kappa(c["tp"], c["fp"], c["fn"], c["tn"]),
        "kappa_ci": kappa_bootstrap_ci(pairs, n_boot=cfg["n_bootstrap"],
                                       seed=cfg["bootstrap_seed"], confidence=confidence),
    }


# ---------------------------------------------------------------- compare

def compare(conn, run_id: str, judge: str, judge_version: str, calib_split: str, cfg: dict,
            confidence: float) -> dict:
    """Grade one judge version against Gaurav's labels on one half and log the comparison.

    Positive class: "ungrounded" (faithfulness) or relevance label 0 / judge
    `irrelevant` (relevance). Unlabelled, unsure and not-judged items are left
    out and counted by reason.
    """
    get_scorable_run(conn, run_id)
    if judge not in JUDGE_METRICS:
        raise ScoringRefused(f"unknown judge {judge!r}; one of {sorted(JUDGE_METRICS)}")
    metric = JUDGE_METRICS[judge]
    judged = {row["query_id"]: row["value"] for row in store.get_scores(conn, run_id, judge_version)
              if row["metric"] == metric}
    if not judged:
        raise ScoringRefused(f"no {metric!r} scores for version {judge_version} in run {run_id}")

    rows = [row for row in store.get_calibration_labels(conn, run_id)
            if calib_split == "all" or row["calib_split"] == calib_split]
    excluded = Counter()
    pairs, ids = [], []
    for row in rows:
        if row["grounded_label"] is None:
            excluded["unlabelled"] += 1
            continue
        if judge == "faithfulness":
            if row["grounded_label"] == "unsure":
                excluded["unsure"] += 1
                continue
            human = row["grounded_label"] == "ungrounded"
        else:
            if row["relevance_label"] is None:
                excluded["no relevance label"] += 1
                continue
            human = row["relevance_label"] == 0
        value = judged.get(row["query_id"])
        if value is None:
            excluded["not judged"] += 1
            continue
        pairs.append((human, value == 1))
        ids.append(row["query_id"])
    if not pairs:
        raise ScoringRefused(f"no usable labels for judge {judge} on split {calib_split}")

    earlier_check_looks = sum(r["judge"] == judge and r["calib_split"] == "check"
                              for r in store.get_calibration_reports(conn, run_id))
    report = {
        "run_id": run_id, "judge": judge, "judge_version": judge_version, "metric": metric,
        "calib_split": calib_split, "confidence": confidence,
        **agreement_report(pairs, cfg, confidence),
        "excluded": dict(excluded),
        "earlier_check_looks": earlier_check_looks,
    }
    if calib_split != "check" or cfg["show_check_items"]:
        report["disagreements"] = [{"query_id": q, "human": h, "judge": j}
                                   for q, (h, j) in zip(ids, pairs) if h != j]
    report["report_id"] = store.write_calibration_report(conn, {
        "run_id": run_id, "judge": judge, "judge_version": judge_version,
        "calib_split": calib_split, "n": report["n"], "n_excluded": sum(excluded.values()),
        "tp": report["tp"], "fp": report["fp"], "fn": report["fn"], "tn": report["tn"],
        "report": report,
    })
    return report


def _fmt_rate(s: dict) -> str:
    if s["undefined"]:
        return "undefined (0/0)"
    return f"{s['rate']:.3f}  {s['successes']}/{s['n']}  [{s['ci_low']:.3f}, {s['ci_high']:.3f}]"


def format_compare(report: dict) -> str:
    """Plain-text comparison: 2x2 table, the four figures with intervals, exclusions, disagreements."""
    positive = report["metric"]
    pct = round(report["confidence"] * 100)
    k, ci = report["kappa"], report["kappa_ci"]
    kappa_text = "undefined" if k is None else f"{k:.3f}"
    ci_text = ("no interval (too many undefined resamples)" if ci["ci_low"] is None
               else f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")
    lines = [
        f"{report['judge']} judge {report['judge_version']} vs Gaurav's labels | "
        f"split {report['calib_split']} | run {report['run_id']}",
        f"positive class: {positive}   ({pct}% intervals: Wilson for rates, bootstrap for kappa)",
        "",
        f"{'':<20}{'human: ' + positive:>22}{'human: not':>14}",
        f"{'judge: ' + positive:<20}{'TP ' + str(report['tp']):>22}{'FP ' + str(report['fp']):>14}",
        f"{'judge: not':<20}{'FN ' + str(report['fn']):>22}{'TN ' + str(report['tn']):>14}",
        "",
        f"  agreement   {_fmt_rate(report['agreement'])}",
        f"  precision   {_fmt_rate(report['precision'])}",
        f"  recall      {_fmt_rate(report['recall'])}",
        f"  kappa       {kappa_text}  {ci_text}  "
        f"({ci['n_undefined']} of {ci['n_boot']} resamples undefined)",
        "",
        f"compared: {report['n']}   excluded: "
        + (", ".join(f"{reason} {count}" for reason, count in sorted(report["excluded"].items()))
           or "none"),
        f"earlier 'check' looks for this judge: {report['earlier_check_looks']}",
    ]
    if "disagreements" in report:
        items = [f"{d['query_id']} (human {'+' if d['human'] else '-'}, "
                 f"judge {'+' if d['judge'] else '-'})" for d in report["disagreements"]]
        lines.append(f"disagreements: {len(items)}" + (": " + ", ".join(items) if items else ""))
    if report["n"] < SMALL_N:
        lines.append("note: n is small, so the intervals are wide; report them with the figures.")
    return "\n".join(lines)


# ---------------------------------------------------------------- status / history

def format_status(rows: list[dict], outputs: list[dict]) -> str:
    """Label counts by split and query type (never judge results)."""
    types = {o["query_id"]: o["type"] for o in outputs}
    lines = [f"{'split':<7}{'type':<14}{'grounded':>10}{'ungrounded':>12}{'unsure':>8}"
             f"{'unlabelled':>12}{'relevance 0/1/2':>18}"]
    for split in CALIB_SPLITS:
        for query_type in QUERY_TYPES:
            group = [r for r in rows if r["calib_split"] == split and types[r["query_id"]] == query_type]
            labels = Counter(r["grounded_label"] for r in group)
            rel = Counter(r["relevance_label"] for r in group)
            lines.append(f"{split:<7}{query_type:<14}{labels['grounded']:>10}{labels['ungrounded']:>12}"
                         f"{labels['unsure']:>8}{labels[None]:>12}"
                         f"{f'{rel[0]}/{rel[1]}/{rel[2]}':>18}")
    total = Counter(r["grounded_label"] for r in rows)
    lines.append(f"total: {len(rows)} answers, {len(rows) - total[None]} labelled, "
                 f"{total[None]} unlabelled")
    return "\n".join(lines)


def format_history(reports: list[dict]) -> str:
    """One line per logged comparison, oldest first."""
    if not reports:
        return "no comparisons yet"
    lines = []
    for r in reports:
        rep = r["report"]
        kappa = "undefined" if rep["kappa"] is None else f"{rep['kappa']:.3f}"
        lines.append(f"#{r['id']} {r['created_at']} {r['judge']} {r['judge_version']} "
                     f"{r['calib_split']}: n={r['n']} TP {r['tp']} FP {r['fp']} FN {r['fn']} "
                     f"TN {r['tn']} | agreement {_fmt_rate(rep['agreement'])} | "
                     f"precision {_fmt_rate(rep['precision'])} | recall {_fmt_rate(rep['recall'])}"
                     f" | kappa {kappa}")
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-id", required=True)
    common.add_argument("--harness-config", default=DEFAULT_HARNESS_CONFIG)
    common.add_argument("--calibration-config", default=DEFAULT_CALIBRATION_CONFIG)
    parser = argparse.ArgumentParser(prog="python -m evalharness.calibration",
                                     description="Calibrate the LLM judges against Gaurav's labels.")
    sub = parser.add_subparsers(dest="command", required=True)
    label = sub.add_parser("label", parents=[common], help="label answers (blind)")
    label.add_argument("--relabel", help="comma-separated ids to label again")
    status = sub.add_parser("status", parents=[common], help="label counts by split and type")
    status.add_argument("--ids", choices=CALIB_SPLITS, help="print only this split's ids")
    comp = sub.add_parser("compare", parents=[common], help="judge vs labels on one split")
    comp.add_argument("--judge", required=True, choices=sorted(JUDGE_METRICS))
    comp.add_argument("--version", required=True)
    comp.add_argument("--split", required=True, choices=(*CALIB_SPLITS, "all"))
    comp.add_argument("--confirm-check", action="store_true",
                      help="required for --split check: each look at check is logged")
    sub.add_parser("history", parents=[common], help="every logged comparison")
    args = parser.parse_args(argv)

    def refuse(problem: str) -> int:
        print(f"error: {problem}", file=sys.stderr)
        return 1

    try:
        harness = load_harness_config(args.harness_config)
        cfg = load_calibration_config(args.calibration_config)
        confidence = load_checks_config(harness)["report_confidence"]
    except ValueError as error:
        return refuse(str(error))
    if args.command == "compare" and args.split == "check" and not args.confirm_check:
        return refuse("comparing on 'check' is the reported figure; pass --confirm-check "
                      "(every look is logged)")

    with closing(store.connect(harness["db_path"])) as conn:
        try:
            if args.command == "label":
                relabel = ([q.strip() for q in args.relabel.split(",") if q.strip()]
                           if args.relabel else None)
                result = label_session(conn, args.run_id, cfg, relabel=relabel)
                print(f"\nlabelled {result['labelled']}, skipped {result['skipped']}, "
                      f"failed outputs {result['failed_outputs']}, remaining {result['remaining']}")
            elif args.command == "status":
                get_scorable_run(conn, args.run_id)
                outputs = store.get_outputs_with_queries(conn, args.run_id)
                _ensure_splits(conn, args.run_id, outputs, cfg)
                rows = store.get_calibration_labels(conn, args.run_id)
                if args.ids:
                    print(",".join(r["query_id"] for r in rows if r["calib_split"] == args.ids))
                else:
                    print(format_status(rows, outputs))
            elif args.command == "compare":
                print(format_compare(compare(conn, args.run_id, args.judge, args.version,
                                             args.split, cfg, confidence)))
            else:
                print(format_history(store.get_calibration_reports(conn, args.run_id)))
        except (ScoringRefused, ValueError) as error:
            return refuse(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
