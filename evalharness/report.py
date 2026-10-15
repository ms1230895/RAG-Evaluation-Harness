"""Baseline report for a stored run (Step 10).

Joins three stored scorings of one run, per answer, and reports the "before"
number Step 11 is measured against:
- the ungrounded rate (faithfulness judge) with a 95% Wilson interval,
  overall, for answerable queries and by query type;
- the failure cause of each ungrounded answer, from the Step 6 `in_context`
  check (did any chunk of a source document reach the prompt?):
      answerable,   in_context 0 -> retrieval_miss   (the source never reached the generator)
      answerable,   in_context 1 -> generation_error (the source was there; the answer added facts)
      unanswerable               -> generation_error (no source exists: an invented answer)
- relevance next to faithfulness (a 2x2 table), a secondary "bad answer" rate
  (ungrounded OR irrelevant), invented answers and false refusals;
- the calibration status of each judge version (Step 9).

No LLM is called and nothing is re-run: every number is read from the
results database. Every judge-based number is labelled "uncalibrated judge"
unless a calibration with positive cases was measured.

CLI:
    python -m evalharness.report --run-id 20261015T152638Z-v0-c7c96975 \\
        [--markdown experiments/baseline_v0.md [--force]] [--log experiments/log.md]
Exit codes: 0 ok; 1 refused (nothing written).
"""

import argparse
import json
import sys
from contextlib import closing
from pathlib import Path

from evalharness import store
from evalharness.checks import (ANSWERABLE_TYPES, ScoringRefused, get_scorable_run,
                                load_checks_config)
from evalharness.judges import (GROUP_TYPES, _coverage_state, load_judges_config,
                                load_relevance_config)
from evalharness.metrics import mean_summary, rate_summary, wilson_interval
from evalharness.runner import DEFAULT_HARNESS_CONFIG, load_harness_config
from evalharness.schemas import QUERY_TYPES
from evalharness.testset import resolve_path

CAUSES = ("retrieval_miss", "generation_error")
REPORT_KEYS = {"primary_metric", "include_bad_answer_rate", "allow_incomplete"}
SCORERS = ("checks", "faithfulness", "relevance")
GROUP_NAMES = ["all", "answerable", *QUERY_TYPES]
CV_HYPOTHESIS = "CV draft: 25% -> 12% (a hypothesis, not a target; the CV follows the measurement)"


# ---------------------------------------------------------------- config and versions

def load_report_config(harness: dict) -> dict:
    """The `report` block of configs/harness.yaml, checked; ValueError names the problem."""
    cfg = harness.get("report")
    if not isinstance(cfg, dict):
        raise ValueError("harness config: 'report' must be a mapping")
    missing, extra = REPORT_KEYS - set(cfg), set(cfg) - REPORT_KEYS
    if missing or extra:
        raise ValueError(f"harness config: report keys missing {sorted(missing)}, "
                         f"unknown {sorted(extra)}")
    if cfg["primary_metric"] != "ungrounded":
        raise ValueError("harness config: 'report.primary_metric' must be 'ungrounded'")
    for key in ("include_bad_answer_rate", "allow_incomplete"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"harness config: 'report.{key}' must be true or false")
    return cfg


def current_versions(harness: dict) -> dict[str, str]:
    """The scorer versions of the CURRENT configs, so the baseline and every Step 11
    run are scored with the same settings by default."""
    return {
        "checks": load_checks_config(harness)["version"],
        "faithfulness": load_judges_config(harness["judges_config"])["version"],
        "relevance": load_relevance_config(harness["judges_config"])["version"],
    }


# ---------------------------------------------------------------- per-answer table

def failure_cause(answerable: bool, in_context: float | None) -> str:
    """Where the RAG chain broke for an ungrounded answer (see the module docstring)."""
    if not answerable:
        return "generation_error"
    if in_context is None:
        raise ValueError("answerable item without an in_context score (Step 6 scores missing)")
    return "retrieval_miss" if in_context == 0 else "generation_error"


def item_table(conn, run_id: str, versions: dict[str, str]) -> list[dict]:
    """One row per output, joined from the three stored scorings (scores table only)."""
    by_version = {}
    for scorer in SCORERS:
        rows = store.get_scores(conn, run_id, versions[scorer])
        if not rows:
            raise ScoringRefused(f"no scores for {scorer} version {versions[scorer]} "
                                 f"in run {run_id}")
        table: dict[str, dict[str, dict]] = {}
        for row in rows:
            table.setdefault(row["query_id"], {})[row["metric"]] = row
        by_version[scorer] = table

    def value(scorer: str, query_id: str, metric: str):
        row = by_version[scorer].get(query_id, {}).get(metric)
        return row["value"] if row is not None else None

    items = []
    for output in store.get_outputs_with_queries(conn, run_id):
        q = output["query_id"]
        answerable = bool(output["source_doc_ids"])
        faith_state = _coverage_state(by_version["faithfulness"].get(q, {}).get("faithfulness"))
        rel_state = _coverage_state(by_version["relevance"].get(q, {}).get("relevance"))
        judged = faith_state == "judged"
        rel_judged = rel_state == "judged"
        ungrounded = value("faithfulness", q, "ungrounded") if judged else None
        irrelevant = value("relevance", q, "irrelevant") if rel_judged else None
        in_context = value("checks", q, "in_context")
        items.append({
            "query_id": q,
            "type": output["type"],
            "answerable": answerable,
            "in_context": in_context,
            "refusal": value("checks", q, "refusal"),
            "hit@4": value("checks", q, "hit@4"),
            "reciprocal_rank": value("checks", q, "reciprocal_rank"),
            "faith_state": faith_state,
            "rel_state": rel_state,
            "ungrounded": ungrounded,
            "irrelevant": irrelevant,
            "faithfulness": value("faithfulness", q, "faithfulness") if judged else None,
            "relevance": value("relevance", q, "relevance") if rel_judged else None,
            "cause": failure_cause(answerable, in_context) if ungrounded == 1 else None,
            "bad": (int(ungrounded == 1 or irrelevant == 1)
                    if ungrounded is not None and irrelevant is not None else None),
        })
    return items


# ---------------------------------------------------------------- report

def _share(successes: int, n: int, confidence: float) -> dict:
    """A share of the ungrounded answers with its Wilson interval (None fields when n = 0)."""
    if n == 0:
        return {"successes": 0, "n": 0, "rate": None, "ci_low": None, "ci_high": None}
    low, high = wilson_interval(successes, n, confidence)
    return {"successes": successes, "n": n, "rate": successes / n, "ci_low": low, "ci_high": high}


def _calibration_status(conn, run_id: str, judge: str, version: str) -> dict:
    """The latest stored check comparison for this judge version (Step 9)."""
    rows = [r for r in store.get_calibration_reports(conn, run_id)
            if r["judge"] == judge and r["judge_version"] == version and r["calib_split"] == "check"]
    if not rows:
        return {"status": "not calibrated"}
    r = rows[-1]
    rep = r["report"]
    base = {"report_id": r["id"], "n": r["n"], "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
            "tn": r["tn"], "agreement": rep["agreement"]}
    if r["tp"] + r["fn"] == 0 or r["tp"] + r["fp"] == 0:
        return {"status": "inconclusive", **base,
                "reason": "no positive cases on at least one side, so precision / recall / "
                          "kappa are undefined"}
    return {"status": "measured", **base, "precision": rep["precision"],
            "recall": rep["recall"], "kappa": rep["kappa"], "kappa_ci": rep["kappa_ci"]}


def build_baseline_report(conn, run_id: str, versions: dict[str, str], confidence: float,
                          cfg: dict, *, allow_heldout: bool = False) -> dict:
    """Everything the baseline needs, computed from item_table() only."""
    run = get_scorable_run(conn, run_id, allow_heldout=allow_heldout)
    items = item_table(conn, run_id, versions)

    coverage = {}
    for judge, key in (("faithfulness", "faith_state"), ("relevance", "rel_state")):
        states = {}
        for item in items:
            states.setdefault(item[key], []).append(item["query_id"])
        coverage[judge] = {state: len(ids) for state, ids in states.items()}
        incomplete = sorted(q for state, ids in states.items() if state != "judged" for q in ids)
        if incomplete and not cfg["allow_incomplete"]:
            raise ScoringRefused(f"{judge} is not complete for run {run_id} "
                                 f"({len(incomplete)} items: {', '.join(incomplete)}); "
                                 "judge them first or set report.allow_incomplete")

    config = json.loads(run["config_json"])
    system = json.loads(run["system_json"]) if run["system_json"] else {}
    generation = config.get("generation") if isinstance(config.get("generation"), dict) else {}
    metadata = {
        "run_id": run_id, "split": run["split"], "status": run["status"],
        "timestamp": run["timestamp"], "git_commit": run["git_commit"],
        "git_dirty": run["git_dirty"], "test_set_sha256": run["test_set_sha256"],
        "config_path": run["config_path"], "n_outputs": len(items), "n_ok": run["n_ok"],
        "n_failed": run["n_failed"], "generator_model": generation.get("model"),
        "generator_model_digest": system.get("generator_model_digest"),
        "index_key": (system.get("index") or {}).get("index_key"),
        "versions": versions,
    }

    groups = {name: [i for i in items if i["type"] in GROUP_TYPES[name]] for name in GROUP_NAMES}
    ungrounded = {name: rate_summary([i["ungrounded"] for i in g], confidence)
                  for name, g in groups.items()}
    causes = {}
    for name, g in groups.items():
        judged = [i for i in g if i["ungrounded"] is not None]
        n_ungrounded = sum(i["ungrounded"] == 1 for i in judged)
        causes[name] = {
            cause: {
                "rate": rate_summary([int(i["cause"] == cause) for i in judged], confidence),
                "share": _share(sum(i["cause"] == cause for i in judged), n_ungrounded, confidence),
            } for cause in CAUSES
        }

    both = [i for i in items if i["ungrounded"] is not None and i["irrelevant"] is not None]
    crosstab = {
        "grounded_relevant": sum(i["ungrounded"] == 0 and i["irrelevant"] == 0 for i in both),
        "ungrounded_relevant": sum(i["ungrounded"] == 1 and i["irrelevant"] == 0 for i in both),
        "grounded_irrelevant": sum(i["ungrounded"] == 0 and i["irrelevant"] == 1 for i in both),
        "ungrounded_irrelevant": sum(i["ungrounded"] == 1 and i["irrelevant"] == 1 for i in both),
        "n": len(both),
    }
    answerable_items = [i for i in items if i["answerable"]]
    report = {
        "kind": "baseline",
        "metadata": metadata,
        "confidence": confidence,
        "coverage": coverage,
        "ungrounded": ungrounded,
        "causes": causes,
        "crosstab": crosstab,
        "bad_answer": ({name: rate_summary([i["bad"] for i in g], confidence)
                        for name, g in groups.items()}
                       if cfg["include_bad_answer_rate"] else None),
        "irrelevant": rate_summary([i["irrelevant"] for i in items], confidence),
        "ungrounded_items": [{"query_id": i["query_id"], "type": i["type"], "cause": i["cause"],
                              "irrelevant": i["irrelevant"]}
                             for i in items if i["ungrounded"] == 1],
        "invented_answers": [{"query_id": i["query_id"], "irrelevant": i["irrelevant"]}
                             for i in items if not i["answerable"] and i["ungrounded"] == 1],
        "false_refusals": [{"query_id": i["query_id"], "type": i["type"],
                            "in_context": i["in_context"]}
                           for i in items if i["answerable"] and i["irrelevant"] == 1
                           and i["refusal"] == 1],
        "retrieval": {
            "in_context": rate_summary([i["in_context"] for i in answerable_items], confidence),
            "hit@4": rate_summary([i["hit@4"] for i in answerable_items], confidence),
            "mrr": mean_summary([i["reciprocal_rank"] for i in answerable_items]),
        },
        "calibration": {judge: _calibration_status(conn, run_id, judge, versions[judge])
                        for judge in ("faithfulness", "relevance")},
    }
    report["judge_label"] = {judge: ("calibrated judge" if status["status"] == "measured"
                                     else "uncalibrated judge")
                             for judge, status in report["calibration"].items()}
    report["notes"] = [
        "Judge-based numbers are the judge's measurement, not verified rates, unless the "
        "calibration status says 'measured'.",
        "Per-type n is small (12-16 on dev): differences inside the intervals are not meaningful.",
        "Each answer is one sample from a cloud model that is not fully deterministic; reruns "
        "are identical only because the LLM cache serves them.",
        CV_HYPOTHESIS,
    ]
    return report


# ---------------------------------------------------------------- formatting

def _pct(x: float | None) -> str:
    """Percent with 2 decimals: k/80 is always exact (7/80 = 8.75%), so the headline never
    depends on how a float like 0.0875 happens to round at 1 decimal."""
    return "-" if x is None else f"{100 * x:.2f}%"


def _fmt_rate(s: dict) -> str:
    """'0.0875  7/80  [0.043, 0.170]', or '-' when n = 0."""
    if s["rate"] is None:
        return "-"
    return f"{s['rate']:.4f}  {s['successes']}/{s['n']}  [{s['ci_low']:.3f}, {s['ci_high']:.3f}]"


def _fmt_rate_pct(s: dict) -> str:
    """'8.75% [4.30%, 17.04%] (7/80)', or '-' when n = 0."""
    if s["rate"] is None:
        return "-"
    return f"{_pct(s['rate'])} [{_pct(s['ci_low'])}, {_pct(s['ci_high'])}] ({s['successes']}/{s['n']})"


def _fmt_calibration(status: dict) -> str:
    if status["status"] == "not calibrated":
        return "not calibrated (no check comparison stored)"
    a = status["agreement"]
    text = (f"{status['status']}: check n={status['n']} TP {status['tp']} FP {status['fp']} "
            f"FN {status['fn']} TN {status['tn']}, agreement {a['successes']}/{a['n']}")
    if status["status"] == "inconclusive":
        return text + f"; {status['reason']}"
    return text


def format_baseline_report(report: dict) -> str:
    """Terminal text, headline first."""
    m, label = report["metadata"], report["judge_label"]
    pct = round(report["confidence"] * 100)
    u = report["ungrounded"]
    lines = [
        f"Baseline report: run {m['run_id']} ({m['split']}, {m['config_path']}, "
        f"git {m['git_commit']}{' dirty' if m['git_dirty'] else ''})",
        f"scorers: checks {m['versions']['checks']} | faithfulness "
        f"{m['versions']['faithfulness']} | relevance {m['versions']['relevance']}",
        "",
        f"HEADLINE  ungrounded rate ({label['faithfulness']}): {_fmt_rate_pct(u['all'])}  "
        f"[{pct}% Wilson CI]",
        "",
        f"Ungrounded rate by group ({label['faithfulness']})",
    ]
    for name in GROUP_NAMES:
        lines.append(f"  {name:<14}{_fmt_rate(u[name])}")
    lines.extend(["", f"Failure cause of ungrounded answers ({label['faithfulness']})",
                  f"  {'group':<14}{'cause':<18}{'rate over judged':<30}share of ungrounded"])
    for name in GROUP_NAMES:
        for cause in CAUSES:
            c = report["causes"][name][cause]
            lines.append(f"  {name:<14}{cause:<18}{_fmt_rate(c['rate']):<30}{_fmt_rate(c['share'])}")
    x = report["crosstab"]
    lines.extend([
        "",
        f"Faithfulness x relevance (both judged, n={x['n']}; {label['faithfulness']} / "
        f"{label['relevance']})",
        f"  {'':<14}{'relevant':>10}{'irrelevant':>12}",
        f"  {'grounded':<14}{x['grounded_relevant']:>10}{x['grounded_irrelevant']:>12}",
        f"  {'ungrounded':<14}{x['ungrounded_relevant']:>10}{x['ungrounded_irrelevant']:>12}",
        "",
        f"Irrelevant rate ({label['relevance']}): {_fmt_rate(report['irrelevant'])}",
    ])
    if report["bad_answer"] is not None:
        lines.append(f"Bad-answer rate, ungrounded OR irrelevant ({label['faithfulness']}):")
        for name in GROUP_NAMES:
            lines.append(f"  {name:<14}{_fmt_rate(report['bad_answer'][name])}")
    ug = ", ".join(f"{i['query_id']} ({i['type']}, {i['cause']})" for i in report["ungrounded_items"])
    inv = ", ".join(i["query_id"] for i in report["invented_answers"])
    fr = ", ".join(f"{i['query_id']} (in_context {i['in_context']:.0f})"
                   for i in report["false_refusals"])
    r = report["retrieval"]
    lines.extend([
        "",
        f"Ungrounded answers:   {len(report['ungrounded_items'])}" + (f": {ug}" if ug else ""),
        f"Invented answers (unanswerable + ungrounded): {len(report['invented_answers'])}"
        + (f": {inv}" if inv else ""),
        f"False refusals (answerable + irrelevant + refusal phrase): "
        f"{len(report['false_refusals'])}" + (f": {fr}" if fr else ""),
        "",
        "Retrieval (answerable, no LLM)",
        f"  in_context    {_fmt_rate(r['in_context'])}",
        f"  hit@4         {_fmt_rate(r['hit@4'])}",
        f"  MRR           {'-' if r['mrr']['mean'] is None else format(r['mrr']['mean'], '.3f')}"
        f"  n={r['mrr']['n']}",
        "",
        "Calibration status (Step 9)",
        f"  faithfulness  {_fmt_calibration(report['calibration']['faithfulness'])}",
        f"  relevance     {_fmt_calibration(report['calibration']['relevance'])}",
        "",
        "Coverage: " + "; ".join(f"{judge} " + ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
                                 for judge, c in report["coverage"].items()),
        "",
        "Notes:",
        *[f"- {note}" for note in report["notes"]],
    ])
    return "\n".join(lines)


def format_baseline_markdown(report: dict) -> str:
    """experiments/baseline_v0.md: the same numbers as markdown tables."""
    m, label = report["metadata"], report["judge_label"]
    pct = round(report["confidence"] * 100)
    u, x, r = report["ungrounded"], report["crosstab"], report["retrieval"]
    lines = [
        f"# Baseline report: {m['config_path']} on {m['split']}",
        "",
        "Generated by `python -m evalharness.report` from `data/results.sqlite`; every number "
        "below is read from the stored run, not typed by hand.",
        "",
        f"**Headline: ungrounded rate {_fmt_rate_pct(u['all'])}, {pct}% Wilson CI "
        f"({label['faithfulness']}).**",
        "",
        "## Run",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Run | `{m['run_id']}` ({m['status']}, started {m['timestamp']}) |",
        f"| Split | {m['split']} ({m['n_outputs']} outputs, {m['n_ok']} ok, {m['n_failed']} failed) |",
        f"| Config | `{m['config_path']}` |",
        f"| Generator | `{m['generator_model']}` digest `{m['generator_model_digest']}` |",
        f"| Git commit | `{m['git_commit']}` (dirty: {bool(m['git_dirty'])}) |",
        f"| Test set SHA-256 | `{m['test_set_sha256']}` |",
        f"| Index key | `{m['index_key']}` |",
        f"| Checks version | `{m['versions']['checks']}` |",
        f"| Faithfulness judge | `{m['versions']['faithfulness']}` ({label['faithfulness']}) |",
        f"| Relevance judge | `{m['versions']['relevance']}` ({label['relevance']}) |",
        "",
        f"## Ungrounded rate by group ({label['faithfulness']})",
        "",
        f"| Group | Ungrounded | n | Rate | {pct}% CI |",
        "|---|---|---|---|---|",
    ]
    for name in GROUP_NAMES:
        s = u[name]
        ci = "-" if s["rate"] is None else f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]"
        rate = "-" if s["rate"] is None else f"{s['rate']:.4f}"
        lines.append(f"| {name} | {s['successes']} | {s['n']} | {rate} | {ci} |")
    lines.extend(["", "## Failure cause of ungrounded answers", "",
                  "| Group | Cause | Rate over judged answers | Share of ungrounded answers |",
                  "|---|---|---|---|"])
    for name in GROUP_NAMES:
        for cause in CAUSES:
            c = report["causes"][name][cause]
            lines.append(f"| {name} | {cause} | {_fmt_rate(c['rate'])} | {_fmt_rate(c['share'])} |")
    lines.extend([
        "",
        "Cause rule: answerable with no source chunk in the prompt = retrieval_miss; answerable "
        "with a source chunk in the prompt = generation_error; unanswerable = generation_error "
        "(invented answer).",
        "",
        f"## Faithfulness x relevance (both judged, n = {x['n']})",
        "",
        "| | Relevant | Irrelevant |",
        "|---|---|---|",
        f"| Grounded | {x['grounded_relevant']} | {x['grounded_irrelevant']} |",
        f"| Ungrounded | {x['ungrounded_relevant']} | {x['ungrounded_irrelevant']} |",
        "",
        f"Irrelevant rate ({label['relevance']}): {_fmt_rate(report['irrelevant'])}",
        "",
    ])
    if report["bad_answer"] is not None:
        lines.extend(["Bad-answer rate (ungrounded OR irrelevant):", "",
                      "| Group | Rate, k/n, CI |", "|---|---|"])
        lines.extend(f"| {name} | {_fmt_rate(report['bad_answer'][name])} |" for name in GROUP_NAMES)
        lines.append("")
    lines.extend(["## Answers behind the numbers", "",
                  "| Query | Type | Cause | Irrelevant |", "|---|---|---|---|"])
    for i in report["ungrounded_items"]:
        lines.append(f"| {i['query_id']} | {i['type']} | {i['cause']} | {i['irrelevant']:.0f} |")
    lines.extend([
        "",
        f"Invented answers (unanswerable + ungrounded): {len(report['invented_answers'])}"
        + (": " + ", ".join(i["query_id"] for i in report["invented_answers"])
           if report["invented_answers"] else ""),
        "",
        f"False refusals (answerable + irrelevant + refusal phrase): {len(report['false_refusals'])}"
        + (": " + ", ".join(f"{i['query_id']} (in_context {i['in_context']:.0f})"
                            for i in report["false_refusals"]) if report["false_refusals"] else ""),
        "",
        "## Retrieval (answerable queries, no LLM)",
        "",
        f"- in_context: {_fmt_rate(r['in_context'])}",
        f"- hit@4: {_fmt_rate(r['hit@4'])}",
        f"- MRR: {'-' if r['mrr']['mean'] is None else format(r['mrr']['mean'], '.3f')} "
        f"(n = {r['mrr']['n']})",
        "",
        "## Calibration status (Step 9)",
        "",
        f"- Faithfulness: {_fmt_calibration(report['calibration']['faithfulness'])}",
        f"- Relevance: {_fmt_calibration(report['calibration']['relevance'])}",
        "",
        "## Notes",
        "",
        *[f"- {note}" for note in report["notes"]],
        "",
    ])
    return "\n".join(lines)


def log_row(report: dict, created_at: str, note: str) -> str:
    """One experiments/log.md row; every number comes from the report."""
    m, s = report["metadata"], report["ungrounded"]["all"]
    rate = (f"{_fmt_rate_pct(s)}, {report['judge_label']['faithfulness']}"
            if s["rate"] is not None else "-")
    return (f"| {created_at[:10]} | `{m['config_path']}` | none (baseline) | `{m['run_id']}` | "
            f"{rate} | {note} |")


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evalharness.report",
                                     description="Baseline report for a stored, scored run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--markdown", help="write the report as markdown to this path")
    parser.add_argument("--force", action="store_true", help="overwrite an existing --markdown file")
    parser.add_argument("--log", help="append one row to this experiment log (markdown table)")
    for scorer in SCORERS:
        parser.add_argument(f"--{scorer}-version", help=f"override the current {scorer} version")
    parser.add_argument("--allow-heldout", action="store_true",
                        help="allow a held-out run (Step 12 only)")
    parser.add_argument("--harness-config", default=DEFAULT_HARNESS_CONFIG)
    args = parser.parse_args(argv)

    def refuse(problem: str) -> int:
        print(f"error: {problem}", file=sys.stderr)
        return 1

    try:
        harness = load_harness_config(args.harness_config)
        cfg = load_report_config(harness)
        confidence = load_checks_config(harness)["report_confidence"]
        versions = current_versions(harness)
    except ValueError as error:
        return refuse(str(error))
    for scorer in SCORERS:
        override = getattr(args, f"{scorer}_version")
        if override:
            versions[scorer] = override

    markdown_path = resolve_path(args.markdown) if args.markdown else None
    log_path = resolve_path(args.log) if args.log else None
    if markdown_path is not None and markdown_path.exists() and not args.force:
        return refuse(f"{markdown_path} exists; pass --force to overwrite it")
    if log_path is not None:
        if not log_path.exists():
            return refuse(f"{log_path} does not exist")
        if f"`{args.run_id}`" in log_path.read_text(encoding="utf-8"):
            return refuse(f"{log_path} already has a row for run {args.run_id}")

    with closing(store.connect(harness["db_path"])) as conn:
        try:
            report = build_baseline_report(conn, args.run_id, versions, confidence, cfg,
                                           allow_heldout=args.allow_heldout)
        except (ScoringRefused, ValueError) as error:
            return refuse(str(error))
        created_at = store._now()
        report_id = store.write_run_report(conn, {"run_id": args.run_id, "kind": "baseline",
                                                  "versions": versions, "report": report,
                                                  "created_at": created_at})

    print(format_baseline_report(report))
    print(f"\nstored as run_reports id {report_id}")
    if markdown_path is not None:
        markdown_path.write_text(format_baseline_markdown(report), encoding="utf-8")
        print(f"wrote {markdown_path}")
    if log_path is not None:
        text = log_path.read_text(encoding="utf-8")
        row = log_row(report, created_at, f"v0 baseline; run_reports id {report_id}")
        log_path.write_text(text + ("" if text.endswith("\n") else "\n") + row + "\n",
                            encoding="utf-8")
        print(f"appended a row to {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
