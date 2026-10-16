"""Data layer of the Streamlit dashboard (Step 15). No streamlit import.

Every function takes an open sqlite3 connection plus ids and returns plain
dicts and lists, so each one can be tested without a browser. No number is
computed here: rates, intervals, flips and failure causes come from the same
functions the CLI reports use (report.item_table, metrics.rate_summary,
experiments.compare_runs, report.build_baseline_report). This module only
reshapes them for the pages.

Read-only: nothing here writes to the database. The one write the dashboard
makes is store.write_review, called from dashboard/app.py.
"""

from collections import defaultdict
from pathlib import Path

from evalharness import store
from evalharness.checks import load_checks_config
from evalharness.experiments import compare_runs, load_experiments_config
from evalharness.metrics import mean_summary, rate_summary
from evalharness.regression import answer_similarity, load_regression_config
from evalharness.report import (_calibration_status, build_baseline_report, current_versions,
                                item_table, load_report_config)

DASHBOARD_KEYS = {"cache_ttl_seconds", "show_heldout", "excerpt_chars", "review_labels"}
REVIEW_LABELS = ["ok", "bad"]   # the only vocabulary the review buttons may write
SHORT_COMMIT_CHARS = 7          # display only


# ---------------------------------------------------------------- config

def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_dashboard_config(harness: dict) -> dict:
    """The checked `dashboard` block of configs/harness.yaml; ValueError names the problem."""
    cfg = harness.get("dashboard")
    if not isinstance(cfg, dict):
        raise ValueError("harness config: 'dashboard' must be a mapping")
    missing, extra = DASHBOARD_KEYS - set(cfg), set(cfg) - DASHBOARD_KEYS
    if missing or extra:
        raise ValueError(f"harness config: dashboard keys missing {sorted(missing)}, "
                         f"unknown {sorted(extra)}")
    for key in ("cache_ttl_seconds", "excerpt_chars"):
        if not _is_int(cfg[key]) or cfg[key] < 1:
            raise ValueError(f"harness config: 'dashboard.{key}' must be an integer >= 1")
    if not isinstance(cfg["show_heldout"], bool):
        raise ValueError("harness config: 'dashboard.show_heldout' must be true or false")
    if cfg["review_labels"] != REVIEW_LABELS:
        raise ValueError(f"harness config: 'dashboard.review_labels' must be {REVIEW_LABELS}")
    return cfg


# ---------------------------------------------------------------- which runs can be shown

def eligible_runs(conn, versions: dict[str, str]) -> dict:
    """Runs the pages can show, and the ones left out with the reason.

    A run is eligible when it is completed, has no failed output and has score rows
    for ALL THREE current scorer versions. Nothing is dropped silently: every other
    run is listed in "excluded". Held-out runs are returned too (split "heldout");
    the pages decide where they may appear.
    """
    scored = store.runs_with_versions(conn, list(versions.values()))
    eligible, excluded = [], []
    for run in store.list_runs(conn):
        if run["status"] != "completed":
            reason = f"status is {run['status']!r}, not completed"
        elif run["n_failed"]:
            reason = f"{run['n_failed']} outputs failed"
        elif run["run_id"] not in scored:
            reason = "no scores for all three current scorer versions"
        else:
            reason = None
        if reason is not None:
            excluded.append({"run_id": run["run_id"], "split": run["split"], "reason": reason})
            continue
        eligible.append({
            "run_id": run["run_id"], "timestamp": run["timestamp"], "split": run["split"],
            "config_path": run["config_path"], "config_name": Path(run["config_path"]).stem,
            "git_commit": run["git_commit"][:SHORT_COMMIT_CHARS], "git_dirty": run["git_dirty"],
            "judge_version": run["judge_version"], "n_ok": run["n_ok"], "n_failed": run["n_failed"],
        })
    return {"eligible": eligible, "excluded": excluded}


# ---------------------------------------------------------------- trends

def trend_table(conn, run_ids: list[str], versions: dict[str, str], confidence: float) -> list[dict]:
    """One row per run, oldest first, all runs on ONE split (dev and held-out never mix).

    Rates are rate_summary dicts (rate, successes, n, ci_low, ci_high); means are
    mean_summary dicts (mean, n) and have no interval. Computed from item_table only.
    """
    runs = []
    for run_id in run_ids:
        run = store.get_run(conn, run_id)
        if run is None:
            raise ValueError(f"no run {run_id!r} in the results database")
        runs.append(run)
    if len({run["split"] for run in runs}) > 1:
        raise ValueError("trend_table: runs from different splits are never shown together")
    rows = []
    for run in sorted(runs, key=lambda r: (r["timestamp"], r["run_id"])):
        items = item_table(conn, run["run_id"], versions)
        answerable = [i for i in items if i["answerable"]]
        rows.append({
            "run_id": run["run_id"], "timestamp": run["timestamp"], "split": run["split"],
            "config_name": Path(run["config_path"]).stem,
            "git_commit": run["git_commit"][:SHORT_COMMIT_CHARS],
            "judge_version": run["judge_version"], "n_outputs": len(items),
            "rates": {
                "ungrounded": rate_summary([i["ungrounded"] for i in items], confidence),
                "bad_answer": rate_summary([i["bad"] for i in items], confidence),
                "irrelevant": rate_summary([i["irrelevant"] for i in items], confidence),
                "in_context": rate_summary([i["in_context"] for i in answerable], confidence),
                "hit@4": rate_summary([i["hit@4"] for i in answerable], confidence),
            },
            "means": {
                "faithfulness": mean_summary([i["faithfulness"] for i in items]),
                "relevance": mean_summary([i["relevance"] for i in items]),
                "mrr": mean_summary([i["reciprocal_rank"] for i in answerable]),
            },
        })
    return rows


# ---------------------------------------------------------------- breakdown

def breakdown(conn, run_id: str, versions: dict[str, str], confidence: float,
              harness: dict) -> dict:
    """The Step 10 baseline report for one dev run, unchanged.

    Raises ScoringRefused when the run is held-out, not completed or not fully judged.
    """
    report_cfg = load_report_config(harness)
    return build_baseline_report(conn, run_id, versions, confidence, report_cfg)


# ---------------------------------------------------------------- run vs run

def compare_two(conn, base_run_id: str, new_run_id: str, versions: dict[str, str],
                harness: dict) -> dict:
    """experiments.compare_runs(base, new) plus the answers that changed wording.

    "changed_answers" lists queries whose word similarity to the base answer is below
    regression.answer_similarity_below (citation markers ignored, as in the gate).
    compare_runs raises ExperimentRefused for the same run twice, a different split,
    test set or query ids, and ScoringRefused for held-out or unfinished runs.
    """
    exp_cfg = load_experiments_config(harness)
    reg_cfg = load_regression_config(harness)
    confidence = load_checks_config(harness)["report_confidence"]
    cfg = {"n_bootstrap": exp_cfg["n_bootstrap"], "bootstrap_seed": exp_cfg["bootstrap_seed"],
           "confidence": confidence}
    result = compare_runs(conn, base_run_id, new_run_id, versions, cfg)

    outputs = {run_id: {o["query_id"]: o for o in store.get_outputs_with_queries(conn, run_id)}
               for run_id in (base_run_id, new_run_id)}
    changed = []
    for query_id, new_out in sorted(outputs[new_run_id].items()):
        old_out = outputs[base_run_id].get(query_id)
        if old_out is None or old_out["answer"] is None or new_out["answer"] is None:
            continue
        similarity = answer_similarity(old_out["answer"], new_out["answer"],
                                       reg_cfg["similarity_ignore_pattern"])
        if similarity < reg_cfg["answer_similarity_below"]:
            changed.append({"query_id": query_id, "type": new_out["type"],
                            "question": new_out["question"], "similarity": similarity,
                            "answer_base": old_out["answer"], "answer_new": new_out["answer"]})
    result["changed_answers"] = changed
    result["similarity_threshold"] = reg_cfg["answer_similarity_below"]
    return result


def stored_gate_results(conn, run_id: str) -> list[dict]:
    """Verdicts of the regression checks already stored for a run (read from run_reports)."""
    out = []
    for row in store.get_run_reports(conn, run_id):
        if row["kind"] != "regression":
            continue
        report = row["report"]
        out.append({"against_run_id": report["against_run_id"], "flag_version": report["flag_version"],
                    "verdict": report["gate"]["verdict"], "delta_points": report["gate"]["delta_points"],
                    "created_at": row["created_at"]})
    return out


# ---------------------------------------------------------------- review queue

def review_queue(conn, run_id: str, versions: dict[str, str]) -> list[dict]:
    """Every flagged answer of a run with what a reviewer needs, unreviewed first.

    Judge reasoning is included here; the page keeps it collapsed so Gaurav forms his own
    view first (Decision 2). A reviews row whose scores are missing gets None reasoning.
    """
    flags = store.get_flag_rows(conn, run_id)
    if not flags:
        return []
    outputs = {o["query_id"]: o for o in store.get_outputs_with_queries(conn, run_id)}
    queries = store.get_queries(conn, [f["query_id"] for f in flags])

    def by_query(version: str) -> dict[str, dict[str, dict]]:
        table: dict[str, dict[str, dict]] = defaultdict(dict)
        for row in store.get_scores(conn, run_id, version):
            table[row["query_id"]][row["metric"]] = row
        return table

    faith, rel = by_query(versions["faithfulness"]), by_query(versions["relevance"])
    claims: dict[str, list[dict]] = defaultdict(list)
    for claim in store.get_judge_claims(conn, run_id, versions["faithfulness"]):
        claims[claim["query_id"]].append({k: claim[k] for k in
                                          ("claim_index", "claim", "label", "reasoning")})
    queue = []
    for flag in flags:
        q = flag["query_id"]
        output, query = outputs.get(q), queries.get(q, {})
        ungrounded, relevance = faith.get(q, {}).get("ungrounded"), rel.get(q, {}).get("relevance")
        queue.append({
            "query_id": q, "type": query.get("type"), "question": query.get("question"),
            "reference_answer": query.get("reference_answer"),
            "answer": output["answer"] if output else None,
            "chunks": [{k: c.get(k) for k in ("chunk_id", "doc_id", "score", "text")}
                       for c in (output["retrieved_chunks"] or [])] if output else [],
            "flag_reasons": flag["flag_reasons"],
            "faithfulness": {"value": faith.get(q, {}).get("faithfulness", {}).get("value"),
                             "ungrounded": ungrounded["value"] if ungrounded else None,
                             "reasoning": ungrounded["reasoning"] if ungrounded else None},
            "claims": claims.get(q, []),
            "relevance": {"value": relevance["value"] if relevance else None,
                          "reasoning": relevance["reasoning"] if relevance else None},
            "human_label": flag["human_label"], "notes": flag["notes"],
            "labeler": flag["labeler"], "reviewed_at": flag["reviewed_at"],
        })
    queue.sort(key=lambda item: (item["human_label"] is not None, item["query_id"]))
    return queue


def review_progress(conn, run_id: str) -> dict:
    """{"flagged", "reviewed", "ok", "bad"} counts of a run's reviews rows."""
    rows = store.get_flag_rows(conn, run_id)
    labels = [r["human_label"] for r in rows if r["human_label"] is not None]
    return {"flagged": len(rows), "reviewed": len(labels),
            "ok": labels.count("ok"), "bad": labels.count("bad")}


def judge_labels(conn, run_id: str, versions: dict[str, str]) -> dict[str, str]:
    """"calibrated judge" only when a stored calibration is `measured`, else "uncalibrated judge"."""
    return {judge: ("calibrated judge"
                    if _calibration_status(conn, run_id, judge, versions[judge])["status"] == "measured"
                    else "uncalibrated judge")
            for judge in ("faithfulness", "relevance")}


def scorer_versions(harness: dict) -> dict[str, str]:
    """The current scorer versions (same call every CLI report uses)."""
    return current_versions(harness)
