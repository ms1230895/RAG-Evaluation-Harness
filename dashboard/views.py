"""Pages of the Streamlit dashboard (Step 15): setup, cached reads and one function per page.

dashboard/app.py only wires these functions into st.navigation; keeping them here lets
tests run any single page. See app.py for the rules every page follows.
"""

import sys
from contextlib import closing
from pathlib import Path

# Same sys.path fix as app.py, so the pages also import when a test loads this module directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from dashboard import figures
from evalharness import dashboard_data as data
from evalharness import runner, store
from evalharness.checks import ScoringRefused, load_checks_config
from evalharness.experiments import ExperimentRefused
from evalharness.regression import NOISE_NOTE, load_regression_config
from evalharness.report import _fmt_rate_pct
from evalharness.schemas import QUERY_TYPES
from evalharness.testset import resolve_path


# ---------------------------------------------------------------- setup (config, DB, versions)

def _setup() -> dict:
    """Load the harness config and check the DB file exists; stop the page with a message if not."""
    try:
        harness = runner.load_harness_config(runner.DEFAULT_HARNESS_CONFIG)
        dash_cfg = data.load_dashboard_config(harness)
        versions = data.scorer_versions(harness)
        confidence = load_checks_config(harness)["report_confidence"]
        labeler = load_regression_config(harness)["labeler"]
    except (ValueError, OSError) as error:
        st.error(f"Cannot load the harness config: {error}")
        st.stop()
    db_path = resolve_path(harness["db_path"])
    if not db_path.exists():   # store.connect would silently create an empty database
        st.error(f"Results database not found: {db_path}. Run an evaluation first.")
        st.stop()
    return {"harness": harness, "dash": dash_cfg, "versions": versions,
            "confidence": confidence, "labeler": labeler, "db_path": str(db_path)}


SETUP = _setup()
HARNESS, DASH, VERSIONS = SETUP["harness"], SETUP["dash"], SETUP["versions"]
CONFIDENCE, LABELER, DB_PATH = SETUP["confidence"], SETUP["labeler"], SETUP["db_path"]
TTL = DASH["cache_ttl_seconds"]
PCT = round(CONFIDENCE * 100)


# ---------------------------------------------------------------- cached reads
# Arguments are plain strings / tuples (never a connection), so Streamlit can cache by them.
# Each read opens and closes its own short-lived connection.

@st.cache_data(ttl=TTL)
def read_runs(db_path: str) -> dict:
    with closing(store.connect(db_path)) as conn:
        return data.eligible_runs(conn, VERSIONS)


@st.cache_data(ttl=TTL)
def read_trends(db_path: str, run_ids: tuple) -> list:
    with closing(store.connect(db_path)) as conn:
        return data.trend_table(conn, list(run_ids), VERSIONS, CONFIDENCE)


@st.cache_data(ttl=TTL)
def read_breakdown(db_path: str, run_id: str) -> dict:
    with closing(store.connect(db_path)) as conn:
        return data.breakdown(conn, run_id, VERSIONS, CONFIDENCE, HARNESS)


@st.cache_data(ttl=TTL)
def read_compare(db_path: str, base_run_id: str, new_run_id: str) -> dict:
    with closing(store.connect(db_path)) as conn:
        return data.compare_two(conn, base_run_id, new_run_id, VERSIONS, HARNESS)


@st.cache_data(ttl=TTL)
def read_gate(db_path: str, run_id: str) -> list:
    with closing(store.connect(db_path)) as conn:
        return data.stored_gate_results(conn, run_id)


@st.cache_data(ttl=TTL)
def read_accepted(db_path: str) -> str | None:
    with closing(store.connect(db_path)) as conn:
        rows = store.get_accepted_runs(conn, "dev")
    return rows[0]["run_id"] if rows else None


@st.cache_data(ttl=TTL)
def read_review_runs(db_path: str) -> list:
    with closing(store.connect(db_path)) as conn:
        return store.runs_with_reviews(conn)


@st.cache_data(ttl=TTL)
def read_queue(db_path: str, run_id: str) -> tuple:
    with closing(store.connect(db_path)) as conn:
        return data.review_queue(conn, run_id, VERSIONS), data.review_progress(conn, run_id)


@st.cache_data(ttl=TTL)
def read_judge_labels(db_path: str, run_id: str) -> dict:
    with closing(store.connect(db_path)) as conn:
        return data.judge_labels(conn, run_id, VERSIONS)


# ---------------------------------------------------------------- small helpers

def _excerpt(text: str | None) -> str:
    """Table cells only: first excerpt_chars characters (the full text is shown in expanders)."""
    if text is None:
        return "-"
    text = " ".join(text.split())
    limit = DASH["excerpt_chars"]
    return text if len(text) <= limit else text[:limit] + "..."


def _run_label(run: dict) -> str:
    return f"{run['config_name']} | {run['timestamp'][:16]} | {run['run_id']}"


def _rate_text(summary: dict) -> str:
    """'8.75% [4.30%, 16.98%] (7/80)' or '-' (n = 0)."""
    return "n = 0" if summary["n"] == 0 else _fmt_rate_pct(summary)


def _dev_runs() -> list[dict]:
    return [r for r in read_runs(DB_PATH)["eligible"] if r["split"] == "dev"]


def sidebar() -> None:
    with st.sidebar:
        st.caption(f"Database: {DB_PATH}")
        st.caption("Scorer versions: " + ", ".join(f"{k} {v}" for k, v in VERSIONS.items()))
        accepted = read_accepted(DB_PATH)
        st.caption(f"Accepted dev run: {accepted or 'none'}")
        st.caption("Judge numbers come from judges that are not calibrated against human "
                   "labels (Step 9 was inconclusive). Treat them as the judges' measurement, "
                   "not as true rates.")


# ---------------------------------------------------------------- page: trends

def page_trends() -> None:
    st.title("Trends across runs")
    runs = read_runs(DB_PATH)
    dev = [r for r in runs["eligible"] if r["split"] == "dev"]
    if not dev:
        st.info("No eligible dev runs yet.")
        return

    metric = st.selectbox("Metric", list(figures.TREND_METRICS),
                          format_func=lambda m: figures.TREND_METRICS[m]["label"])
    rows = read_trends(DB_PATH, tuple(r["run_id"] for r in dev))
    st.plotly_chart(figures.make_trend_figure(rows, metric, CONFIDENCE))
    st.caption("Dev split only. Each point is one run. Points are different configurations "
               "and repeats, not one system drifting over time. At n = 80 one answer is "
               "1.25 points: differences inside the interval bars are not meaningful.")

    table = pd.DataFrame([{
        "run": r["run_id"], "config": r["config_name"], "git": r["git_commit"],
        "ungrounded": _rate_text(r["rates"]["ungrounded"]),
        "bad answer": _rate_text(r["rates"]["bad_answer"]),
        "mean faithfulness": (f"{r['means']['faithfulness']['mean']:.2f}"
                              if r["means"]["faithfulness"]["mean"] is not None else "-"),
        "hit@4": _rate_text(r["rates"]["hit@4"]),
    } for r in rows])
    st.dataframe(table, hide_index=True)

    if runs["excluded"]:
        with st.expander(f"{len(runs['excluded'])} run(s) not shown, with the reason"):
            for item in runs["excluded"]:
                st.text(f"{item['run_id']} ({item['split']}): {item['reason']}")

    if DASH["show_heldout"]:
        held = [r for r in runs["eligible"] if r["split"] == "heldout"]
        st.subheader("Held-out (Step 12 confirmation)")
        st.caption("Run once each, read-only here, never on the dev chart and never used for "
                   "tuning. 20 queries: one answer is 5 points.")
        if held:
            held_rows = read_trends(DB_PATH, tuple(r["run_id"] for r in held))
            st.dataframe(pd.DataFrame([{
                "run": r["run_id"], "config": r["config_name"],
                "ungrounded": _rate_text(r["rates"]["ungrounded"]),
                "bad answer": _rate_text(r["rates"]["bad_answer"]),
            } for r in held_rows]), hide_index=True)
        else:
            st.info("No held-out run stored.")


# ---------------------------------------------------------------- page: compare

def page_compare() -> None:
    st.title("Run vs run")
    dev = _dev_runs()
    if len(dev) < 2:
        st.info("Need at least two eligible dev runs.")
        return
    by_id = {r["run_id"]: r for r in dev}
    ids = list(by_id)
    accepted = read_accepted(DB_PATH)
    left, right = st.columns(2)
    base_default = ids.index(accepted) if accepted in by_id else 0
    base_id = left.selectbox("Reference run (A)", ids, index=base_default,
                             format_func=lambda i: _run_label(by_id[i]))
    new_id = right.selectbox("New run (B)", ids, index=len(ids) - 1,
                             format_func=lambda i: _run_label(by_id[i]))
    if base_id == new_id:
        st.warning("Pick two different runs.")
        return
    try:
        result = read_compare(DB_PATH, base_id, new_id)
    except (ExperimentRefused, ScoringRefused) as error:
        st.error(f"These runs cannot be compared: {error}")
        return

    labels = result["judge_label"]
    st.caption(f"Faithfulness: {labels['faithfulness']}. Both runs scored with the same "
               "current scorer versions on the same test set.")
    base_u = result["baseline"]["ungrounded"]["all"]
    new_u = result["experiment"]["ungrounded"]["all"]
    c1, c2, c3 = st.columns(3)
    c1.metric("A: ungrounded", _rate_text(base_u), help=f"{PCT}% Wilson interval")
    c2.metric("B: ungrounded", _rate_text(new_u), help=f"{PCT}% Wilson interval")
    delta = result["delta_points"]
    c3.metric("B minus A (points)", "-" if delta is None else f"{delta:+.2f}",
              delta_color="off", help="Point estimate only; read the paired counts and p.")

    counts = result["flips"]["counts"]
    p = result["p_value"]
    st.write(f"Paired: improved {counts['improved']}, regressed {counts['regressed']}, "
             f"both ungrounded {counts['both_ungrounded']}, both grounded "
             f"{counts['both_grounded']}, not paired {counts['not_paired']}; "
             + ("no query flipped." if p is None else f"exact paired p = {p:.4f}."))
    diff = result["faithfulness_diff"]
    if diff["mean"] is not None:
        st.write(f"Mean faithfulness difference (B - A): {diff['mean']:+.3f} "
                 f"[{diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}] over {diff['n']} paired answers.")
    st.caption(NOISE_NOTE)
    for gate in read_gate(DB_PATH, new_id):
        if gate["against_run_id"] == base_id:
            st.caption(f"Stored regression check of B against A: {gate['verdict']} "
                       f"({gate['delta_points']:+.2f} points, flags {gate['flag_version']}).")

    # One table with a TEXT label per row (never colour alone).
    details = {}
    table_rows = []
    for item in result["flipped_answers"]:
        marker = "▲ improved" if item["direction"] == "improved" else "▼ regressed"
        table_rows.append({"change": marker, "query": item["query_id"], "type": item["type"],
                           "question": _excerpt(item["question"])})
        details[item["query_id"]] = item
    for item in result["changed_answers"]:
        table_rows.append({"change": f"≠ answer changed (similarity {item['similarity']:.2f})",
                           "query": item["query_id"], "type": item["type"],
                           "question": _excerpt(item["question"])})
        details.setdefault(item["query_id"], {
            "question": item["question"], "answer_baseline": item["answer_base"],
            "answer_experiment": item["answer_new"], "reasoning_baseline": None,
            "reasoning_experiment": None})
    st.subheader("Queries that changed")
    if not table_rows:
        st.info("No query flipped and no answer changed below the similarity threshold "
                f"({result['similarity_threshold']}).")
        return
    st.dataframe(pd.DataFrame(table_rows), hide_index=True)

    chosen = st.selectbox("Show both answers for", sorted(details))
    item = details[chosen]
    st.text(f"Question: {item['question']}")
    col_a, col_b = st.columns(2)
    col_a.markdown("**A (reference)**")
    col_a.text(item["answer_baseline"] or "-")
    col_b.markdown("**B (new)**")
    col_b.text(item["answer_experiment"] or "-")
    with st.expander("Judge reasoning"):
        col_a, col_b = st.columns(2)
        col_a.text(item.get("reasoning_baseline") or "-")
        col_b.text(item.get("reasoning_experiment") or "-")


# ---------------------------------------------------------------- page: review queue

def _save_review(run_id: str, query_id: str, label: str) -> None:
    """Button callback: write the review, clear the cache so the queue re-reads."""
    notes = st.session_state.get(f"notes_{run_id}_{query_id}", "").strip() or None
    try:
        with closing(store.connect(DB_PATH)) as conn:
            store.write_review(conn, run_id, query_id, label, notes, LABELER,
                               DASH["review_labels"])
    except ValueError as error:
        st.session_state["review_error"] = str(error)
        return
    st.session_state.pop("review_error", None)
    st.cache_data.clear()   # the queue is re-sorted: the next unreviewed answer is now current


def page_review_queue() -> None:
    st.title("Review queue")
    run_ids = read_review_runs(DB_PATH)
    if not run_ids:
        st.info("No flagged answers stored yet. Run `python -m evalharness.regression check`.")
        return
    run_id = st.selectbox("Run", run_ids, index=len(run_ids) - 1)
    queue, progress = read_queue(DB_PATH, run_id)
    st.progress(progress["reviewed"] / progress["flagged"] if progress["flagged"] else 0.0)
    st.write(f"{progress['reviewed']} / {progress['flagged']} reviewed "
             f"(ok {progress['ok']}, bad {progress['bad']})")
    st.caption("These are triage labels on flagged answers only. They measure how often a flag "
               "was right, not how many bad answers the flags missed, and they are not "
               "calibration labels for the judges.")
    if not queue:
        return

    key = f"queue_index_{run_id}"
    index = min(st.session_state.get(key, 0), len(queue) - 1)
    nav_prev, nav_next, _ = st.columns([1, 1, 6])
    if nav_prev.button("Previous", disabled=index == 0):
        index -= 1
    if nav_next.button("Next", disabled=index >= len(queue) - 1):
        index += 1
    st.session_state[key] = index
    item = queue[index]

    st.subheader(f"{item['query_id']} ({item['type']})   [{index + 1} of {len(queue)}]")
    st.text(f"Question: {item['question']}")
    st.text(f"Reference answer: {item['reference_answer']}")
    st.markdown("**Generated answer**")
    st.text(item["answer"] or "-")
    st.write("Flag reasons: " + ", ".join(item["flag_reasons"]))
    with st.expander(f"Retrieved chunks ({len(item['chunks'])})"):
        for chunk in item["chunks"]:
            st.text(f"{chunk['doc_id']} / {chunk['chunk_id']}   score {chunk['score']:.3f}")
            st.text(chunk["text"])
    with st.expander("Judge reasoning (collapsed so you form your own view first)"):
        st.text(f"Faithfulness {item['faithfulness']['value']}, ungrounded "
                f"{item['faithfulness']['ungrounded']}: {item['faithfulness']['reasoning']}")
        for claim in item["claims"]:
            st.text(f"claim {claim['claim_index']} [{claim['label']}]: {claim['claim']}\n"
                    f"   {claim['reasoning']}")
        st.text(f"Relevance {item['relevance']['value']}: {item['relevance']['reasoning']}")

    if item["human_label"] is not None:
        st.info(f"Current review: {item['human_label']} by {item['labeler']} at "
                f"{item['reviewed_at']}. Saving again replaces it.")
    st.text_area("Notes (optional)", key=f"notes_{run_id}_{item['query_id']}",
                 value=item["notes"] or "")
    ok_col, bad_col, _ = st.columns([1, 1, 6])
    ok_col.button("OK", on_click=_save_review, args=(run_id, item["query_id"], "ok"))
    bad_col.button("Bad", on_click=_save_review, args=(run_id, item["query_id"], "bad"))
    if "review_error" in st.session_state:
        st.error(st.session_state["review_error"])


# ---------------------------------------------------------------- page: breakdown

def page_breakdown() -> None:
    st.title("Breakdown by query type")
    dev = _dev_runs()
    if not dev:
        st.info("No eligible dev runs yet.")
        return
    by_id = {r["run_id"]: r for r in dev}
    ids = list(by_id)
    accepted = read_accepted(DB_PATH)
    run_id = st.selectbox("Run", ids, index=ids.index(accepted) if accepted in by_id
                          else len(ids) - 1, format_func=lambda i: _run_label(by_id[i]))
    try:
        report = read_breakdown(DB_PATH, run_id)
    except ScoringRefused as error:
        st.error(f"This run cannot be reported: {error}")
        return
    labels = read_judge_labels(DB_PATH, run_id)
    st.write(f"All queries: ungrounded {_rate_text(report['ungrounded']['all'])} "
             f"({labels['faithfulness']})")

    types = list(QUERY_TYPES)
    st.plotly_chart(figures.make_breakdown_figure(report, types))
    if figures.intervals_overlap([report["ungrounded"][t] for t in types]):
        st.caption("Difference is within noise at this sample size: every pair of "
                   "intervals overlaps.")
    st.caption("Per-type n is 12-16 on dev; the intervals are wide.")
    st.plotly_chart(figures.make_cause_figure(report, types))
    st.caption("Retrieval miss = no chunk of a source document reached the prompt. "
               "Generation error = the source was there (or no source exists) and the "
               "answer added unsupported claims.")

    x = report["crosstab"]
    st.write(f"Faithfulness x relevance (n = {x['n']}):")
    st.dataframe(pd.DataFrame(
        {"relevant": [x["grounded_relevant"], x["ungrounded_relevant"]],
         "irrelevant": [x["grounded_irrelevant"], x["ungrounded_irrelevant"]]},
        index=["grounded", "ungrounded"]))
    for judge, status in report["calibration"].items():
        st.caption(f"{judge}: {labels[judge]}; calibration status: {status['status']}.")


