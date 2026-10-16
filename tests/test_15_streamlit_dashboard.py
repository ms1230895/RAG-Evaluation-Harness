"""Step 15 tests: Streamlit dashboard (data layer, store helpers, figures, page smoke tests).

Spec: .claude/specs/15-streamlit-dashboard.md. Offline: tmp DBs, `llm.client.generate` patched to
raise. Hand values come from the `experiment_db` / `dashboard_db` fixtures (see conftest docstrings):
baseline ungrounded 4/9, experiment 1/9; improved q002 q003 q006, regressed q005; McNemar p 0.625.
"""

import ast
import copy
import json
import shutil
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest
import yaml

from dashboard import figures
from evalharness import dashboard_data as dd
from evalharness import runner, store
from evalharness.checks import ScoringRefused
from evalharness.experiments import ExperimentRefused
from evalharness.metrics import rate_summary, wilson_interval

REPO = Path(__file__).resolve().parents[1]
BASE, EXP = "run-baseline-001", "run-exp-001"
CONF = 0.95


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """The dashboard makes no LLM call at all: any call fails the test."""
    import llm.client

    def boom(*args, **kwargs):
        raise AssertionError("dashboard must not call the LLM")
    monkeypatch.setattr(llm.client, "generate", boom)


@pytest.fixture
def harness():
    """The real configs/harness.yaml, loaded read-only."""
    return runner.load_harness_config(runner.DEFAULT_HARNESS_CONFIG)


def hand_summary(successes, n):
    """A rate_summary dict from k successes of n (values 1 then 0)."""
    return rate_summary([1] * successes + [0] * (n - successes), CONF)


# ---------------------------------------------------------------- config

def test_load_dashboard_config_real_block_accepted(harness):
    """Spec Tests/Config: load_dashboard_config accepts the real harness.yaml block."""
    cfg = dd.load_dashboard_config(harness)
    assert cfg["review_labels"] == ["ok", "bad"]
    assert cfg["cache_ttl_seconds"] >= 1 and isinstance(cfg["show_heldout"], bool)


@pytest.mark.parametrize("mutate", [
    lambda b: b.pop("excerpt_chars"),
    lambda b: b.update(surprise=1),
    lambda b: b.update(cache_ttl_seconds=1.5),
    lambda b: b.update(review_labels=["ok", "bad", "maybe"]),
    lambda b: b.update(show_heldout="yes"),
])
def test_load_dashboard_config_bad_block_raises(harness, mutate):
    """Spec Tests/Config: missing key, unknown key, non-integer ttl, other labels raise ValueError."""
    broken = copy.deepcopy(harness)
    mutate(broken["dashboard"])
    with pytest.raises(ValueError):
        dd.load_dashboard_config(broken)


# ---------------------------------------------------------------- eligibility

def test_eligible_runs_lists_excluded_with_reason_and_keeps_heldout_separate(experiment_db):
    """Spec: unfinished / failed / missing-scorer runs are excluded with a reason; held-out has split heldout."""
    db = experiment_db
    db.add_run("run-running", status="running")
    db.add_run("run-aborted", status="aborted")
    db.add_run("run-failed")
    with db.conn:
        db.conn.execute("UPDATE runs SET n_failed = 2 WHERE run_id = 'run-failed'")
    db.add_run("run-noscore", scorers=("checks", "faithfulness"))
    db.add_run("run-held", split="heldout")

    result = dd.eligible_runs(db.conn, db.versions)
    excluded = {e["run_id"]: e["reason"] for e in result["excluded"]}
    assert set(excluded) == {"run-running", "run-aborted", "run-failed", "run-noscore"}
    assert all(isinstance(r, str) and r for r in excluded.values())
    assert "complete" in excluded["run-running"] and "complete" in excluded["run-aborted"]
    assert "fail" in excluded["run-failed"] and "scorer" in excluded["run-noscore"]
    by_id = {r["run_id"]: r for r in result["eligible"]}
    assert set(by_id) == {BASE, EXP, "run-held"}
    assert by_id["run-held"]["split"] == "heldout"
    assert {r["run_id"] for r in result["eligible"] if r["split"] == "dev"} == {BASE, EXP}


def test_eligible_runs_row_has_stem_and_short_commit(experiment_db):
    """Spec: run row has config_name (file stem of config_path) and a short git commit."""
    row = {r["run_id"]: r for r in dd.eligible_runs(experiment_db.conn, experiment_db.versions)["eligible"]}[BASE]
    assert row["config_name"] == "fake"           # config_path "configs/fake.yaml"
    assert row["git_commit"] == experiment_db.commit[:7]
    assert row["n_ok"] == 10 and row["n_failed"] == 0


# ---------------------------------------------------------------- trend_table

def test_trend_table_ungrounded_equals_hand_rate_and_wilson(experiment_db):
    """Spec: trend rates are rate_summary dicts; baseline 4/9, experiment 1/9; oldest first."""
    rows = dd.trend_table(experiment_db.conn, [EXP, BASE], experiment_db.versions, CONF)
    assert [r["run_id"] for r in rows] == [BASE, EXP]      # oldest first, whatever the input order
    base_u, exp_u = rows[0]["rates"]["ungrounded"], rows[1]["rates"]["ungrounded"]
    assert (base_u["successes"], base_u["n"]) == (4, 9)
    assert (exp_u["successes"], exp_u["n"]) == (1, 9)
    assert base_u["rate"] == pytest.approx(4 / 9)
    low, high = wilson_interval(4, 9, CONF)
    assert (base_u["ci_low"], base_u["ci_high"]) == (pytest.approx(low), pytest.approx(high))


def test_trend_table_experiment_retrieval_and_means_have_no_interval(experiment_db):
    """Hand values (exp run): in_context 7/8, hit@4 7/8, MRR 5.25/8, mean faithfulness 8.5/9."""
    row = dd.trend_table(experiment_db.conn, [EXP], experiment_db.versions, CONF)[0]
    assert (row["rates"]["in_context"]["successes"], row["rates"]["in_context"]["n"]) == (7, 8)
    assert (row["rates"]["hit@4"]["successes"], row["rates"]["hit@4"]["n"]) == (7, 8)
    # MRR: (1 + .5 + 1 + 0 + 1 + .25 + .5 + 1) / 8 = 5.25 / 8 = 0.65625
    assert row["means"]["mrr"]["mean"] == pytest.approx(0.65625)
    # faithfulness: eight 1.0 and one 0.5 over the 9 judged answers = 8.5 / 9
    assert row["means"]["faithfulness"]["mean"] == pytest.approx(8.5 / 9)
    assert "ci_low" not in row["means"]["faithfulness"]


def test_trend_table_three_of_ten_matches_wilson(experiment_db):
    """Spec Tests: a hand 3/10 gives metrics.wilson_interval (about [0.1078, 0.6032])."""
    experiment_db.add_simple_run("run-simple", ["q002", "q003", "q006"])
    summary = dd.trend_table(experiment_db.conn, ["run-simple"], experiment_db.versions, CONF)[0]["rates"]["ungrounded"]
    assert (summary["successes"], summary["n"]) == (3, 10)
    assert summary["ci_low"] == pytest.approx(0.1078, abs=1e-3)
    assert summary["ci_high"] == pytest.approx(0.6032, abs=1e-3)


def test_trend_table_mixed_splits_or_unknown_run_raises(experiment_db):
    """Spec: dev and held-out never mix in one table; an unknown run id is an error."""
    experiment_db.add_run("run-held", split="heldout")
    with pytest.raises(ValueError):
        dd.trend_table(experiment_db.conn, [BASE, "run-held"], experiment_db.versions, CONF)
    with pytest.raises(ValueError):
        dd.trend_table(experiment_db.conn, ["no-such-run"], experiment_db.versions, CONF)


# ---------------------------------------------------------------- breakdown

def test_breakdown_equals_baseline_report_with_hand_counts(experiment_db, harness):
    """Spec: breakdown is the Step 10 report unchanged; simple run: 3/10, lookup 2/4, 1 miss + 2 generation errors."""
    from evalharness.report import build_baseline_report, load_report_config

    experiment_db.add_simple_run("run-simple", ["q002", "q003", "q006"])
    got = dd.breakdown(experiment_db.conn, "run-simple", experiment_db.versions, CONF, harness)
    want = build_baseline_report(experiment_db.conn, "run-simple", experiment_db.versions, CONF,
                                 load_report_config(harness))
    assert got == want
    assert (got["ungrounded"]["all"]["successes"], got["ungrounded"]["all"]["n"]) == (3, 10)
    assert (got["ungrounded"]["lookup"]["successes"], got["ungrounded"]["lookup"]["n"]) == (2, 4)
    # q003 has in_context 0 (retrieval miss); q002 and q006 have the source in the prompt
    assert got["causes"]["all"]["retrieval_miss"]["rate"]["successes"] == 1
    assert got["causes"]["all"]["generation_error"]["rate"]["successes"] == 2


def test_breakdown_refuses_incomplete_judging_and_heldout(experiment_db, harness):
    """Spec: ScoringRefused when the run has a judge failure (baseline q008) or is held-out."""
    experiment_db.add_run("run-held", split="heldout", cases=None)
    with pytest.raises(ScoringRefused):
        dd.breakdown(experiment_db.conn, BASE, experiment_db.versions, CONF, harness)
    with pytest.raises(ScoringRefused):
        dd.breakdown(experiment_db.conn, "run-held", experiment_db.versions, CONF, harness)


# ---------------------------------------------------------------- compare_two

def test_compare_two_flips_and_p_equal_hand_values(experiment_db, harness):
    """Spec Tests/Compare: improved / regressed lists equal paired_flips; exact p = 0.625."""
    result = dd.compare_two(experiment_db.conn, BASE, EXP, experiment_db.versions, harness)
    assert result["flips"]["improved"] == ["q002", "q003", "q006"]
    assert result["flips"]["regressed"] == ["q005"]
    assert result["flips"]["counts"]["not_paired"] == 2
    assert result["p_value"] == pytest.approx(0.625)
    assert result["judge_label"]["faithfulness"] == "uncalibrated judge"
    assert result["similarity_threshold"] == pytest.approx(0.6)


def test_compare_two_changed_answers_use_threshold_and_ignore_markers(experiment_db, harness):
    """Spec: answer changed when word similarity < 0.6; 0.588 in, exactly 0.6 out, [n] markers ignored."""
    conn = experiment_db.conn
    answers = {
        # 5 shared words of 8 and 9: 2*5/17 = 0.588 -> changed
        "q001": ("alpha bravo charlie delta echo foxtrot golf hotel",
                 "alpha bravo charlie delta echo xray yankee zulu whiskey"),
        # 3 shared words of 5 and 5: 2*3/10 = 0.6 -> NOT below 0.6
        "q002": ("alpha bravo charlie delta echo", "alpha bravo charlie xray yankee"),
        # markers only: without the ignore pattern 2*1/5 = 0.4 would count as changed
        "q003": ("[1] [2] [3] alpha", "alpha"),
    }
    with conn:
        for query_id, (old, new) in answers.items():
            conn.execute("UPDATE outputs SET answer = ? WHERE run_id = ? AND query_id = ?", (old, BASE, query_id))
            conn.execute("UPDATE outputs SET answer = ? WHERE run_id = ? AND query_id = ?", (new, EXP, query_id))
    changed = dd.compare_two(conn, BASE, EXP, experiment_db.versions, harness)["changed_answers"]
    assert [c["query_id"] for c in changed] == ["q001"]
    assert changed[0]["similarity"] == pytest.approx(10 / 17)
    assert changed[0]["answer_base"].startswith("alpha") and changed[0]["answer_new"].endswith("whiskey")


def test_compare_two_refuses_same_run_other_test_set_and_heldout(experiment_db, harness):
    """Spec: same run twice, different test-set sha256, held-out run are all refused."""
    experiment_db.add_run("run-othersha", sha="cd" * 32)
    experiment_db.add_run("run-held", split="heldout")
    args = (experiment_db.versions, harness)
    with pytest.raises(ExperimentRefused):
        dd.compare_two(experiment_db.conn, BASE, BASE, *args)
    with pytest.raises(ExperimentRefused):
        dd.compare_two(experiment_db.conn, BASE, "run-othersha", *args)
    with pytest.raises((ExperimentRefused, ScoringRefused)):
        dd.compare_two(experiment_db.conn, BASE, "run-held", *args)


def test_judge_labels_uncalibrated_without_measured_calibration(experiment_db):
    """Spec: judge label is 'uncalibrated judge' until a stored calibration is measured."""
    labels = dd.judge_labels(experiment_db.conn, BASE, experiment_db.versions)
    assert labels == {"faithfulness": "uncalibrated judge", "relevance": "uncalibrated judge"}


# ---------------------------------------------------------------- review queue

def test_review_queue_unreviewed_first_then_query_id(dashboard_db):
    """Spec: unreviewed first, then by query_id; a reviewed item moves to the end."""
    conn = dashboard_db.conn
    assert [i["query_id"] for i in dd.review_queue(conn, EXP, dashboard_db.versions)] == ["q001", "q003", "q005"]
    store.write_review(conn, EXP, "q001", "ok", None, "tester", ["ok", "bad"])
    assert [i["query_id"] for i in dd.review_queue(conn, EXP, dashboard_db.versions)] == ["q003", "q005", "q001"]


def test_review_queue_item_carries_chunks_claims_and_reasoning(dashboard_db):
    """Spec: item has question, reference, answer, chunk doc_id/chunk_id/score, claims, both reasonings."""
    item = {i["query_id"]: i for i in dd.review_queue(dashboard_db.conn, EXP, dashboard_db.versions)}["q005"]
    assert item["type"] == "multi_chunk"
    assert item["question"] == "Baseline question number 5?"
    assert item["reference_answer"] == "Reference q005."
    assert item["answer"] == f"Answer {EXP} q005."
    assert item["flag_reasons"] == ["low_relevance", "ungrounded"]
    chunk = item["chunks"][0]
    assert (chunk["doc_id"], chunk["chunk_id"], chunk["score"]) == ("leave_policy", "leave_policy#000", 0.5)
    assert [(c["claim"], c["label"]) for c in item["claims"]] == [("CLAIM-ZERO", "supported"), ("CLAIM-ONE", "contradicted")]
    assert item["claims"][1]["reasoning"] == "claim reasoning one"
    assert item["faithfulness"]["reasoning"] == f"ungrounded reasoning {EXP} q005"
    assert item["relevance"]["reasoning"] == "hand"


def test_review_progress_counts_and_run_without_flags(dashboard_db):
    """Spec: progress is {flagged, reviewed, ok, bad}; a run without flags has an empty queue."""
    conn = dashboard_db.conn
    assert dd.review_progress(conn, EXP) == {"flagged": 3, "reviewed": 0, "ok": 0, "bad": 0}
    store.write_review(conn, EXP, "q001", "ok", None, "tester", ["ok", "bad"])
    store.write_review(conn, EXP, "q003", "bad", "n", "tester", ["ok", "bad"])
    assert dd.review_progress(conn, EXP) == {"flagged": 3, "reviewed": 2, "ok": 1, "bad": 1}
    assert dd.review_queue(conn, BASE, dashboard_db.versions) == []
    assert dd.review_progress(conn, BASE)["flagged"] == 0


# ---------------------------------------------------------------- store.write_review

def test_write_review_stores_label_notes_labeler_and_time(dashboard_db):
    """Spec: write_review sets human_label, notes, labeler, reviewed_at on a flagged row."""
    store.write_review(dashboard_db.conn, EXP, "q003", "bad", "looks wrong", "gaurav", ["ok", "bad"])
    row = {r["query_id"]: r for r in store.get_flag_rows(dashboard_db.conn, EXP)}["q003"]
    assert (row["human_label"], row["notes"], row["labeler"]) == ("bad", "looks wrong", "gaurav")
    assert row["reviewed_at"]
    assert row["flag_reasons"] == ["ungrounded"]
    other = {r["query_id"]: r for r in store.get_flag_rows(dashboard_db.conn, EXP)}["q001"]
    assert other["human_label"] is None and other["reviewed_at"] is None


def test_write_review_refuses_other_label_and_unflagged_row_without_inserting(dashboard_db):
    """Spec: ValueError for a label outside the vocabulary or a row that is not flagged; no row inserted."""
    conn = dashboard_db.conn
    before = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    with pytest.raises(ValueError):
        store.write_review(conn, EXP, "q001", "maybe", None, "gaurav", ["ok", "bad"])
    with pytest.raises(ValueError):
        store.write_review(conn, EXP, "q002", "ok", None, "gaurav", ["ok", "bad"])   # q002 not flagged
    assert conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == before
    assert conn.execute("SELECT COUNT(*) FROM reviews WHERE human_label IS NOT NULL").fetchone()[0] == 0


def test_write_review_changes_only_human_fields(dashboard_db):
    """Spec: scores, outputs, runs, judge_claims and flag_reasons are identical before and after."""
    conn = dashboard_db.conn

    def snapshot():
        tables = {t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY 1, 2, 3, 4")]
                  for t in ("scores", "outputs", "runs", "judge_claims")}
        tables["flags"] = [(r["query_id"], r["flag_reasons"]) for r in store.get_flag_rows(conn, EXP)]
        return tables

    before = snapshot()
    store.write_review(conn, EXP, "q005", "ok", "fine", "gaurav", ["ok", "bad"])
    assert snapshot() == before


def test_write_review_relabel_latest_wins(dashboard_db):
    """Spec Decision 5: re-labelling is allowed and the latest label wins."""
    conn = dashboard_db.conn
    store.write_review(conn, EXP, "q001", "ok", None, "gaurav", ["ok", "bad"])
    store.write_review(conn, EXP, "q001", "bad", "changed my mind", "gaurav", ["ok", "bad"])
    row = {r["query_id"]: r for r in store.get_flag_rows(conn, EXP)}["q001"]
    assert (row["human_label"], row["notes"]) == ("bad", "changed my mind")


def test_replace_flags_keeps_labelled_review_row(dashboard_db):
    """Spec: re-checking a run (delete unlabelled + rewrite flags) never deletes a review."""
    conn = dashboard_db.conn
    store.write_review(conn, EXP, "q001", "ok", None, "gaurav", ["ok", "bad"])
    assert store.delete_unlabelled_flag_rows(conn, EXP) == 2
    rows = store.get_flag_rows(conn, EXP)
    assert [(r["query_id"], r["human_label"]) for r in rows] == [("q001", "ok")]
    store.write_flag_rows(conn, EXP, {"q001": ["answer_changed"], "q004": ["low_relevance"]})
    kept = {r["query_id"]: r for r in store.get_flag_rows(conn, EXP)}["q001"]
    assert kept["human_label"] == "ok"


# ---------------------------------------------------------------- store read helpers

def test_list_runs_filters_by_split_oldest_first(experiment_db):
    """Spec: list_runs returns run rows oldest first, optionally one split."""
    experiment_db.add_run("run-held", split="heldout")
    all_ids = [r["run_id"] for r in store.list_runs(experiment_db.conn)]
    assert set(all_ids) == {BASE, EXP, "run-held"}
    assert [r["run_id"] for r in store.list_runs(experiment_db.conn, "heldout")] == ["run-held"]
    assert {r["run_id"] for r in store.list_runs(experiment_db.conn, "dev")} == {BASE, EXP}
    assert {"run_id", "status", "git_dirty", "judge_version", "n_ok", "n_failed"} <= set(store.list_runs(experiment_db.conn)[0])


def test_runs_with_versions_needs_every_version(experiment_db):
    """Spec: only runs with score rows for ALL given versions are returned."""
    experiment_db.add_run("run-two", scorers=("checks", "faithfulness"))
    versions = list(experiment_db.versions.values())
    assert store.runs_with_versions(experiment_db.conn, versions) == {BASE, EXP}
    assert "run-two" in store.runs_with_versions(experiment_db.conn, versions[:2])


def test_get_queries_returns_reference_and_sources_and_runs_with_reviews(dashboard_db):
    """Spec: get_queries has reference_answer and parsed source_doc_ids; runs_with_reviews lists flagged runs."""
    got = store.get_queries(dashboard_db.conn, ["q005", "q009", "nope"])
    assert set(got) == {"q005", "q009"}
    assert got["q005"]["reference_answer"] == "Reference q005."
    assert got["q005"]["source_doc_ids"] == ["leave_policy", "payroll_policy"]
    assert got["q009"]["source_doc_ids"] == []
    assert store.runs_with_reviews(dashboard_db.conn) == [EXP]


# ---------------------------------------------------------------- schema migration

def test_connect_adds_review_columns_once_and_keeps_old_rows(tmp_path):
    """Spec: old reviews layout gets labeler + reviewed_at once; rerunning connect is a no-op; rows kept."""
    path = tmp_path / "old.sqlite"
    with closing(sqlite3.connect(path)) as raw:
        raw.execute("CREATE TABLE reviews (run_id TEXT NOT NULL, query_id TEXT NOT NULL, flag_reasons TEXT,"
                    " human_label TEXT, notes TEXT, PRIMARY KEY (run_id, query_id))")
        raw.execute("INSERT INTO reviews VALUES ('r1', 'q1', '[\"x\"]', 'ok', 'old note')")
        raw.commit()
    for _ in range(2):
        with closing(store.connect(path)) as conn:
            names = [r[1] for r in conn.execute("PRAGMA table_info(reviews)")]
            assert names.count("labeler") == 1 and names.count("reviewed_at") == 1
            row = conn.execute("SELECT run_id, flag_reasons, human_label, notes, labeler, reviewed_at"
                               " FROM reviews").fetchone()
            assert tuple(row) == ("r1", '["x"]', "ok", "old note", None, None)


def test_fresh_database_reviews_table_has_new_columns(tmp_path):
    """Spec: a new results DB already has labeler and reviewed_at on reviews."""
    with closing(store.connect(tmp_path / "new.sqlite")) as conn:
        names = {r[1] for r in conn.execute("PRAGMA table_info(reviews)")}
    assert names == {"run_id", "query_id", "flag_reasons", "human_label", "notes", "labeler", "reviewed_at"}


# ---------------------------------------------------------------- figures

def test_make_trend_figure_axis_customdata_and_values(experiment_db):
    """Spec Figures: y-axis [0, 1], points at the stored rates, customdata holds k/n, error bars set."""
    rows = dd.trend_table(experiment_db.conn, [BASE, EXP], experiment_db.versions, CONF)
    fig = figures.make_trend_figure(rows, "ungrounded", CONF)
    assert tuple(fig.layout.yaxis.range) == (0, 1)
    ys = sorted(y for trace in fig.data for y in trace.y)
    assert ys == [pytest.approx(1 / 9), pytest.approx(4 / 9)]
    kn = {str(row[3]) for trace in fig.data for row in trace.customdata}
    assert kn == {"4/9", "1/9"}
    assert all(trace.error_y.array is not None for trace in fig.data)


def test_make_trend_figure_rejects_unknown_metric_and_labels_mean(experiment_db):
    """Spec: metric names go through an allow-list; a mean metric says 'no interval'."""
    rows = dd.trend_table(experiment_db.conn, [BASE], experiment_db.versions, CONF)
    with pytest.raises(ValueError):
        figures.make_trend_figure(rows, "latency; DROP TABLE", CONF)
    assert "no interval" in figures.make_trend_figure(rows, "faithfulness", CONF).layout.title.text


def test_make_trend_figure_does_not_draw_n_zero_as_zero(experiment_db):
    """Spec: a rate with n = 0 is not drawn as 0%."""
    rows = dd.trend_table(experiment_db.conn, [BASE, EXP], experiment_db.versions, CONF)
    rows[0] = copy.deepcopy(rows[0])
    rows[0]["rates"]["ungrounded"] = rate_summary([], CONF)
    fig = figures.make_trend_figure(rows, "ungrounded", CONF)
    ys = [y for trace in fig.data for y in trace.y]
    assert ys == [pytest.approx(1 / 9)]


def fake_report():
    """Hand report dict: lookup 3/10, multi_chunk 0/5, unanswerable 0/0."""
    names = ["lookup", "multi_chunk", "unanswerable"]
    ungrounded = {"lookup": hand_summary(3, 10), "multi_chunk": hand_summary(0, 5), "unanswerable": rate_summary([], CONF)}
    causes = {n: {"retrieval_miss": {"rate": hand_summary(1, 10)}, "generation_error": {"rate": hand_summary(2, 10)}}
              for n in names}
    return {"confidence": CONF, "ungrounded": ungrounded, "causes": causes}, names


def test_make_breakdown_figure_axis_kn_and_stable_type_colors():
    """Spec: y-axis [0, 1], k/n per bar, a query type keeps its colour whatever else is drawn."""
    report, names = fake_report()
    fig = figures.make_breakdown_figure(report, names)
    assert tuple(fig.layout.yaxis.range) == (0, 1)
    kn = [str(row[1]) for row in fig.data[0].customdata]
    assert kn == ["3/10", "0/5", "n = 0"]
    colors = dict(zip(names, fig.data[0].marker.color))
    fig2 = figures.make_breakdown_figure(report, ["unanswerable", "lookup"])
    colors2 = dict(zip(["unanswerable", "lookup"], fig2.data[0].marker.color))
    assert colors2["lookup"] == colors["lookup"] and colors2["unanswerable"] == colors["unanswerable"]
    assert len(set(colors.values())) == 3


def test_make_cause_figure_draws_both_causes_on_full_axis():
    """Spec: cause figure has retrieval_miss and generation_error bars at the stored rates."""
    report, names = fake_report()
    fig = figures.make_cause_figure(report, names)
    assert tuple(fig.layout.yaxis.range) == (0, 1)
    assert {t.name for t in fig.data} == {"retrieval_miss", "generation_error"}
    miss = next(t for t in fig.data if t.name == "retrieval_miss")
    assert list(miss.y)[:2] == [pytest.approx(0.1), pytest.approx(0.1)]


def test_intervals_overlap_true_and_false_by_hand():
    """Spec: True when every pair of Wilson intervals overlaps.

    3/10 [0.108, 0.603] vs 4/10 [0.168, 0.687] overlap; 0/40 [0, 0.088] vs 30/40 [0.60, 0.87] do not.
    """
    assert figures.intervals_overlap([hand_summary(3, 10), hand_summary(4, 10)]) is True
    assert figures.intervals_overlap([hand_summary(0, 40), hand_summary(30, 40)]) is False
    # one separated pair among three overlapping ones makes the answer False
    assert figures.intervals_overlap([hand_summary(3, 10), hand_summary(4, 10), hand_summary(0, 40)]) is False


# ---------------------------------------------------------------- layering and safety

DASH_FILES = sorted((REPO / "dashboard").glob("*.py")) + [REPO / "evalharness" / "dashboard_data.py"]


def imported_modules(path):
    mods = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


@pytest.mark.parametrize("path", DASH_FILES, ids=lambda p: p.name)
def test_dashboard_modules_import_nothing_from_rag_or_llm(path):
    """Spec Layering: dashboard/ and dashboard_data.py import nothing from rag/ or llm/."""
    assert not imported_modules(path) & {"rag", "llm"}


@pytest.mark.parametrize("path", [REPO / "evalharness" / "dashboard_data.py", REPO / "dashboard" / "figures.py"],
                         ids=lambda p: p.name)
def test_pure_modules_do_not_import_streamlit(path):
    """Spec: dashboard_data.py and figures.py have no streamlit import."""
    assert "streamlit" not in imported_modules(path)


def test_dashboard_sources_never_use_unsafe_allow_html():
    """Spec Safety: untrusted text never goes through unsafe_allow_html."""
    for path in (REPO / "dashboard").glob("*.py"):
        assert "unsafe_allow_html" not in path.read_text(encoding="utf-8"), path.name


# ---------------------------------------------------------------- AppTest page smoke tests

def forget_views():
    """Drop dashboard.views from sys.modules AND from the package: `from dashboard import views`
    returns the package attribute first, so popping sys.modules alone keeps a stale module."""
    import dashboard

    sys.modules.pop("dashboard.views", None)
    if hasattr(dashboard, "views"):
        delattr(dashboard, "views")


@pytest.fixture
def page_env(dashboard_db, tmp_path, monkeypatch):
    """Dashboard fixture DB with the current scorer versions, a tmp harness YAML and a fresh views import.

    Extra runs: run-simple (dev, 3/10 ungrounded), run-held-001 (heldout), run-noscore (excluded).
    Returns a factory `make(readonly=False)` -> namespace with `.db` (fixture), `.path` (DB the pages use).
    """
    import streamlit as st
    from types import SimpleNamespace

    from evalharness.report import current_versions

    db = dashboard_db
    db.add_simple_run("run-simple", ["q002", "q003", "q006"])
    db.add_run("run-held-001", split="heldout")
    db.add_run("run-noscore", scorers=("checks",))
    current = current_versions(runner.load_harness_config(runner.DEFAULT_HARNESS_CONFIG))
    with db.conn:   # relabel the fixture scorings as the CURRENT versions (the pages use those)
        for scorer, old in db.versions.items():
            db.conn.execute("UPDATE scores SET version = ? WHERE version = ?", (current[scorer], old))
        db.conn.execute("UPDATE judge_claims SET version = ? WHERE version = ?",
                        (current["faithfulness"], db.versions["faithfulness"]))

    def make(readonly=False):
        path = db.path
        if readonly:
            path = tmp_path / "readonly.sqlite"
            shutil.copy(db.path, path)
            path.chmod(0o444)
        harness = yaml.safe_load((REPO / "configs" / "harness.yaml").read_text(encoding="utf-8"))
        harness["db_path"] = str(path)
        harness_path = tmp_path / "harness_pages.yaml"
        harness_path.write_text(yaml.safe_dump(harness), encoding="utf-8")
        monkeypatch.setattr(runner, "DEFAULT_HARNESS_CONFIG", str(harness_path))
        forget_views()
        st.cache_data.clear()
        return SimpleNamespace(db=db, path=path)
    yield make
    forget_views()
    st.cache_data.clear()


def run_page(name):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(f"from dashboard import views\nviews.{name}()")
    at.run(timeout=60)
    return at


def page_text(at):
    parts = []
    for kind in ("markdown", "text", "caption", "error", "warning", "info", "subheader", "title"):
        parts += [str(e.value) for e in getattr(at, kind)]
    return "\n".join(parts)


@pytest.mark.parametrize("page", ["page_trends", "page_compare", "page_review_queue", "page_breakdown"])
def test_each_page_runs_on_fixture_db_without_exception(page_env, page):
    """Spec AppTest smoke: each page renders on the fixture DB without an exception."""
    page_env()
    at = run_page(page)
    assert not at.exception


def test_trends_page_lists_excluded_runs_and_separate_heldout_table(page_env):
    """Spec: held-out runs only in their own table; excluded runs listed, not silently dropped."""
    page_env()
    at = run_page("page_trends")
    assert not at.exception
    text = page_text(at)
    assert "Held-out" in text
    assert any("not shown" in str(e.label) for e in at.expander)   # the excluded-runs expander
    # the dev table must not contain the held-out run
    dev_table = at.dataframe[0].value
    assert "run-held-001" not in set(dev_table["run"])
    assert {BASE, EXP, "run-simple"} <= set(dev_table["run"])


def test_compare_page_shows_hand_flip_counts_and_p(page_env):
    """Spec: compare page prints improved 3 / regressed 1 and the exact paired p 0.6250 (baseline vs experiment)."""
    page_env()
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from dashboard import views\nviews.page_compare()")
    at.run(timeout=60)
    at.selectbox[0].select(BASE)
    at.selectbox[1].select(EXP)
    at.run(timeout=60)
    assert not at.exception
    text = page_text(at)
    assert "improved 3, regressed 1" in text
    assert "p = 0.6250" in text
    assert any("improved" in str(v) for v in at.dataframe[0].value["change"])


def test_breakdown_page_shows_hand_rate_and_uncalibrated_label(page_env):
    """Spec: breakdown of run-simple shows 3/10 and the 'uncalibrated judge' status."""
    page_env()
    at = run_page("page_breakdown")
    assert not at.exception
    text = page_text(at)
    assert "3/10" in text and "uncalibrated judge" in text


def test_review_page_ok_click_stores_row_and_shows_progress(page_env):
    """Spec AppTest: after one OK click the DB has an 'ok' row for q001 and the page shows '1 / 3 reviewed'."""
    env = page_env()
    at = run_page("page_review_queue")
    assert "0 / 3 reviewed" in page_text(at)
    next(b for b in at.button if b.label == "OK").click()
    at.run(timeout=60)
    assert not at.exception
    assert "1 / 3 reviewed" in page_text(at)
    with closing(store.connect(env.path)) as conn:
        rows = conn.execute("SELECT query_id, flag_reasons, human_label, labeler, reviewed_at FROM reviews"
                            " WHERE run_id = ? AND human_label IS NOT NULL", (EXP,)).fetchall()
    assert len(rows) == 1
    assert (rows[0]["query_id"], rows[0]["human_label"]) == ("q001", "ok")
    assert json.loads(rows[0]["flag_reasons"]) == ["answer_changed"]
    assert rows[0]["labeler"] and rows[0]["reviewed_at"]


def test_review_page_renders_hostile_answer_as_plain_text(page_env):
    """Spec Safety: <script> / <b> in an answer reaches the page via st.text, not as HTML."""
    env = page_env()
    hostile = "<script>alert(1)</script><b>bold</b>"
    with env.db.conn:
        env.db.conn.execute("UPDATE outputs SET answer = ? WHERE run_id = ? AND query_id = 'q001'", (hostile, EXP))
    at = run_page("page_review_queue")
    assert not at.exception
    assert any(hostile == str(t.value) for t in at.text)


def test_viewing_pages_render_on_read_only_database(page_env):
    """Spec: with the DB file read-only, the three viewing pages still render."""
    page_env(readonly=True)
    for page in ("page_trends", "page_compare", "page_breakdown"):
        at = run_page(page)
        assert not at.exception, page
