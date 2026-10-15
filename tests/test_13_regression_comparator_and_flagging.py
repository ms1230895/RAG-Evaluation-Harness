"""Step 13 tests: regression comparator, flagging and gate (evalharness/regression.py).

Offline: tmp SQLite DBs built by the `make_env` fixture below, scorer versions patched at
`evalharness.regression.current_versions`, `generate` patched to raise. Expected values are
worked out by hand from the spec (gate examples, similarity 10/11, flag-reason rules).
"""
import ast
import copy
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evalharness import regression, store
from evalharness.checks import ScoringRefused
from evalharness.metrics import wilson_interval
from evalharness.regression import RegressionRefused
from evalharness.schemas import Query

REFUSED = (RegressionRefused, ScoringRefused)
VERSIONS = {"checks": "checks-t13", "faithfulness": "faith-t13", "relevance": "rel-t13"}
SHA = "cd" * 32
IGNORE = r"\[\d+\]"
REGRESSION_BLOCK = {
    "relevance_below": 0.75, "answer_similarity_below": 0.6, "similarity_ignore_pattern": IGNORE,
    "disagreement_rules": ["refusal_vs_relevance_a", "refusal_vs_relevance_b"],
    "max_ungrounded_rise_points": 3.0, "labeler": "gaurav",
}


# ---------------------------------------------------------------- fixtures and helpers

@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Any LLM call from this step is a bug: both import sites raise."""
    def boom(*args, **kwargs):
        raise AssertionError("Step 13 must not call the LLM")
    monkeypatch.setattr("llm.client.generate", boom)
    monkeypatch.setattr("evalharness.judges.generate", boom)


def make_harness(tmp_path, db_path, **regression_overrides):
    return {
        "db_path": str(db_path), "test_set_config": str(tmp_path / "ts.yaml"),
        "max_consecutive_failures": 5, "error_max_chars": 500,
        "system": {"answer": "x:y", "load_config": "x:y", "describe": None},
        "checks": {"hit_ks": [1, 4], "report_confidence": 0.95, "refusal_phrases": ["no mention of"]},
        "experiments": {"base_config": "configs/v0.yaml", "ignore_keys": ["name", "experiment"],
                        "n_bootstrap": 200, "bootstrap_seed": 7,
                        "table_path": str(tmp_path / "table.md")},
        "regression": {**REGRESSION_BLOCK, **regression_overrides},
    }


def block(**overrides):
    return {"regression": {**copy.deepcopy(REGRESSION_BLOCK), **overrides}}


@pytest.fixture
def cfg():
    return regression.load_regression_config(block())


@pytest.fixture
def make_env(tmp_path, make_query, monkeypatch):
    """Factory (n queries, default 10) -> namespace with `.conn`, `.path`, `.harness`,
    `.harness_path`, `.ids`, `.extra_id`, `.add_run(...)`, `.accept(run_id)`.

    Queries q001..q<n> are dev; the last max(2, n // 10) are unanswerable; one extra lookup
    query exists (for the "different query ids" case). Default run: every answer
    "answer <qid>", answerable refusal 0 / unanswerable refusal 1, relevance 1.0, none ungrounded.
    """
    monkeypatch.setattr(regression, "current_versions", lambda harness: dict(VERSIONS))
    opened = []

    def _make(n=10):
        path = tmp_path / f"reg_{len(opened)}.sqlite"
        conn = store.connect(path)
        opened.append(conn)
        n_unans = max(2, n // 10)
        ids = [f"q{i:03d}" for i in range(1, n + 1)]
        extra_id = f"q{n + 1:03d}"
        unans = set(ids[-n_unans:])
        queries = []
        for qid in ids + [extra_id]:
            u = qid in unans
            queries.append(Query.model_validate(make_query(
                id=qid, question=f"Question number {qid}?", type="unanswerable" if u else "lookup",
                reference_answer=f"Reference {qid}.", source_doc_ids=[] if u else ["leave_policy"],
                split="dev", reviewed=True)))
        store.sync_queries(conn, queries)
        chunks = [{"chunk_id": "leave_policy#000", "doc_id": "leave_policy", "text": "Chunk.",
                   "score": 0.5}]

        def add_run(run_id, *, ungrounded=(), split="dev", sha=SHA, dirty=False,
                    status="completed", failed=(), judge_failed=(), query_ids=None,
                    answers=None, relevance=None, refusal=None,
                    scorers=("checks", "faithfulness", "relevance")):
            qids = query_ids or ids
            answers, relevance, refusal = answers or {}, relevance or {}, refusal or {}
            store.start_run(conn, run_id=run_id, split=split, config={"name": run_id},
                            config_path=f"configs/{run_id}.yaml", git_commit="abc1234",
                            git_dirty=dirty, test_set_sha256=sha, system_info={},
                            n_queries=len(qids), llm_calls_start_id=0)
            rows = {"checks": [], "faithfulness": [], "relevance": []}
            for q in qids:
                if q in failed:
                    store.write_output(conn, run_id=run_id, query_id=q, answer=None, chunks=None,
                                       latency_ms=None, error="RuntimeError: boom")
                    for m in ("refusal", "latency_ms"):
                        rows["checks"].append({"query_id": q, "metric": m, "value": None,
                                               "reasoning": "failed output"})
                    for scorer, metrics in (("faithfulness", ("faithfulness", "ungrounded")),
                                            ("relevance", ("relevance", "irrelevant"))):
                        rows[scorer] += [{"query_id": q, "metric": m, "value": None,
                                          "reasoning": "not judged: output failed"} for m in metrics]
                    continue
                store.write_output(conn, run_id=run_id, query_id=q,
                                   answer=answers.get(q, f"answer {q}"), chunks=chunks,
                                   latency_ms=10.0, error=None)
                rows["checks"] += [
                    {"query_id": q, "metric": "refusal",
                     "value": float(refusal.get(q, 1 if q in unans else 0)), "reasoning": "hand"},
                    {"query_id": q, "metric": "latency_ms", "value": 10.0, "reasoning": "hand"},
                    {"query_id": q, "metric": "in_context", "value": None if q in unans else 1.0,
                     "reasoning": "hand"},
                ]
                if q in judge_failed:
                    rows["faithfulness"] += [{"query_id": q, "metric": m, "value": None,
                                              "reasoning": "judge failed: LLMOutputError: bad"}
                                             for m in ("faithfulness", "ungrounded")]
                else:
                    bad = q in ungrounded
                    rows["faithfulness"] += [
                        {"query_id": q, "metric": "faithfulness", "value": 0.5 if bad else 1.0,
                         "reasoning": "hand"},
                        {"query_id": q, "metric": "ungrounded", "value": 1.0 if bad else 0.0,
                         "reasoning": f"JUDGE-REASON {q}"},
                    ]
                rel = relevance.get(q, 1.0)
                rows["relevance"] += [
                    {"query_id": q, "metric": "relevance", "value": rel, "reasoning": "hand"},
                    {"query_id": q, "metric": "irrelevant", "value": float(rel < 0.5),
                     "reasoning": "hand"},
                ]
            if status != "running":
                store.finish_run(conn, run_id, status=status,
                                 n_ok=len(qids) - len(failed), n_failed=len(failed),
                                 llm_calls_summary={})
            for scorer in scorers:
                store.write_scores(conn, run_id, VERSIONS[scorer], rows[scorer])

        harness = make_harness(tmp_path, path)
        env = SimpleNamespace(conn=conn, path=path, ids=ids, extra_id=extra_id, harness=harness,
                              add_run=add_run, n=n)
        env.accept = lambda run_id, note="reference": regression.accept_run(conn, run_id, note,
                                                                           env.harness)
        env.write_harness = lambda h: _dump(tmp_path, h, len(opened))
        env.harness_path = env.write_harness(harness)
        return env

    yield _make
    for conn in opened:
        conn.close()


def _dump(tmp_path, harness, number):
    path = tmp_path / f"harness_{number}.yaml"
    path.write_text(yaml.safe_dump(harness), encoding="utf-8")
    return str(path)


@pytest.fixture
def env(make_env):
    return make_env(10)


def count(path, sql, params=()):
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute(sql, params).fetchone()[0]


def item(**overrides):
    row = {"query_id": "q001", "type": "lookup", "answerable": True, "in_context": 1.0,
           "refusal": 0.0, "hit@4": 1.0, "reciprocal_rank": 1.0, "faith_state": "judged",
           "rel_state": "judged", "ungrounded": 0.0, "irrelevant": 0.0, "faithfulness": 1.0,
           "relevance": 1.0, "cause": None, "bad": 0}
    row.update(overrides)
    return row


def cli(env, *argv):
    return regression.main(["--harness-config", env.harness_path, *argv])


def accepted_and_new(env, new_ungrounded=(), accepted_ungrounded=(), **new_kwargs):
    env.add_run("run-acc", ungrounded=accepted_ungrounded)
    env.accept("run-acc")
    env.add_run("run-new", ungrounded=new_ungrounded, **new_kwargs)
    return "run-new"


# ---------------------------------------------------------------- config

def test_load_regression_config_valid_block_returns_flag_version(cfg):
    """Spec Config: valid block loads and carries flags-<n>-<hash8>."""
    assert cfg["flag_version"].startswith("flags-")
    assert len(cfg["flag_version"].split("-")[-1]) == 8
    assert cfg["max_ungrounded_rise_points"] == 3.0


@pytest.mark.parametrize("change", [
    {"labeler": None},                            # missing key (removed below)
    {"max_ungrounded_rise_points": -1.0},
    {"relevance_below": 1.5},
    {"disagreement_rules": ["no_such_rule"]},
])
def test_load_regression_config_bad_value_raises_valueerror(change):
    """Spec Config: missing key, negative gate, threshold outside [0,1], unknown rule -> ValueError."""
    h = block()
    for key, value in change.items():
        if value is None:
            del h["regression"][key]
        else:
            h["regression"][key] = value
    with pytest.raises(ValueError):
        regression.load_regression_config(h)


@pytest.mark.parametrize("change", [
    {"max_ungrounded_rise_points": 7.5}, {"labeler": "someone_else"},
])
def test_flag_version_gate_or_labeler_change_keeps_version(cfg, change):
    """Spec: the gate value (and labeler) are not part of flag_version."""
    other = regression.load_regression_config(block(**change))
    assert other["flag_version"] == cfg["flag_version"]


@pytest.mark.parametrize("change", [
    {"relevance_below": 0.5}, {"answer_similarity_below": 0.5},
    {"disagreement_rules": ["refusal_vs_relevance_a"]},
])
def test_flag_version_threshold_change_changes_version(cfg, change):
    """Spec: changing a flag threshold or rule set gives a new flag_version."""
    other = regression.load_regression_config(block(**change))
    assert other["flag_version"] != cfg["flag_version"]


# ---------------------------------------------------------------- similarity

@pytest.mark.parametrize("old, new, expected", [
    ("meal bills cannot be claimed", "meal bills cannot be claimed", 1.0),
    # M = 5 matching words, T = 5 + 6 = 11, ratio = 2*5/11 = 10/11
    ("meal bills cannot be claimed", "meal bills cannot be claimed separately", 10 / 11),
    ("meal bills [2] cannot be claimed [10]", "Meal Bills CANNOT be claimed", 1.0),
    ("", "", 1.0),
    ("", "something here", 0.0),
    ("a b c d", "w x y z", 0.0),
])
def test_answer_similarity_cases_match_hand_values(old, new, expected):
    """Spec Similarity: identical, 10/11 worked example, markers and case ignored, empty cases."""
    assert regression.answer_similarity(old, new, IGNORE) == pytest.approx(expected)


# ---------------------------------------------------------------- flag reasons

@pytest.mark.parametrize("overrides, similarity, expected", [
    ({"ungrounded": 1.0, "faithfulness": 0.5}, None, ["ungrounded"]),
    ({"relevance": 0.5}, None, ["low_relevance"]),
    ({"relevance": 1.0}, None, []),
    ({"answerable": True, "refusal": 1.0, "relevance": 1.0}, None, ["judges_disagree"]),      # rule a
    ({"answerable": False, "refusal": 0.0, "relevance": 1.0}, None, ["judges_disagree"]),     # rule b
    ({"answerable": False, "refusal": 1.0, "relevance": 1.0}, None, []),                      # correct refusal
    ({"ungrounded": 1.0, "relevance": 1.0}, None, ["ungrounded"]),    # ungrounded but relevant: no disagreement
    ({}, 0.59, ["answer_changed"]),
    ({}, 0.6, []),
    ({}, None, []),
    ({"faith_state": "judge_failed", "ungrounded": None}, None, ["not_judged"]),
    ({"rel_state": "not_judged", "relevance": None}, 0.0, ["not_judged"]),
    ({"ungrounded": 1.0, "relevance": 0.0}, 0.1,
     ["answer_changed", "low_relevance", "ungrounded"]),              # sorted, unique
])
def test_flag_reasons_table_matches_spec_rules(cfg, overrides, similarity, expected):
    """Spec Flag reasons: each rule at its boundary; not_judged never silently unflagged."""
    assert regression.flag_reasons(item(**overrides), similarity, cfg) == expected


def test_flag_reasons_disagreement_rules_off_gives_no_disagreement():
    """Spec: only rules listed in disagreement_rules are applied."""
    off = regression.load_regression_config(block(disagreement_rules=[]))
    assert regression.flag_reasons(item(refusal=1.0, relevance=1.0), None, off) == []


# ---------------------------------------------------------------- accept

def test_accept_run_valid_run_records_row_with_labeler_and_note(env):
    """Spec Accept: records a row (split dev, accepted_by labeler, note)."""
    env.add_run("run-a")
    row_id = env.accept("run-a", "chosen config")
    row = regression.get_accepted_run(env.conn, "dev")
    assert row["id"] == row_id and row["run_id"] == "run-a"
    assert (row["split"], row["accepted_by"], row["note"]) == ("dev", "gaurav", "chosen config")


@pytest.mark.parametrize("kwargs", [
    {"split": "heldout"},
    {"dirty": True},
    {"judge_failed": ("q003",)},
    {"failed": ("q002",)},
    {"status": "aborted"},
])
def test_accept_run_refused_run_raises_and_writes_nothing(env, kwargs):
    """Spec Accept refusals: held-out, dirty, incomplete judging, not completed."""
    env.add_run("run-bad", **kwargs)
    with pytest.raises(REFUSED):
        env.accept("run-bad")
    assert regression.get_accepted_run(env.conn, "dev") is None
    assert count(env.path, "SELECT COUNT(*) FROM accepted_runs") == 0


def test_accept_run_unknown_run_raises(env):
    """Spec Accept: unknown run is refused."""
    with pytest.raises(REFUSED):
        env.accept("no-such-run")


def test_accept_run_already_accepted_run_is_refused(env):
    """Spec Accept: the current accepted run cannot be accepted again."""
    env.add_run("run-a")
    env.accept("run-a")
    with pytest.raises(RegressionRefused):
        env.accept("run-a")
    assert count(env.path, "SELECT COUNT(*) FROM accepted_runs") == 1


def test_get_accepted_run_newest_row_wins_and_old_rows_stay(env):
    """Spec: accepted run of a split is its newest row; rows are never overwritten."""
    env.add_run("run-a")
    env.add_run("run-b")
    env.accept("run-a")
    env.accept("run-b")
    assert regression.get_accepted_run(env.conn, "dev")["run_id"] == "run-b"
    env.accept("run-a")                      # allowed: it is not the current one
    assert regression.get_accepted_run(env.conn, "dev")["run_id"] == "run-a"
    assert count(env.path, "SELECT COUNT(*) FROM accepted_runs") == 3


# ---------------------------------------------------------------- report refusals

def _no_flag_rows(env):
    assert count(env.path, "SELECT COUNT(*) FROM reviews") == 0
    assert count(env.path, "SELECT COUNT(*) FROM run_reports WHERE kind = 'regression'") == 0


def test_build_regression_report_no_accepted_run_is_refused(env):
    """Spec Report refusals: no accepted run."""
    env.add_run("run-new")
    with pytest.raises(RegressionRefused):
        regression.build_regression_report(env.conn, "run-new", None, env.harness)
    _no_flag_rows(env)


@pytest.mark.parametrize("kwargs", [
    {"split": "heldout"},
    {"sha": "ef" * 32},
    {"query_ids": "swap_last"},
    {"failed": ("q002",)},
    {"judge_failed": ("q004",)},
    {"scorers": ("checks", "faithfulness")},
])
def test_build_regression_report_incomparable_new_run_is_refused_and_writes_nothing(env, kwargs):
    """Spec Report refusals: held-out, other sha, other query ids, failed output, judge failure, missing rows."""
    env.add_run("run-acc")
    env.accept("run-acc")
    if kwargs.get("query_ids") == "swap_last":
        kwargs = {"query_ids": env.ids[:-1] + [env.extra_id]}
    env.add_run("run-new", **kwargs)
    with pytest.raises(REFUSED):
        regression.build_regression_report(env.conn, "run-new", None, env.harness)
    _no_flag_rows(env)


# ---------------------------------------------------------------- gate

@pytest.mark.parametrize("n, acc, new, delta, verdict", [
    (80, 7, 9, 2.5, "PASS"),       # 9/80 = 11.25% vs 7/80 = 8.75%
    (80, 7, 10, 3.75, "FAIL"),     # 10/80 = 12.5%
    (80, 7, 4, -3.75, "PASS"),     # 4/80 = 5%: improvement
    (100, 0, 3, 3.0, "PASS"),      # exactly +3.0 is not "more than 3"
])
def test_build_regression_report_gate_table_matches_spec_examples(make_env, n, acc, new, delta, verdict):
    """Spec Gate: +2.5 PASS, exactly +3.0 PASS, +3.75 FAIL, negative PASS."""
    env = make_env(n)
    env.add_run("run-acc", ungrounded=env.ids[:acc])
    env.accept("run-acc")
    env.add_run("run-new", ungrounded=env.ids[:new])
    gate = regression.build_regression_report(env.conn, "run-new", None, env.harness)["gate"]
    assert gate["delta_points"] == pytest.approx(delta)
    assert gate["verdict"] == verdict


def test_build_regression_report_custom_gate_value_is_honoured(env):
    """Spec Gate: custom gate. +20 points (2 of 10) passes at 25 and fails at 15."""
    run = accepted_and_new(env, new_ungrounded=["q001", "q002"])
    for gate_value, verdict in ((25.0, "PASS"), (15.0, "FAIL")):
        harness = copy.deepcopy(env.harness)
        harness["regression"]["max_ungrounded_rise_points"] = gate_value
        report = regression.build_regression_report(env.conn, run, None, harness)
        assert report["gate"]["delta_points"] == pytest.approx(20.0)
        assert report["gate"]["verdict"] == verdict
        assert report["gate"]["max_points"] == gate_value


def test_main_check_exit_codes_pass_fail_refused(env):
    """Spec CLI: exit 0 PASS, 1 FAIL, 2 refused (nothing written on 2)."""
    env.add_run("run-acc")
    env.accept("run-acc")
    env.add_run("run-pass")
    env.add_run("run-fail", ungrounded=["q001", "q002", "q003", "q004"])   # +40 points
    env.add_run("run-held", split="heldout")
    assert cli(env, "check", "--run-id", "run-pass", "--no-store") == 0
    assert cli(env, "check", "--run-id", "run-fail", "--no-store") == 1
    assert cli(env, "check", "--run-id", "run-held") == 2
    assert cli(env, "check", "--run-id", "no-such-run") == 2
    _no_flag_rows(env)


def test_main_accept_heldout_run_exits_2_and_valid_run_exits_0(env):
    """Spec Done: accept refuses a held-out run (exit 2, no row); a clean dev run exits 0."""
    env.add_run("run-held", split="heldout")
    env.add_run("run-dev")
    assert cli(env, "accept", "--run-id", "run-held", "--note", "x") == 2
    assert count(env.path, "SELECT COUNT(*) FROM accepted_runs") == 0
    assert cli(env, "accept", "--run-id", "run-dev", "--note", "first reference") == 0
    assert count(env.path, "SELECT COUNT(*) FROM accepted_runs WHERE run_id = ? AND accepted_by = ?",
                 ("run-dev", "gaurav")) == 1


# ---------------------------------------------------------------- numbers

def test_build_regression_report_flag_rate_and_flips_match_hand_values(env):
    """Spec Numbers: flag rate with Wilson CI, improved/regressed/p, per-type counts sum."""
    # accepted ungrounded q001-q005; new ungrounded q005, q006:
    # improved q001-q004 (4), regressed q006 (1), both q005; exact two-sided p for 4 vs 1 = 0.375.
    run = accepted_and_new(env, new_ungrounded=["q005", "q006"],
                           accepted_ungrounded=["q001", "q002", "q003", "q004", "q005"])
    report = regression.build_regression_report(env.conn, run, None, env.harness)
    gate = report["gate"]
    assert (gate["improved"], gate["regressed"]) == (4, 1)
    assert gate["p_value"] == pytest.approx(0.375)
    assert (gate["accepted_rate"]["successes"], gate["new_rate"]["successes"]) == (5, 2)
    assert gate["delta_points"] == pytest.approx(-30.0)
    # the 2 ungrounded new answers are the only flags; same answer text so no answer_changed
    assert sorted(f["query_id"] for f in report["flagged"]) == ["q005", "q006"]
    assert report["flag_rate"]["successes"] == 2 and report["flag_rate"]["n"] == 10
    low, high = wilson_interval(2, 10, 0.95)
    assert report["flag_rate"]["ci_low"] == pytest.approx(low)
    assert report["flag_rate"]["ci_high"] == pytest.approx(high)
    assert sum(report["flag_type_counts"].values()) == len(report["flagged"])
    assert report["flag_reason_counts"] == {"ungrounded": 2}


def test_build_regression_report_self_compare_gives_zero_delta_pass_no_changed(env):
    """Spec Done: the accepted run against itself is +0.00 points, PASS, 0 answer_changed."""
    env.add_run("run-acc", ungrounded=["q001", "q002"])
    env.accept("run-acc")
    report = regression.build_regression_report(env.conn, "run-acc", None, env.harness)
    assert report["gate"]["delta_points"] == 0.0 and report["gate"]["verdict"] == "PASS"
    assert (report["gate"]["improved"], report["gate"]["regressed"]) == (0, 0)
    assert report["gate"]["p_value"] is None
    assert "answer_changed" not in report["flag_reason_counts"]
    assert report["against_run_id"] == "run-acc"


def test_build_regression_report_changed_answer_is_flagged_with_similarity(env):
    """Spec: answer_changed when word similarity to the accepted answer is below 0.6."""
    # "answer q001" vs "totally different words here now": no shared word -> 0.0
    run = accepted_and_new(env, answers={"q001": "totally different words here now",
                                         "q002": "ANSWER [3] q002"})   # same words after cleanup
    report = regression.build_regression_report(env.conn, run, None, env.harness)
    assert [f["query_id"] for f in report["flagged"]] == ["q001"]
    assert report["flagged"][0]["reasons"] == ["answer_changed"]
    assert report["flagged"][0]["similarity"] == pytest.approx(0.0)


# ---------------------------------------------------------------- storage

def test_store_flags_writes_one_review_row_per_flag_and_one_regression_report(env):
    """Spec Storage: reviews rows (JSON reasons, NULL label) + one run_reports kind regression."""
    run = accepted_and_new(env, new_ungrounded=["q001", "q002"], relevance={"q003": 0.5})
    report = regression.build_regression_report(env.conn, run, None, env.harness)
    assert regression.store_flags(env.conn, report) == "written"
    rows = store.get_flag_rows(env.conn, run)
    assert [(r["query_id"], r["flag_reasons"]) for r in rows] == [
        ("q001", ["ungrounded"]), ("q002", ["ungrounded"]), ("q003", ["low_relevance"])]
    assert all(r["human_label"] is None and r["notes"] is None for r in rows)
    assert count(env.path, "SELECT COUNT(*) FROM run_reports WHERE run_id = ? AND kind = 'regression'",
                 (run,)) == 1
    with closing(sqlite3.connect(env.path)) as conn:
        raw = conn.execute("SELECT flag_reasons FROM reviews WHERE query_id = 'q003'").fetchone()[0]
    assert json.loads(raw) == ["low_relevance"]


def test_store_flags_rerun_same_version_is_unchanged_and_keeps_human_label(env):
    """Spec Storage: same flag version = unchanged; human_label and notes survive a re-check."""
    run = accepted_and_new(env, new_ungrounded=["q001"])
    report = regression.build_regression_report(env.conn, run, None, env.harness)
    regression.store_flags(env.conn, report)
    with env.conn:
        env.conn.execute("UPDATE reviews SET human_label = 'bad', notes = 'my note' "
                         "WHERE run_id = ? AND query_id = 'q001'", (run,))
    assert regression.store_flags(env.conn, report) == "unchanged"
    row = store.get_flag_rows(env.conn, run)[0]
    assert (row["human_label"], row["notes"]) == ("bad", "my note")
    assert count(env.path, "SELECT COUNT(*) FROM run_reports WHERE kind = 'regression'") == 1


def test_store_flags_other_flag_version_needs_replace_and_keeps_labelled_rows(env):
    """Spec Storage: different version refused without replace; replace deletes only unlabelled rows."""
    run = accepted_and_new(env, new_ungrounded=["q001"], relevance={"q002": 0.5, "q003": 0.5})
    v1 = regression.build_regression_report(env.conn, run, None, env.harness)
    assert sorted(f["query_id"] for f in v1["flagged"]) == ["q001", "q002", "q003"]
    regression.store_flags(env.conn, v1)
    with env.conn:
        env.conn.execute("UPDATE reviews SET human_label = 'bad' WHERE query_id = 'q003'")
    # relevance_below 0.4: 0.5 is no longer low -> only q001 flagged under the new version
    harness2 = copy.deepcopy(env.harness)
    harness2["regression"]["relevance_below"] = 0.4
    v2 = regression.build_regression_report(env.conn, run, None, harness2)
    assert v2["flag_version"] != v1["flag_version"]
    assert [f["query_id"] for f in v2["flagged"]] == ["q001"]
    before = store.get_flag_rows(env.conn, run)
    with pytest.raises(RegressionRefused):
        regression.store_flags(env.conn, v2)
    assert store.get_flag_rows(env.conn, run) == before            # nothing half-written
    assert regression.store_flags(env.conn, v2, replace=True) == "written"
    left = {r["query_id"]: r["human_label"] for r in store.get_flag_rows(env.conn, run)}
    assert left == {"q001": None, "q003": "bad"}                    # q002 (unlabelled) deleted


def test_main_check_no_store_writes_nothing_and_rerun_prints_unchanged(env, capsys):
    """Spec CLI: --no-store writes nothing; a second stored check prints 'unchanged'."""
    accepted_and_new(env, relevance={"q001": 0.5})   # flagged low_relevance, gate unchanged: PASS
    assert cli(env, "check", "--run-id", "run-new", "--no-store") == 0
    _no_flag_rows(env)
    assert cli(env, "check", "--run-id", "run-new") == 0
    capsys.readouterr()
    assert cli(env, "check", "--run-id", "run-new") == 0
    assert "unchanged" in capsys.readouterr().out
    assert count(env.path, "SELECT COUNT(*) FROM reviews WHERE run_id = 'run-new'") == 1
    assert count(env.path, "SELECT COUNT(*) FROM run_reports WHERE kind = 'regression'") == 1


def test_main_flags_lists_stored_flags_without_recompute(env, capsys):
    """Spec CLI: `flags` lists stored flags of a run (reasons, human label, count)."""
    accepted_and_new(env, new_ungrounded=["q004"])
    cli(env, "check", "--run-id", "run-new")
    capsys.readouterr()
    assert cli(env, "flags", "--run-id", "run-new") == 0
    out = capsys.readouterr().out
    assert "q004" in out and "ungrounded" in out and "1 flagged" in out


# ---------------------------------------------------------------- flag quality

def _label(env, run, labels, labeler="gaurav", splits=None):
    store.save_calibration_splits(env.conn, run, {q: "check" for q in (splits or labels)})
    for q, value in labels.items():
        if value is not None:
            store.write_calibration_label(env.conn, run, q, value, 2, None, labeler)


def test_flag_quality_hand_built_labels_give_expected_confusion_counts(env):
    """Spec Flag quality: tp/fp/fn/tn from hand labels; unsure, unlabelled, other labeler apart."""
    env.add_run("run-l", ungrounded=["q001", "q002"])      # q001, q002 flagged; others not
    # q001 human ungrounded -> TP; q002 grounded -> FP; q003 ungrounded -> FN; q004 grounded -> TN
    # q005 unsure; q006 no label; q007 labelled by someone else
    _label(env, "run-l", {"q001": "ungrounded", "q002": "grounded", "q003": "ungrounded",
                          "q004": "grounded", "q005": "unsure", "q006": None},
           splits=["q001", "q002", "q003", "q004", "q005", "q006", "q007"])
    store.write_calibration_label(env.conn, "run-l", "q007", "ungrounded", 2, None, "someone_else")
    result = regression.flag_quality(env.conn, "run-l", env.harness)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
    assert (result["unsure"], result["unlabelled"], result["other_labeler"]) == (1, 1, 1)
    assert result["precision"]["rate"] == pytest.approx(0.5)     # 1 / (1 + 1)
    assert result["recall"]["rate"] == pytest.approx(0.5)        # 1 / (1 + 1)
    low, high = wilson_interval(1, 2, 0.95)
    assert result["precision"]["ci_low"] == pytest.approx(low)
    assert result["recall"]["ci_high"] == pytest.approx(high)


def test_flag_quality_no_ungrounded_labels_is_undefined_and_printed_as_such(env, capsys):
    """Spec Flag quality: zero positives -> None rates and 'undefined (0/0)' text."""
    env.add_run("run-l")
    _label(env, "run-l", {"q001": "grounded", "q002": "grounded", "q003": "grounded"})
    result = regression.flag_quality(env.conn, "run-l", env.harness)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (0, 0, 0, 3)
    assert result["precision"]["rate"] is None and result["recall"]["rate"] is None
    assert cli(env, "thresholds", "--run-id", "run-l") == 0
    assert "undefined (0/0)" in capsys.readouterr().out


# ---------------------------------------------------------------- honesty strings and files

def test_format_report_names_uncalibrated_judge_gate_n_and_noise_note(env):
    """Spec Honesty: 'uncalibrated judge', gate value, n and the noise note in text and markdown."""
    run = accepted_and_new(env, new_ungrounded=["q001"])
    report = regression.build_regression_report(env.conn, run, None, env.harness)
    for text in (regression.format_report(report), regression.format_markdown(report)):
        assert "uncalibrated judge" in text
        assert "3.00" in text                              # gate value
        assert "10 queries" in text                        # n
        assert report["noise_note"] in text
        assert "JUDGE-REASON q001" in text                 # judge reasoning of the flagged answer
    assert regression.format_report(report).splitlines()[1].startswith(report["gate"]["verdict"])


def test_main_check_markdown_and_log_written_and_existing_markdown_refused(env, tmp_path):
    """Spec CLI: --markdown / --log write files; an existing markdown needs --force (else exit 2)."""
    accepted_and_new(env, relevance={"q001": 0.5})   # flagged low_relevance, gate unchanged: PASS
    md, log = tmp_path / "latest.md", tmp_path / "log.md"
    log.write_text("| date | config |\n", encoding="utf-8")
    assert cli(env, "check", "--run-id", "run-new", "--markdown", str(md), "--log", str(log)) == 0
    assert "Verdict: PASS" in md.read_text(encoding="utf-8")
    assert "`run-new`" in log.read_text(encoding="utf-8")
    assert cli(env, "check", "--run-id", "run-new", "--markdown", str(md)) == 2
    assert cli(env, "check", "--run-id", "run-new", "--markdown", str(md), "--force") == 0


def test_check_makes_no_llm_call_even_with_generate_patched_to_raise(env):
    """Spec No LLM: a full check runs with `generate` raising (autouse fixture)."""
    accepted_and_new(env, relevance={"q001": 0.5})   # flagged low_relevance, gate unchanged: PASS
    assert cli(env, "check", "--run-id", "run-new") == 0


# ---------------------------------------------------------------- layering

def test_regression_module_has_no_direct_import_of_rag_or_llm():
    """Spec Layering: regression.py imports nothing from rag/ or llm/ directly."""
    source = Path(regression.__file__).read_text(encoding="utf-8")
    modules = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    banned = [m for m in modules if m.split(".")[0] in ("rag", "llm")]
    assert banned == []
