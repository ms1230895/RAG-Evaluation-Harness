"""Tests for Step 10: the baseline report (evalharness/report.py), written from the spec.

Spec: .claude/specs/10-baseline-report.md. All offline: tmp DBs from the `baseline_db`
fixture (conftest), no LLM, never data/results.sqlite, never the real experiments/ files.

The spec does not name the keys of the report dict, so most number checks search the report
for a "rate dict" (a dict with `successes` and `n`, the shape of metrics.rate_summary) instead
of reading fixed paths. Item-table fields named in the spec are read directly.

Hand values (see the `baseline_db` docstring): faithfulness judged 9 (q008 judge failed),
ungrounded q002 q003 q006 q010 = 4/9.
"""

import ast
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
import yaml

from evalharness import report as report_module
from evalharness import store
from evalharness.checks import ScoringRefused
from evalharness.metrics import wilson_interval
from evalharness.report import (
    CAUSES,
    build_baseline_report,
    current_versions,
    failure_cause,
    format_baseline_markdown,
    format_baseline_report,
    item_table,
    load_report_config,
    log_row,
    main,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_HARNESS = PROJECT_ROOT / "configs" / "harness.yaml"
REAL_JUDGES = PROJECT_ROOT / "configs" / "judges.yaml"
CONFIDENCE = 0.95
CFG = {"primary_metric": "ungrounded", "include_bad_answer_rate": True, "allow_incomplete": True}
CFG_STRICT = {**CFG, "allow_incomplete": False}
UNCAL = "uncalibrated judge"


# ---------------------------------------------------------------- helpers

def build(db, cfg=None, versions=None, **kwargs):
    """Build the baseline report for the fixture run (allow_incomplete true by default)."""
    return build_baseline_report(db.conn, db.run_id, versions or db.versions, CONFIDENCE,
                                 cfg or CFG, **kwargs)


def walk(obj):
    """Yield every dict and list inside obj (obj included)."""
    yield obj
    if isinstance(obj, dict):
        for value in obj.values():
            yield from walk(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from walk(value)


def rate_dicts(report, successes, n):
    """Every dict in the report with `successes` and `n` equal to the given counts."""
    return [d for d in walk(report) if isinstance(d, dict)
            and d.get("successes") == successes and d.get("n") == n]


def bounds(d):
    """The interval of a rate dict: ci_low/ci_high, low/high or a `ci` pair."""
    for low_key, high_key in (("ci_low", "ci_high"), ("low", "high")):
        if low_key in d and high_key in d:
            return d[low_key], d[high_key]
    if "ci" in d:
        return tuple(d["ci"])
    raise AssertionError(f"rate dict has no interval: {d}")


def assert_rate(report, successes, n):
    """Spec: a rate successes/n exists in the report with the Wilson interval for those counts."""
    found = rate_dicts(report, successes, n)
    assert found, f"no rate {successes}/{n} anywhere in the report"
    low, high = wilson_interval(successes, n, CONFIDENCE)
    for d in found:
        got_low, got_high = bounds(d)
        assert got_low == pytest.approx(low, abs=1e-9), f"{successes}/{n}: wrong lower bound"
        assert got_high == pytest.approx(high, abs=1e-9), f"{successes}/{n}: wrong upper bound"
        if d.get("rate") is not None:
            assert d["rate"] == pytest.approx(successes / n), f"{successes}/{n}: wrong rate"
    return found


def values_of_key(obj, part):
    """All values whose (lowercased) key contains `part`."""
    return [v for d in walk(obj) if isinstance(d, dict) for k, v in d.items()
            if part in str(k).lower()]


def ids_in(obj):
    """Query ids found in obj: strings like 'q010' and 'query_id' values."""
    found = set()
    for node in walk(obj):
        if isinstance(node, dict) and isinstance(node.get("query_id"), str):
            found.add(node["query_id"])
        elif isinstance(node, list):
            found.update(x for x in node if isinstance(x, str) and x.startswith("q"))
    return found


def list_ids(report, key_part):
    """Ids inside the lists stored under keys that contain `key_part`; fails if no such key."""
    lists = [v for v in values_of_key(report, key_part) if isinstance(v, (list, tuple))]
    assert lists, f"no list under a key containing {key_part!r} in the report"
    return set().union(*(ids_in(item) for item in lists)) | {
        x for lst in lists for x in lst if isinstance(x, str)}


def state_of(row, judge_prefix):
    """The state field of an item row for one judge (key contains 'state' and the prefix)."""
    keys = [k for k in row if "state" in k and k.startswith(judge_prefix)]
    assert len(keys) == 1, f"expected one state key for {judge_prefix!r}, got {keys}: {sorted(row)}"
    return row[keys[0]]


def rows_by_id(rows):
    return {row["query_id"]: row for row in rows}


def sql_one(path, sql, params=()):
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute(sql, params).fetchone()


def write_harness(tmp_path, db_path, *, allow_incomplete=True, include_bad=True):
    """Tmp harness YAML: tmp db_path, valid checks and report blocks, the real judges.yaml read-only."""
    config = {
        "db_path": str(db_path),
        "test_set_config": str(tmp_path / "test_set.yaml"),
        "max_consecutive_failures": 5,
        "error_max_chars": 500,
        "judges_config": str(REAL_JUDGES),
        "system": {"answer": "somewhere:answer", "load_config": "somewhere:load", "describe": None},
        "checks": {"hit_ks": [1, 4], "report_confidence": 0.95,
                   "refusal_phrases": ["no mention of", "does not mention", "cannot find"]},
        "report": {"primary_metric": "ungrounded", "include_bad_answer_rate": include_bad,
                   "allow_incomplete": allow_incomplete},
    }
    path = tmp_path / "harness_report_test.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


def cli(db, tmp_path, *extra, allow_incomplete=True, include_bad=True, versions=True):
    """Run main() against the fixture DB; the three version flags pin the fixture's scoring."""
    argv = ["--run-id", db.run_id,
            "--harness-config", write_harness(tmp_path, db.path, allow_incomplete=allow_incomplete,
                                              include_bad=include_bad)]
    if versions:
        argv += ["--checks-version", db.versions["checks"],
                 "--faithfulness-version", db.versions["faithfulness"],
                 "--relevance-version", db.versions["relevance"]]
    return main(argv + list(extra))


@pytest.fixture
def no_llm(monkeypatch):
    """Make any LLM call from the judges or the shared client fail loudly."""
    import llm.client
    from evalharness import judges

    def boom(*args, **kwargs):
        raise AssertionError("the report must not call an LLM")

    monkeypatch.setattr(judges, "generate", boom)
    monkeypatch.setattr(llm.client, "generate", boom)


# ---------------------------------------------------------------- failure_cause

class TestFailureCause:
    def test_failure_cause_answerable_not_in_context_returns_retrieval_miss(self):
        """Spec: answerable and in_context 0 -> retrieval_miss."""
        assert failure_cause(True, 0) == "retrieval_miss"

    def test_failure_cause_answerable_in_context_returns_generation_error(self):
        """Spec: answerable and in_context 1 -> generation_error."""
        assert failure_cause(True, 1) == "generation_error"

    def test_failure_cause_float_zero_counts_as_not_in_context(self):
        """Spec: in_context == 0 (stored as a float) -> retrieval_miss."""
        assert failure_cause(True, 0.0) == "retrieval_miss"

    def test_failure_cause_unanswerable_without_in_context_returns_generation_error(self):
        """Spec: unanswerable (in_context None) -> generation_error (invented answer)."""
        assert failure_cause(False, None) == "generation_error"

    def test_failure_cause_answerable_without_in_context_raises_value_error(self):
        """Spec: an answerable item with in_context None raises ValueError."""
        with pytest.raises(ValueError):
            failure_cause(True, None)

    def test_causes_constant_lists_both_causes(self):
        """Spec: CAUSES = ("retrieval_miss", "generation_error")."""
        assert tuple(CAUSES) == ("retrieval_miss", "generation_error")


# ---------------------------------------------------------------- config

class TestReportConfig:
    def test_load_report_config_real_harness_has_approved_values(self):
        """Spec Config changes: report block is ungrounded / true / false (read-only real file)."""
        harness = yaml.safe_load(REAL_HARNESS.read_text(encoding="utf-8"))
        cfg = load_report_config(harness)
        assert cfg["primary_metric"] == "ungrounded"
        assert cfg["include_bad_answer_rate"] is True
        assert cfg["allow_incomplete"] is False

    def test_load_report_config_missing_block_raises(self):
        """Spec: the report block is required; a harness without it is refused."""
        assert load_report_config({"report": dict(CFG)})["allow_incomplete"] is True  # control
        with pytest.raises(ValueError):
            load_report_config({})

    @pytest.mark.parametrize("key", ["primary_metric", "include_bad_answer_rate", "allow_incomplete"])
    def test_load_report_config_missing_key_raises(self, key):
        """Spec: each of the three report keys is required."""
        cfg = dict(CFG)
        del cfg[key]
        with pytest.raises(ValueError):
            load_report_config({"report": cfg})

    def test_load_report_config_non_boolean_flag_raises(self):
        """Spec: include_bad_answer_rate / allow_incomplete are true/false values."""
        with pytest.raises(ValueError):
            load_report_config({"report": {**CFG, "allow_incomplete": "no"}})

    def test_load_report_config_unknown_primary_metric_raises(self):
        """Spec: the headline metric is ungrounded; other values are rejected."""
        with pytest.raises(ValueError):
            load_report_config({"report": {**CFG, "primary_metric": "faithfulness"}})


class TestCurrentVersions:
    @pytest.fixture
    def real_harness(self):
        return yaml.safe_load(REAL_HARNESS.read_text(encoding="utf-8"))

    def test_current_versions_returns_the_three_scorer_keys(self, real_harness):
        """Spec: current_versions returns checks, faithfulness and relevance versions."""
        versions = current_versions(real_harness)
        assert set(versions) == {"checks", "faithfulness", "relevance"}

    def test_current_versions_match_the_loaders_of_the_real_configs(self, real_harness):
        """Spec: versions come from load_checks_config / load_judges_config / load_relevance_config."""
        from evalharness.checks import load_checks_config
        from evalharness.judges import load_judges_config, load_relevance_config

        versions = current_versions(real_harness)
        assert versions["checks"] == load_checks_config(real_harness)["version"]
        assert versions["faithfulness"] == load_judges_config(real_harness["judges_config"])["version"]
        assert versions["relevance"] == load_relevance_config(real_harness["judges_config"])["version"]

    def test_current_versions_real_configs_equal_the_baseline_versions(self, real_harness):
        """Spec Overview: the v0 baseline was scored by checks-1-66c5a50a, faith-v2-d130a39f, rel-v2-c825e4f7."""
        assert current_versions(real_harness) == {
            "checks": "checks-1-66c5a50a", "faithfulness": "faith-v2-d130a39f",
            "relevance": "rel-v2-c825e4f7"}


# ---------------------------------------------------------------- item_table

# query_id: (type, answerable, in_context, refusal, ungrounded, irrelevant, faithfulness, relevance, cause)
EXPECTED_ITEMS = {
    "q001": ("lookup", True, 1, 0, 0, 0, 1.0, 1.0, None),
    "q002": ("lookup", True, 1, 0, 1, 0, 0.5, 1.0, "generation_error"),
    "q003": ("lookup", True, 0, 0, 1, 0, 0.6, 0.5, "retrieval_miss"),
    "q004": ("lookup", True, 0, 1, 0, 1, 1.0, 0.0, None),
    "q005": ("multi_chunk", True, 1, 0, 0, 0, 1.0, 1.0, None),
    "q006": ("multi_chunk", True, 1, 0, 1, 0, 0.75, 1.0, "generation_error"),
    "q007": ("paraphrased", True, 1, 0, 0, 0, 1.0, 1.0, None),
    "q008": ("paraphrased", True, 1, 0, None, 0, None, 1.0, None),
    "q009": ("unanswerable", False, None, 1, 0, 0, 1.0, 1.0, None),
    "q010": ("unanswerable", False, None, 0, 1, 1, 0.0, 0.0, "generation_error"),
}


class TestItemTable:
    def test_item_table_returns_one_row_per_output(self, baseline_db):
        """Spec: one row per output."""
        rows = item_table(baseline_db.conn, baseline_db.run_id, baseline_db.versions)
        assert sorted(r["query_id"] for r in rows) == sorted(EXPECTED_ITEMS), "expected q001..q010 once each"

    @pytest.mark.parametrize("query_id", sorted(EXPECTED_ITEMS))
    def test_item_table_fields_match_hand_values(self, baseline_db, query_id):
        """Spec: type, answerable, in_context, refusal, ungrounded, irrelevant, faithfulness, relevance, cause."""
        rows = rows_by_id(item_table(baseline_db.conn, baseline_db.run_id, baseline_db.versions))
        row = rows[query_id]
        (qtype, answerable, in_context, refusal, ungrounded, irrelevant,
         faithfulness, relevance, cause) = EXPECTED_ITEMS[query_id]
        assert row["type"] == qtype
        assert row["answerable"] is answerable or row["answerable"] == answerable
        assert row["in_context"] == in_context
        assert row["refusal"] == refusal
        assert row["ungrounded"] == ungrounded
        assert row["irrelevant"] == irrelevant
        assert row["faithfulness"] == (pytest.approx(faithfulness) if faithfulness is not None else None)
        assert row["relevance"] == (pytest.approx(relevance) if relevance is not None else None)
        assert row["cause"] == cause

    def test_item_table_cause_set_only_for_ungrounded_rows(self, baseline_db):
        """Spec: cause is None unless the item is ungrounded."""
        rows = item_table(baseline_db.conn, baseline_db.run_id, baseline_db.versions)
        for row in rows:
            if row["ungrounded"] == 1:
                assert row["cause"] in CAUSES, f"{row['query_id']}: ungrounded needs a cause"
            else:
                assert row["cause"] is None, f"{row['query_id']}: cause set without ungrounded"

    def test_item_table_states_are_judged_except_the_failed_faithfulness_item(self, baseline_db):
        """Spec: state is judged / judge_failed per judge; q008 faithfulness failed."""
        rows = rows_by_id(item_table(baseline_db.conn, baseline_db.run_id, baseline_db.versions))
        assert state_of(rows["q008"], "faith") == "judge_failed"
        assert state_of(rows["q008"], "rel") == "judged"
        for query_id in set(rows) - {"q008"}:
            assert state_of(rows[query_id], "faith") == "judged", query_id
            assert state_of(rows[query_id], "rel") == "judged", query_id

    def test_item_table_not_judged_reasoning_gives_not_judged_state(self, baseline_db):
        """Spec: NULL rows with reasoning 'not judged: ...' are state not_judged."""
        for metric in ("faithfulness", "ungrounded", "n_claims"):
            baseline_db.set_score(baseline_db.versions["faithfulness"], "q007", metric, None,
                                  "not judged: blank answer")
        rows = rows_by_id(item_table(baseline_db.conn, baseline_db.run_id, baseline_db.versions))
        assert state_of(rows["q007"], "faith") == "not_judged"
        assert rows["q007"]["ungrounded"] is None, "not-judged item must not become grounded or ungrounded"

    def test_item_table_deleted_judge_rows_give_missing_state(self, baseline_db):
        """Spec: an item with no rows for a judge version has state 'missing'."""
        with baseline_db.conn:
            baseline_db.conn.execute(
                "DELETE FROM scores WHERE run_id = ? AND version = ? AND query_id = ?",
                (baseline_db.run_id, baseline_db.versions["faithfulness"], "q007"))
        rows = rows_by_id(item_table(baseline_db.conn, baseline_db.run_id, baseline_db.versions))
        assert state_of(rows["q007"], "faith") == "missing"
        assert rows["q007"]["ungrounded"] is None

    @pytest.mark.parametrize("scorer", ["checks", "faithfulness", "relevance"])
    def test_item_table_version_without_rows_is_refused(self, baseline_db, scorer):
        """Spec: ScoringRefused if any of the three versions has no rows for the run."""
        item_table(baseline_db.conn, baseline_db.run_id, baseline_db.versions)  # control: accepted
        versions = {**baseline_db.versions, scorer: "no-such-version"}
        with pytest.raises(ScoringRefused):
            item_table(baseline_db.conn, baseline_db.run_id, versions)

    def test_item_table_reads_the_requested_faithfulness_version(self, baseline_db):
        """Spec: the table is read from the scores of the versions given."""
        baseline_db.add_faith_version("faith-bl-other")
        versions = {**baseline_db.versions, "faithfulness": "faith-bl-other"}
        rows = item_table(baseline_db.conn, baseline_db.run_id, versions)
        assert all(r["ungrounded"] == 0 for r in rows), "the other version marks nothing ungrounded"
        assert all(r["cause"] is None for r in rows)

    def test_item_table_follows_stored_in_context_value(self, baseline_db):
        """Spec: the cause comes from the stored in_context score."""
        baseline_db.set_score(baseline_db.versions["checks"], "q002", "in_context", 0.0)
        rows = rows_by_id(item_table(baseline_db.conn, baseline_db.run_id, baseline_db.versions))
        assert rows["q002"]["cause"] == "retrieval_miss", "in_context 0 moves q002 to retrieval_miss"


# ---------------------------------------------------------------- report numbers

class TestReportNumbers:
    def test_wilson_hand_value_for_four_of_nine(self):
        """Hand check of the numbers used below: Wilson 95% for 4/9 is about [0.189, 0.733]."""
        # p = 0.4444, z^2 = 3.8416, n = 9; denominator = 1 + 3.8416/9 = 1.42684
        # centre = (0.4444 + 3.8416/18) / 1.42684 = 0.65786 / 1.42684 = 0.46105
        # half = 1.96 * sqrt(0.24691/9 + 3.8416/324) / 1.42684 = 1.96 * 0.19823 / 1.42684 = 0.27230
        low, high = wilson_interval(4, 9, CONFIDENCE)
        assert low == pytest.approx(0.1888, abs=1e-3)
        assert high == pytest.approx(0.7334, abs=1e-3)

    def test_ungrounded_rate_all_is_four_of_nine_with_wilson_interval(self, baseline_db):
        """Spec: ungrounded rate overall = 4/9 (q002 q003 q006 q010 of 9 judged), Wilson CI."""
        assert_rate(build(baseline_db), 4, 9)

    def test_ungrounded_rate_answerable_is_three_of_seven(self, baseline_db):
        """Spec: answerable only: q001-q007 judged (q008 failed); ungrounded q002 q003 q006 = 3/7."""
        assert_rate(build(baseline_db), 3, 7)

    def test_ungrounded_rate_lookup_is_two_of_four(self, baseline_db):
        """Spec: lookup q001-q004, ungrounded q002 q003 = 2/4."""
        assert_rate(build(baseline_db), 2, 4)

    def test_ungrounded_rate_multi_chunk_and_unanswerable_are_one_of_two(self, baseline_db):
        """Spec: multi_chunk (q006) and unanswerable (q010) are each 1/2, so at least two 1/2 rates exist."""
        found = assert_rate(build(baseline_db), 1, 2)
        assert len(found) >= 2, "multi_chunk and unanswerable both give 1/2"

    def test_ungrounded_rate_paraphrased_excludes_failed_item(self, baseline_db):
        """Spec: paraphrased has q007 judged and q008 failed -> 0/1, not 0/2."""
        assert_rate(build(baseline_db), 0, 1)

    def test_failed_judge_item_is_excluded_from_every_denominator(self, baseline_db):
        """Spec: judge-failed items are excluded from denominators (no 4/10 ungrounded rate)."""
        report = build(baseline_db)
        assert not rate_dicts(report, 4, 10), "q008 (judge failed) must not be counted as grounded"

    def test_failed_judge_item_is_counted(self, baseline_db):
        """Spec: judge-failed items are counted (a *fail* field holds 1 or lists q008)."""
        report = build(baseline_db)
        counted = [v for v in values_of_key(report, "fail")
                   if v == 1 or (isinstance(v, (list, tuple))
                                 and "q008" in ids_in(v) | {x for x in v if isinstance(x, str)})]
        assert counted, "the report does not count the judge-failed item"

    def test_ungrounded_count_matches_sql_on_the_scores_table(self, baseline_db):
        """Spec Definition of done: SUM(value), COUNT(value) from SQL equal the report's k and n."""
        k, n = sql_one(baseline_db.path,
                       "SELECT SUM(value), COUNT(value) FROM scores WHERE run_id = ? AND version = ?"
                       " AND metric = 'ungrounded'", (baseline_db.run_id, baseline_db.versions["faithfulness"]))
        assert (k, n) == (4, 9), "hand fixture: 4 ungrounded of 9 judged"
        assert rate_dicts(build(baseline_db), int(k), int(n))

    def test_cause_counts_match_sql_join(self, baseline_db):
        """Spec Definition of done: ungrounded join in_context gives the same cause counts."""
        with closing(sqlite3.connect(baseline_db.path)) as conn:
            rows = conn.execute(
                "SELECT c.value FROM scores u JOIN scores c ON c.run_id = u.run_id AND c.query_id = u.query_id"
                " WHERE u.run_id = ? AND u.version = ? AND u.metric = 'ungrounded' AND u.value = 1"
                " AND c.version = ? AND c.metric = 'in_context' AND c.value = 0",
                (baseline_db.run_id, baseline_db.versions["faithfulness"],
                 baseline_db.versions["checks"])).fetchall()
        assert len(rows) == 1, "hand fixture: one ungrounded answer without the source in context (q003)"

    def test_retrieval_miss_cause_is_one_rate_and_one_share(self, baseline_db):
        """Spec: retrieval_miss = q003: rate over judged 1/9, share of ungrounded 1/4."""
        report = build(baseline_db)
        subtrees = [v for key, v in _items(report) if key == "retrieval_miss"]
        assert subtrees, "no 'retrieval_miss' entry in the report"
        assert any(rate_dicts(s, 1, 9) and rate_dicts(s, 1, 4) for s in subtrees), \
            "retrieval_miss needs a 1/9 rate and a 1/4 share"

    def test_generation_error_cause_is_three_rate_and_three_share(self, baseline_db):
        """Spec: generation_error = q002 q006 q010: rate 3/9, share 3/4."""
        report = build(baseline_db)
        subtrees = [v for key, v in _items(report) if key == "generation_error"]
        assert subtrees, "no 'generation_error' entry in the report"
        assert any(rate_dicts(s, 3, 9) and rate_dicts(s, 3, 4) for s in subtrees), \
            "generation_error needs a 3/9 rate and a 3/4 share"

    def test_cause_rates_and_shares_use_wilson_intervals(self, baseline_db):
        """Spec: each cause rate and share carries a Wilson interval."""
        report = build(baseline_db)
        for successes, n in [(1, 9), (3, 9), (1, 4), (3, 4)]:
            assert_rate(report, successes, n)

    def test_causes_by_type_unanswerable_invented_answer_is_generation_error(self, baseline_db):
        """Spec: by type, the one unanswerable ungrounded answer (q010) is a generation error 1/2 of the type."""
        report = build(baseline_db)
        assert any(rate_dicts(s, 1, 2) for key, s in _items(report) if key == "generation_error"), \
            "generation_error by type should hold a 1/2 rate (unanswerable or multi_chunk)"

    def test_crosstab_has_hand_counts_and_sums_to_judged_n(self, baseline_db):
        """Spec: 2x2 ungrounded x irrelevant over the 9 items judged by both: cells 4, 1, 3, 1."""
        # not ungrounded + relevant: q001 q005 q007 q009 = 4
        # not ungrounded + irrelevant: q004 = 1
        # ungrounded + relevant: q002 q003 q006 = 3
        # ungrounded + irrelevant: q010 = 1; total 9 (q008 has no faithfulness verdict)
        report = build(baseline_db)
        candidates = []
        for node in walk(report):
            if not isinstance(node, dict):
                continue
            cells = [v for k, v in _leaf_ints(node) if k not in ("n", "total")]
            if sorted(cells) == [1, 1, 3, 4]:
                candidates.append(node)
        assert candidates, "no 2x2 table with cells 4, 1, 3, 1 found"
        assert sum(_cells(candidates[0])) == 9, "the 2x2 table must sum to the judged n (9)"

    def test_crosstab_cell_for_invented_answers_is_one(self, baseline_db):
        """Spec: the ungrounded+irrelevant cell holds the invented answer (q010), so a cell with key naming both is 1."""
        report = build(baseline_db)
        named = [v for k, v in _items(report)
                 if "ungrounded" in k and "irrelevant" in k and isinstance(v, int)]
        assert named, "no cell keyed by ungrounded and irrelevant"
        assert 1 in named

    def test_bad_answer_rate_is_five_of_nine(self, baseline_db):
        """Spec: bad answer = ungrounded OR irrelevant: q002 q003 q006 q004 q010 = 5 of 9."""
        assert_rate(build(baseline_db), 5, 9)

    def test_bad_answer_rate_absent_when_config_turns_it_off(self, baseline_db):
        """Spec Config: include_bad_answer_rate false -> no bad-answer rate (5/9) in the report."""
        report = build(baseline_db, cfg={**CFG, "include_bad_answer_rate": False})
        assert not rate_dicts(report, 5, 9), "bad-answer rate must not be reported when switched off"

    def test_invented_answer_list_holds_the_unanswerable_ungrounded_item(self, baseline_db):
        """Spec: invented-answer list = unanswerable and ungrounded (q010)."""
        assert list_ids(build(baseline_db), "invented") == {"q010"}

    def test_false_refusal_list_holds_the_answerable_irrelevant_item(self, baseline_db):
        """Spec: false-refusal list = answerable and irrelevant (q004)."""
        assert list_ids(build(baseline_db), "false_refusal") == {"q004"}

    def test_invented_list_follows_the_stored_ungrounded_value(self, baseline_db):
        """Spec: numbers come from the DB: q010 stored as grounded -> no invented answer."""
        baseline_db.set_score(baseline_db.versions["faithfulness"], "q010", "ungrounded", 0.0)
        assert list_ids(build(baseline_db), "invented") == set()

    def test_false_refusal_list_follows_the_stored_irrelevant_value(self, baseline_db):
        """Spec: numbers come from the DB: q004 stored as relevant -> no false refusal."""
        baseline_db.set_score(baseline_db.versions["relevance"], "q004", "irrelevant", 0.0)
        assert list_ids(build(baseline_db), "false_refusal") == set()

    def test_retrieval_summary_in_context_is_six_of_eight_answerable(self, baseline_db):
        """Spec: retrieval summary: in_context over the 8 answerable items (q001 q002 q005 q006 q007 q008 hit) = 6/8."""
        assert_rate(build(baseline_db), 6, 8)

    def test_retrieval_summary_mrr_matches_hand_value(self, baseline_db):
        """Spec: MRR over answerable items = (1+0.5+0+0+1+0.25+0.5+1)/8 = 4.25/8 = 0.53125."""
        report = build(baseline_db)
        floats = [x for node in walk(report) if isinstance(node, dict)
                  for x in node.values() if isinstance(x, float)]
        assert any(x == pytest.approx(0.53125) for x in floats), "MRR 0.53125 not found in the report"

    def test_changing_one_stored_ungrounded_value_changes_the_report(self, baseline_db):
        """Spec: numbers come from the DB: q001 stored as ungrounded -> 5/9, no more 4/9."""
        before = build(baseline_db)
        baseline_db.set_score(baseline_db.versions["faithfulness"], "q001", "ungrounded", 1.0)
        after = build(baseline_db)
        assert json.dumps(before, sort_keys=True) != json.dumps(after, sort_keys=True)
        # Only the ungrounded rates: after the change 4/9 is still a legitimate value
        # elsewhere (generation_error over judged = q001, q002, q006, q010 = 4/9).
        assert rate_dicts(before["ungrounded"], 4, 9) and not rate_dicts(after["ungrounded"], 4, 9)
        assert_rate(after["ungrounded"], 5, 9)

    def test_changed_ungrounded_item_in_context_gets_generation_error(self, baseline_db):
        """Spec: q001 (in_context 1) turned ungrounded is a generation error: share 4/5."""
        baseline_db.set_score(baseline_db.versions["faithfulness"], "q001", "ungrounded", 1.0)
        report = build(baseline_db)
        subtrees = [v for key, v in _items(report) if key == "generation_error"]
        assert any(rate_dicts(s, 4, 5) for s in subtrees)

    def test_changing_in_context_moves_an_item_between_causes(self, baseline_db):
        """Spec: q002 in_context 1 -> 0 moves it to retrieval_miss: both causes become 2 of 4 ungrounded."""
        baseline_db.set_score(baseline_db.versions["checks"], "q002", "in_context", 0.0)
        report = build(baseline_db)
        miss = [v for key, v in _items(report) if key == "retrieval_miss"]
        gen = [v for key, v in _items(report) if key == "generation_error"]
        assert any(rate_dicts(s, 2, 4) and rate_dicts(s, 2, 9) for s in miss)
        assert any(rate_dicts(s, 2, 4) and rate_dicts(s, 2, 9) for s in gen)

    def test_report_is_deterministic(self, baseline_db):
        """Determinism: the same DB and versions give the same report."""
        first = json.dumps(build(baseline_db), sort_keys=True, default=str)
        second = json.dumps(build(baseline_db), sort_keys=True, default=str)
        assert first == second

    def test_report_selects_scoring_by_version(self, baseline_db):
        """Spec: the requested faithfulness version is used: the other version has 0 ungrounded of 10."""
        baseline_db.add_faith_version("faith-bl-other")
        report = build(baseline_db, versions={**baseline_db.versions, "faithfulness": "faith-bl-other"})
        assert_rate(report, 0, 10)
        assert not rate_dicts(report, 4, 9)

    def test_build_does_not_write_scores_or_reports(self, baseline_db):
        """Spec: reading only: build changes no scores row and writes no run_reports row."""
        before = sql_one(baseline_db.path, "SELECT COUNT(*) FROM scores")[0]
        build(baseline_db)
        assert sql_one(baseline_db.path, "SELECT COUNT(*) FROM scores")[0] == before
        assert sql_one(baseline_db.path, "SELECT COUNT(*) FROM run_reports")[0] == 0


def _items(obj):
    """Yield (key, value) for every dict entry anywhere in obj."""
    for node in walk(obj):
        if isinstance(node, dict):
            for key, value in node.items():
                yield str(key), value


def _leaf_ints(node):
    """(key, int) pairs for the int leaves of a dict, looking one dict level down."""
    for key, value in node.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            yield key, value
        elif isinstance(value, dict):
            yield from _leaf_ints(value)


def _cells(node):
    return [v for k, v in _leaf_ints(node) if k not in ("n", "total")]


# ---------------------------------------------------------------- run refusals and incomplete judging

class TestRefusals:
    def test_build_unknown_run_is_refused(self, baseline_db):
        """Spec: an unknown run is refused."""
        with pytest.raises(ScoringRefused):
            build_baseline_report(baseline_db.conn, "no-such-run", baseline_db.versions, CONFIDENCE, CFG)

    def test_build_run_not_completed_is_refused(self, baseline_db):
        """Spec: a run that is not completed is refused."""
        build(baseline_db)  # control: the completed run is accepted
        with baseline_db.conn:
            baseline_db.conn.execute("UPDATE runs SET status = 'aborted' WHERE run_id = ?",
                                     (baseline_db.run_id,))
        with pytest.raises(ScoringRefused):
            build(baseline_db)

    def test_build_heldout_run_is_refused_without_flag(self, baseline_db):
        """Spec: a held-out run is refused unless allow_heldout is given."""
        with baseline_db.conn:
            baseline_db.conn.execute("UPDATE runs SET split = 'heldout' WHERE run_id = ?",
                                     (baseline_db.run_id,))
        with pytest.raises(ScoringRefused):
            build(baseline_db)

    def test_build_heldout_run_is_accepted_with_flag(self, baseline_db):
        """Spec: allow_heldout=True lets a held-out run through (Step 12)."""
        with baseline_db.conn:
            baseline_db.conn.execute("UPDATE runs SET split = 'heldout' WHERE run_id = ?",
                                     (baseline_db.run_id,))
        assert_rate(build(baseline_db, allow_heldout=True), 4, 9)

    @pytest.mark.parametrize("scorer", ["checks", "faithfulness", "relevance"])
    def test_build_version_with_no_rows_is_refused(self, baseline_db, scorer):
        """Spec: a version with no rows is refused instead of judged."""
        with pytest.raises(ScoringRefused):
            build(baseline_db, versions={**baseline_db.versions, scorer: "no-such-version"})

    def test_judge_failure_refuses_when_allow_incomplete_is_false(self, baseline_db):
        """Spec Decision 3: allow_incomplete false -> a failed judge item refuses the report."""
        with pytest.raises((ScoringRefused, ValueError)):
            build(baseline_db, cfg=CFG_STRICT)

    def test_judge_failure_is_reported_when_allow_incomplete_is_true(self, baseline_db):
        """Spec Decision 3: allow_incomplete true -> the failure is reported and counted (4/9, not refused)."""
        assert_rate(build(baseline_db, cfg=CFG), 4, 9)

    def test_missing_items_refuse_when_allow_incomplete_is_false(self, baseline_db):
        """Spec: judge with missing items refuses under allow_incomplete false."""
        with baseline_db.conn:
            baseline_db.conn.execute(
                "DELETE FROM scores WHERE run_id = ? AND version = ? AND query_id = ?",
                (baseline_db.run_id, baseline_db.versions["relevance"], "q003"))
        with pytest.raises((ScoringRefused, ValueError)):
            build(baseline_db, cfg=CFG_STRICT)

    def test_not_judged_item_is_excluded_not_counted_as_grounded(self, baseline_db):
        """Spec: not-judged items are excluded: q007 not judged -> 4/8 ungrounded."""
        for metric in ("faithfulness", "ungrounded", "n_claims"):
            baseline_db.set_score(baseline_db.versions["faithfulness"], "q007", metric, None,
                                  "not judged: blank answer")
        assert_rate(build(baseline_db), 4, 8)

    def test_complete_judging_passes_under_allow_incomplete_false(self, baseline_db):
        """Spec: with no failed items the strict config accepts the run; q008 grounded -> 4/10."""
        v = baseline_db.versions["faithfulness"]
        baseline_db.set_score(v, "q008", "faithfulness", 1.0, "fixed")
        baseline_db.set_score(v, "q008", "ungrounded", 0.0, "fixed")
        baseline_db.set_score(v, "q008", "n_claims", 1.0, "fixed")
        assert_rate(build(baseline_db, cfg=CFG_STRICT), 4, 10)


# ---------------------------------------------------------------- metadata and versions

class TestMetadata:
    def test_report_records_run_id_versions_sha_commit_and_dirty_flag(self, baseline_db):
        """Spec: the report records the three versions, the test-set SHA, the git commit and the dirty flag."""
        report = build(baseline_db)
        text = json.dumps(report, default=str)
        for needed in [baseline_db.run_id, baseline_db.sha, baseline_db.commit,
                       *baseline_db.versions.values()]:
            assert needed in text, f"{needed!r} missing from the report"
        dirty = values_of_key(report, "dirty")
        assert dirty, "no git dirty flag recorded"
        assert any(d in (True, 1) for d in dirty), "fixture run was dirty (git_dirty True)"

    def test_markdown_records_all_metadata(self, baseline_db):
        """Spec: experiments/baseline_v0.md names run_id, three versions, SHA and commit."""
        markdown = format_baseline_markdown(build(baseline_db))
        for needed in [baseline_db.run_id, baseline_db.sha, baseline_db.commit,
                       *baseline_db.versions.values()]:
            assert needed in markdown, f"{needed!r} missing from the markdown"

    def test_text_report_records_run_id_and_versions(self, baseline_db):
        """Spec: the printed report names the exact scoring."""
        text = format_baseline_report(build(baseline_db))
        for needed in [baseline_db.run_id, *baseline_db.versions.values()]:
            assert needed in text, f"{needed!r} missing from the printed report"


# ---------------------------------------------------------------- calibration status

class TestCalibrationStatus:
    def test_no_calibration_report_says_not_calibrated(self, baseline_db):
        """Spec: with no report the status is 'not calibrated'."""
        text = format_baseline_report(build(baseline_db)).lower()
        assert "not calibrated" in text
        assert "inconclusive" not in text

    def test_check_report_without_positives_says_inconclusive_with_agreement_and_n(self, baseline_db):
        """Spec: stored check report with 0 positives -> 'inconclusive' and agreement 20/20."""
        baseline_db.add_calibration_report("faithfulness", baseline_db.versions["faithfulness"],
                                           tp=0, fp=0, fn=0, tn=20)
        text = format_baseline_report(build(baseline_db))
        assert "inconclusive" in text.lower()
        assert "20/20" in text, "agreement 20/20 with its n should be printed"

    def test_inconclusive_status_is_also_in_the_markdown(self, baseline_db):
        """Spec: the calibration status is part of the stored markdown report."""
        baseline_db.add_calibration_report("faithfulness", baseline_db.versions["faithfulness"],
                                           tp=0, fp=0, fn=0, tn=20)
        markdown = format_baseline_markdown(build(baseline_db))
        assert "inconclusive" in markdown.lower()
        assert "20/20" in markdown

    def test_relevance_without_report_stays_not_calibrated_when_only_faithfulness_has_one(self, baseline_db):
        """Spec: the status is per judge version; relevance has no report -> 'not calibrated'."""
        baseline_db.add_calibration_report("faithfulness", baseline_db.versions["faithfulness"],
                                           tp=0, fp=0, fn=0, tn=20)
        assert "not calibrated" in format_baseline_report(build(baseline_db)).lower()

    def test_report_for_another_judge_version_is_ignored(self, baseline_db):
        """Spec: calibration status of each judge version: a report for another version does not count."""
        baseline_db.add_calibration_report("faithfulness", "faith-some-other-version",
                                           tp=0, fp=0, fn=0, tn=20)
        text = format_baseline_report(build(baseline_db)).lower()
        assert "inconclusive" not in text

    def test_tune_half_report_is_not_the_calibration_status(self, baseline_db):
        """Spec: the status uses the latest 'check' comparison; a tune comparison is ignored."""
        baseline_db.add_calibration_report("faithfulness", baseline_db.versions["faithfulness"],
                                           tp=0, fp=0, fn=0, tn=20, calib_split="tune")
        assert "inconclusive" not in format_baseline_report(build(baseline_db)).lower()


class TestUncalibratedLabel:
    def test_text_report_labels_judge_numbers_uncalibrated(self, baseline_db):
        """Spec (per-section reading, Gaurav 2026-10-16): the headline line and the heading of
        every judge-based section say 'uncalibrated judge'; table rows under a heading need not."""
        lines = format_baseline_report(build(baseline_db)).splitlines()
        headline = [line for line in lines if line.startswith("HEADLINE")]
        assert len(headline) == 1 and "4/9" in headline[0]
        assert UNCAL in headline[0], "headline line must say 'uncalibrated judge'"
        for heading in ("Ungrounded rate by group", "Failure cause", "Irrelevant rate",
                        "Bad-answer rate"):
            found = [line for line in lines if line.startswith(heading)]
            assert found and all(UNCAL in line for line in found), f"{heading} heading unlabelled"

    def test_markdown_labels_judge_numbers_uncalibrated(self, baseline_db):
        """Spec: the markdown headline says 'uncalibrated judge'."""
        markdown = format_baseline_markdown(build(baseline_db))
        headline = [line for line in markdown.splitlines() if "4/9" in line]
        assert headline, "headline rate 4/9 not in the markdown"
        assert any(UNCAL in line for line in headline)


# ---------------------------------------------------------------- output formats

def _pct(x):
    # 2 decimals (Gaurav 2026-10-16): k/80 is exact (7/80 = 8.75%), so no float tie at 1 decimal.
    return f"{100 * x:.2f}%"


class TestTextAndMarkdown:
    def test_markdown_has_headline_rate_with_interval(self, baseline_db):
        """Spec: the markdown holds the headline rate 44.44% with its Wilson CI and 4/9."""
        # Wilson(4, 9) = [0.1888, 0.7333] -> 18.88% and 73.33%
        low, high = wilson_interval(4, 9, CONFIDENCE)
        markdown = format_baseline_markdown(build(baseline_db))
        assert "4/9" in markdown
        assert _pct(4 / 9) in markdown, "44.44% expected"
        assert _pct(low) in markdown and _pct(high) in markdown

    def test_text_report_has_headline_rate_with_interval(self, baseline_db):
        """Spec: the printed report shows rate, 95% interval and k/n."""
        low, high = wilson_interval(4, 9, CONFIDENCE)
        text = format_baseline_report(build(baseline_db))
        assert "4/9" in text and _pct(4 / 9) in text
        assert _pct(low) in text and _pct(high) in text

    @pytest.mark.parametrize("qtype, counts", [("lookup", "2/4"), ("multi_chunk", "1/2"),
                                               ("paraphrased", "0/1"), ("unanswerable", "1/2")])
    def test_markdown_per_type_table_has_each_type_with_counts(self, baseline_db, qtype, counts):
        """Spec: the per-type table gives every query type with its k and n (separate cells)."""
        k, n = counts.split("/")
        row = f"| {qtype} | {k} | {n} |"
        assert any(line.startswith(row) for line in
                   format_baseline_markdown(build(baseline_db)).splitlines()), \
            f"no table row starting {row!r}"

    def test_markdown_answerable_row_has_three_of_seven(self, baseline_db):
        """Spec: the report gives the answerable rate 3/7."""
        assert any(line.startswith("| answerable | 3 | 7 |") for line in
                   format_baseline_markdown(build(baseline_db)).splitlines())

    @pytest.mark.parametrize("cause, rate", [("retrieval miss", "1/9"), ("generation error", "3/9")])
    def test_markdown_cause_table_has_each_cause_with_its_rate(self, baseline_db, cause, rate):
        """Spec: the cause table lists each cause with its count over judged answers."""
        lines = [line.lower().replace("_", " ") for line in format_baseline_markdown(build(baseline_db)).splitlines()]
        assert [line for line in lines if cause in line and rate in line], f"no {cause} row with {rate}"

    def test_markdown_shares_of_ungrounded_are_shown(self, baseline_db):
        """Spec: each cause also as a share of the ungrounded answers (1/4 and 3/4)."""
        markdown = format_baseline_markdown(build(baseline_db))
        assert "1/4" in markdown and "3/4" in markdown

    def test_text_report_shows_cause_counts(self, baseline_db):
        """Spec: the printed report shows the failure causes."""
        text = format_baseline_report(build(baseline_db)).lower().replace("_", " ")
        assert "retrieval miss" in text and "generation error" in text

    def test_markdown_states_the_bad_answer_rate(self, baseline_db):
        """Spec Decision 2: the bad-answer rate 5/9 is reported next to the headline."""
        assert "5/9" in format_baseline_markdown(build(baseline_db))

    def test_markdown_lists_invented_answer_and_false_refusal_ids(self, baseline_db):
        """Spec: the invented-answer (q010) and false-refusal (q004) items are named."""
        markdown = format_baseline_markdown(build(baseline_db))
        assert "q010" in markdown and "q004" in markdown

    def test_markdown_states_cv_hypothesis_next_to_the_number(self, baseline_db):
        """Spec step rule: say where the CV hypothesis (25% -> 12%) stands."""
        markdown = format_baseline_markdown(build(baseline_db))
        assert "25%" in markdown and "12%" in markdown

    def test_markdown_warns_about_small_per_type_n(self, baseline_db):
        """Spec: per-type rates carry the note that differences inside the intervals are not meaningful."""
        markdown = format_baseline_markdown(build(baseline_db)).lower()
        assert "not meaningful" in markdown

    def test_markdown_mentions_single_sample_per_query(self, baseline_db):
        """Spec: the report says each answer is one sample from a not fully deterministic model."""
        markdown = format_baseline_markdown(build(baseline_db)).lower()
        assert "sample" in markdown and "deterministic" in markdown


class TestLogRow:
    CREATED = "2026-10-16T10:11:12+00:00"

    def test_log_row_is_a_single_markdown_table_row(self, baseline_db):
        """Spec: log_row returns one experiments/log.md table row (6 cells, one line)."""
        row = log_row(build(baseline_db), self.CREATED, "v0 baseline")
        assert "\n" not in row.strip(), "a log row is one line"
        assert row.strip().startswith("|") and row.strip().endswith("|")
        cells = row.strip().strip("|").split("|")
        assert len(cells) == 6, f"date, config, change, run_id, result, note; got {cells}"

    def test_log_row_has_date_config_run_id_and_note(self, baseline_db):
        """Spec: the row holds the date, config, run_id and note."""
        row = log_row(build(baseline_db), self.CREATED, "my baseline note")
        assert "2026-10-16" in row
        assert "configs/fake.yaml" in row
        assert baseline_db.run_id in row
        assert "my baseline note" in row
        assert "baseline" in row.lower(), "the change column says baseline"

    def test_log_row_result_text_has_rate_interval_counts_and_label(self, baseline_db):
        """Spec: result like '8.75% [4.30%, 16.98%] (7/80), uncalibrated judge' -> here 44.44% [18.88%, 73.33%] (4/9)."""
        low, high = wilson_interval(4, 9, CONFIDENCE)
        row = log_row(build(baseline_db), self.CREATED, "note")
        assert f"{_pct(4 / 9)} [{_pct(low)}, {_pct(high)}]" in row
        assert "4/9" in row
        assert UNCAL in row


# ---------------------------------------------------------------- store: run_reports

class TestRunReportsStore:
    def test_run_reports_table_has_the_spec_columns(self, baseline_db):
        """Spec Data changes: run_reports(id, run_id, kind, versions_json, report_json, created_at)."""
        with closing(sqlite3.connect(baseline_db.path)) as conn:
            columns = [row[1] for row in conn.execute("PRAGMA table_info(run_reports)")]
        assert columns == ["id", "run_id", "kind", "versions_json", "report_json", "created_at"]

    def test_write_and_get_run_report_round_trip(self, baseline_db):
        """Spec: write_run_report returns an id and get_run_reports returns the stored row."""
        # Assumption: the input row has the keys run_id, kind, versions, report, created_at.
        report = build(baseline_db)
        report_id = store.write_run_report(baseline_db.conn, {
            "run_id": baseline_db.run_id, "kind": "baseline", "versions": baseline_db.versions,
            "report": report, "created_at": "2026-10-16T10:00:00+00:00"})
        rows = store.get_run_reports(baseline_db.conn, baseline_db.run_id)
        assert len(rows) == 1 and rows[0]["id"] == report_id
        assert rows[0]["kind"] == "baseline" and rows[0]["run_id"] == baseline_db.run_id

    def test_get_run_reports_for_other_run_is_empty(self, baseline_db):
        """Spec: get_run_reports is per run."""
        assert store.get_run_reports(baseline_db.conn, "other-run") == []


# ---------------------------------------------------------------- CLI main

class TestMain:
    def test_main_ok_prints_report_and_exits_zero(self, baseline_db, tmp_path, capsys, no_llm):
        """Spec CLI: exit 0 and the report is printed (headline 4/9, uncalibrated judge)."""
        assert cli(baseline_db, tmp_path) == 0
        out = capsys.readouterr().out
        assert "4/9" in out and UNCAL in out and baseline_db.run_id in out

    def test_main_makes_no_llm_call(self, baseline_db, tmp_path, no_llm):
        """Spec: no LLM call; the patched generate functions raise if called."""
        assert cli(baseline_db, tmp_path) == 0

    def test_main_stores_one_run_report_per_call(self, baseline_db, tmp_path):
        """Spec: every report invocation stores one run_reports row (2 calls -> 2 rows)."""
        assert cli(baseline_db, tmp_path) == 0
        assert len(store.get_run_reports(baseline_db.conn, baseline_db.run_id)) == 1
        assert cli(baseline_db, tmp_path) == 0
        assert len(store.get_run_reports(baseline_db.conn, baseline_db.run_id)) == 2

    def test_main_stored_row_has_kind_versions_and_report(self, baseline_db, tmp_path):
        """Spec: the row holds kind 'baseline', the three versions and the full report JSON."""
        assert cli(baseline_db, tmp_path) == 0
        kind, versions_json, report_json = sql_one(
            baseline_db.path, "SELECT kind, versions_json, report_json FROM run_reports WHERE run_id = ?",
            (baseline_db.run_id,))
        assert kind == "baseline"
        assert json.loads(versions_json) == baseline_db.versions
        assert baseline_db.run_id in report_json
        assert rate_dicts(json.loads(report_json), 4, 9), "stored report must hold the 4/9 rate"

    def test_main_version_flags_override_current_versions(self, baseline_db, tmp_path, capsys):
        """Spec: a version flag is used when given: other faithfulness version -> 0/10 ungrounded."""
        baseline_db.add_faith_version("faith-bl-other")
        argv = ["--run-id", baseline_db.run_id,
                "--harness-config", write_harness(tmp_path, baseline_db.path),
                "--checks-version", baseline_db.versions["checks"],
                "--faithfulness-version", "faith-bl-other",
                "--relevance-version", baseline_db.versions["relevance"]]
        assert main(argv) == 0
        assert "0/10" in capsys.readouterr().out
        stored = json.loads(sql_one(baseline_db.path, "SELECT versions_json FROM run_reports")[0])
        assert stored["faithfulness"] == "faith-bl-other"

    def test_main_without_version_flags_uses_current_versions_which_have_no_rows(self, baseline_db, tmp_path):
        """Spec: by default current_versions() is used; the fixture DB has no rows for them -> exit 1."""
        assert cli(baseline_db, tmp_path, versions=False) == 1
        assert store.get_run_reports(baseline_db.conn, baseline_db.run_id) == []

    def test_main_unknown_run_exits_one(self, baseline_db, tmp_path):
        """Spec exit codes: an unknown run -> 1, nothing stored."""
        argv = ["--run-id", "no-such-run", "--harness-config", write_harness(tmp_path, baseline_db.path),
                "--checks-version", baseline_db.versions["checks"],
                "--faithfulness-version", baseline_db.versions["faithfulness"],
                "--relevance-version", baseline_db.versions["relevance"]]
        assert main(argv) == 1
        assert sql_one(baseline_db.path, "SELECT COUNT(*) FROM run_reports")[0] == 0

    def test_main_run_not_completed_exits_one(self, baseline_db, tmp_path):
        """Spec exit codes: a run that is not completed -> 1."""
        with baseline_db.conn:
            baseline_db.conn.execute("UPDATE runs SET status = 'aborted' WHERE run_id = ?",
                                     (baseline_db.run_id,))
        assert cli(baseline_db, tmp_path) == 1

    def test_main_heldout_run_exits_one_without_flag(self, baseline_db, tmp_path):
        """Spec exit codes: a held-out run without --allow-heldout -> 1, nothing stored."""
        with baseline_db.conn:
            baseline_db.conn.execute("UPDATE runs SET split = 'heldout' WHERE run_id = ?",
                                     (baseline_db.run_id,))
        assert cli(baseline_db, tmp_path) == 1
        assert sql_one(baseline_db.path, "SELECT COUNT(*) FROM run_reports")[0] == 0

    def test_main_heldout_run_exits_zero_with_flag(self, baseline_db, tmp_path):
        """Spec CLI: --allow-heldout lets a held-out run through."""
        with baseline_db.conn:
            baseline_db.conn.execute("UPDATE runs SET split = 'heldout' WHERE run_id = ?",
                                     (baseline_db.run_id,))
        assert cli(baseline_db, tmp_path, "--allow-heldout") == 0

    @pytest.mark.parametrize("flag, scorer", [("--checks-version", "checks"),
                                              ("--faithfulness-version", "faithfulness"),
                                              ("--relevance-version", "relevance")])
    def test_main_version_with_no_rows_exits_one(self, baseline_db, tmp_path, flag, scorer):
        """Spec exit codes: a version with no rows -> 1, nothing stored."""
        argv = ["--run-id", baseline_db.run_id,
                "--harness-config", write_harness(tmp_path, baseline_db.path)]
        for name, flag_name in [("checks", "--checks-version"), ("faithfulness", "--faithfulness-version"),
                                ("relevance", "--relevance-version")]:
            argv += [flag_name, "no-such-version" if name == scorer else baseline_db.versions[name]]
        assert main(argv) == 1
        assert sql_one(baseline_db.path, "SELECT COUNT(*) FROM run_reports")[0] == 0

    def test_main_judge_failure_exits_one_when_allow_incomplete_is_false(self, baseline_db, tmp_path):
        """Spec exit codes: a judge with failed items and allow_incomplete false -> 1."""
        assert cli(baseline_db, tmp_path, allow_incomplete=False) == 1

    def test_main_judge_failure_exits_zero_when_allow_incomplete_is_true(self, baseline_db, tmp_path):
        """Spec Decision 3: allow_incomplete true -> the failure is reported, exit 0."""
        assert cli(baseline_db, tmp_path, allow_incomplete=True) == 0

    def test_main_bad_answer_rate_follows_the_config_switch(self, baseline_db, tmp_path, capsys):
        """Spec Config: include_bad_answer_rate false -> 5/9 is not printed."""
        assert cli(baseline_db, tmp_path, include_bad=True) == 0
        assert "5/9" in capsys.readouterr().out
        assert cli(baseline_db, tmp_path, include_bad=False) == 0
        assert "5/9" not in capsys.readouterr().out

    def test_main_without_markdown_or_log_flag_writes_no_files(self, baseline_db, tmp_path):
        """Spec: without --markdown and --log the report is only printed and stored."""
        before = set(tmp_path.rglob("*"))
        assert cli(baseline_db, tmp_path) == 0
        new = {p for p in set(tmp_path.rglob("*")) - before if p.suffix in (".md",)}
        assert not new, f"unexpected markdown files: {new}"


class TestMainMarkdown:
    def test_main_markdown_writes_the_report_file(self, baseline_db, tmp_path):
        """Spec: --markdown writes the markdown report."""
        target = tmp_path / "baseline_v0.md"
        assert cli(baseline_db, tmp_path, "--markdown", str(target)) == 0
        text = target.read_text(encoding="utf-8")
        assert "4/9" in text and baseline_db.run_id in text and UNCAL in text

    def test_main_markdown_existing_file_is_refused_and_kept(self, baseline_db, tmp_path):
        """Spec: --markdown refuses to overwrite without --force (exit 1, file untouched)."""
        target = tmp_path / "baseline_v0.md"
        target.write_text("OLD CONTENT", encoding="utf-8")
        assert cli(baseline_db, tmp_path, "--markdown", str(target)) == 1
        assert target.read_text(encoding="utf-8") == "OLD CONTENT"

    def test_main_markdown_existing_file_is_overwritten_with_force(self, baseline_db, tmp_path):
        """Spec: --force allows overwriting an existing --markdown file."""
        target = tmp_path / "baseline_v0.md"
        target.write_text("OLD CONTENT", encoding="utf-8")
        assert cli(baseline_db, tmp_path, "--markdown", str(target), "--force") == 0
        assert "OLD CONTENT" not in target.read_text(encoding="utf-8")


class TestMainLog:
    HEADER = "| date | config | change | run | result | note |\n|---|---|---|---|---|---|\n"

    def test_main_log_appends_one_row_for_the_run(self, baseline_db, tmp_path):
        """Spec: --log appends exactly one row with the run_id and the result text."""
        log = tmp_path / "log.md"
        log.write_text(self.HEADER, encoding="utf-8")
        assert cli(baseline_db, tmp_path, "--log", str(log)) == 0
        text = log.read_text(encoding="utf-8")
        assert text.startswith(self.HEADER), "existing log content must be kept"
        rows = [line for line in text.splitlines() if baseline_db.run_id in line]
        assert len(rows) == 1 and "4/9" in rows[0] and UNCAL in rows[0]

    def test_main_log_duplicate_run_id_is_refused(self, baseline_db, tmp_path):
        """Spec: a second --log call for the same run exits 1 and adds no row."""
        log = tmp_path / "log.md"
        log.write_text(self.HEADER, encoding="utf-8")
        assert cli(baseline_db, tmp_path, "--log", str(log)) == 0
        assert cli(baseline_db, tmp_path, "--log", str(log)) == 1
        rows = [line for line in log.read_text(encoding="utf-8").splitlines() if baseline_db.run_id in line]
        assert len(rows) == 1, "still exactly one row for the run"


# ---------------------------------------------------------------- layering

def test_report_module_imports_neither_rag_nor_llm_directly():
    """Spec Layering: evalharness/report.py imports neither rag nor llm."""
    tree = ast.parse((PROJECT_ROOT / "evalharness" / "report.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert name.split(".")[0] not in ("rag", "llm"), f"report.py imports {name}"


def test_report_module_exposes_the_documented_interface():
    """Spec Interfaces: the public names exist."""
    for name in ["CAUSES", "load_report_config", "current_versions", "failure_cause", "item_table",
                 "build_baseline_report", "format_baseline_report", "format_baseline_markdown",
                 "log_row", "main"]:
        assert hasattr(report_module, name), f"evalharness.report.{name} missing"
