"""Step 06 tests: deterministic checks and retrieval metrics (offline, tmp DBs only).

Covers evalharness/metrics.py (pure formulas), evalharness/checks.py (per-output
checks, score_run, report, CLI) and the new `scores` functions in evalharness/store.py.
Every expected number is worked out by hand in a comment. Nothing here calls an LLM,
loads an embedding model or touches data/results.sqlite.

Fixtures `scored_run_db` and `checks_config` live in tests/conftest.py; the 8 hand-made
outputs of the fixture run are described in the `scored_run_db` docstring.
"""

import ast
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
import yaml

from evalharness import checks, metrics, store
from evalharness.checks import (CHECKS_CODE_VERSION, ScoringRefused, build_report, find_refusal_phrase,
                                format_report, is_empty_answer, load_checks_config, metric_names,
                                normalize_text, score_output, score_run)
from evalharness.metrics import (doc_recall, first_source_rank, hit_at_k, mean_summary,
                                 percentile_summary, rate_summary, reciprocal_rank, wilson_interval)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_HARNESS_CONFIG = PROJECT_ROOT / "configs" / "harness.yaml"

# Worked example from the spec: retrieved doc ids in rank order, source doc leave_policy.
WORKED_CHUNKS = ["intern_leave_policy", "intern_leave_policy", "leave_policy", "internship_program_rules"]


# ---------------------------------------------------------------- helpers

def by_metric(rows):
    """{metric: row} for a list of score rows; fails if a metric appears twice."""
    result = {row["metric"]: row for row in rows}
    assert len(result) == len(rows), "a metric appears more than once"
    return result


def chunks_for(*doc_ids):
    """Contract chunk dicts for the given doc ids, in rank order (chunk_id = '<doc>#<rank-1>')."""
    return [{"chunk_id": f"{doc}#{i:03d}", "doc_id": doc, "text": f"text {i}", "score": 0.5}
            for i, doc in enumerate(doc_ids)]


def make_output(answer="Employees get 8 casual leaves.", chunks=None, latency_ms=100.0, error=None):
    """An outputs row as store.get_outputs() returns it (retrieved_chunks already parsed)."""
    return {"run_id": "r1", "query_id": "q001", "answer": answer,
            "retrieved_chunks": chunks if chunks is not None else chunks_for("leave_policy"),
            "latency_ms": latency_ms, "error": error}


def make_query_dict(sources, type_="lookup"):
    """The query side of score_output: type and source_doc_ids."""
    return {"id": "q001", "type": type_, "source_doc_ids": sources}


def count_scores(conn, run_id=None):
    """Number of rows in scores (optionally for one run)."""
    if run_id is None:
        return conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    return conn.execute("SELECT COUNT(*) FROM scores WHERE run_id = ?", (run_id,)).fetchone()[0]


def add_run(conn, run_id, *, split="dev", status="completed", output_ids=("q001",)):
    """Insert a minimal fake run (queries must already be in the DB) with one output per id."""
    store.start_run(conn, run_id=run_id, split=split, config={"name": "fake"},
                    config_path="configs/fake.yaml", git_commit="abc1234", git_dirty=False,
                    test_set_sha256="0" * 64, system_info={}, n_queries=len(output_ids),
                    llm_calls_start_id=0)
    for query_id in output_ids:
        store.write_output(conn, run_id=run_id, query_id=query_id,
                           answer="Employees get 8 casual leaves.",
                           chunks=chunks_for("leave_policy"), latency_ms=10.0, error=None)
    if status != "running":
        store.finish_run(conn, run_id, status=status, n_ok=len(output_ids), n_failed=0,
                         llm_calls_summary={}, note=None)


def valid_harness(**checks_overrides):
    """A harness-config dict with a valid `checks` block (for load_checks_config)."""
    block = {"hit_ks": [1, 4], "report_confidence": 0.95,
             "refusal_phrases": ["no mention of", "does not specify"]}
    block.update(checks_overrides)
    return {"checks": block}


def write_harness_yaml(tmp_path, db_path, checks_block=None):
    """Write a tmp harness YAML whose db_path is the given tmp DB; return its path as str."""
    block = checks_block or {"hit_ks": [1, 2], "report_confidence": 0.95,
                             "refusal_phrases": ["no mention of", "does not mention",
                                                 "doesn't mention", "cannot find"]}
    config = {
        "db_path": str(db_path),
        "test_set_config": str(tmp_path / "test_set.yaml"),
        "max_consecutive_failures": 5,
        "error_max_chars": 500,
        "system": {"answer": "somewhere:answer", "load_config": "somewhere:load", "describe": None},
        "checks": block,
    }
    path = tmp_path / "harness_test.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


# ================================================================ metrics.py

class TestFirstSourceRank:
    def test_first_source_rank_source_at_first_position_returns_one(self):
        """Spec: rank is the 1-based position of the first chunk whose doc_id is a source doc."""
        assert first_source_rank(["leave_policy", "it_policy"], ["leave_policy"]) == 1

    def test_first_source_rank_after_two_non_source_chunks_returns_three(self):
        """Spec worked example: first source at position 3."""
        assert first_source_rank(WORKED_CHUNKS, ["leave_policy"]) == 3

    def test_first_source_rank_no_source_retrieved_returns_none(self):
        """Spec: None if no chunk is from a source doc."""
        assert first_source_rank(["it_policy", "payroll_policy"], ["leave_policy"]) is None

    def test_first_source_rank_empty_chunk_list_returns_none(self):
        """Spec: no chunks means no source chunk, so None (not an error)."""
        assert first_source_rank([], ["leave_policy"]) is None

    def test_first_source_rank_duplicate_source_chunks_count_the_first_one(self):
        """Spec: duplicate source chunks count once, at the first position."""
        assert first_source_rank(["x", "leave_policy", "leave_policy"], ["leave_policy"]) == 2

    def test_first_source_rank_two_sources_uses_the_earlier_one(self):
        """Spec: any source doc counts, so the earliest of several sources wins."""
        # chunks x, b, a with sources [a, b]: b is at position 2, a at position 3 -> 2
        assert first_source_rank(["x", "b", "a"], ["a", "b"]) == 2

    def test_first_source_rank_empty_sources_raises_value_error(self):
        """Spec: ValueError when source_doc_ids is empty (unanswerable: undefined)."""
        assert first_source_rank(["a"], ["a"]) == 1  # control: same call with a source works
        with pytest.raises(ValueError):
            first_source_rank(["a"], [])


class TestHitAtK:
    @pytest.mark.parametrize("k, expected", [(1, 0), (2, 0), (3, 1), (4, 1)])
    def test_hit_at_k_worked_example_matches_hand_calculation(self, k, expected):
        """Spec worked example: first source at rank 3 -> hit@1 = 0, hit@3 = 1, hit@4 = 1."""
        # source leave_policy is at position 3: hit@k = 1 exactly when k >= 3
        assert hit_at_k(WORKED_CHUNKS, ["leave_policy"], k) == expected

    def test_hit_at_k_spec_example_returns_zero(self):
        """Spec example: hit_at_k(['a','a','b','c'], ['b'], 1) -> 0."""
        assert hit_at_k(["a", "a", "b", "c"], ["b"], 1) == 0

    @pytest.mark.parametrize("k", [0, -1, 5])
    def test_hit_at_k_k_out_of_range_raises_value_error(self, k):
        """Spec: ValueError if k < 1 or k > len(chunk_doc_ids)."""
        assert hit_at_k(WORKED_CHUNKS, ["leave_policy"], 4) == 1  # control: k = 4 of 4 is valid
        with pytest.raises(ValueError):
            hit_at_k(WORKED_CHUNKS, ["leave_policy"], k)

    def test_hit_at_k_empty_sources_raises_value_error(self):
        """Spec: ValueError when there are no source docs."""
        with pytest.raises(ValueError):
            hit_at_k(WORKED_CHUNKS, [], 2)


class TestReciprocalRank:
    @pytest.mark.parametrize("chunks, sources, expected", [
        (["a", "b"], ["a"], 1.0),                    # rank 1 -> 1/1
        (["x", "y", "a", "z"], ["a"], 1 / 3),        # rank 3 -> 1/3
        (["x", "y"], ["a"], 0.0),                    # not retrieved -> 0.0
    ])
    def test_reciprocal_rank_hand_calculation(self, chunks, sources, expected):
        """Spec: RR = 1 / first_source_rank, or 0.0 if no source doc was retrieved."""
        assert reciprocal_rank(chunks, sources) == pytest.approx(expected)

    def test_reciprocal_rank_worked_example_is_one_third(self):
        """Spec worked example: rank 3 gives 0.3333."""
        assert reciprocal_rank(WORKED_CHUNKS, ["leave_policy"]) == pytest.approx(0.3333, abs=1e-4)

    def test_mrr_over_one_third_and_zero_matches_hand_calculation(self):
        """Spec: MRR of RR values [1, 1/3, 0] via mean_summary is 0.4444."""
        # (1 + 0.3333 + 0) / 3 = 1.3333 / 3 = 0.4444
        summary = mean_summary([1.0, 1 / 3, 0.0])
        assert summary["mean"] == pytest.approx(0.4444, abs=1e-4)
        assert summary["n"] == 3

    def test_reciprocal_rank_empty_sources_raises_value_error(self):
        """Spec: undefined for unanswerable queries (no sources), so ValueError."""
        with pytest.raises(ValueError):
            reciprocal_rank(["a"], [])


class TestDocRecall:
    def test_doc_recall_one_of_two_sources_found_returns_half(self):
        """Spec: retrieved [A, A, C, D] with sources [A, B] gives doc recall 0.5."""
        # distinct sources {A, B}; found {A}; 1 / 2 = 0.5
        assert doc_recall(["A", "A", "C", "D"], ["A", "B"]) == pytest.approx(0.5)

    def test_doc_recall_all_sources_found_returns_one(self):
        """Spec: every source doc retrieved gives 1.0 (the 'all found' case)."""
        assert doc_recall(["B", "x", "A"], ["A", "B"]) == pytest.approx(1.0)

    def test_doc_recall_no_source_found_returns_zero(self):
        """Spec: no source doc retrieved gives 0.0."""
        assert doc_recall(["x", "y"], ["A", "B"]) == pytest.approx(0.0)

    def test_doc_recall_repeated_source_id_counts_distinct_docs(self):
        """Spec: share of *distinct* source docs; a repeated source id counts once."""
        # distinct sources {A, B}; found {A}; 1 / 2 = 0.5
        assert doc_recall(["A"], ["A", "A", "B"]) == pytest.approx(0.5)


class TestWilsonInterval:
    @pytest.mark.parametrize("successes, n, low, high", [
        (9, 12, 0.4677, 0.9111),    # spec worked example
        (12, 12, 0.7575, 1.0),      # all successes: normal approx would claim [1, 1]
        (0, 12, 0.0, 0.2425),       # no successes
        (3, 5, 0.2307, 0.8824),     # p = .6, n = 5: center 0.98416/1.76832 = 0.5566, half 0.3258
        (4, 5, 0.3755, 0.9638),     # p = .8, n = 5: center 1.18416/1.76832 = 0.6697, half 0.2941
        (1, 2, 0.0945, 0.9055),     # p = .5, n = 2: center 0.5, half 1.1843/2.9208 = 0.4055
        (1, 1, 0.2065, 1.0),        # n = 1: center 2.9208/4.8416 = 0.6033, half 0.3967
    ])
    def test_wilson_interval_matches_hand_calculation(self, successes, n, low, high):
        """Spec: Wilson score interval at 95% (z = 1.96) for small hand-checked cases."""
        got_low, got_high = wilson_interval(successes, n, 0.95)
        assert got_low == pytest.approx(low, abs=1e-3), f"{successes}/{n} low bound"
        assert got_high == pytest.approx(high, abs=1e-3), f"{successes}/{n} high bound"

    @pytest.mark.parametrize("n", [1, 2, 5, 12, 30, 100])
    def test_wilson_interval_bounds_always_inside_zero_one(self, n):
        """Spec: both bounds are clamped to [0, 1], including at 0/n and n/n."""
        for successes in (0, n // 2, n):
            low, high = wilson_interval(successes, n, 0.95)
            assert 0.0 <= low <= high <= 1.0, f"{successes}/{n} gave ({low}, {high})"

    def test_wilson_interval_n_over_n_upper_bound_is_exactly_one(self):
        """Spec: float error can give 1.0000000002 at n/n; the bound must be clamped to 1.0."""
        assert wilson_interval(12, 12, 0.95)[1] <= 1.0
        assert wilson_interval(12, 12, 0.95)[1] == pytest.approx(1.0)

    def test_wilson_interval_n_zero_raises_value_error(self):
        """Spec: ValueError if n < 1."""
        assert wilson_interval(0, 1, 0.95)[1] <= 1.0  # control: n = 1 works
        with pytest.raises(ValueError):
            wilson_interval(0, 0, 0.95)

    @pytest.mark.parametrize("successes, n", [(13, 12), (-1, 12)])
    def test_wilson_interval_successes_outside_zero_to_n_raises_value_error(self, successes, n):
        """Spec: ValueError if successes is not in 0..n."""
        assert wilson_interval(12, 12, 0.95)[0] > 0  # control: 12/12 works
        with pytest.raises(ValueError):
            wilson_interval(successes, n, 0.95)

    def test_wilson_interval_higher_confidence_gives_wider_interval(self):
        """Spec: z grows with confidence, so a 99% interval is wider than a 95% one."""
        low95, high95 = wilson_interval(9, 12, 0.95)
        low99, high99 = wilson_interval(9, 12, 0.99)
        assert (high99 - low99) > (high95 - low95), "99% interval must be wider than 95%"

    def test_wilson_interval_contains_the_observed_rate(self):
        """Wilson interval for 9/12 contains p = 0.75."""
        low, high = wilson_interval(9, 12, 0.95)
        assert low < 0.75 < high


class TestSummaries:
    def test_rate_summary_counts_na_separately_and_excludes_it(self):
        """Spec: None values are counted in n_na and excluded from n and the rate."""
        # values [1, 0, 1, None, 1, None]: n = 4, n_na = 2, successes = 3, rate = 3/4 = 0.75
        summary = rate_summary([1, 0, 1, None, 1, None], 0.95)
        assert summary["n"] == 4
        assert summary["n_na"] == 2
        assert summary["successes"] == 3
        assert summary["rate"] == pytest.approx(0.75)

    def test_rate_summary_ci_is_wilson_interval_of_successes_and_n(self):
        """Spec: rate_summary returns the Wilson interval of successes/n as ci_low and ci_high."""
        summary = rate_summary([1, 1, 1, 0], 0.95)
        low, high = wilson_interval(3, 4, 0.95)
        assert summary["ci_low"] == pytest.approx(low)
        assert summary["ci_high"] == pytest.approx(high)

    def test_rate_summary_nine_of_twelve_matches_spec_interval(self):
        """Spec: 9 of 12 gives rate 0.75 and Wilson (0.4677, 0.9111)."""
        summary = rate_summary([1] * 9 + [0] * 3, 0.95)
        assert summary["rate"] == pytest.approx(0.75)
        assert summary["ci_low"] == pytest.approx(0.4677, abs=1e-3)
        assert summary["ci_high"] == pytest.approx(0.9111, abs=1e-3)

    def test_rate_summary_all_none_gives_n_zero_and_no_rate(self):
        """Spec: with nothing applicable, n = 0 and rate/CI are None (no division by zero)."""
        summary = rate_summary([None, None], 0.95)
        assert summary["n"] == 0
        assert summary["n_na"] == 2
        assert summary["rate"] is None
        assert summary["ci_low"] is None
        assert summary["ci_high"] is None

    def test_rate_summary_empty_list_gives_n_zero_and_no_rate(self):
        """Spec: an empty list behaves like all-None: n = 0, rate None."""
        summary = rate_summary([], 0.95)
        assert summary["n"] == 0 and summary["rate"] is None

    def test_mean_summary_skips_none_values(self):
        """Spec: mean over the applicable values only, with n and n_na."""
        # values 1.0 and 0.5 are applicable: mean = 1.5 / 2 = 0.75
        summary = mean_summary([1.0, None, 0.5])
        assert summary["n"] == 2
        assert summary["n_na"] == 1
        assert summary["mean"] == pytest.approx(0.75)

    def test_mean_summary_all_none_gives_mean_none(self):
        """Spec: mean is None when n == 0."""
        summary = mean_summary([None, None, None])
        assert summary["n"] == 0
        assert summary["n_na"] == 3
        assert summary["mean"] is None

    def test_percentile_summary_matches_numpy_linear_percentiles(self):
        """Spec: p50, p95 and max via numpy.percentile (linear interpolation), None skipped."""
        # sorted values 10, 20, 30, 40: p50 index 1.5 -> 25; p95 index 0.95 * 3 = 2.85 -> 30 + 0.85 * 10 = 38.5
        summary = percentile_summary([40, None, 10, 30, 20])
        assert summary["n"] == 4
        assert summary["n_na"] == 1
        assert summary["p50"] == pytest.approx(25.0)
        assert summary["p95"] == pytest.approx(38.5)
        assert summary["max"] == pytest.approx(40.0)
        assert summary["p95"] == pytest.approx(float(np.percentile([10, 20, 30, 40], 95)))

    def test_percentile_summary_single_value_repeats_it(self):
        """A single value is its own p50, p95 and max."""
        summary = percentile_summary([5.0])
        assert (summary["p50"], summary["p95"], summary["max"]) == (5.0, 5.0, 5.0)

    def test_percentile_summary_all_none_does_not_raise_and_has_n_zero(self):
        """Spec: n and n_na are reported; all-None must not crash (p50/p95 values unspecified)."""
        summary = percentile_summary([None, None])
        assert summary["n"] == 0
        assert summary["n_na"] == 2


# ================================================================ checks.py: text helpers

class TestNormalizeAndRefusal:
    def test_normalize_text_lowercases_and_collapses_whitespace(self):
        """Spec: lowercase, whitespace runs to one space, stripped."""
        assert normalize_text("  Hello\n\n  WORLD \t again ") == "hello world again"

    def test_normalize_text_curly_apostrophe_becomes_straight(self):
        """Spec: curly apostrophes become straight."""
        assert normalize_text("It doesn’t say") == "it doesn't say"

    def test_normalize_text_curly_double_quotes_become_straight(self):
        """Spec: curly quotes become straight."""
        assert normalize_text("“quoted”") == '"quoted"'

    def test_normalize_text_keeps_non_ascii_letters(self):
        """Edge case: non-ASCII letters survive (only lowercased)."""
        assert normalize_text("CafÉ Urlaub") == "café urlaub"

    @pytest.mark.parametrize("answer, expected", [
        ("", True), ("   ", True), ("  \n", True), ("\t\n ", True),
        ("0", False), (".", False), ("No.", False),
    ])
    def test_is_empty_answer_only_whitespace_is_empty(self, answer, expected):
        """Spec: empty means answer.strip() == ''; '0' is not empty."""
        assert is_empty_answer(answer) is expected

    def test_find_refusal_phrase_matches_across_a_line_break(self):
        """Spec: matching happens on the normalised text, so a line break inside a phrase still matches."""
        assert find_refusal_phrase("The context does not\nmention leave.", ["does not mention"]) == "does not mention"

    def test_find_refusal_phrase_matches_curly_apostrophe(self):
        """Spec: 'doesn’t mention' (curly) matches the straight-apostrophe phrase."""
        assert find_refusal_phrase("It doesn’t mention pets.", ["doesn't mention"]) == "doesn't mention"

    def test_find_refusal_phrase_is_case_insensitive(self):
        """Spec: normalisation lowercases the answer."""
        assert find_refusal_phrase("THERE IS NO MENTION OF PETS", ["no mention of"]) == "no mention of"

    def test_find_refusal_phrase_returns_first_phrase_in_list_order(self):
        """Spec: the first configured phrase (list order) that is found is returned."""
        answer = "There is no mention of X and no information on Y."
        # both phrases occur; "no information" comes first in the list although later in the text
        assert find_refusal_phrase(answer, ["no information", "no mention of"]) == "no information"
        assert find_refusal_phrase(answer, ["no mention of", "no information"]) == "no mention of"

    def test_find_refusal_phrase_no_listed_phrase_returns_none(self):
        """Spec control: an answer with none of the phrases is not a refusal."""
        assert find_refusal_phrase("Interns get 12 days of leave.", ["no mention of", "cannot find"]) is None

    def test_find_refusal_phrase_empty_answer_returns_none(self):
        """Edge case: an empty answer contains no phrase."""
        assert find_refusal_phrase("", ["no mention of"]) is None


# ================================================================ checks.py: load_checks_config

class TestLoadChecksConfig:
    def test_load_checks_config_valid_block_passes_and_has_version(self):
        """Spec control: a valid block loads and gets 'checks-<code version>-<8 hex>'."""
        config = load_checks_config(valid_harness())
        assert re.fullmatch(rf"checks-{CHECKS_CODE_VERSION}-[0-9a-f]{{8}}", config["version"]), config["version"]
        assert config["hit_ks"] == [1, 4]
        assert config["refusal_phrases"] == ["no mention of", "does not specify"]

    def test_load_checks_config_real_harness_yaml_loads(self):
        """Spec: the real configs/harness.yaml (read-only) loads: hit_ks [1, 4], 19 phrases, no 'not covered'."""
        with open(REAL_HARNESS_CONFIG, encoding="utf-8") as f:
            harness = yaml.safe_load(f)
        config = load_checks_config(harness)
        assert config["hit_ks"] == [1, 4]
        assert len(config["refusal_phrases"]) == 19
        assert "not covered" not in config["refusal_phrases"]
        assert "multi_doc_hit" not in config
        assert config["report_confidence"] == pytest.approx(0.95)

    @pytest.mark.parametrize("key", ["hit_ks", "refusal_phrases", "report_confidence"])
    def test_load_checks_config_missing_key_raises_value_error_naming_it(self, key):
        """Spec: a missing key raises ValueError that names it."""
        load_checks_config(valid_harness())  # control: the full block is accepted
        harness = valid_harness()
        del harness["checks"][key]
        with pytest.raises(ValueError, match=key):
            load_checks_config(harness)

    def test_load_checks_config_missing_checks_block_raises_value_error(self):
        """Spec: no `checks` block at all raises ValueError."""
        with pytest.raises(ValueError, match="checks"):
            load_checks_config({})

    @pytest.mark.parametrize("bad_ks", [[4, 1], [1, 1], [0, 4], [-1, 2], []])
    def test_load_checks_config_bad_hit_ks_raises_value_error(self, bad_ks):
        """Spec: hit_ks must be sorted, distinct integers >= 1 (unsorted, duplicate, k < 1, empty)."""
        load_checks_config(valid_harness(hit_ks=[1, 4]))  # control
        with pytest.raises(ValueError, match="hit_ks"):
            load_checks_config(valid_harness(hit_ks=bad_ks))

    @pytest.mark.parametrize("bad_phrases", [
        ["No mention of"],            # uppercase
        ["no mention of", ""],        # empty
        ["no mention of", "   "],     # blank
    ])
    def test_load_checks_config_bad_phrase_raises_value_error(self, bad_phrases):
        """Spec: phrases must be lowercase and non-empty."""
        load_checks_config(valid_harness(refusal_phrases=["no mention of"]))  # control
        with pytest.raises(ValueError, match="refusal"):
            load_checks_config(valid_harness(refusal_phrases=bad_phrases))

    def test_load_checks_config_version_identical_across_two_loads(self):
        """Spec: the version is deterministic: two loads of the same settings agree."""
        assert load_checks_config(valid_harness())["version"] == load_checks_config(valid_harness())["version"]

    def test_load_checks_config_adding_a_phrase_changes_version(self):
        """Spec: any change to the phrases gives a new version."""
        base = load_checks_config(valid_harness())["version"]
        more = load_checks_config(valid_harness(
            refusal_phrases=["no mention of", "does not specify", "cannot find"]))["version"]
        assert base != more

    def test_load_checks_config_changing_hit_ks_changes_version(self):
        """Spec: any change to the ks gives a new version."""
        assert (load_checks_config(valid_harness(hit_ks=[1, 4]))["version"]
                != load_checks_config(valid_harness(hit_ks=[1, 2]))["version"])

    def test_load_checks_config_report_confidence_does_not_change_version(self):
        """Spec: report_confidence is not part of the version hash."""
        a = load_checks_config(valid_harness(report_confidence=0.95))["version"]
        b = load_checks_config(valid_harness(report_confidence=0.9))["version"]
        assert a == b

    def test_metric_names_for_ks_one_and_four_lists_the_eight_spec_metrics(self):
        """Spec: metric_names for ks [1, 4] is the 8 listed names (set; order not specified)."""
        names = metric_names({"hit_ks": [1, 4]})
        assert sorted(names) == sorted(["empty_answer", "refusal", "latency_ms", "hit@1", "hit@4",
                                        "in_context", "reciprocal_rank", "doc_recall"])
        assert len(names) == len(set(names)) == 8

    def test_metric_names_follows_hit_ks(self):
        """Spec: one hit@k metric per configured k."""
        names = metric_names({"hit_ks": [1, 2, 3]})
        assert [n for n in names if n.startswith("hit@")] == ["hit@1", "hit@2", "hit@3"]
        assert len(names) == 9


# ================================================================ checks.py: score_output

class TestScoreOutput:
    def test_score_output_hit_at_rank_one_gives_full_scores(self, checks_config):
        """Spec: source doc at rank 1 -> hit@1 = hit@2 = in_context = 1, RR = 1, doc_recall = 1."""
        output = make_output(chunks=chunks_for("leave_policy", "payroll_policy", "it_policy"))
        rows = by_metric(score_output(output, make_query_dict(["leave_policy"]), checks_config))
        assert rows["hit@1"]["value"] == 1
        assert rows["hit@2"]["value"] == 1
        assert rows["in_context"]["value"] == 1
        assert rows["reciprocal_rank"]["value"] == pytest.approx(1.0)
        assert rows["doc_recall"]["value"] == pytest.approx(1.0)
        assert "rank 1" in rows["hit@1"]["reasoning"]

    def test_score_output_metric_set_equals_metric_names_one_row_each(self, checks_config):
        """Spec: exactly one row per metric in metric_names(), always."""
        output = make_output(chunks=chunks_for("leave_policy", "it_policy"))
        rows = score_output(output, make_query_dict(["leave_policy"]), checks_config)
        assert sorted(r["metric"] for r in rows) == sorted(metric_names(checks_config))
        for row in rows:
            assert set(row) >= {"metric", "value", "reasoning"}
            assert row["reasoning"], f"{row['metric']} has an empty reasoning"

    def test_score_output_hit_at_rank_three_matches_hand_calculation(self, checks_config):
        """Spec: source at rank 3 -> hit@1 = 0, hit@2 = 0, in_context = 1, RR = 1/3, reasoning shows rank and chunk."""
        chunks = chunks_for("it_policy", "leave_policy", "payroll_policy")
        output = make_output(chunks=chunks)
        rows = by_metric(score_output(output, make_query_dict(["payroll_policy"]), checks_config))
        # first payroll_policy chunk is at position 3 (chunk id payroll_policy#002); RR = 1/3 = 0.3333
        assert rows["hit@1"]["value"] == 0
        assert rows["hit@2"]["value"] == 0
        assert rows["in_context"]["value"] == 1
        assert rows["reciprocal_rank"]["value"] == pytest.approx(0.3333, abs=1e-4)
        assert "rank 3" in rows["hit@1"]["reasoning"]
        assert "payroll_policy#002" in rows["in_context"]["reasoning"]
        assert "1/3" in rows["reciprocal_rank"]["reasoning"]

    def test_score_output_retrieval_miss_gives_zeros_with_reason(self, checks_config):
        """Spec: no source doc retrieved -> hit = 0, in_context = 0, RR = 0.0, doc_recall = 0.0 (values, not NULL)."""
        output = make_output(chunks=chunks_for("leave_policy", "payroll_policy", "payroll_policy"))
        rows = by_metric(score_output(output, make_query_dict(["it_policy"]), checks_config))
        assert rows["hit@1"]["value"] == 0
        assert rows["hit@2"]["value"] == 0
        assert rows["in_context"]["value"] == 0
        assert rows["reciprocal_rank"]["value"] == pytest.approx(0.0)
        assert rows["doc_recall"]["value"] == pytest.approx(0.0)
        assert "no source doc retrieved" in rows["reciprocal_rank"]["reasoning"]

    def test_score_output_multi_doc_partial_hit_has_hit_one_and_recall_half(self, checks_config):
        """Spec: multi-doc partial case: hit = 1 (any source doc) and doc_recall = 0.5."""
        output = make_output(chunks=chunks_for("leave_policy", "leave_policy", "it_policy"))
        query = make_query_dict(["leave_policy", "payroll_policy"], type_="multi_chunk")
        rows = by_metric(score_output(output, query, checks_config))
        # distinct sources {leave, payroll}; found {leave}; 1 / 2 = 0.5
        assert rows["hit@1"]["value"] == 1
        assert rows["in_context"]["value"] == 1
        assert rows["doc_recall"]["value"] == pytest.approx(0.5)
        assert "1 of 2" in rows["doc_recall"]["reasoning"]
        assert "leave_policy" in rows["doc_recall"]["reasoning"]

    def test_score_output_source_beyond_top_k_counts_for_in_context_not_hit_k(self, checks_config):
        """Spec: in_context uses all retrieved chunks whatever top_k is; hit@k only the first k."""
        chunks = chunks_for("it_policy", "payroll_policy", "it_policy", "payroll_policy", "leave_policy")
        rows = by_metric(score_output(make_output(chunks=chunks), make_query_dict(["leave_policy"]),
                                      checks_config))
        # ks are [1, 2]; source at position 5: hit@1 = hit@2 = 0, in_context = 1, RR = 1/5 = 0.2
        assert rows["hit@2"]["value"] == 0
        assert rows["in_context"]["value"] == 1
        assert rows["reciprocal_rank"]["value"] == pytest.approx(0.2)

    def test_score_output_fewer_chunks_than_k_gives_null_hit_for_that_k(self, checks_config):
        """Spec: hit@k is NULL when fewer than k chunks were retrieved; smaller ks and in_context still scored."""
        output = make_output(chunks=chunks_for("leave_policy"))  # 1 chunk, ks are [1, 2]
        rows = by_metric(score_output(output, make_query_dict(["leave_policy"]), checks_config))
        assert rows["hit@1"]["value"] == 1
        assert rows["hit@2"]["value"] is None
        assert rows["hit@2"]["reasoning"], "a NULL value needs a reason"
        assert rows["in_context"]["value"] == 1

    def test_score_output_zero_chunks_gives_null_hits_and_zero_in_context(self, checks_config):
        """Spec: an ok output with no chunks has fewer than k chunks (hit NULL) and no source doc in context (0)."""
        output = make_output(chunks=[])
        rows = by_metric(score_output(output, make_query_dict(["leave_policy"]), checks_config))
        assert rows["hit@1"]["value"] is None
        assert rows["hit@2"]["value"] is None
        assert rows["in_context"]["value"] == 0
        assert rows["reciprocal_rank"]["value"] == pytest.approx(0.0)

    def test_score_output_unanswerable_query_has_null_retrieval_rows_with_reason(self, checks_config):
        """Spec: unanswerable -> hit@k, in_context, RR, doc_recall are NULL with 'unanswerable: no source docs'."""
        output = make_output(chunks=chunks_for("leave_policy", "payroll_policy"))
        rows = by_metric(score_output(output, make_query_dict([], type_="unanswerable"), checks_config))
        for metric in ("hit@1", "hit@2", "in_context", "reciprocal_rank", "doc_recall"):
            assert rows[metric]["value"] is None, f"{metric} must be NULL, never 0"
            assert rows[metric]["reasoning"] == "unanswerable: no source docs"

    def test_score_output_unanswerable_query_still_gets_answer_checks(self, checks_config):
        """Spec: answer checks (empty, refusal, latency) apply to unanswerable queries too."""
        output = make_output(answer="There is no mention of pets.", latency_ms=42.0,
                             chunks=chunks_for("leave_policy"))
        rows = by_metric(score_output(output, make_query_dict([], type_="unanswerable"), checks_config))
        assert rows["empty_answer"]["value"] == 0
        assert rows["refusal"]["value"] == 1
        assert rows["latency_ms"]["value"] == pytest.approx(42.0)

    def test_score_output_failed_output_gives_every_row_null_with_error_text(self, checks_config):
        """Spec: a failed output has every metric NULL and a reasoning 'output failed: <stored error>'."""
        output = make_output(answer=None, chunks=None, latency_ms=None,
                             error="ContextLimitError: prompt too long")
        rows = score_output(output, make_query_dict(["leave_policy"]), checks_config)
        assert sorted(r["metric"] for r in rows) == sorted(metric_names(checks_config))
        for row in rows:
            assert row["value"] is None, f"{row['metric']} must be NULL for a failed output"
            assert "ContextLimitError: prompt too long" in row["reasoning"]
            assert "output failed" in row["reasoning"]

    def test_score_output_failed_unanswerable_output_has_all_rows_null(self, checks_config):
        """Spec: failed output on an unanswerable query still gives one NULL row per metric."""
        output = make_output(answer=None, chunks=None, latency_ms=None, error="TimeoutError: slow")
        rows = score_output(output, make_query_dict([], type_="unanswerable"), checks_config)
        assert len(rows) == len(metric_names(checks_config))
        assert all(r["value"] is None for r in rows)

    def test_score_output_blank_answer_marks_empty_with_reason(self, checks_config):
        """Spec: blank answer -> empty_answer = 1, reasoning 'answer is blank'."""
        rows = by_metric(score_output(make_output(answer="  \n"), make_query_dict(["leave_policy"]),
                                      checks_config))
        assert rows["empty_answer"]["value"] == 1
        assert rows["empty_answer"]["reasoning"] == "answer is blank"

    def test_score_output_normal_answer_reasoning_has_char_count(self, checks_config):
        """Spec: non-empty answer -> empty_answer = 0, reasoning 'answer has <n> chars'."""
        answer = "Employees get 8 casual leaves."
        rows = by_metric(score_output(make_output(answer=answer), make_query_dict(["leave_policy"]),
                                      checks_config))
        assert rows["empty_answer"]["value"] == 0
        assert f"{len(answer)} chars" in rows["empty_answer"]["reasoning"]

    def test_score_output_refusal_phrase_found_reports_matched_phrase(self, checks_config):
        """Spec: refusal = 1 with reasoning naming the matched phrase."""
        rows = by_metric(score_output(make_output(answer="Based on the context, there is no mention of pets."),
                                      make_query_dict(["leave_policy"]), checks_config))
        assert rows["refusal"]["value"] == 1
        assert "no mention of" in rows["refusal"]["reasoning"]

    def test_score_output_answer_without_phrase_is_not_a_refusal(self, checks_config):
        """Spec: no listed phrase -> refusal = 0, reasoning 'no refusal phrase'."""
        rows = by_metric(score_output(make_output(answer="Interns get 12 days of casual leave."),
                                      make_query_dict(["leave_policy"]), checks_config))
        assert rows["refusal"]["value"] == 0
        assert rows["refusal"]["reasoning"] == "no refusal phrase"

    def test_score_output_non_ascii_answer_is_scored(self, checks_config):
        """Edge case: non-ASCII answer text works (char count, no refusal)."""
        answer = "Der Urlaub beträgt 12 Tage – café ’ 休假"
        rows = by_metric(score_output(make_output(answer=answer), make_query_dict(["leave_policy"]),
                                      checks_config))
        assert rows["empty_answer"]["value"] == 0
        assert rows["refusal"]["value"] == 0
        assert f"{len(answer)} chars" in rows["empty_answer"]["reasoning"]

    def test_score_output_latency_row_copies_stored_latency(self, checks_config):
        """Spec: latency_ms row has the stored latency as its value (reasoning 'from outputs.latency_ms')."""
        rows = by_metric(score_output(make_output(latency_ms=123.5), make_query_dict(["leave_policy"]),
                                      checks_config))
        assert rows["latency_ms"]["value"] == pytest.approx(123.5)
        assert rows["latency_ms"]["reasoning"] == "from outputs.latency_ms"

    def test_score_output_uses_stored_chunk_order_not_scores(self, checks_config):
        """Spec: ranks use the stored chunk order; never re-sort chunks by score."""
        chunks = chunks_for("it_policy", "leave_policy")
        chunks[0]["score"] = 0.1   # the first stored chunk has the lowest score on purpose
        chunks[1]["score"] = 0.9
        rows = by_metric(score_output(make_output(chunks=chunks), make_query_dict(["leave_policy"]),
                                      checks_config))
        # source is second in stored order: RR = 1/2 = 0.5, hit@1 = 0
        assert rows["reciprocal_rank"]["value"] == pytest.approx(0.5)
        assert rows["hit@1"]["value"] == 0


# ================================================================ checks.py: score_run

class TestScoreRun:
    def test_score_run_first_call_writes_outputs_times_metrics_rows(self, scored_run_db, checks_config):
        """Spec: score_run returns 'written' and stores n_outputs x len(metric_names) rows."""
        status = score_run(scored_run_db.conn, scored_run_db.run_id, checks_config)
        rows = store.get_scores(scored_run_db.conn, scored_run_db.run_id, checks_config["version"])
        # 8 outputs x 8 metrics (3 answer checks + hit@1 + hit@2 + in_context + RR + doc_recall) = 64
        assert status == "written"
        assert len(rows) == 8 * len(metric_names(checks_config)) == 64

    def test_score_run_every_output_has_every_metric_exactly_once(self, scored_run_db, checks_config):
        """Spec: every output gets exactly one row per metric, never skipped; every row has a reasoning."""
        score_run(scored_run_db.conn, scored_run_db.run_id, checks_config)
        rows = store.get_scores(scored_run_db.conn, scored_run_db.run_id, checks_config["version"])
        pairs = [(r["query_id"], r["metric"]) for r in rows]
        assert len(pairs) == len(set(pairs)), "duplicate (query, metric) rows"
        assert {r["query_id"] for r in rows} == {f"q00{i}" for i in range(1, 9)}
        for query_id in {r["query_id"] for r in rows}:
            metrics_for_query = sorted(r["metric"] for r in rows if r["query_id"] == query_id)
            assert metrics_for_query == sorted(metric_names(checks_config)), query_id
        assert all(r["reasoning"] for r in rows), "every score row needs a reasoning"
        assert all(r["version"] == checks_config["version"] for r in rows)
        assert all(r["scored_at"] for r in rows)

    def test_score_run_stored_values_match_hand_calculation(self, scored_run_db, checks_config):
        """Spec: stored values follow the fixture's hand-made cases (hit, partial hit, failed, n/a)."""
        score_run(scored_run_db.conn, scored_run_db.run_id, checks_config)
        rows = store.get_scores(scored_run_db.conn, scored_run_db.run_id, checks_config["version"])
        value = {(r["query_id"], r["metric"]): r["value"] for r in rows}
        # q001: leave_policy at rank 1 -> RR 1.0
        assert value[("q001", "reciprocal_rank")] == pytest.approx(1.0)
        # q002: payroll_policy at rank 3 -> RR 1/3
        assert value[("q002", "reciprocal_rank")] == pytest.approx(1 / 3)
        assert value[("q002", "hit@2")] == 0
        # q003: it_policy never retrieved
        assert value[("q003", "in_context")] == 0
        # q004: sources leave + payroll, only leave retrieved -> recall 1 / 2
        assert value[("q004", "doc_recall")] == pytest.approx(0.5)
        # q005: failed output -> every metric NULL
        assert all(value[("q005", m)] is None for m in metric_names(checks_config))
        # q006: blank answer, one chunk -> empty 1, hit@2 NULL (fewer than 2 chunks)
        assert value[("q006", "empty_answer")] == 1
        assert value[("q006", "hit@2")] is None
        # q007: unanswerable -> refusal 1, retrieval NULL
        assert value[("q007", "refusal")] == 1
        assert value[("q007", "hit@1")] is None
        # latencies come from outputs
        assert value[("q004", "latency_ms")] == pytest.approx(400.0)

    def test_score_run_second_call_returns_unchanged_and_writes_nothing(self, scored_run_db, checks_config):
        """Spec: scoring the same run with the same version again is a no-op ('unchanged')."""
        conn, run_id = scored_run_db.conn, scored_run_db.run_id
        score_run(conn, run_id, checks_config)
        before = store.get_scores(conn, run_id, checks_config["version"])
        status = score_run(conn, run_id, checks_config)
        after = store.get_scores(conn, run_id, checks_config["version"])
        assert status == "unchanged"
        assert after == before, "a no-op must not touch the stored rows (scored_at included)"
        assert len(after) == 64

    def test_score_run_tampered_stored_value_raises_scoring_refused_and_keeps_it(self, scored_run_db, checks_config):
        """Spec: different results under the same version are refused, never overwritten."""
        conn, run_id = scored_run_db.conn, scored_run_db.run_id
        score_run(conn, run_id, checks_config)
        with conn:
            conn.execute("UPDATE scores SET value = ? WHERE run_id = ? AND query_id = ? AND metric = ?",
                         (0.0, run_id, "q001", "in_context"))
        with pytest.raises(ScoringRefused):
            score_run(conn, run_id, checks_config)
        stored = conn.execute("SELECT value FROM scores WHERE run_id = ? AND query_id = ? AND metric = ?",
                              (run_id, "q001", "in_context")).fetchone()[0]
        assert stored == 0.0, "the tampered value must not be overwritten"

    def test_score_run_new_version_is_stored_next_to_the_old_one(self, scored_run_db, checks_config):
        """Spec: a changed setting gets a new version; both versions stay stored."""
        conn, run_id = scored_run_db.conn, scored_run_db.run_id
        score_run(conn, run_id, checks_config)
        second = {**checks_config, "version": "checks-test-v2"}
        assert score_run(conn, run_id, second) == "written"
        assert count_scores(conn, run_id) == 128  # 64 rows per version
        assert len(store.get_scores(conn, run_id, "checks-test-v1")) == 64
        assert len(store.get_scores(conn, run_id, "checks-test-v2")) == 64

    @pytest.mark.parametrize("status", ["running", "aborted", "interrupted", "invalid"])
    def test_score_run_non_completed_run_is_refused_and_writes_nothing(self, scored_run_db, checks_config, status):
        """Spec: only completed runs are scored."""
        add_run(scored_run_db.conn, f"run-{status}", status=status)
        with pytest.raises(ScoringRefused):
            score_run(scored_run_db.conn, f"run-{status}", checks_config)
        assert count_scores(scored_run_db.conn) == 0

    def test_score_run_unknown_run_id_is_refused(self, scored_run_db, checks_config):
        """Spec: an unknown run_id raises ScoringRefused."""
        with pytest.raises(ScoringRefused):
            score_run(scored_run_db.conn, "does-not-exist", checks_config)

    def test_score_run_heldout_run_refused_without_flag(self, scored_run_db, checks_config):
        """Spec: a held-out run is refused without allow_heldout, and nothing is written."""
        add_run(scored_run_db.conn, "run-heldout", split="heldout")
        with pytest.raises(ScoringRefused):
            score_run(scored_run_db.conn, "run-heldout", checks_config)
        assert count_scores(scored_run_db.conn) == 0

    def test_score_run_heldout_run_accepted_with_allow_heldout(self, scored_run_db, checks_config):
        """Spec: allow_heldout=True lets a (fake) held-out run be scored."""
        add_run(scored_run_db.conn, "run-heldout", split="heldout")
        assert score_run(scored_run_db.conn, "run-heldout", checks_config, allow_heldout=True) == "written"
        assert count_scores(scored_run_db.conn, "run-heldout") == 8  # 1 output x 8 metrics

    def test_score_run_dev_run_with_allow_heldout_flag_is_still_scored(self, scored_run_db, checks_config):
        """The allow_heldout flag does not change how a dev run is scored."""
        assert score_run(scored_run_db.conn, scored_run_db.run_id, checks_config, allow_heldout=True) == "written"

    def test_score_run_write_failure_midway_leaves_zero_rows(self, scored_run_db, checks_config, monkeypatch):
        """Spec: rows are written in one transaction; a failure for the last output leaves 0 rows."""
        real_score_output = checks.score_output

        def failing_on_last(output, query, config):
            rows = real_score_output(output, query, config)
            if output["query_id"] == "q008":
                rows[0] = {**rows[0], "reasoning": None}   # NOT NULL violation in the last output
            return rows

        monkeypatch.setattr(checks, "score_output", failing_on_last)
        with pytest.raises((sqlite3.IntegrityError, ValueError, TypeError)):
            score_run(scored_run_db.conn, scored_run_db.run_id, checks_config)
        assert count_scores(scored_run_db.conn) == 0, "partial scores must not be left behind"

    def test_scoring_refused_is_an_exception_class(self):
        """Spec: ScoringRefused is an Exception subclass."""
        assert issubclass(ScoringRefused, Exception)


# ================================================================ checks.py: build_report

@pytest.fixture
def report(scored_run_db, checks_config):
    """Score the fixture run, then build its report (confidence 0.95)."""
    score_run(scored_run_db.conn, scored_run_db.run_id, checks_config)
    return build_report(scored_run_db.conn, scored_run_db.run_id, checks_config["version"], 0.95)


class TestBuildReport:
    def test_report_overall_counts_show_ok_and_failed(self, report):
        """Spec: counts (outputs, ok, failed) are reported, failures not hidden."""
        overall = report["groups"]["all"]
        assert (overall["outputs"], overall["ok"], overall["failed"]) == (8, 7, 1)

    def test_report_per_type_counts(self, report):
        """Spec: per-type counts: lookup 3, multi_chunk 2 (1 failed), paraphrased 1, unanswerable 2."""
        groups = report["groups"]
        assert (groups["lookup"]["outputs"], groups["lookup"]["failed"]) == (3, 0)
        assert (groups["multi_chunk"]["outputs"], groups["multi_chunk"]["ok"],
                groups["multi_chunk"]["failed"]) == (2, 1, 1)
        assert groups["paraphrased"]["outputs"] == 1
        assert groups["unanswerable"]["outputs"] == 2

    def test_report_failed_output_listed_by_query_id(self, report):
        """Spec: failed outputs are shown, not hidden (q005 failed)."""
        assert report["failed"] == ["q005"]

    def test_report_overall_hit_at_one_matches_hand_calculation(self, report):
        """Spec: hit@1 over answerable ok outputs, with n, successes, n/a and Wilson CI."""
        summary = report["groups"]["all"]["hit@1"]
        # values: q001 1, q002 0, q003 0, q004 1, q006 1 -> n = 5, successes 3, rate 0.6
        # n/a: q005 (failed), q007, q008 (unanswerable) -> 3
        # Wilson 3/5 = (0.2307, 0.8824)
        assert (summary["n"], summary["successes"], summary["n_na"]) == (5, 3, 3)
        assert summary["rate"] == pytest.approx(0.6)
        assert summary["ci_low"] == pytest.approx(0.2307, abs=1e-3)
        assert summary["ci_high"] == pytest.approx(0.8824, abs=1e-3)

    def test_report_hit_at_two_skips_outputs_with_fewer_than_two_chunks(self, report):
        """Spec: hit@2 is NULL for q006 (one chunk), so it counts as n/a, not 0."""
        summary = report["groups"]["all"]["hit@2"]
        # values: q001 1, q002 0, q003 0, q004 1 -> n = 4, successes 2; n/a: q005, q006, q007, q008 = 4
        assert (summary["n"], summary["successes"], summary["n_na"]) == (4, 2, 4)
        assert summary["rate"] == pytest.approx(0.5)

    def test_report_lists_hit_metrics_for_configured_ks(self, report):
        """Spec: one hit@k section per configured k."""
        assert report["hit_metrics"] == ["hit@1", "hit@2"]
        assert "hit@1" in report["groups"]["all"] and "hit@2" in report["groups"]["all"]

    def test_report_in_context_matches_hand_calculation(self, report):
        """Spec: in_context over answerable ok outputs with Wilson CI."""
        summary = report["groups"]["all"]["in_context"]
        # values: q001 1, q002 1, q003 0, q004 1, q006 1 -> n = 5, successes 4, rate 0.8; Wilson 4/5 = (0.3755, 0.9638)
        assert (summary["n"], summary["successes"], summary["n_na"]) == (5, 4, 3)
        assert summary["rate"] == pytest.approx(0.8)
        assert summary["ci_low"] == pytest.approx(0.3755, abs=1e-3)
        assert summary["ci_high"] == pytest.approx(0.9638, abs=1e-3)

    def test_report_mrr_overall_matches_hand_calculation(self, report):
        """Spec: MRR = mean RR over scored answerable outputs, reported with n."""
        summary = report["groups"]["all"]["reciprocal_rank"]
        # RR: q001 1, q002 1/3, q003 0, q004 1, q006 1 -> (1 + 0.3333 + 0 + 1 + 1) / 5 = 3.3333 / 5 = 0.6667
        assert summary["n"] == 5
        assert summary["mean"] == pytest.approx(0.6667, abs=1e-4)

    def test_report_doc_recall_overall_matches_hand_calculation(self, report):
        """Spec: doc recall mean over scored answerable outputs with n."""
        summary = report["groups"]["all"]["doc_recall"]
        # recall: q001 1, q002 1, q003 0, q004 0.5, q006 1 -> 3.5 / 5 = 0.7
        assert summary["n"] == 5
        assert summary["mean"] == pytest.approx(0.7)

    def test_report_empty_answers_counted(self, report):
        """Spec: empty answers are reported: only q006 is blank among the 7 ok outputs."""
        summary = report["groups"]["all"]["empty_answer"]
        # ok outputs: 7 (q005 failed -> n/a); blank: q006 -> 1 of 7
        assert (summary["n"], summary["successes"], summary["n_na"]) == (7, 1, 1)
        assert report["groups"]["paraphrased"]["empty_answer"]["successes"] == 1

    def test_report_refusals_split_into_correct_and_false(self, report):
        """Spec: refusals are split: correct (unanswerable q007) vs false (answerable q003, curly apostrophe)."""
        assert report["correct_refusals"] == ["q007"]
        assert report["false_refusals"] == ["q003"]
        assert report["unanswerable_not_refused"] == ["q008"]

    def test_report_unanswerable_refusal_rate_has_wilson_ci(self, report):
        """Spec: refusal rate on unanswerable: 1 of 2 with Wilson (0.0945, 0.9055)."""
        summary = report["groups"]["unanswerable"]["refusal"]
        assert (summary["n"], summary["successes"]) == (2, 1)
        assert summary["rate"] == pytest.approx(0.5)
        assert summary["ci_low"] == pytest.approx(0.0945, abs=1e-3)
        assert summary["ci_high"] == pytest.approx(0.9055, abs=1e-3)

    def test_report_retrieval_miss_list_has_the_in_context_zero_queries(self, report):
        """Spec: the report lists the query ids with in_context = 0 (q003)."""
        assert report["retrieval_misses"] == ["q003"]

    def test_report_unanswerable_group_has_no_retrieval_rates(self, report):
        """Spec: retrieval metrics are n/a for unanswerable queries: n = 0, rate None, n_na = 2."""
        group = report["groups"]["unanswerable"]
        assert group["hit@1"]["n"] == 0
        assert group["hit@1"]["n_na"] == 2
        assert group["hit@1"]["rate"] is None
        assert group["reciprocal_rank"]["mean"] is None

    def test_report_lookup_group_numbers_match_hand_calculation(self, report):
        """Spec: per-type numbers: lookup q001-q003."""
        group = report["groups"]["lookup"]
        # hit@1: 1, 0, 0 -> 1/3; in_context: 1, 1, 0 -> 2/3
        # MRR (1 + 1/3 + 0) / 3 = 0.4444; recall (1 + 1 + 0) / 3 = 0.6667
        assert (group["hit@1"]["successes"], group["hit@1"]["n"]) == (1, 3)
        assert (group["in_context"]["successes"], group["in_context"]["n"]) == (2, 3)
        assert group["reciprocal_rank"]["mean"] == pytest.approx(0.4444, abs=1e-4)
        assert group["doc_recall"]["mean"] == pytest.approx(0.6667, abs=1e-4)

    def test_report_multi_chunk_group_shows_failed_as_not_applicable(self, report):
        """Spec: a failed output is n/a, not 0: multi_chunk has q004 (scored) and q005 (failed)."""
        group = report["groups"]["multi_chunk"]
        # in_context: q004 1, q005 None -> n = 1, successes 1, n_na = 1; recall q004 only = 0.5
        assert (group["in_context"]["n"], group["in_context"]["successes"], group["in_context"]["n_na"]) == (1, 1, 1)
        assert group["doc_recall"]["mean"] == pytest.approx(0.5)
        assert group["reciprocal_rank"]["mean"] == pytest.approx(1.0)

    def test_report_paraphrased_group_hit_at_two_is_all_not_applicable(self, report):
        """Spec: q006 has one chunk, so hit@2 is n/a: n = 0, rate None, no division by zero."""
        summary = report["groups"]["paraphrased"]["hit@2"]
        assert (summary["n"], summary["n_na"]) == (0, 1)
        assert summary["rate"] is None

    def test_report_latency_percentiles_match_hand_calculation(self, report):
        """Spec: latency p50, p95, max over ok outputs."""
        summary = report["groups"]["all"]["latency_ms"]
        # sorted [50, 100, 150, 200, 250, 300, 400]; p50 = 200; p95 index 0.95 * 6 = 5.7 -> 300 + 0.7 * 100 = 370; max 400
        assert summary["n"] == 7
        assert summary["n_na"] == 1
        assert summary["p50"] == pytest.approx(200.0)
        assert summary["p95"] == pytest.approx(370.0)
        assert summary["max"] == pytest.approx(400.0)

    def test_report_latency_source_says_real_calls_and_no_cache_hits(self, report):
        """Spec: with 0 cache hits the source says '<n> real calls, 0 cache hits' and no repeat warning."""
        assert "7 real calls" in report["latency_source"]
        assert "0 cache hits" in report["latency_source"]
        assert "repeat" not in report["latency_source"].lower()

    def test_report_latency_source_with_cache_hits_warns_latencies_repeat(self, scored_run_db, checks_config):
        """Spec: with cache hits > 0 the source adds that those latencies repeat earlier model latencies."""
        conn = scored_run_db.conn
        with conn:
            conn.execute("UPDATE runs SET llm_calls_json = ? WHERE run_id = ?",
                         ('{"gen-model": {"calls": 7, "hits": 7, "misses": 0, "errors": 0}}',
                          scored_run_db.run_id))
        score_run(conn, scored_run_db.run_id, checks_config)
        text = build_report(conn, scored_run_db.run_id, checks_config["version"], 0.95)["latency_source"]
        assert "7 cache hits" in text
        assert "repeat" in text.lower()

    def test_report_reads_numbers_from_scores_table(self, scored_run_db, checks_config):
        """Spec: the report reads back from `scores`: corrupting a stored row changes the report."""
        conn, run_id = scored_run_db.conn, scored_run_db.run_id
        score_run(conn, run_id, checks_config)
        with conn:
            conn.execute("UPDATE scores SET value = ? WHERE run_id = ? AND query_id = ? AND metric = ?",
                         (0.0, run_id, "q001", "in_context"))
        changed = build_report(conn, run_id, checks_config["version"], 0.95)
        # in_context: q001 now 0 -> successes 3 of 5; misses q001 and q003
        assert changed["groups"]["all"]["in_context"]["successes"] == 3
        assert changed["retrieval_misses"] == ["q001", "q003"]

    def test_report_ignores_scores_of_other_versions(self, scored_run_db, checks_config):
        """Spec: the report is for one version; another version's rows do not leak in."""
        conn, run_id = scored_run_db.conn, scored_run_db.run_id
        score_run(conn, run_id, checks_config)
        store.write_scores(conn, run_id, "other-version",
                           [{"query_id": "q001", "metric": "in_context", "value": 0.0, "reasoning": "x"}])
        summary = build_report(conn, run_id, checks_config["version"], 0.95)["groups"]["all"]["in_context"]
        assert summary["successes"] == 4

    def test_report_ci_width_follows_the_confidence_argument(self, scored_run_db, checks_config):
        """Spec: build_report takes the confidence level; 0.99 gives a wider interval than 0.95."""
        conn, run_id = scored_run_db.conn, scored_run_db.run_id
        score_run(conn, run_id, checks_config)
        narrow = build_report(conn, run_id, checks_config["version"], 0.95)["groups"]["all"]["in_context"]
        wide = build_report(conn, run_id, checks_config["version"], 0.99)["groups"]["all"]["in_context"]
        assert (wide["ci_high"] - wide["ci_low"]) > (narrow["ci_high"] - narrow["ci_low"])


class TestFormatReport:
    def test_format_report_returns_text_with_run_id_and_retrieval_misses(self, report):
        """Spec: plain-text report names the run and lists the retrieval-miss ids."""
        text = format_report(report)
        assert isinstance(text, str)
        assert report["run_id"] in text
        assert "q003" in text

    def test_format_report_shows_successes_over_n_and_the_wilson_bounds(self, report):
        """Spec: every rate shows n, successes and its CI (hit@1: 3 of 5, Wilson 0.23 to 0.88)."""
        text = format_report(report)
        assert re.search(r"3\s*/\s*5|3 of 5", text), "hit@1 successes and n must be printed"
        assert "0.23" in text and "0.88" in text, "the Wilson bounds of 3/5 must be printed"

    def test_format_report_calls_refusal_a_phrase_match(self, report):
        """Spec: the report calls it 'refusal (phrase match)'."""
        assert "refusal (phrase match" in format_report(report).lower()

    def test_format_report_states_that_per_type_n_is_small(self, report):
        """Spec: the report says per-type n is 12-16 on dev, so small differences are not meaningful."""
        text = format_report(report)
        assert "12-16" in text

    def test_format_report_shows_failed_count_and_version(self, report):
        """Spec: failed counts are shown, not hidden; the scores version is printed."""
        text = format_report(report)
        assert "failed" in text.lower()
        assert report["version"] in text


# ================================================================ checks.py: main (CLI)

class TestMain:
    def test_main_scores_fixture_run_exit_zero_and_prints_report(self, scored_run_db, tmp_path, capsys):
        """Spec: exit 0, prints 'scored run <id> with <version>: written | 8 outputs x 8 metrics = 64 rows' then the report."""
        harness = write_harness_yaml(tmp_path, scored_run_db.path)
        code = checks.main(["--run-id", scored_run_db.run_id, "--harness-config", harness])
        out = capsys.readouterr().out
        assert code == 0
        assert f"scored run {scored_run_db.run_id}" in out
        assert "written" in out
        assert "8 outputs x 8 metrics = 64 rows" in out
        assert "q003" in out, "the report (with the retrieval-miss list) must be printed"

    def test_main_writes_rows_into_the_tmp_db(self, scored_run_db, tmp_path):
        """Spec: main stores 64 rows under the version computed from the YAML settings."""
        harness = write_harness_yaml(tmp_path, scored_run_db.path)
        assert checks.main(["--run-id", scored_run_db.run_id, "--harness-config", harness]) == 0
        with closing(sqlite3.connect(scored_run_db.path)) as conn:
            rows = conn.execute("SELECT COUNT(*), COUNT(DISTINCT version) FROM scores WHERE run_id = ?",
                                (scored_run_db.run_id,)).fetchone()
            version = conn.execute("SELECT DISTINCT version FROM scores").fetchone()[0]
        assert tuple(rows) == (64, 1)
        assert re.fullmatch(rf"checks-{CHECKS_CODE_VERSION}-[0-9a-f]{{8}}", version)

    def test_main_second_call_prints_unchanged_exit_zero_row_count_stays(self, scored_run_db, tmp_path, capsys):
        """Spec: re-running prints 'unchanged', exits 0, and the row count stays the same."""
        harness = write_harness_yaml(tmp_path, scored_run_db.path)
        args = ["--run-id", scored_run_db.run_id, "--harness-config", harness]
        assert checks.main(args) == 0
        capsys.readouterr()
        assert checks.main(args) == 0
        out = capsys.readouterr().out
        assert "unchanged" in out
        with closing(sqlite3.connect(scored_run_db.path)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 64

    def test_main_unknown_run_id_exits_one_and_writes_nothing(self, scored_run_db, tmp_path):
        """Spec: an unknown run_id exits 1."""
        harness = write_harness_yaml(tmp_path, scored_run_db.path)
        assert checks.main(["--run-id", "does-not-exist", "--harness-config", harness]) == 1
        assert count_scores(scored_run_db.conn) == 0

    @pytest.mark.parametrize("status", ["running", "aborted"])
    def test_main_non_completed_run_exits_one_and_writes_nothing(self, scored_run_db, tmp_path, status):
        """Spec: a run that is not completed exits 1."""
        add_run(scored_run_db.conn, f"run-{status}", status=status)
        harness = write_harness_yaml(tmp_path, scored_run_db.path)
        assert checks.main(["--run-id", f"run-{status}", "--harness-config", harness]) == 1
        assert count_scores(scored_run_db.conn) == 0

    def test_main_heldout_run_without_flag_exits_one_and_with_flag_exits_zero(self, scored_run_db, tmp_path):
        """Spec: held-out run exits 1 without --allow-heldout; with the flag it is scored (exit 0)."""
        add_run(scored_run_db.conn, "run-heldout", split="heldout")
        harness = write_harness_yaml(tmp_path, scored_run_db.path)
        assert checks.main(["--run-id", "run-heldout", "--harness-config", harness]) == 1
        assert count_scores(scored_run_db.conn, "run-heldout") == 0
        assert checks.main(["--run-id", "run-heldout", "--harness-config", harness, "--allow-heldout"]) == 0
        assert count_scores(scored_run_db.conn, "run-heldout") == 8

    def test_main_bad_checks_config_exits_one(self, scored_run_db, tmp_path):
        """Spec: a bad config (here an uppercase phrase) exits 1 and writes nothing."""
        good = write_harness_yaml(tmp_path, scored_run_db.path)
        assert checks.main(["--run-id", scored_run_db.run_id, "--harness-config", good]) == 0  # control
        bad_block = {"hit_ks": [1, 2], "report_confidence": 0.95, "refusal_phrases": ["No Mention Of"]}
        bad = write_harness_yaml(tmp_path, scored_run_db.path, bad_block)
        assert checks.main(["--run-id", scored_run_db.run_id, "--harness-config", bad]) == 1


# ================================================================ store.py

class TestStoreScores:
    def test_scores_table_has_exactly_the_spec_columns(self, results_db):
        """Spec: scores columns are run_id, query_id, metric, version, value, reasoning, scored_at."""
        info = results_db.execute("PRAGMA table_info(scores)").fetchall()
        assert {row[1] for row in info} == {"run_id", "query_id", "metric", "version", "value",
                                            "reasoning", "scored_at"}

    def test_scores_primary_key_is_run_query_metric_version(self, results_db):
        """Spec: PRIMARY KEY (run_id, query_id, metric, version)."""
        info = results_db.execute("PRAGMA table_info(scores)").fetchall()
        assert {row[1] for row in info if row[5] > 0} == {"run_id", "query_id", "metric", "version"}

    def test_scores_reasoning_and_version_are_not_null_columns(self, results_db):
        """Spec: reasoning, version and scored_at are NOT NULL; value is nullable."""
        info = {row[1]: row for row in results_db.execute("PRAGMA table_info(scores)").fetchall()}
        for column in ("reasoning", "version", "scored_at"):
            assert info[column][3] == 1, f"{column} must be NOT NULL"
        assert info["value"][3] == 0, "value must be nullable (NULL = not applicable)"

    def test_write_scores_same_metric_under_two_versions_is_allowed(self, scored_run_db):
        """Spec: the PK allows the same (run, query, metric) under two versions."""
        conn, run_id = scored_run_db.conn, scored_run_db.run_id
        row = {"query_id": "q001", "metric": "refusal", "value": 0.0, "reasoning": "r"}
        store.write_scores(conn, run_id, "v1", [row])
        store.write_scores(conn, run_id, "v2", [{**row, "value": 1.0}])
        assert [r["value"] for r in store.get_scores(conn, run_id, "v1")] == [0.0]
        assert [r["value"] for r in store.get_scores(conn, run_id, "v2")] == [1.0]

    def test_write_scores_same_key_same_version_twice_is_rejected(self, scored_run_db):
        """Spec: never overwrite: a duplicate (run, query, metric, version) is rejected."""
        conn, run_id = scored_run_db.conn, scored_run_db.run_id
        row = {"query_id": "q001", "metric": "refusal", "value": 0.0, "reasoning": "r"}
        store.write_scores(conn, run_id, "v1", [row])
        with pytest.raises(sqlite3.IntegrityError):
            store.write_scores(conn, run_id, "v1", [row])
        assert count_scores(conn) == 1

    def test_write_scores_null_value_with_reasoning_is_stored_as_null(self, scored_run_db):
        """Spec: a NULL value with a reasoning is accepted and stays NULL (not 0)."""
        conn, run_id = scored_run_db.conn, scored_run_db.run_id
        store.write_scores(conn, run_id, "v1", [{"query_id": "q007", "metric": "hit@1", "value": None,
                                                 "reasoning": "unanswerable: no source docs"}])
        (row,) = store.get_scores(conn, run_id, "v1")
        assert row["value"] is None
        assert row["reasoning"] == "unanswerable: no source docs"

    def test_write_scores_null_reasoning_is_rejected_and_nothing_stored(self, scored_run_db):
        """Spec: reasoning is NOT NULL; the whole batch is rolled back (all or nothing)."""
        conn, run_id = scored_run_db.conn, scored_run_db.run_id
        good = {"query_id": "q001", "metric": "refusal", "value": 0.0, "reasoning": "ok"}
        bad = {"query_id": "q002", "metric": "refusal", "value": 0.0, "reasoning": None}
        with pytest.raises(sqlite3.IntegrityError):
            store.write_scores(conn, run_id, "v1", [good, bad])
        assert count_scores(conn) == 0, "the good row must be rolled back with the bad one"

    def test_write_scores_stores_run_id_version_and_scored_at(self, scored_run_db):
        """Spec: rows carry the run_id, the version and a scored_at timestamp."""
        conn, run_id = scored_run_db.conn, scored_run_db.run_id
        store.write_scores(conn, run_id, "v1", [{"query_id": "q001", "metric": "refusal", "value": 1.0,
                                                 "reasoning": "r"}])
        (row,) = store.get_scores(conn, run_id, "v1")
        assert (row["run_id"], row["version"], row["query_id"], row["metric"]) == (run_id, "v1", "q001", "refusal")
        assert row["scored_at"], "scored_at must be set"

    def test_write_scores_hostile_reasoning_is_stored_verbatim(self, scored_run_db):
        """Spec: SQL in a reasoning string is stored as plain text and the table survives."""
        conn, run_id = scored_run_db.conn, scored_run_db.run_id
        hostile = "'); DROP TABLE scores; --"
        store.write_scores(conn, run_id, "v1", [{"query_id": "q001", "metric": "refusal", "value": 0.0,
                                                 "reasoning": hostile}])
        assert store.get_scores(conn, run_id, "v1")[0]["reasoning"] == hostile
        assert count_scores(conn) == 1

    def test_get_scores_returns_rows_ordered_by_query_then_metric(self, scored_run_db):
        """Spec: get_scores orders by query_id, metric."""
        conn, run_id = scored_run_db.conn, scored_run_db.run_id
        rows = [{"query_id": q, "metric": m, "value": 1.0, "reasoning": "r"}
                for q, m in [("q002", "refusal"), ("q001", "refusal"), ("q002", "empty_answer"),
                             ("q001", "empty_answer")]]
        store.write_scores(conn, run_id, "v1", rows)
        got = [(r["query_id"], r["metric"]) for r in store.get_scores(conn, run_id, "v1")]
        assert got == [("q001", "empty_answer"), ("q001", "refusal"),
                       ("q002", "empty_answer"), ("q002", "refusal")]

    def test_get_scores_filters_by_run_and_version(self, scored_run_db):
        """Spec: get_scores returns rows for one run and one version only."""
        conn, run_id = scored_run_db.conn, scored_run_db.run_id
        add_run(conn, "run-other")
        row = {"query_id": "q001", "metric": "refusal", "value": 0.0, "reasoning": "r"}
        store.write_scores(conn, run_id, "v1", [row])
        store.write_scores(conn, run_id, "v2", [row])
        store.write_scores(conn, "run-other", "v1", [row])
        assert len(store.get_scores(conn, run_id, "v1")) == 1
        assert store.get_scores(conn, run_id, "v3") == []

    def test_get_outputs_with_queries_adds_type_split_and_parsed_sources(self, scored_run_db):
        """Spec: get_outputs rows plus the query's type, split and parsed source_doc_ids, in query_id order."""
        rows = store.get_outputs_with_queries(scored_run_db.conn, scored_run_db.run_id)
        assert [r["query_id"] for r in rows] == [f"q00{i}" for i in range(1, 9)]
        by_id = {r["query_id"]: r for r in rows}
        assert by_id["q004"]["type"] == "multi_chunk"
        assert by_id["q004"]["source_doc_ids"] == ["leave_policy", "payroll_policy"]
        assert by_id["q004"]["split"] == "dev"
        assert by_id["q005"]["split"] == "heldout"   # the query's own split, as stored in `queries`
        assert by_id["q007"]["type"] == "unanswerable"
        assert by_id["q007"]["source_doc_ids"] == []
        assert by_id["q001"]["answer"] == "Employees get 8 casual leaves per year."

    def test_get_outputs_with_queries_keeps_failed_output_error(self, scored_run_db):
        """Spec: failed outputs are returned too, with their error text."""
        by_id = {r["query_id"]: r for r in
                 store.get_outputs_with_queries(scored_run_db.conn, scored_run_db.run_id)}
        assert by_id["q005"]["error"] == "ContextLimitError: prompt too long"
        assert by_id["q005"]["answer"] is None


class TestScoresMigration:
    OLD_SCORES = (
        "CREATE TABLE scores (run_id TEXT NOT NULL, query_id TEXT NOT NULL, metric TEXT NOT NULL,"
        " value REAL, reasoning TEXT, PRIMARY KEY (run_id, query_id, metric))"
    )

    def _make_old_db(self, path, with_row):
        """A DB file holding only an old-style scores table (no version column)."""
        with closing(sqlite3.connect(path)) as conn:
            conn.execute(self.OLD_SCORES)
            if with_row:
                conn.execute("INSERT INTO scores (run_id, query_id, metric, value, reasoning)"
                             " VALUES (?, ?, ?, ?, ?)", ("r1", "q001", "hit@4", 1.0, "old row"))
            conn.commit()

    def test_connect_replaces_old_empty_scores_table_with_new_schema(self, tmp_path):
        """Spec: an old-style scores table with 0 rows is dropped and recreated with the version column."""
        path = tmp_path / "old_empty.sqlite"
        self._make_old_db(path, with_row=False)
        with closing(store.connect(path)) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(scores)").fetchall()}
        assert {"version", "scored_at"} <= columns

    def test_connect_with_old_scores_table_holding_a_row_raises_and_keeps_the_row(self, tmp_path):
        """Spec: an old-style table with rows is never dropped: raise and stop, the row survives."""
        path = tmp_path / "old_with_row.sqlite"
        self._make_old_db(path, with_row=True)
        with pytest.raises(Exception):
            store.connect(path).close()
        with closing(sqlite3.connect(path)) as conn:
            rows = conn.execute("SELECT run_id, query_id, metric, value, reasoning FROM scores").fetchall()
            columns = {row[1] for row in conn.execute("PRAGMA table_info(scores)").fetchall()}
        assert [tuple(r) for r in rows] == [("r1", "q001", "hit@4", 1.0, "old row")]
        assert "version" not in columns, "the old table must be left as it was"

    def test_connect_on_new_schema_twice_keeps_stored_scores(self, scored_run_db, checks_config):
        """Spec: create_tables is repeatable; reconnecting never drops scores that have the new schema."""
        score_run(scored_run_db.conn, scored_run_db.run_id, checks_config)
        with closing(store.connect(scored_run_db.path)) as conn:
            assert count_scores(conn) == 64


# ================================================================ layering

class TestLayering:
    @pytest.mark.parametrize("module_file", ["checks.py", "metrics.py"])
    def test_step6_modules_do_not_import_rag_or_llm(self, module_file):
        """Spec layering guard: evalharness/checks.py and metrics.py import neither `rag` nor `llm`."""
        path = PROJECT_ROOT / "evalharness" / module_file
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        top_levels = {name.split(".")[0] for name in imported}
        assert not top_levels & {"rag", "llm"}, f"{module_file} imports {top_levels & {'rag', 'llm'}}"
