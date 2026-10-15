"""Step 09 tests: judge calibration (evalharness/calibration.py and the calibration store functions).

All offline, no LLM call (calibration makes none). Tmp DBs only (never data/results.sqlite).
The labelling session is driven with `scripted_input` and a list-collecting `print_fn`.
Expected numbers are worked out by hand in comments (spec worked example: TP 3, FP 1, FN 2, TN 34).
Fixtures `calibration_config`, `make_calibration_db` and `calibration_db` live in tests/conftest.py.

Convention used everywhere: a pair is (human_ungrounded, judge_ungrounded); "positive" = ungrounded.
"""

import ast
import io
import json
import random
import re
import sqlite3
import sys
from contextlib import closing, suppress
from pathlib import Path

import pytest
import yaml

from evalharness import calibration, store
from evalharness.checks import ScoringRefused
from evalharness.metrics import wilson_interval

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN = "run-calib-001"
FAITH_A = "faith-test-A"
FAITH_B = "faith-test-B"
REL_A = "rel-test-A"
ALL_IDS = [f"q00{i}" for i in range(1, 9)]
TYPES = {"q001": "lookup", "q002": "lookup", "q003": "lookup", "q004": "multi_chunk",
         "q005": "multi_chunk", "q006": "paraphrased", "q007": "unanswerable",
         "q008": "unanswerable"}

# Hand-made calibration data for the compare tests (judge flags are in tests/conftest.py).
#   tune  = q001 q002 q003 q004      check = q005 q006 q007 q008
SPLITS = {"q001": "tune", "q002": "tune", "q003": "tune", "q004": "tune",
          "q005": "check", "q006": "check", "q007": "check", "q008": "check"}
# id: (grounded_label, relevance_label); q008 is left unlabelled on purpose.
LABELS = {"q001": ("ungrounded", 2), "q002": ("ungrounded", 1), "q003": ("grounded", 0),
          "q004": ("unsure", 2), "q005": ("grounded", 0), "q006": ("ungrounded", 0),
          "q007": ("grounded", 2)}


# ---------------------------------------------------------------- helpers

class Printer:
    """A `print_fn` that collects everything the tool prints."""

    def __init__(self):
        self.lines = []

    def __call__(self, *args, **kwargs):
        self.lines.append(" ".join(str(a) for a in args))

    @property
    def text(self):
        return "\n".join(self.lines)


def ids_in_order(text):
    """Query ids (q001...) in the order they first appear in `text`."""
    seen = []
    for found in re.findall(r"q\d{3}", text):
        if found not in seen:
            seen.append(found)
    return seen


def seed_labels(conn, run_id=RUN, splits=None, labels=None):
    """Store the hand-made split and labels through the store functions (like Gaurav's sessions would)."""
    store.save_calibration_splits(conn, run_id, splits or SPLITS)
    for query_id, (grounded, relevance) in (labels or LABELS).items():
        store.write_calibration_label(conn, run_id, query_id, grounded, relevance, None, "gaurav")


def label_rows(conn, run_id=RUN):
    """{query_id: row dict} of calibration_labels."""
    return {row["query_id"]: row for row in store.get_calibration_labels(conn, run_id)}


def init_splits(db, cfg, scripted_input):
    """Run a session that quits at once: it only assigns and stores the split (no label). Returns {id: split}."""
    calibration.label_session(db.conn, db.run_id, cfg, input_fn=scripted_input(["q"]),
                              print_fn=Printer())
    return {q: row["calib_split"] for q, row in label_rows(db.conn, db.run_id).items()}


def queue_order(db, cfg, scripted_input, **kwargs):
    """The order in which a session shows the items: skip every item ('s') and read the printed ids."""
    out = Printer()
    calibration.label_session(db.conn, db.run_id, cfg, input_fn=scripted_input(["s"] * 8),
                              print_fn=out, **kwargs)
    return ids_in_order(out.text)


def bounds(value):
    """(low, high) of an interval given as a tuple, or a dict with ci_low/ci_high or low/high; None if absent."""
    if value is None:
        return None
    if isinstance(value, dict):
        for low_key, high_key in (("ci_low", "ci_high"), ("low", "high"), ("lower", "upper")):
            if low_key in value:
                if value[low_key] is None:
                    return None
                return value[low_key], value[high_key]
        for key in ("ci", "wilson", "interval"):
            if key in value:
                return bounds(value[key])
        raise AssertionError(f"cannot find the interval bounds in {value!r}")
    return tuple(value)


def is_undefined(rate):
    """True when a rate is reported as undefined (the string 'undefined', None, or a dict flagged undefined)."""
    if rate is None or rate == "undefined":
        return True
    return isinstance(rate, dict) and (rate.get("undefined") is True or rate.get("rate") == "undefined")


def check_looks(report):
    """The number of earlier 'check' looks in a compare report (key found by name: has 'check' and 'look')."""
    keys = [k for k in report if "check" in k and "look" in k]
    assert keys, f"compare report has no key about earlier check looks: {sorted(report)}"
    return report[keys[0]]


def pairs_worked_example():
    """TP 3, FP 1, FN 2, TN 34 as (human_ungrounded, judge_ungrounded) pairs."""
    return ([(True, True)] * 3 + [(False, True)] * 1 + [(True, False)] * 2 + [(False, False)] * 34)


def write_yaml(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def cli_args(tmp_path, db_path, cfg):
    """['--harness-config', ..., '--calibration-config', ...] for main(): tmp YAMLs, db_path in tmp_path."""
    harness = {
        "db_path": str(db_path),
        "test_set_config": str(tmp_path / "test_set.yaml"),
        "max_consecutive_failures": 5,
        "error_max_chars": 500,
        "system": {"answer": "somewhere:answer", "load_config": "somewhere:load", "describe": None},
        "checks": {"hit_ks": [1, 2], "report_confidence": 0.95,
                   "refusal_phrases": ["no mention of", "does not mention"]},
    }
    return ["--harness-config", write_yaml(tmp_path, "harness_test.yaml", harness),
            "--calibration-config", write_yaml(tmp_path, "calibration_test.yaml", cfg)]


def compare_call(db, cfg, judge, version, split):
    return calibration.compare(db.conn, db.run_id, judge, version, split, cfg, 0.95)


def last_report_row(db):
    return store.get_calibration_reports(db.conn, db.run_id)[-1]


# ================================================================ config

class TestCalibrationConfig:
    def test_grounded_labels_constant_matches_spec(self):
        """Spec Interfaces: GROUNDED_LABELS = ("grounded", "ungrounded", "unsure")."""
        assert tuple(calibration.GROUNDED_LABELS) == ("grounded", "ungrounded", "unsure")

    def test_real_calibration_yaml_loads_with_spec_values(self):
        """Spec Config changes: configs/calibration.yaml (read-only) holds the approved values."""
        cfg = calibration.load_calibration_config(PROJECT_ROOT / "configs" / "calibration.yaml")
        assert cfg["labeler"] == "gaurav"
        assert cfg["split_seed"] == 2027
        assert cfg["check_fraction"] == 0.5
        assert cfg["forced_tune_ids"] == ["q001", "q029", "q053", "q057", "q058", "q068", "q069",
                                          "q071", "q082", "q085", "q097"]
        assert cfg["order_seed"] == 11
        assert cfg["label_relevance"] is True
        assert cfg["show_check_items"] is False
        assert cfg["n_bootstrap"] == 2000
        assert cfg["bootstrap_seed"] == 7
        assert cfg["subset_seed"] == 2028, "Further-reduced note: subset_seed 2028"
        assert cfg["labeling_guide"].strip()

    def test_valid_config_written_to_tmp_loads_unchanged(self, tmp_path, calibration_config):
        """Control for the refusal tests: the conftest config is accepted as it is."""
        path = write_yaml(tmp_path, "c.yaml", calibration_config)
        assert calibration.load_calibration_config(path) == calibration_config

    @pytest.mark.parametrize("key", [
        "labeler", "split_seed", "check_fraction", "forced_tune_ids", "order_seed",
        "label_relevance", "show_check_items", "n_bootstrap", "bootstrap_seed", "subset_seed",
        "labeling_guide"])
    def test_load_config_missing_key_raises_value_error_naming_key(self, tmp_path, calibration_config, key):
        """Spec: load_calibration_config raises ValueError naming the problem (missing key)."""
        del calibration_config[key]
        path = write_yaml(tmp_path, "c.yaml", calibration_config)
        with pytest.raises(ValueError, match=key):
            calibration.load_calibration_config(path)

    @pytest.mark.parametrize("key, bad", [
        ("check_fraction", 1.5), ("check_fraction", -0.1), ("check_fraction", "half"),
        ("forced_tune_ids", "q001"), ("label_relevance", "yes"), ("show_check_items", "no"),
        ("split_seed", "abc"), ("n_bootstrap", "many"), ("n_bootstrap", -5)])
    def test_load_config_bad_value_raises_value_error_naming_key(self, tmp_path, calibration_config, key, bad):
        """Spec: a malformed config value raises ValueError that names the key."""
        calibration_config[key] = bad
        path = write_yaml(tmp_path, "c.yaml", calibration_config)
        with pytest.raises(ValueError, match=key):
            calibration.load_calibration_config(path)


# ================================================================ metrics by hand

class TestConfusion:
    def test_confusion_worked_example_counts(self):
        """Spec: pairs (human, judge) -> tp/fp/fn/tn/n; FP = judge says ungrounded, human grounded."""
        # 3 x (True, True) = TP 3; 1 x (False, True) = FP 1; 2 x (True, False) = FN 2;
        # 34 x (False, False) = TN 34; n = 3 + 1 + 2 + 34 = 40
        result = calibration.confusion(pairs_worked_example())
        assert (result["tp"], result["fp"], result["fn"], result["tn"], result["n"]) == (3, 1, 2, 34, 40)

    def test_confusion_empty_list_gives_all_zero(self):
        """Edge case: no pairs gives zero in every cell."""
        result = calibration.confusion([])
        assert (result["tp"], result["fp"], result["fn"], result["tn"], result["n"]) == (0, 0, 0, 0, 0)

    def test_confusion_cells_are_not_swapped(self):
        """Spec table: judge says ungrounded + human grounded is FP; judge grounded + human ungrounded is FN."""
        assert calibration.confusion([(False, True)])["fp"] == 1
        assert calibration.confusion([(True, False)])["fn"] == 1


class TestCohenKappa:
    def test_cohen_kappa_worked_example_is_0_625(self):
        """Spec worked example: TP 3, FP 1, FN 2, TN 34 -> kappa 0.625."""
        # p_o = 37/40 = 0.925; judge+ = 4/40 = 0.10; human+ = 5/40 = 0.125
        # p_e = 0.10 * 0.125 + 0.90 * 0.875 = 0.0125 + 0.7875 = 0.80
        # kappa = (0.925 - 0.80) / (1 - 0.80) = 0.125 / 0.20 = 0.625
        assert calibration.cohen_kappa(3, 1, 2, 34) == pytest.approx(0.625)

    def test_cohen_kappa_perfect_agreement_is_one(self):
        """Spec: perfect agreement -> 1.0."""
        # TP 5, TN 5: p_o = 1; judge+ = 0.5, human+ = 0.5; p_e = 0.25 + 0.25 = 0.5
        # kappa = (1 - 0.5) / (1 - 0.5) = 1.0
        assert calibration.cohen_kappa(5, 0, 0, 5) == pytest.approx(1.0)

    def test_cohen_kappa_agreement_at_chance_level_is_zero(self):
        """Spec: agreement at chance level -> 0.0."""
        # TP 1, FP 1, FN 1, TN 1: p_o = 2/4 = 0.5; judge+ = 0.5, human+ = 0.5; p_e = 0.5
        # kappa = (0.5 - 0.5) / 0.5 = 0
        assert calibration.cohen_kappa(1, 1, 1, 1) == pytest.approx(0.0)

    def test_cohen_kappa_always_grounded_judge_scores_zero_despite_high_agreement(self):
        """Spec lesson: a judge that always says grounded gets 35/40 agreement but kappa 0."""
        # TP 0, FP 0, FN 5, TN 35: p_o = 35/40 = 0.875; judge+ = 0; human+ = 5/40 = 0.125
        # p_e = 0 * 0.125 + 1 * 0.875 = 0.875; kappa = (0.875 - 0.875) / 0.125 = 0
        assert calibration.cohen_kappa(0, 0, 5, 35) == pytest.approx(0.0)

    def test_cohen_kappa_complete_disagreement_is_minus_one(self):
        """Boundary: raters always disagree -> kappa -1."""
        # FP 5, FN 5: p_o = 0; judge+ = 5/10 = 0.5; human+ = 0.5; p_e = 0.5
        # kappa = (0 - 0.5) / 0.5 = -1
        assert calibration.cohen_kappa(0, 5, 5, 0) == pytest.approx(-1.0)

    @pytest.mark.parametrize("tp, fp, fn, tn", [(0, 0, 0, 10), (10, 0, 0, 0)])
    def test_cohen_kappa_both_raters_one_label_only_is_none(self, tp, fp, fn, tn):
        """Spec: both raters give one label only (p_e = 1) -> None (undefined), never 0 or 1."""
        # all negative: judge+ = 0, human+ = 0, p_e = 0 + 1 = 1 -> 1 - p_e = 0 -> undefined
        # all positive: judge+ = 1, human+ = 1, p_e = 1 + 0 = 1 -> undefined
        assert calibration.cohen_kappa(tp, fp, fn, tn) is None


class TestAgreementReport:
    def test_agreement_report_worked_example_counts_and_rates(self, calibration_config):
        """Spec worked example: agreement 37/40, precision 3/4, recall 3/5, kappa 0.625."""
        report = calibration.agreement_report(pairs_worked_example(), calibration_config, 0.95)
        assert (report["tp"], report["fp"], report["fn"], report["tn"]) == (3, 1, 2, 34)
        # agreement = (3 + 34) / 40 = 0.925
        assert report["agreement"]["successes"] == 37
        assert report["agreement"]["n"] == 40
        assert report["agreement"]["rate"] == pytest.approx(0.925)
        # precision = 3 / (3 + 1) = 0.75
        assert (report["precision"]["successes"], report["precision"]["n"]) == (3, 4)
        assert report["precision"]["rate"] == pytest.approx(0.75)
        # recall = 3 / (3 + 2) = 0.60
        assert (report["recall"]["successes"], report["recall"]["n"]) == (3, 5)
        assert report["recall"]["rate"] == pytest.approx(0.60)
        assert report["kappa"] == pytest.approx(0.625)

    def test_agreement_report_wilson_intervals_match_hand_values(self, calibration_config):
        """Spec: agreement, precision and recall each get a Wilson interval (95%)."""
        report = calibration.agreement_report(pairs_worked_example(), calibration_config, 0.95)
        # Wilson 37/40: center (0.925 + 1.96^2/80) / (1 + 1.96^2/40) = 0.8878,
        #   half width 0.0864 -> [0.8014, 0.9742]
        low, high = bounds(report["agreement"])
        assert low == pytest.approx(0.8014, abs=2e-3)
        assert high == pytest.approx(0.9742, abs=2e-3)
        # Wilson 3/4 = [0.3007, 0.9544] (hand calculation: center 0.6275, half width 0.3269)
        low, high = bounds(report["precision"])
        assert low == pytest.approx(0.3007, abs=2e-3)
        assert high == pytest.approx(0.9544, abs=2e-3)
        # Wilson 3/5 = [0.2307, 0.8824] (value recorded in tests/README.md, Step 7)
        low, high = bounds(report["recall"])
        assert low == pytest.approx(0.2307, abs=2e-3)
        assert high == pytest.approx(0.8824, abs=2e-3)

    def test_agreement_report_uses_the_step_6_wilson_interval(self, calibration_config):
        """Spec: Wilson interval (Step 6), at the confidence given: 90% is narrower than 99%."""
        pairs = pairs_worked_example()
        low90, high90 = bounds(calibration.agreement_report(pairs, calibration_config, 0.90)["agreement"])
        low99, high99 = bounds(calibration.agreement_report(pairs, calibration_config, 0.99)["agreement"])
        assert (low90, high90) == pytest.approx(wilson_interval(37, 40, 0.90))
        assert (low99, high99) == pytest.approx(wilson_interval(37, 40, 0.99))
        assert high99 - low99 > high90 - low90

    def test_agreement_report_kappa_has_bootstrap_interval_around_point(self, calibration_config):
        """Spec: kappa gets a bootstrap interval; on the worked example low <= 0.625 <= high."""
        report = calibration.agreement_report(pairs_worked_example(), calibration_config, 0.95)
        ci_keys = [k for k in report if "kappa" in k and k != "kappa"]
        assert ci_keys, f"report has no kappa interval key: {sorted(report)}"
        low, high = bounds(report[ci_keys[0]])
        assert low <= report["kappa"] <= high

    def test_agreement_report_judge_flags_nothing_precision_undefined(self, calibration_config):
        """Spec: judge flags nothing -> precision is 0/0, reported as undefined, not 0 or 1."""
        # pairs: human ungrounded x1, human grounded x2, judge never flags
        # precision = 0 / (0 + 0) undefined; recall = 0 / (0 + 1) = 0 (defined)
        pairs = [(True, False), (False, False), (False, False)]
        report = calibration.agreement_report(pairs, calibration_config, 0.95)
        assert is_undefined(report["precision"]), f"precision was {report['precision']!r}"
        assert not is_undefined(report["recall"])
        assert report["recall"]["rate"] == pytest.approx(0.0)

    def test_agreement_report_human_flags_nothing_recall_undefined(self, calibration_config):
        """Spec: Gaurav labels nothing ungrounded -> recall is 0/0, reported as undefined."""
        # pairs: human never ungrounded; judge flags one of three
        # recall = 0 / (0 + 0) undefined; precision = 0 / (0 + 1) = 0 (defined)
        pairs = [(False, True), (False, False), (False, False)]
        report = calibration.agreement_report(pairs, calibration_config, 0.95)
        assert is_undefined(report["recall"]), f"recall was {report['recall']!r}"
        assert not is_undefined(report["precision"])
        assert report["precision"]["rate"] == pytest.approx(0.0)

    def test_agreement_report_one_label_only_kappa_none_and_both_rates_undefined(self, calibration_config):
        """Spec: both raters give one label only -> kappa 0/0 (None); precision and recall 0/0 too."""
        report = calibration.agreement_report([(False, False)] * 5, calibration_config, 0.95)
        assert report["kappa"] is None
        assert is_undefined(report["precision"])
        assert is_undefined(report["recall"])
        # agreement is still defined: 5 / 5
        assert report["agreement"]["rate"] == pytest.approx(1.0)


class TestKappaBootstrapCi:
    def test_same_seed_gives_identical_interval(self):
        """Spec: identical across two calls with the same seed."""
        pairs = pairs_worked_example()
        first = calibration.kappa_bootstrap_ci(pairs, n_boot=500, seed=7, confidence=0.95)
        second = calibration.kappa_bootstrap_ci(pairs, n_boot=500, seed=7, confidence=0.95)
        assert bounds(first) == bounds(second)

    def test_different_seeds_give_different_intervals(self):
        """Spec: different with another seed (five seeds give at least two distinct intervals)."""
        pairs = pairs_worked_example()
        results = {bounds(calibration.kappa_bootstrap_ci(pairs, n_boot=500, seed=s, confidence=0.95))
                   for s in (1, 2, 3, 4, 5)}
        assert len(results) >= 2, "the seed has no effect on the bootstrap interval"

    def test_interval_contains_point_estimate_on_worked_example(self):
        """Spec: lower <= point <= upper on the worked example (point kappa 0.625)."""
        low, high = bounds(calibration.kappa_bootstrap_ci(pairs_worked_example(), n_boot=2000,
                                                          seed=7, confidence=0.95))
        assert low <= 0.625 <= high, f"interval [{low}, {high}] misses 0.625"
        assert low < high

    def test_higher_confidence_gives_a_wider_or_equal_interval(self):
        """Spec: percentile bootstrap at the given confidence (same seed, same resamples)."""
        pairs = pairs_worked_example()
        low80, high80 = bounds(calibration.kappa_bootstrap_ci(pairs, n_boot=1000, seed=7, confidence=0.80))
        low99, high99 = bounds(calibration.kappa_bootstrap_ci(pairs, n_boot=1000, seed=7, confidence=0.99))
        assert (high99 - low99) >= (high80 - low80)

    def test_all_resamples_undefined_returns_no_interval(self):
        """Spec: None when fewer than half the resamples have a defined kappa."""
        # every pair is (False, False): every resample has p_e = 1, so kappa is undefined each time
        result = calibration.kappa_bootstrap_ci([(False, False)] * 10, n_boot=200, seed=7,
                                                confidence=0.95)
        assert bounds(result) is None


# ================================================================ splits

class TestAssignCalibrationSplits:
    @staticmethod
    def _types():
        """20 ids: lookup q001-q010, multi_chunk q011-q015, paraphrased q016-q018, unanswerable q019-q020."""
        types = {}
        for number in range(1, 21):
            qid = f"q{number:03d}"
            types[qid] = ("lookup" if number <= 10 else "multi_chunk" if number <= 15
                          else "paraphrased" if number <= 18 else "unanswerable")
        return types

    def _assign(self, seed=2027, fraction=0.4, forced=("q001", "q002", "q003", "q004")):
        return calibration.assign_calibration_splits(self._types(), seed=seed, check_fraction=fraction,
                                                     forced_tune_ids=list(forced))

    def test_every_id_gets_exactly_one_valid_split(self):
        """Spec: every output gets exactly one split, "tune" or "check"."""
        splits = self._assign()
        assert set(splits) == set(self._types())
        assert set(splits.values()) <= {"tune", "check"}

    def test_per_type_check_counts_equal_rounded_fraction_of_eligible(self):
        """Spec: in each type, round(check_fraction * eligible) ids go to check."""
        splits = self._assign()
        # lookup: 10 ids - 4 forced = 6 eligible -> round(0.4 * 6) = round(2.4) = 2
        # multi_chunk: 5 eligible -> round(2.0) = 2
        # paraphrased: 3 eligible -> round(1.2) = 1
        # unanswerable: 2 eligible -> round(0.8) = 1
        expected = {"lookup": 2, "multi_chunk": 2, "paraphrased": 1, "unanswerable": 1}
        for query_type, count in expected.items():
            got = sum(1 for q, t in self._types().items() if t == query_type and splits[q] == "check")
            assert got == count, f"{query_type}: expected {count} check ids, got {got}"

    def test_forced_ids_are_always_tune(self):
        """Spec: forced ids are always tune (also across many seeds and a big fraction)."""
        forced = ["q001", "q002", "q003", "q004", "q011"]
        for seed in range(30):
            splits = self._assign(seed=seed, fraction=0.9, forced=forced)
            assert all(splits[q] == "tune" for q in forced), f"seed {seed} put a forced id in check"

    def test_same_seed_and_input_give_the_same_split(self):
        """Spec: deterministic for the same input."""
        assert self._assign(seed=2027) == self._assign(seed=2027)

    def test_different_seeds_give_different_splits(self):
        """Spec: the seed drives the draw (three seeds give at least two distinct splits)."""
        results = {tuple(sorted(self._assign(seed=s).items())) for s in (1, 2, 3)}
        assert len(results) >= 2

    def test_no_forced_ids_every_id_is_eligible(self):
        """Edge case: empty forced list -> counts use all ids of each type."""
        splits = self._assign(forced=())
        # lookup 10 * 0.4 = 4; multi 5 * 0.4 = 2; paraphrased 3 * 0.4 -> round(1.2) = 1; unanswerable -> 1
        assert sum(1 for q in self._types() if splits[q] == "check") == 4 + 2 + 1 + 1


class TestSelectLabelSubset:
    @staticmethod
    def _example():
        """12 ids: lookup q001-q006, multi_chunk q007-q009, paraphrased q010-q011, unanswerable q012."""
        types = {}
        for number in range(1, 13):
            qid = f"q{number:03d}"
            types[qid] = ("lookup" if number <= 6 else "multi_chunk" if number <= 9
                          else "paraphrased" if number <= 11 else "unanswerable")
        return types

    def test_largest_remainder_allocation_with_tie_broken_by_type_name(self):
        """Spec: largest-remainder allocation per type; tied leftovers go to the type name that sorts first."""
        # n = 6 of 12: exact quotas lookup 6*6/12 = 3.0, multi 6*3/12 = 1.5, para 6*2/12 = 1.0,
        # unanswerable 6*1/12 = 0.5; floors 3 + 1 + 1 + 0 = 5, one leftover slot;
        # remainders: multi 0.5, unanswerable 0.5 (tie) -> "multi_chunk" < "unanswerable" -> multi gets it
        # quotas: lookup 3, multi_chunk 2, paraphrased 1, unanswerable 0
        types = self._example()
        chosen = calibration.select_label_subset(types, 6, seed=2028)
        counts = {t: sum(1 for q in chosen if types[q] == t) for t in set(types.values())}
        assert counts == {"lookup": 3, "multi_chunk": 2, "paraphrased": 1, "unanswerable": 0}
        assert len(chosen) == len(set(chosen)) == 6

    def test_tie_between_two_types_goes_to_alphabetically_first_type(self):
        """Spec: ties broken by type name. 1 lookup + 1 paraphrased, n = 1 -> the lookup id."""
        # exact quotas 0.5 and 0.5, floors 0 + 0, one leftover; remainders tie -> "lookup" < "paraphrased"
        chosen = calibration.select_label_subset({"q001": "lookup", "q002": "paraphrased"}, 1, seed=1)
        assert sorted(chosen) == ["q001"]

    def test_leftover_goes_to_largest_remainder_not_to_name_order(self):
        """Spec: leftovers go to the largest remainders. 2 lookup + 1 unanswerable, n = 2."""
        # exact: lookup 2*2/3 = 1.333, unanswerable 2*1/3 = 0.667; floors 1 + 0 = 1, one leftover;
        # remainders lookup 0.333, unanswerable 0.667 -> unanswerable gets it (name order would pick lookup)
        types = {"q001": "lookup", "q002": "lookup", "q003": "unanswerable"}
        chosen = calibration.select_label_subset(types, 2, seed=1)
        assert "q003" in chosen
        assert sum(1 for q in chosen if types[q] == "lookup") == 1

    def test_draw_uses_one_seeded_random_sample_per_type_in_sorted_type_order(self):
        """Spec: one random.Random(seed), rng.sample(sorted ids, quota) per type, types in sorted order."""
        types = self._example()
        quotas = {"lookup": 3, "multi_chunk": 2, "paraphrased": 1, "unanswerable": 0}   # see test above
        rng = random.Random(2028)
        expected = []
        for query_type in sorted(quotas):
            ids = sorted(q for q, t in types.items() if t == query_type)
            expected += rng.sample(ids, quotas[query_type])
        assert sorted(calibration.select_label_subset(types, 6, seed=2028)) == sorted(expected)

    def test_same_input_and_seed_give_the_same_subset(self):
        """Spec: fixed seeded subset, so a resumed session always shows the same answers."""
        types = self._example()
        assert (calibration.select_label_subset(types, 5, seed=2028)
                == calibration.select_label_subset(types, 5, seed=2028))

    def test_subset_of_full_size_returns_every_id(self):
        """Boundary: n equal to the number of ids returns all of them."""
        types = self._example()
        assert sorted(calibration.select_label_subset(types, 12, seed=1)) == sorted(types)

    def test_different_seeds_give_different_subsets(self):
        """Spec: the seed drives the draw (five seeds give at least two distinct subsets)."""
        types = {f"q{n:03d}": "lookup" for n in range(1, 21)}
        results = {tuple(sorted(calibration.select_label_subset(types, 5, seed=s))) for s in (1, 2, 3, 4, 5)}
        assert len(results) >= 2


# ================================================================ store

class TestCalibrationStore:
    def test_calibration_tables_have_the_spec_columns(self, calibration_db):
        """Spec Data and schema changes: exact columns of calibration_labels and calibration_reports."""
        def columns(table):
            return {row[1] for row in calibration_db.conn.execute(f"PRAGMA table_info({table})")}
        assert columns("calibration_labels") == {"run_id", "query_id", "calib_split", "grounded_label",
                                                 "relevance_label", "notes", "labeler", "labeled_at"}
        assert columns("calibration_reports") == {"id", "run_id", "judge", "judge_version",
                                                  "calib_split", "n", "n_excluded", "tp", "fp", "fn",
                                                  "tn", "report_json", "created_at"}

    def test_save_calibration_splits_inserts_unlabelled_rows(self, calibration_db):
        """Spec: save_calibration_splits inserts the split rows; labels start NULL (not labelled yet)."""
        store.save_calibration_splits(calibration_db.conn, RUN, SPLITS)
        rows = label_rows(calibration_db.conn)
        assert {q: r["calib_split"] for q, r in rows.items()} == SPLITS
        assert all(r["grounded_label"] is None and r["relevance_label"] is None
                   and r["labeler"] is None and r["labeled_at"] is None for r in rows.values())

    def test_save_calibration_splits_same_values_twice_is_accepted(self, calibration_db):
        """Spec: inserts once; storing the identical split again does not fail or duplicate rows."""
        store.save_calibration_splits(calibration_db.conn, RUN, SPLITS)
        store.save_calibration_splits(calibration_db.conn, RUN, dict(SPLITS))
        assert len(store.get_calibration_labels(calibration_db.conn, RUN)) == 8

    def test_save_calibration_splits_refuses_to_change_a_stored_split(self, calibration_db):
        """Spec: raises if the rows already exist with different values; the stored split stays."""
        store.save_calibration_splits(calibration_db.conn, RUN, SPLITS)
        changed = dict(SPLITS, q001="check")
        with pytest.raises((ValueError, RuntimeError, sqlite3.IntegrityError)):
            store.save_calibration_splits(calibration_db.conn, RUN, changed)
        assert label_rows(calibration_db.conn)["q001"]["calib_split"] == "tune"

    def test_save_calibration_splits_rejects_unknown_split_value(self, calibration_db):
        """Spec: CHECK (calib_split IN ('tune', 'check')) rejects other values; nothing is stored."""
        with pytest.raises((sqlite3.IntegrityError, ValueError)):
            store.save_calibration_splits(calibration_db.conn, RUN, {"q001": "dev"})
        assert store.get_calibration_labels(calibration_db.conn, RUN) == []

    def test_save_calibration_splits_unknown_run_is_rejected_by_foreign_key(self, calibration_db):
        """Spec: run_id REFERENCES runs(run_id); an unknown run id is refused."""
        with pytest.raises((sqlite3.IntegrityError, ValueError)):
            store.save_calibration_splits(calibration_db.conn, "no-such-run", {"q001": "tune"})

    def test_write_calibration_label_stores_label_labeler_and_timestamp(self, calibration_db):
        """Spec: write_calibration_label stores the label, relevance, notes, labeler and labeled_at."""
        store.save_calibration_splits(calibration_db.conn, RUN, SPLITS)
        store.write_calibration_label(calibration_db.conn, RUN, "q002", "ungrounded", 1, "my note", "gaurav")
        row = label_rows(calibration_db.conn)["q002"]
        assert (row["grounded_label"], row["relevance_label"], row["notes"], row["labeler"]) == (
            "ungrounded", 1, "my note", "gaurav")
        assert isinstance(row["labeled_at"], str) and row["labeled_at"], "labeled_at must be set with the label"

    def test_write_calibration_label_is_committed_at_once(self, calibration_db):
        """Spec: one row, committed at once: a second connection sees it without any extra commit."""
        store.save_calibration_splits(calibration_db.conn, RUN, SPLITS)
        store.write_calibration_label(calibration_db.conn, RUN, "q003", "grounded", 2, None, "gaurav")
        with closing(sqlite3.connect(calibration_db.path)) as other:
            row = other.execute(
                "SELECT grounded_label, relevance_label FROM calibration_labels"
                " WHERE run_id = ? AND query_id = ?", (RUN, "q003")).fetchone()
        assert row == ("grounded", 2)

    def test_write_calibration_label_rejects_bad_grounded_label(self, calibration_db):
        """Spec: CHECK (grounded_label IN grounded/ungrounded/unsure) rejects a bad label."""
        store.save_calibration_splits(calibration_db.conn, RUN, SPLITS)
        with pytest.raises((sqlite3.IntegrityError, ValueError)):
            store.write_calibration_label(calibration_db.conn, RUN, "q001", "maybe", 1, None, "gaurav")
        assert label_rows(calibration_db.conn)["q001"]["grounded_label"] is None

    @pytest.mark.parametrize("bad", [3, -1])
    def test_write_calibration_label_rejects_bad_relevance_label(self, calibration_db, bad):
        """Spec: CHECK (relevance_label IN (0, 1, 2)) rejects other numbers."""
        store.save_calibration_splits(calibration_db.conn, RUN, SPLITS)
        with pytest.raises((sqlite3.IntegrityError, ValueError)):
            store.write_calibration_label(calibration_db.conn, RUN, "q001", "grounded", bad, None, "gaurav")
        assert label_rows(calibration_db.conn)["q001"]["relevance_label"] is None

    def test_write_calibration_label_without_relevance_stores_null(self, calibration_db):
        """Spec: relevance_label is NULL unless Decision 5 (label_relevance) is on."""
        store.save_calibration_splits(calibration_db.conn, RUN, SPLITS)
        store.write_calibration_label(calibration_db.conn, RUN, "q001", "unsure", None, None, "gaurav")
        row = label_rows(calibration_db.conn)["q001"]
        assert row["grounded_label"] == "unsure" and row["relevance_label"] is None

    def test_hostile_note_is_stored_verbatim(self, calibration_db):
        """Spec: a hostile note (SQL, quotes, non-ASCII) is stored verbatim; the table survives."""
        hostile = "Robert'); DROP TABLE calibration_labels; -- \"quoted\" é中文 \n newline"
        store.save_calibration_splits(calibration_db.conn, RUN, SPLITS)
        store.write_calibration_label(calibration_db.conn, RUN, "q001", "grounded", 2, hostile, "gaurav")
        assert label_rows(calibration_db.conn)["q001"]["notes"] == hostile
        assert len(store.get_calibration_labels(calibration_db.conn, RUN)) == 8

    def test_get_calibration_labels_other_run_is_empty(self, calibration_db):
        """Spec: labels are per run_id."""
        store.save_calibration_splits(calibration_db.conn, RUN, SPLITS)
        assert store.get_calibration_labels(calibration_db.conn, "other-run") == []

    def test_get_calibration_reports_starts_empty(self, calibration_db):
        """Spec: calibration_reports is empty until a compare is logged."""
        assert store.get_calibration_reports(calibration_db.conn, RUN) == []


# ================================================================ label_session

class TestLabelSession:
    def test_first_session_assigns_and_stores_split_for_every_output(self, calibration_db, calibration_config,
                                                                     scripted_input):
        """Spec: on the first session it assigns and stores the tune/check split for every output."""
        result = calibration.label_session(
            calibration_db.conn, RUN, calibration_config, input_fn=scripted_input(["q"]), print_fn=Printer())
        rows = label_rows(calibration_db.conn)
        assert sorted(rows) == ALL_IDS, "one split row per output"
        assert all(r["calib_split"] in ("tune", "check") for r in rows.values())
        assert all(r["grounded_label"] is None for r in rows.values()), "quitting at once labels nothing"
        assert (result["labelled"], result["remaining"]) == (0, 8)

    def test_stored_split_matches_assign_calibration_splits_and_forces_tune(self, calibration_db,
                                                                            calibration_config, scripted_input):
        """Spec: split = assign_calibration_splits(config values); forced ids are tune."""
        stored = init_splits(calibration_db, calibration_config, scripted_input)
        expected = calibration.assign_calibration_splits(
            TYPES, seed=calibration_config["split_seed"], check_fraction=calibration_config["check_fraction"],
            forced_tune_ids=calibration_config["forced_tune_ids"])
        assert stored == expected
        assert stored["q001"] == "tune"
        # fraction 0.4: lookup eligible q002 q003 -> round(0.8) = 1; multi_chunk 2 -> 1;
        # paraphrased q006 -> round(0.4) = 0; unanswerable 2 -> 1; so 3 check, 5 tune
        assert sorted(stored.values()).count("check") == 3

    def test_split_is_not_changed_by_a_later_session(self, calibration_db, calibration_config, scripted_input):
        """Spec: the split is assigned once and never changed (a session with another seed cannot move ids)."""
        before = init_splits(calibration_db, calibration_config, scripted_input)
        other = dict(calibration_config, split_seed=999, forced_tune_ids=[])
        with suppress(ValueError, ScoringRefused):   # refusing the changed split is allowed, moving ids is not
            calibration.label_session(calibration_db.conn, RUN, other, input_fn=scripted_input(["q"]),
                                      print_fn=Printer())
        after = {q: r["calib_split"] for q, r in label_rows(calibration_db.conn).items()}
        assert after == before

    def test_keys_g_u_question_store_labels_relevance_labeler_and_timestamp(self, calibration_db,
                                                                            calibration_config, scripted_input):
        """Spec: g, u, ? and the 0/1/2 relevance answers are stored with labeler and labeled_at."""
        order = queue_order(calibration_db, calibration_config, scripted_input)
        inp = scripted_input(["g", "2", "u", "0", "?", "1", "q"])
        result = calibration.label_session(calibration_db.conn, RUN, calibration_config, input_fn=inp,
                                           print_fn=Printer())
        rows = label_rows(calibration_db.conn)
        assert (rows[order[0]]["grounded_label"], rows[order[0]]["relevance_label"]) == ("grounded", 2)
        assert (rows[order[1]]["grounded_label"], rows[order[1]]["relevance_label"]) == ("ungrounded", 0)
        assert (rows[order[2]]["grounded_label"], rows[order[2]]["relevance_label"]) == ("unsure", 1)
        for query_id in order[:3]:
            assert rows[query_id]["labeler"] == "tester", "labeler comes from the config"
            assert isinstance(rows[query_id]["labeled_at"], str) and rows[query_id]["labeled_at"]
        assert all(rows[q]["grounded_label"] is None for q in order[3:])
        assert (result["labelled"], result["remaining"]) == (3, 5)

    def test_label_relevance_off_asks_no_relevance_question(self, calibration_db, calibration_config,
                                                            scripted_input):
        """Spec: relevance is asked only when label_relevance is on; otherwise relevance_label stays NULL."""
        calibration_config["label_relevance"] = False
        order = queue_order(calibration_db, calibration_config, scripted_input)
        inp = scripted_input(["g", "u", "?", "q"])   # the fake raises if a relevance prompt needs more keys
        calibration.label_session(calibration_db.conn, RUN, calibration_config, input_fn=inp, print_fn=Printer())
        rows = label_rows(calibration_db.conn)
        assert [rows[q]["grounded_label"] for q in order[:3]] == ["grounded", "ungrounded", "unsure"]
        assert all(rows[q]["relevance_label"] is None for q in order[:3])

    def test_quit_keeps_labels_and_second_session_resumes_with_the_rest(self, calibration_db,
                                                                         calibration_config, scripted_input):
        """Spec: q saves labels so far (each committed at once); a second session shows only the rest."""
        order = queue_order(calibration_db, calibration_config, scripted_input)
        calibration.label_session(calibration_db.conn, RUN, calibration_config,
                                  input_fn=scripted_input(["g", "2", "u", "1", "q"]), print_fn=Printer())
        out = Printer()
        inp = scripted_input(["s"] * 8)
        calibration.label_session(calibration_db.conn, RUN, calibration_config, input_fn=inp, print_fn=out)
        shown = ids_in_order(out.text)
        assert shown == order[2:], "second session must resume with the 6 unlabelled items, same order"
        rows = label_rows(calibration_db.conn)
        assert rows[order[0]]["grounded_label"] == "grounded", "earlier labels survive"
        assert rows[order[1]]["grounded_label"] == "ungrounded"

    def test_skip_key_leaves_item_unlabelled_and_counts_it(self, calibration_db, calibration_config,
                                                            scripted_input):
        """Spec: s leaves an item unlabelled (skip for now)."""
        order = queue_order(calibration_db, calibration_config, scripted_input)
        result = calibration.label_session(calibration_db.conn, RUN, calibration_config,
                                           input_fn=scripted_input(["s", "g", "2", "q"]), print_fn=Printer())
        rows = label_rows(calibration_db.conn)
        assert rows[order[0]]["grounded_label"] is None, "skipped item stays unlabelled"
        assert rows[order[1]]["grounded_label"] == "grounded"
        assert result["skipped"] == 1 and result["labelled"] == 1

    def test_note_key_stores_note_on_that_item_only(self, calibration_db, calibration_config, scripted_input):
        """Spec: n adds a note first; the note belongs to the item being labelled, not the next one."""
        order = queue_order(calibration_db, calibration_config, scripted_input)
        inp = scripted_input(["n", "check the leave table", "g", "2", "u", "1", "q"])
        calibration.label_session(calibration_db.conn, RUN, calibration_config, input_fn=inp, print_fn=Printer())
        rows = label_rows(calibration_db.conn)
        assert rows[order[0]]["notes"] == "check the leave table"
        assert rows[order[0]]["grounded_label"] == "grounded"
        assert rows[order[1]]["notes"] is None

    def test_invalid_key_asks_again_without_storing_anything(self, calibration_db, calibration_config,
                                                              scripted_input):
        """Spec: an invalid key re-asks (the item is still labelled by the next valid key)."""
        order = queue_order(calibration_db, calibration_config, scripted_input)
        inp = scripted_input(["x", "zz", "", "u", "1", "q"])
        result = calibration.label_session(calibration_db.conn, RUN, calibration_config, input_fn=inp,
                                           print_fn=Printer())
        rows = label_rows(calibration_db.conn)
        assert rows[order[0]]["grounded_label"] == "ungrounded"
        assert result["labelled"] == 1
        assert inp.answers == [], "every scripted key was consumed"

    @pytest.mark.parametrize("bad_answers", [["5"], ["abc", "3"], [""]])
    def test_invalid_relevance_answer_asks_again(self, calibration_db, calibration_config, scripted_input,
                                                 bad_answers):
        """Spec: relevance must be 0, 1 or 2; anything else re-asks."""
        order = queue_order(calibration_db, calibration_config, scripted_input)
        inp = scripted_input(["g", *bad_answers, "1", "q"])
        calibration.label_session(calibration_db.conn, RUN, calibration_config, input_fn=inp, print_fn=Printer())
        row = label_rows(calibration_db.conn)[order[0]]
        assert (row["grounded_label"], row["relevance_label"]) == ("grounded", 1)

    def test_failed_outputs_are_skipped_and_counted(self, make_calibration_db, calibration_config,
                                                    scripted_input):
        """Spec: failed outputs (outputs.error) are skipped and counted, never shown or labelled."""
        db = make_calibration_db(failed_ids=("q005",))
        out = Printer()
        inp = scripted_input(["g", "2"] * 7)
        result = calibration.label_session(db.conn, RUN, calibration_config, input_fn=inp, print_fn=out)
        assert "q005" not in ids_in_order(out.text), "a failed output has nothing to label"
        assert label_rows(db.conn)["q005"]["grounded_label"] is None
        assert result["labelled"] == 7
        counted = result["skipped"] == 1 or any("fail" in k and v == 1 for k, v in result.items())
        assert counted, f"the failed output must be counted, got {result}"
        assert inp.answers == []

    def test_order_is_a_seeded_shuffle_of_all_items(self, make_calibration_db, calibration_config,
                                                    scripted_input):
        """Spec: seeded shuffled order: all items once, same seed same order, another seed another order."""
        first = queue_order(make_calibration_db(), calibration_config, scripted_input)
        again = queue_order(make_calibration_db(), calibration_config, scripted_input)
        assert sorted(first) == ALL_IDS, "every item shown exactly once"
        assert first == again, "same seed, same order"
        orders = set()
        for seed in (1, 2, 3):
            orders.add(tuple(queue_order(make_calibration_db(), dict(calibration_config, order_seed=seed),
                                         scripted_input)))
        assert len(orders) >= 2, "the order_seed must change the order"
        assert any(list(o) != ALL_IDS for o in orders), "items must not come in plain id order"

    def test_progress_guide_question_answer_and_chunks_are_shown(self, calibration_db, calibration_config,
                                                                 scripted_input):
        """Spec: shows progress (1/8), query id, type, question, answer and the stored chunks (rank, doc, text)."""
        order = queue_order(calibration_db, calibration_config, scripted_input)
        out = Printer()
        calibration.label_session(calibration_db.conn, RUN, calibration_config,
                                  input_fn=scripted_input(["q"]), print_fn=out)
        first = order[0]
        text = out.text
        assert calibration_config["labeling_guide"] in text, "the labeling guide is printed at the start"
        assert re.search(r"\b1/8\b", text), "progress 1/8"
        assert first in text and TYPES[first] in text
        assert calibration_db.answers[first] in text
        assert "CHUNK-TEXT-%s-A" % first in text and "CHUNK-TEXT-%s-B" % first in text
        assert "leave_policy" in text and "payroll_policy" in text, "doc ids of the chunks"
        # stored order: rank 1 (score 0.4) before rank 2 (score 0.9); never re-sorted by score
        assert text.index("CHUNK-TEXT-%s-A" % first) < text.index("CHUNK-TEXT-%s-B" % first)
        assert text.index(calibration_config["labeling_guide"]) < text.index(first)

    def test_printed_text_never_contains_judge_output_or_reference_answer(self, calibration_db,
                                                                         calibration_config, scripted_input):
        """Spec (blindness): no judge reasoning, claim text, score values or reference answer is ever shown."""
        # Control: the planted markers really are in the DB (judge rows and reference answers).
        reasoning = [r["reasoning"] for r in store.get_scores(calibration_db.conn, RUN, FAITH_A)]
        assert any("REASONING-MARKER-q001" in r for r in reasoning)
        assert any("CLAIM-MARKER-q001" in c["claim"] for c in store.get_judge_claims(calibration_db.conn, RUN, FAITH_A))
        reference = calibration_db.conn.execute(
            "SELECT reference_answer FROM queries WHERE id = ?", ("q001",)).fetchone()[0]
        assert reference == "REFERENCE-MARKER-q001"
        # Label all 8 items so every item is printed.
        inp = scripted_input(["g", "2"] * 8)
        out = Printer()
        calibration.label_session(calibration_db.conn, RUN, calibration_config, input_fn=inp, print_fn=out)
        shown = out.text + "\n" + "\n".join(inp.prompts)
        assert inp.answers == [] and len(ids_in_order(out.text)) == 8
        assert calibration_db.markers, "fixture must plant markers"
        for marker in calibration_db.markers:
            assert marker not in shown, f"blindness broken: {marker!r} was shown to the labeller"

    def test_no_label_is_written_without_a_label_key(self, calibration_db, calibration_config, scripted_input):
        """Spec rule: only Gaurav's keystrokes create labels; skipping and quitting write none."""
        calibration.label_session(calibration_db.conn, RUN, calibration_config,
                                  input_fn=scripted_input(["s", "s", "n", "a note", "s", "q"]), print_fn=Printer())
        rows = store.get_calibration_labels(calibration_db.conn, RUN)
        assert len(rows) == 8
        assert all(r["grounded_label"] is None and r["labeler"] is None and r["labeled_at"] is None
                   for r in rows), "no label may appear without a g/u/? key"

    def test_held_out_run_is_refused_and_nothing_is_written(self, make_calibration_db, calibration_config,
                                                            scripted_input):
        """Spec: a held-out run is refused (exit 1 on the CLI); no calibration rows appear."""
        db = make_calibration_db(split="heldout")
        with pytest.raises(ScoringRefused):
            calibration.label_session(db.conn, RUN, calibration_config, input_fn=scripted_input(["q"]),
                                      print_fn=Printer())
        assert store.get_calibration_labels(db.conn, RUN) == []

    def test_unknown_and_not_completed_runs_are_refused(self, make_calibration_db, calibration_config,
                                                        scripted_input):
        """Spec: unknown or not-completed runs are refused."""
        db = make_calibration_db(status="running")
        with pytest.raises(ScoringRefused):
            calibration.label_session(db.conn, RUN, calibration_config, input_fn=scripted_input(["q"]),
                                      print_fn=Printer())
        with pytest.raises(ScoringRefused):
            calibration.label_session(db.conn, "no-such-run", calibration_config,
                                      input_fn=scripted_input(["q"]), print_fn=Printer())
        assert store.get_calibration_labels(db.conn, RUN) == []


class TestLabelSessionRelabel:
    @staticmethod
    def _labelled(db, cfg, scripted_input):
        """Splits from a session, then three labels written through the store (stand-ins for earlier sessions)."""
        init_splits(db, cfg, scripted_input)
        store.write_calibration_label(db.conn, RUN, "q001", "grounded", 2, None, "tester")
        store.write_calibration_label(db.conn, RUN, "q002", "ungrounded", 1, None, "tester")
        store.write_calibration_label(db.conn, RUN, "q003", "unsure", 0, None, "tester")

    def test_relabel_confirmed_overwrites_only_the_named_id(self, calibration_db, calibration_config,
                                                             scripted_input):
        """Spec: relabel re-asks only the named ids and overwrites after a y confirmation."""
        self._labelled(calibration_db, calibration_config, scripted_input)
        out = Printer()
        inp = scripted_input(["y", "g", "0"])
        calibration.label_session(calibration_db.conn, RUN, calibration_config, input_fn=inp, print_fn=out,
                                  relabel=["q002"])
        rows = label_rows(calibration_db.conn)
        assert (rows["q002"]["grounded_label"], rows["q002"]["relevance_label"]) == ("grounded", 0)
        assert (rows["q001"]["grounded_label"], rows["q001"]["relevance_label"]) == ("grounded", 2)
        assert (rows["q003"]["grounded_label"], rows["q003"]["relevance_label"]) == ("unsure", 0)
        assert ids_in_order(out.text) == ["q002"], "only the named id is shown"
        assert inp.answers == []

    def test_relabel_declined_keeps_the_old_label(self, calibration_db, calibration_config, scripted_input):
        """Spec: overwrite needs a y/n confirmation; n keeps the stored label."""
        self._labelled(calibration_db, calibration_config, scripted_input)
        calibration.label_session(calibration_db.conn, RUN, calibration_config,
                                  input_fn=scripted_input(["n"]), print_fn=Printer(), relabel=["q002"])
        row = label_rows(calibration_db.conn)["q002"]
        assert (row["grounded_label"], row["relevance_label"]) == ("ungrounded", 1)


class TestLabelSessionSubset:
    def test_only_split_check_queues_only_check_items(self, calibration_db, calibration_config, scripted_input):
        """Spec (change 2026-10-15): only_split="check" queues only that half; tune stays unlabelled."""
        splits = init_splits(calibration_db, calibration_config, scripted_input)
        check_ids = sorted(q for q, s in splits.items() if s == "check")
        out = Printer()
        inp = scripted_input(["g", "2"] * len(check_ids))
        result = calibration.label_session(calibration_db.conn, RUN, calibration_config, input_fn=inp,
                                           print_fn=out, only_split="check")
        assert sorted(ids_in_order(out.text)) == check_ids
        rows = label_rows(calibration_db.conn)
        assert all(rows[q]["grounded_label"] == "grounded" for q in check_ids)
        assert all(r["grounded_label"] is None for q, r in rows.items() if q not in check_ids)
        assert result["labelled"] == len(check_ids)
        assert re.search(rf"\b1/{len(check_ids)}\b", out.text), "progress counts the queued half only"

    def test_only_split_tune_queues_only_tune_items(self, calibration_db, calibration_config, scripted_input):
        """Spec: only_split accepts "tune" too."""
        splits = init_splits(calibration_db, calibration_config, scripted_input)
        tune_ids = sorted(q for q, s in splits.items() if s == "tune")
        out = Printer()
        calibration.label_session(calibration_db.conn, RUN, calibration_config,
                                  input_fn=scripted_input(["s"] * 8), print_fn=out, only_split="tune")
        assert sorted(ids_in_order(out.text)) == tune_ids

    def test_limit_labels_the_seeded_stratified_subset_of_the_half(self, calibration_db, calibration_config,
                                                                    scripted_input):
        """Spec: limit queues select_label_subset(that half's types, limit, seed=subset_seed)."""
        splits = init_splits(calibration_db, calibration_config, scripted_input)
        check_types = {q: TYPES[q] for q, s in splits.items() if s == "check"}
        expected = sorted(calibration.select_label_subset(check_types, 2, seed=calibration_config["subset_seed"]))
        out = Printer()
        calibration.label_session(calibration_db.conn, RUN, calibration_config,
                                  input_fn=scripted_input(["g", "2"] * 2), print_fn=out,
                                  only_split="check", limit=2)
        assert sorted(ids_in_order(out.text)) == expected
        rows = label_rows(calibration_db.conn)
        assert sorted(q for q, r in rows.items() if r["grounded_label"] is not None) == expected
        # check half is 1 lookup + 1 multi_chunk + 1 unanswerable; n = 2: exact quotas 0.667 each,
        # floors 0, two leftovers tie -> type-name order: lookup and multi_chunk get one each
        assert sorted(TYPES[q] for q in expected) == ["lookup", "multi_chunk"]

    def test_limit_subset_is_the_same_after_a_quit_and_resume(self, calibration_db, calibration_config,
                                                              scripted_input):
        """Spec: a fixed subset, so a resumed session shows only the rest of the same answers."""
        splits = init_splits(calibration_db, calibration_config, scripted_input)
        check_types = {q: TYPES[q] for q, s in splits.items() if s == "check"}
        subset = calibration.select_label_subset(check_types, 2, seed=calibration_config["subset_seed"])
        first_out = Printer()
        calibration.label_session(calibration_db.conn, RUN, calibration_config,
                                  input_fn=scripted_input(["g", "2", "q"]), print_fn=first_out,
                                  only_split="check", limit=2)
        done = ids_in_order(first_out.text)[0]
        second_out = Printer()
        result = calibration.label_session(calibration_db.conn, RUN, calibration_config,
                                           input_fn=scripted_input(["g", "2"]), print_fn=second_out,
                                           only_split="check", limit=2)
        rest = [q for q in subset if q != done]
        assert ids_in_order(second_out.text) == rest
        assert result["labelled"] == 1
        labelled = sorted(q for q, r in label_rows(calibration_db.conn).items() if r["grounded_label"])
        assert labelled == sorted(subset), "exactly the subset is labelled, nothing outside it"

    def test_limit_without_only_split_is_refused(self, calibration_db, calibration_config, scripted_input):
        """Spec: limit needs only_split."""
        with pytest.raises((ScoringRefused, ValueError)):
            calibration.label_session(calibration_db.conn, RUN, calibration_config,
                                      input_fn=scripted_input(["q"]), print_fn=Printer(), limit=2)
        assert all(r["grounded_label"] is None for r in store.get_calibration_labels(calibration_db.conn, RUN))


# ================================================================ compare

class TestCompare:
    def test_faithfulness_tune_counts_match_hand_fixture(self, calibration_db, calibration_config):
        """Spec: faithfulness `ungrounded` vs labels on tune; counts match the hand-made fixture."""
        seed_labels(calibration_db.conn)
        report = compare_call(calibration_db, calibration_config, "faithfulness", FAITH_A, "tune")
        # tune ids: q001 (human ungrounded, judge 0) -> FN; q002 (ungrounded, judge 1) -> TP;
        # q003 (grounded, judge 0) -> TN; q004 unsure -> excluded. TP 1, FP 0, FN 1, TN 1, n 3.
        row = last_report_row(calibration_db)
        assert (row["tp"], row["fp"], row["fn"], row["tn"], row["n"]) == (1, 0, 1, 1, 3)
        assert (report["tp"], report["fp"], report["fn"], report["tn"], report["n"]) == (1, 0, 1, 1, 3)
        # agreement 2/3; precision 1/1; recall 1/2
        assert (report["agreement"]["successes"], report["agreement"]["n"]) == (2, 3)
        assert (report["precision"]["successes"], report["precision"]["n"]) == (1, 1)
        assert (report["recall"]["successes"], report["recall"]["n"]) == (1, 2)
        # kappa: p_o = 2/3; judge+ = 1/3; human+ = 2/3; p_e = (1/3)(2/3) + (2/3)(1/3) = 4/9
        # kappa = (2/3 - 4/9) / (1 - 4/9) = (2/9) / (5/9) = 0.4
        assert report["kappa"] == pytest.approx(0.4)

    def test_faithfulness_check_excludes_not_judged_and_unlabelled_items(self, calibration_db, calibration_config):
        """Spec: judge-NULL and unlabelled items are left out and counted (n_excluded)."""
        seed_labels(calibration_db.conn)
        report = compare_call(calibration_db, calibration_config, "faithfulness", FAITH_A, "check")
        # check ids: q005 (grounded, judge 0) -> TN; q006 (ungrounded, judge 1) -> TP;
        # q007 judge NULL -> excluded; q008 unlabelled -> excluded. TP 1, TN 1, n 2, n_excluded 2
        row = last_report_row(calibration_db)
        assert (row["tp"], row["fp"], row["fn"], row["tn"], row["n"], row["n_excluded"]) == (1, 0, 0, 1, 2, 2)
        # kappa: p_o = 1; judge+ = 0.5; human+ = 0.5; p_e = 0.5; kappa = 0.5 / 0.5 = 1.0
        assert report["kappa"] == pytest.approx(1.0)

    def test_faithfulness_all_combines_both_halves_and_counts_all_exclusions(self, calibration_db,
                                                                              calibration_config):
        """Spec: split "all" pairs every labelled item; unsure + not judged + unlabelled are counted."""
        seed_labels(calibration_db.conn)
        report = compare_call(calibration_db, calibration_config, "faithfulness", FAITH_A, "all")
        # tune (TP 1, FN 1, TN 1) + check (TP 1, TN 1): TP 2, FP 0, FN 1, TN 2, n 5
        # excluded: q004 unsure, q007 not judged, q008 unlabelled = 3
        row = last_report_row(calibration_db)
        assert (row["tp"], row["fp"], row["fn"], row["tn"], row["n"], row["n_excluded"]) == (2, 0, 1, 2, 5, 3)
        assert row["calib_split"] == "all"
        # agreement 4/5; precision 2/2; recall 2/3
        assert (report["agreement"]["successes"], report["agreement"]["n"]) == (4, 5)
        assert (report["recall"]["successes"], report["recall"]["n"]) == (2, 3)
        # kappa: p_o = 0.8; judge+ = 2/5; human+ = 3/5; p_e = 0.4*0.6 + 0.6*0.4 = 0.48
        # kappa = (0.8 - 0.48) / 0.52 = 0.6154
        assert report["kappa"] == pytest.approx(0.6154, abs=1e-4)

    def test_other_judge_version_is_compared_against_its_own_flags(self, calibration_db, calibration_config):
        """Spec: a judge version is paired with its own stored flags (versions side by side)."""
        seed_labels(calibration_db.conn)
        compare_call(calibration_db, calibration_config, "faithfulness", FAITH_B, "tune")
        # faith-test-B tune: q001 (ungrounded, B flags 1) TP; q002 (ungrounded, B 0) FN;
        # q003 (grounded, B 0) TN; q004 unsure excluded -> TP 1, FN 1, TN 1
        row = last_report_row(calibration_db)
        assert (row["judge_version"], row["tp"], row["fp"], row["fn"], row["tn"]) == (FAITH_B, 1, 0, 1, 1)

    def test_relevance_irrelevant_flag_is_compared_with_relevance_label_zero(self, calibration_db,
                                                                              calibration_config):
        """Spec: relevance judge `irrelevant` vs relevance_label == 0 (positive = irrelevant)."""
        seed_labels(calibration_db.conn)
        report = compare_call(calibration_db, calibration_config, "relevance", REL_A, "check")
        # check ids: q005 relevance label 0 (human +), judge irrelevant 1 -> TP;
        # q006 label 0 (+), judge 0 -> FN; q007 label 2 (-), judge 0 -> TN; q008 unlabelled -> excluded
        row = last_report_row(calibration_db)
        assert (row["judge"], row["tp"], row["fp"], row["fn"], row["tn"], row["n"], row["n_excluded"]) == (
            "relevance", 1, 0, 1, 1, 3, 1)
        # precision 1/1; recall 1/2; agreement 2/3
        assert (report["recall"]["successes"], report["recall"]["n"]) == (1, 2)
        # kappa: p_o = 2/3; judge+ = 1/3; human+ = 2/3; p_e = 4/9; kappa = 0.4
        assert report["kappa"] == pytest.approx(0.4)

    def test_relevance_without_relevance_labels_is_refused(self, calibration_db, calibration_config):
        """Spec: the relevance comparison needs relevance labels; with none there is nothing to compare."""
        labels = {q: (g, None) for q, (g, _) in LABELS.items()}
        seed_labels(calibration_db.conn, labels=labels)
        with pytest.raises((ScoringRefused, ValueError)):
            compare_call(calibration_db, calibration_config, "relevance", REL_A, "check")
        assert store.get_calibration_reports(calibration_db.conn, RUN) == []

    def test_compare_writes_one_report_row_per_call_with_full_report_json(self, calibration_db,
                                                                          calibration_config):
        """Spec: every look is logged: one calibration_reports row per call, with the full report as JSON."""
        seed_labels(calibration_db.conn)
        compare_call(calibration_db, calibration_config, "faithfulness", FAITH_A, "tune")
        compare_call(calibration_db, calibration_config, "faithfulness", FAITH_A, "tune")
        rows = store.get_calibration_reports(calibration_db.conn, RUN)
        assert len(rows) == 2, "identical second look is logged again"
        assert rows[0]["id"] < rows[1]["id"]
        assert (rows[0]["judge"], rows[0]["judge_version"], rows[0]["calib_split"], rows[0]["run_id"]) == (
            "faithfulness", FAITH_A, "tune", RUN)
        assert rows[0]["n_excluded"] == 1 and rows[0]["created_at"]
        with closing(sqlite3.connect(calibration_db.path)) as other:
            raw = other.execute("SELECT report_json FROM calibration_reports WHERE run_id = ? ORDER BY id",
                                (RUN,)).fetchall()
        parsed = json.loads(raw[0][0])
        assert {"tp", "fp", "fn", "tn", "kappa"} <= set(parsed)

    def test_compare_does_not_change_labels_or_judge_scores(self, calibration_db, calibration_config):
        """Spec: compare only reads labels and scores; it writes only the report row."""
        seed_labels(calibration_db.conn)
        labels_before = store.get_calibration_labels(calibration_db.conn, RUN)
        scores_before = store.get_scores(calibration_db.conn, RUN, FAITH_A)
        compare_call(calibration_db, calibration_config, "faithfulness", FAITH_A, "all")
        assert store.get_calibration_labels(calibration_db.conn, RUN) == labels_before
        assert store.get_scores(calibration_db.conn, RUN, FAITH_A) == scores_before

    def test_check_looks_are_counted_per_judge_and_tune_looks_do_not_count(self, calibration_db,
                                                                            calibration_config):
        """Spec: the report says how many earlier "check" looks exist for this judge."""
        seed_labels(calibration_db.conn)
        compare_call(calibration_db, calibration_config, "faithfulness", FAITH_A, "tune")
        compare_call(calibration_db, calibration_config, "faithfulness", FAITH_A, "tune")
        first = compare_call(calibration_db, calibration_config, "faithfulness", FAITH_A, "check")
        assert check_looks(first) == 0, "tune looks are not check looks"
        compare_call(calibration_db, calibration_config, "relevance", REL_A, "check")
        second = compare_call(calibration_db, calibration_config, "faithfulness", FAITH_A, "check")
        assert check_looks(second) == 1, "one earlier faithfulness check look; the relevance look is another judge"
        third = compare_call(calibration_db, calibration_config, "faithfulness", FAITH_B, "check")
        assert check_looks(third) == 2

    def test_check_report_lists_no_item_ids_when_show_check_items_false(self, calibration_db, calibration_config):
        """Spec: check lists counts only (no query ids) unless show_check_items is set."""
        seed_labels(calibration_db.conn)
        report = compare_call(calibration_db, calibration_config, "faithfulness", FAITH_B, "check")
        # B on check has one disagreement: q006 (human ungrounded, judge 0)
        text = calibration.format_compare(report)
        assert not re.search(r"q\d{3}", text), f"check output leaked item ids:\n{text}"
        stored = json.dumps(last_report_row(calibration_db)["report"])
        assert not re.search(r"q\d{3}", stored), "the stored report must not list check item ids either"

    def test_check_report_lists_disagreements_when_show_check_items_true(self, calibration_db, calibration_config):
        """Spec: with show_check_items set, check also lists the disagreements."""
        calibration_config["show_check_items"] = True
        seed_labels(calibration_db.conn)
        report = compare_call(calibration_db, calibration_config, "faithfulness", FAITH_B, "check")
        assert "q006" in calibration.format_compare(report)

    def test_tune_report_lists_disagreements_with_human_and_judge_verdicts(self, calibration_db, calibration_config):
        """Spec: for tune it lists the disagreements (query id, human, judge)."""
        seed_labels(calibration_db.conn)
        report = compare_call(calibration_db, calibration_config, "faithfulness", FAITH_A, "tune")
        text = calibration.format_compare(report)
        # only q001 disagrees (human ungrounded, judge grounded); q002 and q003 agree
        assert "q001" in text
        assert "q002" not in text and "q003" not in text

    def test_all_report_lists_disagreements(self, calibration_db, calibration_config):
        """Spec: for tune and all it lists the disagreements."""
        seed_labels(calibration_db.conn)
        report = compare_call(calibration_db, calibration_config, "faithfulness", FAITH_A, "all")
        assert "q001" in calibration.format_compare(report)

    def test_format_compare_prints_confusion_figures_and_undefined_zero_over_zero(self, calibration_db,
                                                                                   calibration_config):
        """Spec: 0/0 cases print as "undefined (0/0)" (precision, recall and kappa), never 0 or 1."""
        # Only grounded labels, judge B flags nothing: TN 2 -> precision, recall and kappa are all 0/0
        store.save_calibration_splits(calibration_db.conn, RUN, {"q005": "check", "q007": "check"})
        store.write_calibration_label(calibration_db.conn, RUN, "q005", "grounded", 2, None, "gaurav")
        store.write_calibration_label(calibration_db.conn, RUN, "q007", "grounded", 2, None, "gaurav")
        report = compare_call(calibration_db, calibration_config, "faithfulness", FAITH_B, "check")
        text = calibration.format_compare(report)
        assert text.count("undefined (0/0)") >= 3, f"expected 3 undefined figures:\n{text}"
        for word in ("agreement", "precision", "recall", "kappa"):
            assert word in text.lower()
        assert FAITH_B in text

    def test_format_compare_names_version_split_counts_and_confidence(self, calibration_db, calibration_config):
        """Spec: the printed report has the 2x2 table, the four figures with intervals, and n."""
        seed_labels(calibration_db.conn)
        report = compare_call(calibration_db, calibration_config, "faithfulness", FAITH_A, "tune")
        text = calibration.format_compare(report)
        assert FAITH_A in text and "tune" in text
        for cell in ("TP", "FP", "FN", "TN"):
            assert cell in text
        for word in ("agreement", "precision", "recall", "kappa"):
            assert word in text.lower()

    def test_unknown_version_is_refused_and_logs_nothing(self, calibration_db, calibration_config):
        """Spec: an unknown judge version is refused; no report row is written."""
        seed_labels(calibration_db.conn)
        with pytest.raises((ScoringRefused, ValueError)):
            compare_call(calibration_db, calibration_config, "faithfulness", "faith-no-such-version", "tune")
        assert store.get_calibration_reports(calibration_db.conn, RUN) == []

    def test_unknown_judge_name_is_refused(self, calibration_db, calibration_config):
        """Spec: judge is "faithfulness" or "relevance"; anything else is refused."""
        seed_labels(calibration_db.conn)
        with pytest.raises((ScoringRefused, ValueError)):
            compare_call(calibration_db, calibration_config, "style", FAITH_A, "tune")

    def test_split_without_labels_is_refused(self, calibration_db, calibration_config):
        """Spec: a split without labels is refused (nothing to compare)."""
        store.save_calibration_splits(calibration_db.conn, RUN, SPLITS)   # splits only, nothing labelled
        with pytest.raises((ScoringRefused, ValueError)):
            compare_call(calibration_db, calibration_config, "faithfulness", FAITH_A, "tune")
        assert store.get_calibration_reports(calibration_db.conn, RUN) == []

    def test_held_out_run_is_refused(self, make_calibration_db, calibration_config):
        """Spec: a held-out run is refused."""
        db = make_calibration_db(split="heldout")
        with pytest.raises(ScoringRefused):
            compare_call(db, calibration_config, "faithfulness", FAITH_A, "tune")
        assert store.get_calibration_reports(db.conn, RUN) == []


# ================================================================ main (CLI)

class TestMain:
    def _compare(self, tmp_path, db, cfg, version, split, *extra, judge="faithfulness"):
        return calibration.main(["compare", "--run-id", RUN, "--judge", judge, "--version", version,
                                 "--split", split, *extra, *cli_args(tmp_path, db.path, cfg)])

    def test_compare_tune_exits_zero_prints_figures_and_disagreements(self, tmp_path, calibration_db,
                                                                      calibration_config, capsys):
        """Spec CLI: compare --split tune prints the figures and the disagreements; exit 0; one report row."""
        seed_labels(calibration_db.conn)
        code = self._compare(tmp_path, calibration_db, calibration_config, FAITH_A, "tune")
        out = capsys.readouterr().out
        assert code == 0
        for word in ("agreement", "precision", "recall", "kappa"):
            assert word in out.lower()
        assert "q001" in out, "the disagreement is listed"
        assert len(store.get_calibration_reports(calibration_db.conn, RUN)) == 1

    def test_compare_check_without_confirm_flag_exits_one_and_logs_nothing(self, tmp_path, calibration_db,
                                                                            calibration_config, capsys):
        """Spec: compare --split check without --confirm-check is refused (exit 1); no look is logged."""
        seed_labels(calibration_db.conn)
        code = self._compare(tmp_path, calibration_db, calibration_config, FAITH_A, "check")
        capsys.readouterr()
        assert code == 1
        assert store.get_calibration_reports(calibration_db.conn, RUN) == []

    def test_compare_check_with_confirm_flag_exits_zero_and_logs_a_check_look(self, tmp_path, calibration_db,
                                                                               calibration_config, capsys):
        """Spec: --confirm-check makes the look deliberate; it is logged as a check row."""
        seed_labels(calibration_db.conn)
        code = self._compare(tmp_path, calibration_db, calibration_config, FAITH_A, "check", "--confirm-check")
        out = capsys.readouterr().out
        assert code == 0
        rows = store.get_calibration_reports(calibration_db.conn, RUN)
        assert [(r["calib_split"], r["judge_version"]) for r in rows] == [("check", FAITH_A)]
        assert not re.search(r"q\d{3}", out), "check output shows counts only (show_check_items false)"

    def test_compare_unknown_version_exits_one(self, tmp_path, calibration_db, calibration_config, capsys):
        """Spec: an unknown version gives exit 1."""
        seed_labels(calibration_db.conn)
        code = self._compare(tmp_path, calibration_db, calibration_config, "faith-nope", "tune")
        capsys.readouterr()
        assert code == 1
        assert store.get_calibration_reports(calibration_db.conn, RUN) == []

    def test_compare_split_without_labels_exits_one(self, tmp_path, calibration_db, calibration_config, capsys):
        """Spec: a split without labels gives exit 1."""
        store.save_calibration_splits(calibration_db.conn, RUN, SPLITS)
        code = self._compare(tmp_path, calibration_db, calibration_config, FAITH_A, "tune")
        capsys.readouterr()
        assert code == 1

    def test_compare_relevance_judge_through_cli(self, tmp_path, calibration_db, calibration_config, capsys):
        """Spec CLI: --judge relevance compares `irrelevant` with relevance labels."""
        seed_labels(calibration_db.conn)
        code = self._compare(tmp_path, calibration_db, calibration_config, REL_A, "check", "--confirm-check",
                             judge="relevance")
        capsys.readouterr()
        assert code == 0
        assert last_report_row(calibration_db)["judge"] == "relevance"

    @pytest.mark.parametrize("command", ["label", "status", "compare"])
    def test_unknown_run_exits_one(self, tmp_path, calibration_db, calibration_config, capsys, command):
        """Spec exit codes: an unknown run is refused with exit 1."""
        extra = ["--judge", "faithfulness", "--version", FAITH_A, "--split", "tune"] if command == "compare" else []
        code = calibration.main([command, "--run-id", "no-such-run", *extra,
                                 *cli_args(tmp_path, calibration_db.path, calibration_config)])
        capsys.readouterr()
        assert code == 1

    @pytest.mark.parametrize("command", ["label", "status"])
    def test_held_out_run_exits_one_and_writes_no_labels(self, tmp_path, make_calibration_db,
                                                          calibration_config, capsys, command):
        """Spec exit codes: a held-out run is refused; no calibration row is created."""
        db = make_calibration_db(split="heldout")
        code = calibration.main([command, "--run-id", RUN, *cli_args(tmp_path, db.path, calibration_config)])
        capsys.readouterr()
        assert code == 1
        assert store.get_calibration_labels(db.conn, RUN) == []

    def test_not_completed_run_exits_one(self, tmp_path, make_calibration_db, calibration_config, capsys):
        """Spec exit codes: a run that is not completed is refused."""
        db = make_calibration_db(status="running")
        code = calibration.main(["status", "--run-id", RUN, *cli_args(tmp_path, db.path, calibration_config)])
        capsys.readouterr()
        assert code == 1

    def test_bad_calibration_config_exits_one(self, tmp_path, calibration_db, calibration_config, capsys):
        """Spec exit codes: a bad calibration config is refused."""
        args = cli_args(tmp_path, calibration_db.path, calibration_config)
        del calibration_config["check_fraction"]
        args[3] = write_yaml(tmp_path, "bad_calibration.yaml", calibration_config)
        code = calibration.main(["status", "--run-id", RUN, *args])
        capsys.readouterr()
        assert code == 1

    def test_label_with_limit_but_no_split_exits_one(self, tmp_path, calibration_db, calibration_config, capsys):
        """Spec: --limit needs --split; refused before any question is asked."""
        code = calibration.main(["label", "--run-id", RUN, "--limit", "2",
                                 *cli_args(tmp_path, calibration_db.path, calibration_config)])
        capsys.readouterr()
        assert code == 1

    def test_label_split_check_limit_two_reads_keystrokes_and_stores_labels(self, tmp_path, calibration_db,
                                                                            calibration_config, monkeypatch,
                                                                            capsys):
        """Spec CLI: label --split check --limit N reads the real input() (stdin here) and saves each label."""
        monkeypatch.setattr(sys, "stdin", io.StringIO("g\n2\nq\n"))
        code = calibration.main(["label", "--run-id", RUN, "--split", "check", "--limit", "2",
                                 *cli_args(tmp_path, calibration_db.path, calibration_config)])
        capsys.readouterr()
        assert code == 0
        labelled = {q: r for q, r in label_rows(calibration_db.conn).items() if r["grounded_label"]}
        assert len(labelled) == 1, "one label, then q"
        (row,) = labelled.values()
        assert row["calib_split"] == "check"
        assert (row["grounded_label"], row["relevance_label"], row["labeler"]) == ("grounded", 2, "tester")

    def test_status_prints_counts_by_split_and_type_and_no_judge_results(self, tmp_path, calibration_db,
                                                                         calibration_config, scripted_input, capsys):
        """Spec: status prints labels by split and type; it never shows a judge result."""
        init_splits(calibration_db, calibration_config, scripted_input)
        calibration.label_session(calibration_db.conn, RUN, calibration_config,
                                  input_fn=scripted_input(["u", "1", "q"]), print_fn=Printer())
        code = calibration.main(["status", "--run-id", RUN,
                                 *cli_args(tmp_path, calibration_db.path, calibration_config)])
        out = capsys.readouterr().out
        assert code == 0
        for word in ("tune", "check", "lookup", "unanswerable", "unlabelled", "ungrounded"):
            assert word in out.lower(), f"status output lacks {word!r}"
        for marker in calibration_db.markers:
            assert marker not in out, f"status leaked judge or reference text {marker!r}"

    def test_status_ids_prints_only_that_splits_ids_comma_separated(self, tmp_path, calibration_db,
                                                                   calibration_config, scripted_input, capsys):
        """Spec: status --ids tune|check prints only the comma-separated ids of that split."""
        splits = init_splits(calibration_db, calibration_config, scripted_input)
        for half in ("tune", "check"):
            code = calibration.main(["status", "--run-id", RUN, "--ids", half,
                                     *cli_args(tmp_path, calibration_db.path, calibration_config)])
            out = capsys.readouterr().out.strip()
            assert code == 0
            assert sorted(out.split(",")) == sorted(q for q, s in splits.items() if s == half)

    def test_history_lists_every_stored_comparison_in_time_order(self, tmp_path, calibration_db,
                                                                 calibration_config, capsys):
        """Spec: history prints every stored row in time order: version, split, n and the figures."""
        seed_labels(calibration_db.conn)
        assert self._compare(tmp_path, calibration_db, calibration_config, FAITH_A, "tune") == 0
        assert self._compare(tmp_path, calibration_db, calibration_config, FAITH_B, "check", "--confirm-check") == 0
        capsys.readouterr()
        code = calibration.main(["history", "--run-id", RUN,
                                 *cli_args(tmp_path, calibration_db.path, calibration_config)])
        out = capsys.readouterr().out
        assert code == 0
        assert FAITH_A in out and FAITH_B in out
        assert out.index(FAITH_A) < out.index(FAITH_B), "oldest comparison first"
        assert "tune" in out and "check" in out
        assert "kappa" in out.lower() and "recall" in out.lower() and "precision" in out.lower()

    def test_history_without_comparisons_exits_zero(self, tmp_path, calibration_db, calibration_config, capsys):
        """Edge case: history on a run with no comparison prints something and does not fail."""
        code = calibration.main(["history", "--run-id", RUN,
                                 *cli_args(tmp_path, calibration_db.path, calibration_config)])
        capsys.readouterr()
        assert code == 0


# ================================================================ layering

def _imported_roots(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
    return roots


class TestLayering:
    def test_calibration_module_imports_neither_rag_nor_llm_nor_an_sdk(self):
        """Spec Tests/Layering: calibration.py imports neither rag nor llm.client; no LLM call is possible."""
        roots = _imported_roots(PROJECT_ROOT / "evalharness" / "calibration.py")
        for forbidden in ("rag", "llm", "openai", "tenacity"):
            assert forbidden not in roots, f"calibration.py must not import {forbidden}"

    def test_calibration_module_does_not_use_scikit_learn(self):
        """Spec New dependencies: Cohen's kappa is written by hand; scikit-learn is not approved."""
        roots = _imported_roots(PROJECT_ROOT / "evalharness" / "calibration.py")
        assert "sklearn" not in roots
