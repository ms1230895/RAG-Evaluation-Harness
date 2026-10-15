# tests/test_04_test_set.py
# Step 04: test set schema, loader/writer, checks, split assignment, review tool,
# CLI, and the committed data/test_set.jsonl (read-only).
# Expected values come from .claude/specs/04-test-set.md, worked out by hand.
import ast
import json
import random
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from evalharness.schemas import Query
from evalharness.testset import (
    assign_splits,
    check_doc_ids,
    check_ready,
    corpus_doc_ids,
    counts,
    filter_split,
    load_test_set,
    load_testset_config,
    main,
    review,
    save_test_set,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_TEST_SET = PROJECT_ROOT / "data" / "test_set.jsonl"
REAL_CONFIG = PROJECT_ROOT / "configs" / "test_set.yaml"
REAL_DOCS = PROJECT_ROOT / "data" / "docs"

TYPES = ["lookup", "multi_chunk", "paraphrased", "unanswerable"]
ANSWERABLE_TYPES = ["lookup", "multi_chunk", "paraphrased"]
SCHEMA_ORDER = ["id", "question", "type", "reference_answer", "source_doc_ids", "split", "reviewed"]


# ---------------------------------------------------------------- helpers

def write_jsonl(path, rows, trailing_newline=True):
    """Write dict rows as JSONL (one object per line)."""
    text = "\n".join(json.dumps(row) for row in rows)
    if trailing_newline:
        text += "\n"
    Path(path).write_text(text, encoding="utf-8")


def big_set(sizes, make_query):
    """Reviewed, unsplit queries: sizes = {type: n}. Ids q001.. in type order, unique questions."""
    queries = []
    number = 0
    for qtype, n in sizes.items():
        for _ in range(n):
            number += 1
            sources = [] if qtype == "unanswerable" else ["leave_policy"]
            queries.append(Query.model_validate(make_query(
                id=f"q{number:03d}", question=f"Distinct question number {number}?", type=qtype,
                source_doc_ids=sources, reviewed=True, split=None)))
    return queries


def unsplit(queries):
    """Copy of queries with split = None."""
    return [q.model_copy(update={"split": None}) for q in queries]


def config_for(sizes_heldout, sizes_total, seed, docs_dir):
    """Config dict with expected_counts built from per-type totals and heldout numbers."""
    return {
        "path": "unused.jsonl",
        "docs_dir": str(docs_dir),
        "split_seed": seed,
        "expected_counts": {
            t: {"dev": sizes_total[t] - sizes_heldout[t], "heldout": sizes_heldout[t]} for t in TYPES
        },
    }


def heldout_ids(queries):
    """Set of ids whose split is heldout."""
    return {q.id for q in queries if q.split == "heldout"}


# ---------------------------------------------------------------- Schema (Query)

class TestQuerySchema:
    def test_valid_query_passes(self, make_query):
        """Spec Tests: a correct lookup and a correct unanswerable both validate."""
        lookup = Query.model_validate(make_query())
        unanswerable = Query.model_validate(
            make_query(id="q091", type="unanswerable", source_doc_ids=[], split="heldout", reviewed=True))
        assert lookup.id == "q001" and lookup.split is None and lookup.reviewed is False
        assert unanswerable.type == "unanswerable" and unanswerable.source_doc_ids == []

    def test_valid_query_from_json_line(self):
        """Spec Interfaces: Query.model_validate_json parses the documented example line."""
        line = ('{"id": "q001", "question": "Q?", "type": "lookup", "reference_answer": "A.", '
                '"source_doc_ids": ["leave_policy"], "split": null, "reviewed": false}')
        query = Query.model_validate_json(line)
        assert query.source_doc_ids == ["leave_policy"]

    @pytest.mark.parametrize("split", ["dev", "heldout", None])
    def test_split_values_allowed(self, make_query, split):
        """Spec Interfaces: split is dev, heldout or None (drafting)."""
        assert Query.model_validate(make_query(split=split)).split == split

    def test_extra_field_rejected(self, make_query):
        """Spec: extra="forbid", unknown fields are rejected."""
        with pytest.raises(ValidationError):
            Query.model_validate(make_query(difficulty="hard"))

    @pytest.mark.parametrize("bad_type", ["Lookup", "factoid", "", "multi-chunk"])
    def test_bad_type_rejected(self, make_query, bad_type):
        """Spec: unknown query types are rejected."""
        with pytest.raises(ValidationError):
            Query.model_validate(make_query(type=bad_type))

    @pytest.mark.parametrize("bad_split", ["train", "test", "Dev", "", "held-out"])
    def test_bad_split_rejected(self, make_query, bad_split):
        """Spec: split must be dev, heldout or None."""
        with pytest.raises(ValidationError):
            Query.model_validate(make_query(split=bad_split))

    @pytest.mark.parametrize("bad_id", ["7", "q7", "Q007", "q0007", "", "007", "q00a"])
    def test_bad_id_rejected(self, make_query, bad_id):
        """Spec: id must match ^q\\d{3}$."""
        with pytest.raises(ValidationError):
            Query.model_validate(make_query(id=bad_id))

    @pytest.mark.parametrize("field", ["question", "reference_answer"])
    @pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
    def test_blank_question_or_answer_rejected(self, make_query, field, blank):
        """Spec: question and reference_answer must be non-blank after strip."""
        with pytest.raises(ValidationError):
            Query.model_validate(make_query(**{field: blank}))

    def test_strict_types_reviewed_string_rejected(self, make_query):
        """Spec: strict=True, "reviewed": "true" is rejected."""
        with pytest.raises(ValidationError):
            Query.model_validate(make_query(reviewed="true"))

    def test_strict_types_source_doc_ids_string_rejected(self, make_query):
        """Spec: strict=True, "source_doc_ids": "leave_policy" (a string, not a list) is rejected."""
        with pytest.raises(ValidationError):
            Query.model_validate(make_query(source_doc_ids="leave_policy"))

    def test_strict_types_int_id_rejected(self, make_query):
        """Spec: strict=True, an int id is rejected."""
        with pytest.raises(ValidationError):
            Query.model_validate(make_query(id=7))

    def test_unanswerable_needs_no_sources(self, make_query):
        """Spec: type unanswerable <=> source_doc_ids == [], so unanswerable with sources fails."""
        # Control: the same row with [] is valid, so the failure below is about the sources.
        Query.model_validate(make_query(type="unanswerable", source_doc_ids=[]))
        with pytest.raises(ValidationError):
            Query.model_validate(make_query(type="unanswerable", source_doc_ids=["leave_policy"]))

    @pytest.mark.parametrize("qtype", ANSWERABLE_TYPES)
    def test_answerable_needs_sources(self, make_query, qtype):
        """Spec: lookup / multi_chunk / paraphrased with source_doc_ids == [] are rejected."""
        # Control: with a source the row is valid.
        Query.model_validate(make_query(type=qtype, source_doc_ids=["leave_policy"]))
        with pytest.raises(ValidationError):
            Query.model_validate(make_query(type=qtype, source_doc_ids=[]))

    @pytest.mark.parametrize("bad_doc", ["Leave_Policy", "leave-policy", "leave policy", "", "leave.md", "leave/policy"])
    def test_bad_doc_id_rejected(self, make_query, bad_doc):
        """Spec: each source doc ID must match ^[a-z0-9_]+$."""
        with pytest.raises(ValidationError):
            Query.model_validate(make_query(source_doc_ids=[bad_doc]))

    def test_duplicate_doc_id_rejected(self, make_query):
        """Spec: source_doc_ids has no duplicates."""
        with pytest.raises(ValidationError):
            Query.model_validate(make_query(type="multi_chunk",
                                            source_doc_ids=["leave_policy", "leave_policy"]))


# ---------------------------------------------------------------- Loader and writer

class TestLoadAndSave:
    def test_load_returns_queries_in_file_order(self, tmp_path, make_query):
        """Spec: load_test_set returns one Query per JSONL line."""
        rows = [make_query(id="q001"), make_query(id="q002", question="Another question?")]
        write_jsonl(tmp_path / "ts.jsonl", rows)
        queries = load_test_set(tmp_path / "ts.jsonl")
        assert [q.id for q in queries] == ["q001", "q002"]
        assert all(isinstance(q, Query) for q in queries)

    def test_load_reports_line_number_for_validation_error(self, tmp_path, make_query):
        """Spec: a validation error on line 3 raises ValueError mentioning line 3."""
        rows = [make_query(id="q001"), make_query(id="q002", question="Second?"),
                make_query(id="q003", question="Third?", type="not_a_type")]
        write_jsonl(tmp_path / "ts.jsonl", rows)
        with pytest.raises(ValueError, match=r"(?i)line 3\b"):
            load_test_set(tmp_path / "ts.jsonl")

    def test_load_reports_line_number_for_bad_json(self, tmp_path, make_query):
        """Spec: bad JSON on line 3 raises ValueError mentioning line 3."""
        good = [json.dumps(make_query(id="q001")), json.dumps(make_query(id="q002", question="Second?"))]
        (tmp_path / "ts.jsonl").write_text("\n".join(good + ["{not json"]) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"(?i)line 3\b"):
            load_test_set(tmp_path / "ts.jsonl")

    def test_duplicate_id_rejected(self, tmp_path, make_query):
        """Spec: a duplicate id raises ValueError."""
        rows = [make_query(id="q001"), make_query(id="q001", question="A different question?")]
        write_jsonl(tmp_path / "ts.jsonl", rows)
        with pytest.raises(ValueError, match=r"(?i)duplicate"):
            load_test_set(tmp_path / "ts.jsonl")

    def test_duplicate_question_rejected_when_case_and_spacing_differ(self, tmp_path, make_query):
        """Spec: duplicate questions are compared lowercased with whitespace collapsed."""
        rows = [make_query(id="q001", question="How many leaves?"),
                make_query(id="q002", question="  how   MANY\tleaves?  ")]
        write_jsonl(tmp_path / "ts.jsonl", rows)
        with pytest.raises(ValueError, match=r"(?i)duplicate"):
            load_test_set(tmp_path / "ts.jsonl")

    def test_blank_line_in_middle_rejected(self, tmp_path, make_query):
        """Spec: blank lines are an error."""
        first = json.dumps(make_query(id="q001"))
        second = json.dumps(make_query(id="q002", question="Second?"))
        (tmp_path / "ts.jsonl").write_text(first + "\n\n" + second + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_test_set(tmp_path / "ts.jsonl")

    @pytest.mark.parametrize("trailing_newline", [True, False])
    def test_trailing_newline_ok(self, tmp_path, make_query, trailing_newline):
        """Spec: a trailing newline (or none) is not a blank line."""
        write_jsonl(tmp_path / "ts.jsonl",
                    [make_query(id="q001"), make_query(id="q002", question="Second?")],
                    trailing_newline=trailing_newline)
        assert len(load_test_set(tmp_path / "ts.jsonl")) == 2

    def test_save_load_roundtrip(self, tmp_path, ready_queries):
        """Spec: queries are equal after save then load."""
        path = tmp_path / "out" / "ts.jsonl"
        path.parent.mkdir()
        save_test_set(ready_queries, path)
        assert load_test_set(path) == ready_queries

    def test_save_sorts_by_id_with_schema_field_order(self, tmp_path, ready_queries):
        """Spec: output sorted by id, fields in schema order, one object per line."""
        path = tmp_path / "out" / "ts.jsonl"
        path.parent.mkdir()
        save_test_set(list(reversed(ready_queries)), path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 8, "Expected one line per query"
        objects = [json.loads(line) for line in lines]
        assert [o["id"] for o in objects] == [f"q{i:03d}" for i in range(1, 9)]
        assert all(list(o.keys()) == SCHEMA_ORDER for o in objects)

    def test_save_leaves_no_temp_file(self, tmp_path, ready_queries):
        """Spec: atomic write via temp file + os.replace, nothing else left in the directory."""
        out = tmp_path / "out"
        out.mkdir()
        path = out / "ts.jsonl"
        save_test_set(ready_queries, path)
        save_test_set(ready_queries[:3], path)  # overwrite an existing file
        assert [p.name for p in out.iterdir()] == ["ts.jsonl"], "Only the target file may remain"
        assert len(load_test_set(path)) == 3, "Second save must replace the first"

    def test_save_keeps_non_ascii_unescaped(self, tmp_path, make_query):
        """Spec: ensure_ascii=False, non-ASCII text is written as is and survives a round trip."""
        query = Query.model_validate(make_query(question="Is the fee Rs 1,000 or ₹1,000 – café?"))
        path = tmp_path / "ts.jsonl"
        save_test_set([query], path)
        text = path.read_text(encoding="utf-8")
        assert "₹" in text and "café" in text, "Non-ASCII characters must be written raw"
        assert "\\u" not in text, "No \\u escapes expected"
        assert load_test_set(path) == [query]


# ---------------------------------------------------------------- Checks

class TestChecks:
    def test_corpus_doc_ids_returns_md_stems_only(self, tmp_path):
        """Spec: corpus_doc_ids = stems of *.md files in docs_dir."""
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "leave_policy.md").write_text("# Leave\n", encoding="utf-8")
        (docs / "it_policy.md").write_text("# IT\n", encoding="utf-8")
        (docs / "notes.txt").write_text("not a doc", encoding="utf-8")
        assert corpus_doc_ids(docs) == {"leave_policy", "it_policy"}

    def test_check_doc_ids_names_unknown(self, make_query):
        """Spec: one problem string per unknown source doc ID, naming it."""
        queries = [
            Query.model_validate(make_query(id="q001", type="multi_chunk",
                                            source_doc_ids=["leave_policy", "ghost_policy"])),
            Query.model_validate(make_query(id="q002", question="Second?", source_doc_ids=["phantom_doc"])),
        ]
        problems = check_doc_ids(queries, {"leave_policy"})
        assert len(problems) == 2, f"Expected one problem per unknown ID, got {problems}"
        assert any("ghost_policy" in p for p in problems)
        assert any("phantom_doc" in p for p in problems)

    def test_check_doc_ids_all_known_returns_empty(self, ready_queries):
        """Spec: [] if all source doc IDs are known."""
        known = {"leave_policy", "payroll_policy", "it_policy"}
        assert check_doc_ids(ready_queries, known) == []

    def test_counts_shape(self, ready_queries):
        """Spec: counts gives {type: {dev, heldout, unassigned}} for all 4 types."""
        # ready_queries: lookup q001 dev, q002 dev, q003 heldout; multi_chunk q004 dev, q005 heldout;
        # paraphrased q006 dev; unanswerable q007 dev, q008 heldout.
        result = counts(ready_queries)
        assert result == {
            "lookup": {"dev": 2, "heldout": 1, "unassigned": 0},
            "multi_chunk": {"dev": 1, "heldout": 1, "unassigned": 0},
            "paraphrased": {"dev": 1, "heldout": 0, "unassigned": 0},
            "unanswerable": {"dev": 1, "heldout": 1, "unassigned": 0},
        }

    def test_counts_unassigned_split_counted(self, ready_queries):
        """Spec: queries with split None are counted as unassigned."""
        # q002 (lookup, dev) loses its split -> lookup: dev 1, heldout 1, unassigned 1.
        modified = [q.model_copy(update={"split": None}) if q.id == "q002" else q for q in ready_queries]
        assert counts(modified)["lookup"] == {"dev": 1, "heldout": 1, "unassigned": 1}

    def test_counts_all_types_present_for_empty_list(self):
        """Spec: all 4 types appear even when a type has 0 queries."""
        result = counts([])
        assert set(result) == set(TYPES)
        assert all(v == {"dev": 0, "heldout": 0, "unassigned": 0} for v in result.values())

    def test_counts_type_with_zero_queries_still_listed(self, make_query):
        """Spec: a type with 0 queries is present with zeros while other types have counts."""
        only_lookup = [Query.model_validate(make_query(split="dev"))]
        result = counts(only_lookup)
        assert result["lookup"] == {"dev": 1, "heldout": 0, "unassigned": 0}
        assert result["unanswerable"] == {"dev": 0, "heldout": 0, "unassigned": 0}

    def test_check_ready_correct_set_returns_empty(self, ready_queries, testset_config):
        """Spec: check_ready returns [] when reviewed, split, counts exact and doc IDs known."""
        assert check_ready(ready_queries, testset_config) == []

    def test_check_ready_lists_unreviewed_id(self, ready_queries, testset_config):
        """Spec: an unreviewed query id appears in the problems."""
        modified = [q.model_copy(update={"reviewed": False}) if q.id == "q004" else q for q in ready_queries]
        problems = check_ready(modified, testset_config)
        assert any("q004" in p for p in problems), f"q004 not named in {problems}"

    def test_check_ready_lists_missing_split(self, ready_queries, testset_config):
        """Spec: a query with split None appears in the problems."""
        modified = [q.model_copy(update={"split": None}) if q.id == "q002" else q for q in ready_queries]
        problems = check_ready(modified, testset_config)
        assert any("q002" in p for p in problems), f"q002 not named in {problems}"

    def test_check_ready_lists_count_mismatch(self, ready_queries, testset_config):
        """Spec: type x split counts that differ from config appear in the problems."""
        # Drop q006: paraphrased dev is 0, config expects 1.
        modified = [q for q in ready_queries if q.id != "q006"]
        problems = check_ready(modified, testset_config)
        assert any("paraphrased" in p for p in problems), f"paraphrased mismatch not named in {problems}"

    def test_check_ready_lists_unknown_doc_id(self, ready_queries, testset_config):
        """Spec: an unknown source doc ID appears in the problems."""
        modified = [q.model_copy(update={"source_doc_ids": ["ghost_policy"]}) if q.id == "q006" else q
                    for q in ready_queries]
        problems = check_ready(modified, testset_config)
        assert any("ghost_policy" in p for p in problems), f"ghost_policy not named in {problems}"

    def test_check_ready_lists_each_problem_together(self, ready_queries, testset_config):
        """Spec: check_ready lists every reason at once (unreviewed, no split, count, doc ID)."""
        modified = []
        for q in ready_queries:
            if q.id == "q006":
                continue  # drops paraphrased dev: count mismatch
            if q.id == "q004":
                q = q.model_copy(update={"reviewed": False})
            if q.id == "q002":
                q = q.model_copy(update={"split": None})
            if q.id == "q005":
                q = q.model_copy(update={"source_doc_ids": ["ghost_policy"]})
            modified.append(q)
        problems = check_ready(modified, testset_config)
        for marker in ("q004", "q002", "paraphrased", "ghost_policy"):
            assert any(marker in p for p in problems), f"{marker} not named in {problems}"


class TestFilterSplit:
    def test_filter_split_returns_one_split_in_id_order(self, ready_queries):
        """Spec: filter_split returns the queries of one split, in id order."""
        shuffled = [ready_queries[i] for i in (7, 2, 5, 0, 3, 6, 1, 4)]
        dev = filter_split(shuffled, "dev")
        heldout = filter_split(shuffled, "heldout")
        # dev ids: q001 q002 q004 q006 q007; heldout ids: q003 q005 q008
        assert [q.id for q in dev] == ["q001", "q002", "q004", "q006", "q007"]
        assert [q.id for q in heldout] == ["q003", "q005", "q008"]

    def test_filter_split_excludes_unassigned(self, ready_queries):
        """Spec: a query with split None is in neither split."""
        modified = [q.model_copy(update={"split": None}) if q.id == "q001" else q for q in ready_queries]
        assert "q001" not in [q.id for q in filter_split(modified, "dev")]
        assert "q001" not in [q.id for q in filter_split(modified, "heldout")]


# ---------------------------------------------------------------- Config

class TestConfig:
    def test_load_testset_config_real_file(self):
        """Spec Tests: the real configs/test_set.yaml loads, with the spec's values."""
        config = load_testset_config(REAL_CONFIG)
        assert config["path"] == "data/test_set.jsonl"
        assert config["docs_dir"] == "data/docs"
        assert config["split_seed"] == 2026
        assert config["expected_counts"] == {
            "lookup": {"dev": 40, "heldout": 10},
            "multi_chunk": {"dev": 16, "heldout": 4},
            "paraphrased": {"dev": 12, "heldout": 3},
            "unanswerable": {"dev": 12, "heldout": 3},
        }

    @pytest.mark.parametrize("missing", ["path", "docs_dir", "split_seed", "expected_counts"])
    def test_load_testset_config_missing_key_raises(self, tmp_path, testset_config, missing):
        """Spec: a missing required key raises ValueError naming the problem."""
        data = {k: v for k, v in testset_config.items() if k != missing}
        (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ValueError, match=missing):
            load_testset_config(tmp_path / "cfg.yaml")

    @pytest.mark.parametrize("missing_type", TYPES)
    def test_load_testset_config_missing_type_raises(self, tmp_path, testset_config, missing_type):
        """Spec: every type must be present in expected_counts."""
        data = dict(testset_config)
        data["expected_counts"] = {k: v for k, v in testset_config["expected_counts"].items()
                                   if k != missing_type}
        (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ValueError, match=missing_type):
            load_testset_config(tmp_path / "cfg.yaml")

    @pytest.mark.parametrize("bad_value", [-1, 2.5, "ten"])
    def test_load_testset_config_bad_count_raises(self, tmp_path, testset_config, bad_value):
        """Spec: counts must be non-negative ints."""
        data = dict(testset_config)
        data["expected_counts"] = {**testset_config["expected_counts"],
                                   "lookup": {"dev": bad_value, "heldout": 1}}
        (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ValueError):
            load_testset_config(tmp_path / "cfg.yaml")

    def test_load_testset_config_valid_tmp_file_loads(self, tmp_path, testset_config):
        """Spec: a well-formed config (zero counts allowed) loads and round-trips its values."""
        # Control for the failure tests above: only the injected defect makes them fail.
        data = dict(testset_config)
        data["expected_counts"] = {**testset_config["expected_counts"],
                                   "paraphrased": {"dev": 0, "heldout": 0}}
        (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
        loaded = load_testset_config(tmp_path / "cfg.yaml")
        assert loaded["split_seed"] == 7
        assert loaded["expected_counts"]["paraphrased"] == {"dev": 0, "heldout": 0}


# ---------------------------------------------------------------- Split assignment

class TestAssignSplits:
    SIZES = {"lookup": 30, "multi_chunk": 10, "paraphrased": 10, "unanswerable": 10}
    HELDOUT = {"lookup": 10, "multi_chunk": 3, "paraphrased": 3, "unanswerable": 3}

    def test_assign_splits_exact_counts(self, make_query, testset_docs_dir):
        """Spec: per type, heldout and dev counts match the config."""
        queries = big_set(self.SIZES, make_query)
        config = config_for(self.HELDOUT, self.SIZES, 2026, testset_docs_dir)
        result = counts(assign_splits(queries, config))
        # lookup 30 -> 20 dev + 10 heldout; the other types 10 -> 7 dev + 3 heldout.
        assert result["lookup"] == {"dev": 20, "heldout": 10, "unassigned": 0}
        for qtype in ("multi_chunk", "paraphrased", "unanswerable"):
            assert result[qtype] == {"dev": 7, "heldout": 3, "unassigned": 0}, qtype

    def test_assign_splits_tiny_set_matches_config(self, ready_queries, testset_config):
        """Spec: assign_splits on the tiny set gives exactly the config's counts, then check_ready is []."""
        result = assign_splits(unsplit(ready_queries), testset_config)
        assert counts(result) == counts(ready_queries)
        assert check_ready(result, testset_config) == []

    def test_assign_splits_deterministic_for_same_seed(self, make_query, testset_docs_dir):
        """Spec: the same seed gives the same heldout ids."""
        queries = big_set(self.SIZES, make_query)
        config = config_for(self.HELDOUT, self.SIZES, 2026, testset_docs_dir)
        first = heldout_ids(assign_splits(queries, config))
        second = heldout_ids(assign_splits(queries, config))
        # heldout total = 10 + 3 + 3 + 3 = 19
        assert first == second
        assert len(first) == 19

    def test_assign_splits_different_seed_changes_ids(self, make_query, testset_docs_dir):
        """Spec: another seed gives different heldout ids (set is large enough that this is certain)."""
        queries = big_set(self.SIZES, make_query)
        with_seed_1 = heldout_ids(assign_splits(queries, config_for(self.HELDOUT, self.SIZES, 1, testset_docs_dir)))
        with_seed_2 = heldout_ids(assign_splits(queries, config_for(self.HELDOUT, self.SIZES, 2, testset_docs_dir)))
        assert with_seed_1 != with_seed_2

    def test_assign_splits_uses_seeded_sample_of_sorted_ids(self, make_query, testset_docs_dir):
        """Spec: ids sorted, random.Random(split_seed).sample() picks the heldout ids."""
        sizes = {"lookup": 20, "multi_chunk": 0, "paraphrased": 0, "unanswerable": 0}
        heldout = {"lookup": 5, "multi_chunk": 0, "paraphrased": 0, "unanswerable": 0}
        queries = big_set(sizes, make_query)
        shuffled = list(reversed(queries))  # input order must not matter: ids are sorted first
        config = config_for(heldout, sizes, 2026, testset_docs_dir)
        expected = set(random.Random(2026).sample(sorted(q.id for q in queries), 5))
        assert heldout_ids(assign_splits(shuffled, config)) == expected

    def test_assign_splits_refuses_when_split_already_set(self, ready_queries, testset_config):
        """Spec: ValueError if any query already has a split."""
        queries = unsplit(ready_queries)
        assign_splits(queries, testset_config)  # control: a clean set is accepted
        queries[0] = queries[0].model_copy(update={"split": "dev"})
        with pytest.raises(ValueError):
            assign_splits(queries, testset_config)

    def test_assign_splits_refuses_unreviewed_query(self, ready_queries, testset_config):
        """Spec: ValueError if any query is not reviewed."""
        queries = unsplit(ready_queries)
        assign_splits(queries, testset_config)  # control
        queries[3] = queries[3].model_copy(update={"reviewed": False})
        with pytest.raises(ValueError):
            assign_splits(queries, testset_config)

    def test_assign_splits_refuses_type_total_differing_from_config(self, ready_queries, testset_config):
        """Spec: ValueError if a type's total differs from the config total."""
        queries = unsplit(ready_queries)
        assign_splits(queries, testset_config)  # control
        # Drop one unanswerable query: total 1, config expects 1 + 1 = 2.
        fewer = [q for q in queries if q.id != "q008"]
        with pytest.raises(ValueError):
            assign_splits(fewer, testset_config)

    def test_assign_splits_changes_only_split(self, make_query, testset_docs_dir):
        """Spec: returns new Query objects; no field other than split changes."""
        queries = big_set(self.SIZES, make_query)
        config = config_for(self.HELDOUT, self.SIZES, 2026, testset_docs_dir)
        result = assign_splits(queries, config)
        assert len(result) == len(queries)
        by_id = {q.id: q for q in result}
        for original in queries:
            new = by_id[original.id]
            assert new is not original, "Expected new Query objects"
            assert new.split in ("dev", "heldout")
            assert new.model_dump(exclude={"split"}) == original.model_dump(exclude={"split"})
        assert all(q.split is None for q in queries), "Input queries must not be modified"


# ---------------------------------------------------------------- Review tool

class TestReview:
    def _draft(self, tmp_path, make_query, rows=None):
        """Write unreviewed draft rows to tmp_path/ts.jsonl and return (path, loaded queries)."""
        rows = rows or [
            make_query(id="q001", question="Question one?"),
            make_query(id="q002", question="Question two?", source_doc_ids=["payroll_policy"]),
        ]
        path = tmp_path / "ts.jsonl"
        write_jsonl(path, rows)
        return path, load_test_set(path)

    def test_review_accept_sets_reviewed_and_saves(self, tmp_path, make_query, testset_docs_dir, scripted_input):
        """Spec: [a]ccept sets reviewed=True and saves the file."""
        path, queries = self._draft(tmp_path, make_query)
        inp = scripted_input(["a", "a"])
        result = review(queries, path, testset_docs_dir, input_fn=inp)
        assert [q.reviewed for q in result] == [True, True]
        saved = load_test_set(path)
        assert [q.reviewed for q in saved] == [True, True], "Accepted state must be in the saved file"
        assert saved[0].question == "Question one?", "Accept must not change other fields"

    def test_review_accept_only_marks_current_query(self, tmp_path, make_query, testset_docs_dir, scripted_input):
        """Spec: only a key pressed for a query sets reviewed=True for that query."""
        path, queries = self._draft(tmp_path, make_query)
        review(queries, path, testset_docs_dir, input_fn=scripted_input(["a", "s"]))
        saved = {q.id: q for q in load_test_set(path)}
        assert saved["q001"].reviewed is True
        assert saved["q002"].reviewed is False

    def test_review_skip_leaves_unreviewed(self, tmp_path, make_query, testset_docs_dir, scripted_input):
        """Spec: [s]kip leaves the query unreviewed."""
        path, queries = self._draft(tmp_path, make_query)
        result = review(queries, path, testset_docs_dir, input_fn=scripted_input(["s", "s"]))
        assert [q.reviewed for q in result] == [False, False]
        assert [q.reviewed for q in load_test_set(path)] == [False, False]

    def test_review_only_walks_unreviewed_queries(self, tmp_path, make_query, testset_docs_dir, scripted_input):
        """Spec: the tool goes through each unreviewed query; reviewed ones are not shown again."""
        rows = [make_query(id="q001", question="Question one?", reviewed=True),
                make_query(id="q002", question="Question two?")]
        path, queries = self._draft(tmp_path, make_query, rows)
        inp = scripted_input(["a"])
        result = review(queries, path, testset_docs_dir, input_fn=inp)
        assert inp.answers == [], "Exactly one key should have been consumed"
        assert [q.reviewed for q in result] == [True, True]
        assert [q.id for q in load_test_set(path) if q.reviewed] == ["q001", "q002"]

    def test_review_walks_in_id_order(self, tmp_path, make_query, testset_docs_dir, scripted_input):
        """Spec: for each unreviewed query in id order (accept first, skip second)."""
        rows = [make_query(id="q002", question="Question two?"),
                make_query(id="q001", question="Question one?")]
        path, queries = self._draft(tmp_path, make_query, rows)
        review(queries, path, testset_docs_dir, input_fn=scripted_input(["a", "s"]))
        saved = {q.id: q.reviewed for q in load_test_set(path)}
        assert saved == {"q001": True, "q002": False}, "First shown query must be q001"

    def test_review_shows_query_and_source_text(self, tmp_path, make_query, testset_docs_dir,
                                                scripted_input, capsys):
        """Spec: prints id, type, question, reference answer, source doc IDs and full text of each source doc."""
        rows = [make_query(id="q001", question="Which leave applies to staff?",
                           reference_answer="Eight days.", source_doc_ids=["leave_policy"])]
        path, queries = self._draft(tmp_path, make_query, rows)
        review(queries, path, testset_docs_dir, input_fn=scripted_input(["q"]))
        out = capsys.readouterr().out
        for expected in ("q001", "lookup", "Which leave applies to staff?", "Eight days.", "leave_policy",
                         "Employees get 8 Casual leaves per year.", "Carry forward is not allowed."):
            assert expected in out, f"{expected!r} missing from review output"

    @pytest.mark.parametrize("field,new_value", [
        ("question", "A completely new question?"),
        ("reference_answer", "A corrected reference answer."),
    ])
    def test_review_edit_text_field_applies_and_saves(self, tmp_path, make_query, testset_docs_dir,
                                                      scripted_input, field, new_value):
        """Spec: [e]dit a field, enter a new value, then accept; both edit and accept are saved."""
        path, queries = self._draft(tmp_path, make_query, [make_query(id="q001")])
        result = review(queries, path, testset_docs_dir, input_fn=scripted_input(["e", field, new_value, "a"]))
        assert getattr(result[0], field) == new_value
        assert result[0].reviewed is True
        saved = load_test_set(path)[0]
        assert getattr(saved, field) == new_value and saved.reviewed is True

    def test_review_edit_source_doc_ids_applies(self, tmp_path, make_query, testset_docs_dir, scripted_input):
        """Spec: source_doc_ids is an editable field (single ID typed as the new value)."""
        path, queries = self._draft(tmp_path, make_query, [make_query(id="q001")])
        result = review(queries, path, testset_docs_dir,
                        input_fn=scripted_input(["e", "source_doc_ids", "payroll_policy", "a"]))
        assert result[0].source_doc_ids == ["payroll_policy"]
        assert load_test_set(path)[0].source_doc_ids == ["payroll_policy"]

    def test_review_edit_type_asks_for_sources_and_applies_them(self, tmp_path, make_query, testset_docs_dir,
                                                                scripted_input):
        """Spec: editing type also asks for source_doc_ids; unanswerable -> lookup needs new sources."""
        rows = [make_query(id="q001", type="unanswerable", source_doc_ids=[])]
        path, queries = self._draft(tmp_path, make_query, rows)
        inp = scripted_input(["e", "type", "lookup", "payroll_policy", "a"])
        result = review(queries, path, testset_docs_dir, input_fn=inp)
        assert inp.answers == [], "Type edit must consume exactly the follow-up prompt"
        assert result[0].type == "lookup"
        assert result[0].source_doc_ids == ["payroll_policy"]

    def test_review_edit_type_enter_keeps_current_sources(self, tmp_path, make_query, testset_docs_dir,
                                                          scripted_input):
        """Spec: the follow-up source_doc_ids prompt keeps the current value on Enter (empty input)."""
        path, queries = self._draft(tmp_path, make_query, [make_query(id="q001")])
        result = review(queries, path, testset_docs_dir,
                        input_fn=scripted_input(["e", "type", "multi_chunk", "", "a"]))
        assert result[0].type == "multi_chunk"
        assert result[0].source_doc_ids == ["leave_policy"]

    def test_review_edit_type_to_unanswerable_clears_sources(self, tmp_path, make_query, testset_docs_dir,
                                                             scripted_input):
        """Spec: "-" at the type follow-up clears source_doc_ids, so a lookup can become unanswerable."""
        path, queries = self._draft(tmp_path, make_query, [make_query(id="q001")])
        inp = scripted_input(["e", "type", "unanswerable", "-", "a"])
        result = review(queries, path, testset_docs_dir, input_fn=inp)
        assert inp.answers == []
        assert result[0].type == "unanswerable"
        assert result[0].source_doc_ids == []
        saved = load_test_set(path)[0]
        assert saved.type == "unanswerable" and saved.source_doc_ids == [] and saved.reviewed is True

    def test_review_edit_source_doc_ids_multiple_and_none(self, tmp_path, make_query, testset_docs_dir,
                                                          scripted_input):
        """Spec: source_doc_ids typed comma-separated; "-" means none (invalid for a lookup, so rejected)."""
        path, queries = self._draft(tmp_path, make_query, [make_query(id="q001")])
        result = review(queries, path, testset_docs_dir,
                        input_fn=scripted_input(["e", "source_doc_ids", "leave_policy, payroll_policy", "s"]))
        assert result[0].source_doc_ids == ["leave_policy", "payroll_policy"]

        path, queries = self._draft(tmp_path, make_query, [make_query(id="q001")])
        original = queries[0]
        result = review(queries, path, testset_docs_dir,
                        input_fn=scripted_input(["e", "source_doc_ids", "-", "s"]))
        assert result == [original], "A lookup with no sources is invalid; the edit must be rejected"

    def test_review_edit_revalidates_and_rejects_invalid_value(self, tmp_path, make_query, testset_docs_dir,
                                                               scripted_input):
        """Spec: an edit to an invalid value (unanswerable with sources) is rejected, query unchanged."""
        rows = [make_query(id="q001", type="unanswerable", source_doc_ids=[])]
        path, queries = self._draft(tmp_path, make_query, rows)
        original = queries[0]
        before = path.read_bytes()
        result = review(queries, path, testset_docs_dir,
                        input_fn=scripted_input(["e", "source_doc_ids", "leave_policy", "s"]))
        assert result == [original], "Rejected edit must leave the query unchanged"
        assert load_test_set(path) == [original]
        assert path.read_bytes() == before, "File must not change when the only action was a rejected edit"

    def test_review_edit_rejects_blank_question(self, tmp_path, make_query, testset_docs_dir, scripted_input):
        """Spec: edits are re-validated, a blank question is rejected."""
        path, queries = self._draft(tmp_path, make_query, [make_query(id="q001", question="Keep me?")])
        result = review(queries, path, testset_docs_dir,
                        input_fn=scripted_input(["e", "question", "   ", "s"]))
        assert result[0].question == "Keep me?"

    def test_review_delete_removes_query_from_saved_file(self, tmp_path, make_query, testset_docs_dir,
                                                         scripted_input):
        """Spec: [d]elete removes the query; the deleted id is gone from the saved file."""
        rows = [make_query(id="q001", question="Question one?"),
                make_query(id="q002", question="Question two?"),
                make_query(id="q003", question="Question three?")]
        path, queries = self._draft(tmp_path, make_query, rows)
        result = review(queries, path, testset_docs_dir, input_fn=scripted_input(["d", "a", "s"]))
        assert [q.id for q in result] == ["q002", "q003"]
        saved = load_test_set(path)
        assert [q.id for q in saved] == ["q002", "q003"], "q001 must be gone from the file"
        assert [q.reviewed for q in saved] == [True, False]

    def test_review_quit_saves_work_so_far_and_stops(self, tmp_path, make_query, testset_docs_dir,
                                                     scripted_input):
        """Spec: [q]uit saves what was done so far and asks for nothing more."""
        rows = [make_query(id="q001", question="Question one?"),
                make_query(id="q002", question="Question two?"),
                make_query(id="q003", question="Question three?")]
        path, queries = self._draft(tmp_path, make_query, rows)
        inp = scripted_input(["a", "q", "a"])
        review(queries, path, testset_docs_dir, input_fn=inp)
        assert inp.answers == ["a"], "Nothing may be asked after quit"
        saved = {q.id: q.reviewed for q in load_test_set(path)}
        assert saved == {"q001": True, "q002": False, "q003": False}

    def test_review_grep_prints_matches(self, tmp_path, make_query, testset_docs_dir, scripted_input, capsys):
        """Spec: g then a word prints matching corpus lines (whole word, any case); file unchanged."""
        rows = [make_query(id="q001", question="Is there a gym?", type="unanswerable",
                           reference_answer="The Kestrel Labs HR policies do not cover this.",
                           source_doc_ids=[])]
        path, queries = self._draft(tmp_path, make_query, rows)
        before = path.read_bytes()
        review(queries, path, testset_docs_dir, input_fn=scripted_input(["g", "CASUAL", "s"]))
        out = capsys.readouterr().out
        assert "Employees get 8 Casual leaves per year." in out, "Whole-word, case-insensitive match expected"
        assert "Casualties are not a laptop topic." not in out, "'Casualties' is not the whole word 'casual'"
        assert "Salary is paid on the last working day." not in out, "Non-matching line must not be printed"
        assert path.read_bytes() == before, "grep must not change the file"

    def test_review_unknown_key_does_not_change_query(self, tmp_path, make_query, testset_docs_dir,
                                                      scripted_input):
        """Spec: only the defined keys act; an unknown key never sets reviewed=True."""
        path, queries = self._draft(tmp_path, make_query, [make_query(id="q001")])
        result = review(queries, path, testset_docs_dir, input_fn=scripted_input(["x", "s"]))
        assert result[0].reviewed is False


# ---------------------------------------------------------------- CLI

class TestCli:
    def _write_config(self, tmp_path, testset_config):
        """Write testset_config as YAML and return its path as a string."""
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(testset_config), encoding="utf-8")
        return str(path)

    def test_summary_ready_set_exits_zero(self, tmp_path, ready_queries, testset_config, capsys):
        """Spec CLI: summary exits 0 when ready and prints reviewed n/total."""
        save_test_set(ready_queries, testset_config["path"])
        code = main(["summary", "--config", self._write_config(tmp_path, testset_config)])
        out = capsys.readouterr().out.lower()
        assert code == 0
        assert "reviewed 8/8" in out, f"Expected 'reviewed 8/8' in summary, got:\n{out}"

    def test_summary_not_ready_exits_one(self, tmp_path, ready_queries, testset_config, capsys):
        """Spec CLI: summary exits 1 if the set is not ready (here one query unreviewed)."""
        modified = [q.model_copy(update={"reviewed": False}) if q.id == "q004" else q for q in ready_queries]
        save_test_set(modified, testset_config["path"])
        code = main(["summary", "--config", self._write_config(tmp_path, testset_config)])
        out = capsys.readouterr().out.lower()
        assert code == 1
        assert "reviewed 7/8" in out, f"Expected 'reviewed 7/8' in summary, got:\n{out}"

    def test_summary_lists_number_not_in_source_for_lookup(self, tmp_path, ready_queries, testset_config, capsys):
        """Spec CLI: summary lists a lookup reference-answer number missing from its source docs."""
        # leave_policy.md has the number 8 only, so 99 is not found.
        modified = [q.model_copy(update={"reference_answer": "99 casual leaves."}) if q.id == "q001" else q
                    for q in ready_queries]
        save_test_set(modified, testset_config["path"])
        main(["summary", "--config", self._write_config(tmp_path, testset_config)])
        out = capsys.readouterr().out
        assert "99" in out, "Number missing from the source docs must be listed"

    def test_summary_does_not_check_numbers_for_multi_chunk(self, tmp_path, ready_queries, testset_config, capsys):
        """Spec CLI: the number check covers lookup and paraphrased only."""
        modified = [q.model_copy(update={"reference_answer": "777 combined days."}) if q.id == "q004" else q
                    for q in ready_queries]
        save_test_set(modified, testset_config["path"])
        main(["summary", "--config", self._write_config(tmp_path, testset_config)])
        out = capsys.readouterr().out
        assert "777" not in out, "multi_chunk numbers are not part of the number check"

    def test_assign_splits_command_writes_splits(self, tmp_path, ready_queries, testset_config):
        """Spec CLI: assign-splits writes dev / heldout into the file, matching the config counts."""
        save_test_set(unsplit(ready_queries), testset_config["path"])
        code = main(["assign-splits", "--config", self._write_config(tmp_path, testset_config)])
        assert code == 0
        saved = load_test_set(testset_config["path"])
        assert counts(saved) == counts(ready_queries)
        assert check_ready(saved, testset_config) == []

    def test_assign_splits_command_refuses_second_run(self, tmp_path, ready_queries, testset_config):
        """Spec DoD: re-running assign-splits refuses because splits are already set; file unchanged."""
        save_test_set(unsplit(ready_queries), testset_config["path"])
        config_path = self._write_config(tmp_path, testset_config)
        assert main(["assign-splits", "--config", config_path]) == 0
        after_first = Path(testset_config["path"]).read_bytes()
        code = main(["assign-splits", "--config", config_path])
        assert code != 0, "Second run must fail"
        assert Path(testset_config["path"]).read_bytes() == after_first, "Second run must not change the file"


# ---------------------------------------------------------------- Real files (read-only)

NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def numbers_in(text):
    """Set of numbers in text, commas removed, kept as whole-number strings."""
    return {m.replace(",", "") for m in NUMBER_RE.findall(text)}


class TestRealFiles:
    def test_number_helper_treats_5_and_15_as_different(self):
        """Spec: numbers are compared as whole numbers, commas removed on both sides."""
        assert "5" not in numbers_in("You get 15 days"), "5 must not match inside 15"
        assert numbers_in("Fee is 1,500.50 and 12 days") == {"1500.50", "12"}
        assert numbers_in("Rs 10,000") == numbers_in("Rs 10000") == {"10000"}

    def test_real_test_set_loads(self):
        """Spec Tests (reads committed data/test_set.jsonl read-only): the file loads with no error."""
        queries = load_test_set(REAL_TEST_SET)
        assert len(queries) == 100, f"Expected 100 queries, got {len(queries)}"

    def test_real_test_set_ready(self):
        """Spec Tests (reads data/test_set.jsonl read-only): check_ready is [] after review and assign-splits.

        Expected to fail until Gaurav has reviewed every query and run assign-splits.
        """
        config = load_testset_config(REAL_CONFIG)
        config = {**config, "docs_dir": str(PROJECT_ROOT / config["docs_dir"])}
        problems = check_ready(load_test_set(REAL_TEST_SET), config)
        assert problems == [], f"{len(problems)} problems, first ones:\n" + "\n".join(problems[:10])

    def test_real_numbers_in_sources(self):
        """Spec Tests (reads data/test_set.jsonl and data/docs read-only): lookup/paraphrased numbers are in sources."""
        bad = []
        for query in load_test_set(REAL_TEST_SET):
            if query.type not in ("lookup", "paraphrased"):
                continue
            source_text = " ".join(
                (REAL_DOCS / f"{doc_id}.md").read_text(encoding="utf-8") for doc_id in query.source_doc_ids)
            missing = numbers_in(query.reference_answer) - numbers_in(source_text)
            if missing:
                bad.append(f"{query.id}: {sorted(missing)} not in {query.source_doc_ids}")
        assert bad == [], "Numbers missing from source docs:\n" + "\n".join(bad)

    def test_real_every_doc_used(self):
        """Spec Tests (reads data/test_set.jsonl and data/docs read-only): every corpus doc is a source at least once."""
        all_docs = {p.stem for p in REAL_DOCS.glob("*.md")}
        assert len(all_docs) == 22, f"Expected 22 corpus docs, found {len(all_docs)}"
        used = {d for q in load_test_set(REAL_TEST_SET) for d in q.source_doc_ids}
        assert all_docs - used == set(), f"Docs never used as a source: {sorted(all_docs - used)}"


# ---------------------------------------------------------------- Architecture

@pytest.mark.parametrize("module", ["schemas.py", "testset.py"])
def test_testset_does_not_import_rag_or_llm(module):
    """Spec: evalharness/schemas.py and testset.py import neither rag nor llm."""
    path = PROJECT_ROOT / "evalharness" / module
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        assert not any(n.split(".")[0] in ("rag", "llm") for n in names), f"{path} imports rag or llm: {names}"
