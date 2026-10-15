"""Load, check, review and split the test set (Step 4).

The test set is data/test_set.jsonl: one JSON object per line, each a
`Query` (evalharness/schemas.py). This module:
- loads it strictly (line numbers in every error, no duplicate ids or questions),
- checks it is final (all reviewed, all split, counts exactly as configured),
- lets Gaurav review each query next to its source documents,
- assigns the stratified dev / heldout split once, with a fixed seed.

It knows nothing about the RAG pipeline: it imports neither `rag` nor `llm`,
and lists corpus doc IDs by reading filenames only.

CLI:
    python -m evalharness.testset summary         # exit 0 only when the set is ready
    python -m evalharness.testset review          # interactive review
    python -m evalharness.testset assign-splits   # once, after review
"""

import argparse
import json
import os
import random
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import yaml
from pydantic import ValidationError

from evalharness.schemas import QUERY_TYPES, SPLITS, Query

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "configs/test_set.yaml"
REQUIRED_CONFIG_KEYS = ("path", "docs_dir", "split_seed", "expected_counts")

# Fields the review tool lets Gaurav edit (id, split and reviewed are not editable).
EDITABLE_FIELDS = ("question", "reference_answer", "type", "source_doc_ids")

# A number as written in the docs: "8", "1,500", "2.5". Commas are removed
# before comparing, so "1,500" in an answer matches "1500" in a doc.
NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path: relative paths are relative to the repo root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


# ---------------------------------------------------------------- config


def load_testset_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Read configs/test_set.yaml and check its shape. ValueError names the problem."""
    path = resolve_path(path)
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    for key in REQUIRED_CONFIG_KEYS:
        if key not in config:
            raise ValueError(f"{path}: missing key '{key}'")
    for key in ("path", "docs_dir"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{path}: '{key}' must be a non-empty string")
    # bool is a subclass of int in Python, so exclude it explicitly.
    if not isinstance(config["split_seed"], int) or isinstance(config["split_seed"], bool):
        raise ValueError(f"{path}: 'split_seed' must be an integer")

    expected = config["expected_counts"]
    if not isinstance(expected, dict):
        raise ValueError(f"{path}: 'expected_counts' must be a mapping of type -> split -> count")
    for qtype in expected:
        if qtype not in QUERY_TYPES:
            raise ValueError(f"{path}: unknown query type '{qtype}' in expected_counts")
    for qtype in QUERY_TYPES:
        if qtype not in expected:
            raise ValueError(f"{path}: expected_counts is missing type '{qtype}'")
        per_split = expected[qtype]
        if not isinstance(per_split, dict) or set(per_split) != set(SPLITS):
            raise ValueError(f"{path}: expected_counts.{qtype} must have exactly the keys {list(SPLITS)}")
        for split, n in per_split.items():
            if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                raise ValueError(f"{path}: expected_counts.{qtype}.{split} must be an integer >= 0")
    return config


# ---------------------------------------------------------------- load / save


def _normalise_question(question: str) -> str:
    """Lowercase, whitespace collapsed: two questions that differ only in
    case or spacing count as duplicates."""
    return " ".join(question.lower().split())


def load_test_set(path: str | Path) -> list[Query]:
    """Parse the JSONL file into Query objects, in file order.

    Raises ValueError naming the line number on bad JSON, a validation error,
    a blank line (only one trailing newline is allowed), a duplicate id or a
    duplicate question.
    """
    path = resolve_path(path)
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # the file's final newline is not a blank line

    queries = []
    id_line = {}        # id -> line number where it first appeared
    question_line = {}  # normalised question -> line number
    for line_no, line in enumerate(lines, start=1):
        where = f"{path.name} line {line_no}"
        if not line.strip():
            raise ValueError(f"{where}: blank line")
        try:
            # Bad JSON and schema errors both raise ValidationError here.
            query = Query.model_validate_json(line)
        except ValidationError as error:
            raise ValueError(f"{where}: {error}") from error

        if query.id in id_line:
            raise ValueError(f"{where}: duplicate id {query.id} (first on line {id_line[query.id]})")
        key = _normalise_question(query.question)
        if key in question_line:
            raise ValueError(f"{where}: duplicate question (first on line {question_line[key]})")
        id_line[query.id] = line_no
        question_line[key] = line_no
        queries.append(query)
    return queries


def save_test_set(queries: list[Query], path: str | Path) -> None:
    """Write the queries as JSONL, sorted by id, fields in schema order.

    Writes to a temp file in the same folder, then os.replace() swaps it in,
    so a crash mid-write never leaves a half-written test set.
    """
    path = resolve_path(path)
    ids = [q.id for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("cannot save: duplicate query ids")

    ordered = sorted(queries, key=lambda q: q.id)
    text = "".join(json.dumps(q.model_dump(), ensure_ascii=False) + "\n" for q in ordered)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------- checks


def corpus_doc_ids(docs_dir: str | Path) -> set[str]:
    """Doc IDs in the corpus = stems of the *.md files (contents are not read)."""
    docs_dir = resolve_path(docs_dir)
    if not docs_dir.is_dir():
        raise ValueError(f"docs_dir {docs_dir} is not a directory")
    return {p.stem for p in docs_dir.glob("*.md")}


def check_doc_ids(queries: list[Query], known: set[str]) -> list[str]:
    """One problem per source doc ID that is not in the corpus."""
    problems = []
    for query in sorted(queries, key=lambda q: q.id):
        for doc_id in query.source_doc_ids:
            if doc_id not in known:
                problems.append(f"{query.id}: unknown source doc '{doc_id}'")
    return problems


def counts(queries: list[Query]) -> dict[str, dict[str, int]]:
    """{type: {"dev": n, "heldout": n, "unassigned": n}}, every type present."""
    table = {qtype: {"dev": 0, "heldout": 0, "unassigned": 0} for qtype in QUERY_TYPES}
    for query in queries:
        table[query.type][query.split or "unassigned"] += 1
    return table


def check_ready(queries: list[Query], config: dict) -> list[str]:
    """Every reason the test set is not final yet. [] means ready.

    Type totals are always compared with the config. The per-split counts
    are compared only once every query has a split (before assign-splits
    they are all "unassigned", and listing 8 mismatches would be noise).
    """
    problems = []
    ordered = sorted(queries, key=lambda q: q.id)

    unreviewed = [q.id for q in ordered if not q.reviewed]
    if unreviewed:
        problems.append(f"{len(unreviewed)} not reviewed: {', '.join(unreviewed)}")
    unsplit = [q.id for q in ordered if q.split is None]
    if unsplit:
        problems.append(f"{len(unsplit)} without a split: {', '.join(unsplit)}")

    table = counts(queries)
    expected = config["expected_counts"]
    for qtype in QUERY_TYPES:
        have = sum(table[qtype].values())
        want = sum(expected[qtype].values())
        if have != want:
            problems.append(f"{qtype}: {have} queries, expected {want}")
        elif not unsplit:
            for split in SPLITS:
                if table[qtype][split] != expected[qtype][split]:
                    problems.append(
                        f"{qtype} {split}: {table[qtype][split]} queries, "
                        f"expected {expected[qtype][split]}"
                    )

    problems += check_doc_ids(queries, corpus_doc_ids(config["docs_dir"]))
    return problems


def filter_split(queries: list[Query], split: str) -> list[Query]:
    """The queries of one split, in id order."""
    if split not in SPLITS:
        raise ValueError(f"unknown split '{split}', expected one of {list(SPLITS)}")
    return sorted((q for q in queries if q.split == split), key=lambda q: q.id)


def numbers_not_in_sources(query: Query, docs_dir: str | Path) -> list[str]:
    """Numbers in the reference answer that appear nowhere in its source docs.

    A review aid for lookup and paraphrased queries, whose answers should
    copy the doc's numbers exactly. Numbers are compared as whole numbers,
    so "5" in an answer does not match "15" in a doc.
    """
    docs_dir = resolve_path(docs_dir)
    source_text = " ".join(
        (docs_dir / f"{doc_id}.md").read_text(encoding="utf-8") for doc_id in query.source_doc_ids
    )
    in_sources = {n.replace(",", "") for n in NUMBER_PATTERN.findall(source_text)}
    in_answer = [n.replace(",", "") for n in NUMBER_PATTERN.findall(query.reference_answer)]
    return [n for n in in_answer if n not in in_sources]


def grep_corpus(word: str, docs_dir: str | Path) -> list[str]:
    """Every corpus line containing `word` (whole word, any case), as
    "<doc_id>:<line>: <text>". Used to confirm a topic is absent."""
    docs_dir = resolve_path(docs_dir)
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    hits = []
    for path in sorted(docs_dir.glob("*.md")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"{path.stem}:{line_no}: {line.strip()}")
    return hits


# ---------------------------------------------------------------- split


def assign_splits(queries: list[Query], config: dict) -> list[Query]:
    """Assign dev / heldout once, stratified by type, with config["split_seed"].

    For each type (in QUERY_TYPES order), the ids are sorted and
    random.Random(split_seed).sample() picks expected_counts[type]["heldout"]
    of them as heldout; the rest are dev. Nobody hand-picks held-out queries,
    and the same file and seed always give the same split.
    """
    already = [q.id for q in queries if q.split is not None]
    if already:
        raise ValueError(
            f"splits are already assigned ({len(already)} queries); "
            "they are assigned once and never re-rolled"
        )
    unreviewed = [q.id for q in queries if not q.reviewed]
    if unreviewed:
        raise ValueError(f"review every query first; not reviewed: {', '.join(sorted(unreviewed))}")

    table = counts(queries)
    expected = config["expected_counts"]
    for qtype in QUERY_TYPES:
        have = sum(table[qtype].values())
        want = sum(expected[qtype].values())
        if have != want:
            raise ValueError(f"{qtype}: {have} queries, config expects {want}")

    rng = random.Random(config["split_seed"])
    heldout_ids = set()
    for qtype in QUERY_TYPES:
        ids = sorted(q.id for q in queries if q.type == qtype)
        heldout_ids.update(rng.sample(ids, expected[qtype]["heldout"]))

    return [
        q.model_copy(update={"split": "heldout" if q.id in heldout_ids else "dev"})
        for q in sorted(queries, key=lambda q: q.id)
    ]


# ---------------------------------------------------------------- review


def _show(query: Query, docs_dir: Path) -> None:
    """Print one query and the full text of each of its source docs."""
    print("\n" + "=" * 78)
    print(f"{query.id}  [{query.type}]")
    print(f"Question:         {query.question}")
    print(f"Reference answer: {query.reference_answer}")
    print(f"Source docs:      {', '.join(query.source_doc_ids) or '(none)'}")
    if query.type == "unanswerable":
        print("\nNo source docs. Use [g]rep to check the topic is absent from the corpus.")
        return
    if query.type in ("lookup", "paraphrased"):
        missing = numbers_not_in_sources(query, docs_dir)
        if missing:
            print(f"WARNING: numbers not found in the source docs: {', '.join(missing)}")
    for doc_id in query.source_doc_ids:
        print(f"\n----- {doc_id} " + "-" * max(0, 70 - len(doc_id)))
        print((docs_dir / f"{doc_id}.md").read_text(encoding="utf-8").rstrip())


def _edit(query: Query, input_fn) -> Query | None:
    """Ask for one field and its new value; return the re-validated query,
    or None if the edit is rejected.

    source_doc_ids is typed comma-separated. Editing `type` also asks for
    source_doc_ids (Enter keeps them), because changing a lookup to
    unanswerable needs both changed at once to stay valid.
    """
    field = input_fn(f"field to edit ({', '.join(EDITABLE_FIELDS)}) > ").strip()
    if field not in EDITABLE_FIELDS:
        print(f"Not an editable field: {field!r}")
        return None

    data = query.model_dump()
    print(f"Current {field}: {data[field]}")
    raw = input_fn("new value > ")
    if field == "source_doc_ids":
        data[field] = [d.strip() for d in raw.split(",") if d.strip()]
    else:
        data[field] = raw.strip()
    if field == "type":
        raw_docs = input_fn("source_doc_ids, comma-separated (Enter keeps current) > ")
        if raw_docs.strip():
            data["source_doc_ids"] = [d.strip() for d in raw_docs.split(",") if d.strip()]

    try:
        return Query.model_validate(data)
    except ValidationError as error:
        print(f"Edit rejected, query unchanged:\n{error}")
        return None


def review(queries: list[Query], path: str | Path, docs_dir: str | Path, input_fn=input) -> list[Query]:
    """Walk through every unreviewed query, in id order.

    Keys: [a]ccept (sets reviewed=True), [e]dit, [d]elete (the id is retired),
    [g]rep (search the corpus for a word), [s]kip, [q]uit.
    The file is saved after every change, so quitting loses nothing.
    Only a key pressed here sets reviewed=True; nothing else in the code does.
    """
    docs_dir = resolve_path(docs_dir)
    by_id = {q.id: q for q in queries}
    pending = sorted(q.id for q in queries if not q.reviewed)
    print(f"{len(pending)} queries to review.")

    def save() -> None:
        save_test_set(list(by_id.values()), path)

    for qid in pending:
        show = True
        while True:
            if show:
                _show(by_id[qid], docs_dir)
                show = False
            try:
                key = input_fn("\n[a]ccept [e]dit [d]elete [g]rep [s]kip [q]uit > ").strip().lower()
            except EOFError:  # Ctrl-D: treat as quit
                key = "q"

            if key == "a":
                by_id[qid] = by_id[qid].model_copy(update={"reviewed": True})
                save()
                print(f"{qid} accepted.")
                break
            elif key == "e":
                edited = _edit(by_id[qid], input_fn)
                if edited is not None:
                    by_id[qid] = edited
                    save()
                    show = True  # show the edited query again before accepting
            elif key == "d":
                del by_id[qid]
                save()
                print(f"{qid} deleted (its id is retired).")
                break
            elif key == "g":
                word = input_fn("word to search > ").strip()
                if word:
                    hits = grep_corpus(word, docs_dir)
                    print("\n".join(hits) if hits else f"No corpus line contains '{word}'.")
            elif key == "s":
                break
            elif key == "q":
                print("Quit. Everything done so far is saved.")
                return sorted(by_id.values(), key=lambda q: q.id)
            else:
                print(f"Unknown key: {key!r}")

    return sorted(by_id.values(), key=lambda q: q.id)


# ---------------------------------------------------------------- CLI


def summary(config: dict) -> int:
    """Print the counts table, review progress, per-doc coverage and every
    problem. Returns 0 if the test set is ready, else 1."""
    queries = load_test_set(config["path"])
    table = counts(queries)

    print(f"{len(queries)} queries in {config['path']}\n")
    print(f"{'type':<14}{'dev':>6}{'heldout':>9}{'unassigned':>12}{'total':>7}")
    for qtype in QUERY_TYPES:
        row = table[qtype]
        print(f"{qtype:<14}{row['dev']:>6}{row['heldout']:>9}{row['unassigned']:>12}{sum(row.values()):>7}")
    col = {key: sum(table[t][key] for t in QUERY_TYPES) for key in ("dev", "heldout", "unassigned")}
    print(f"{'all':<14}{col['dev']:>6}{col['heldout']:>9}{col['unassigned']:>12}{len(queries):>7}")

    reviewed = sum(q.reviewed for q in queries)
    print(f"\nreviewed {reviewed}/{len(queries)}")

    known = corpus_doc_ids(config["docs_dir"])
    per_doc = Counter(doc_id for q in queries for doc_id in q.source_doc_ids)
    print("\nqueries per source doc:")
    for doc_id in sorted(known):
        print(f"  {doc_id:<32}{per_doc[doc_id]:>3}")
    unused = sorted(d for d in known if per_doc[d] == 0)
    if unused:
        print(f"docs never used as a source: {', '.join(unused)}")

    number_notes = []
    for query in sorted(queries, key=lambda q: q.id):
        if query.type in ("lookup", "paraphrased") and set(query.source_doc_ids) <= known:
            missing = numbers_not_in_sources(query, config["docs_dir"])
            if missing:
                number_notes.append(f"  {query.id}: {', '.join(missing)}")
    if number_notes:
        print("\nnumbers in reference answers not found in their source docs:")
        print("\n".join(number_notes))

    problems = check_ready(queries, config)
    if problems:
        print("\nNOT READY:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nREADY: all reviewed, all split, counts as configured, doc IDs known.")
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=DEFAULT_CONFIG, help="test set config YAML")
    parser = argparse.ArgumentParser(prog="python -m evalharness.testset", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("summary", parents=[common], help="counts, review progress, problems")
    commands.add_parser("review", parents=[common], help="review unreviewed queries one by one")
    commands.add_parser("assign-splits", parents=[common], help="assign dev / heldout once")
    args = parser.parse_args(argv)

    try:
        config = load_testset_config(args.config)
        if args.command == "summary":
            return summary(config)
        if args.command == "review":
            queries = review(load_test_set(config["path"]), config["path"], config["docs_dir"])
            left = sum(not q.reviewed for q in queries)
            print(f"{left} queries still unreviewed.")
            return 0
        # assign-splits
        queries = assign_splits(load_test_set(config["path"]), config)
        save_test_set(queries, config["path"])
        heldout = filter_split(queries, "heldout")
        print(f"Assigned {len(queries) - len(heldout)} dev and {len(heldout)} heldout queries "
              f"(seed {config['split_seed']}).")
        return 0
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
