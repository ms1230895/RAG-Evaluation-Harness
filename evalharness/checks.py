"""Deterministic checks and retrieval metrics for a stored run (Step 6).

Reads a completed run's outputs (Step 5) and the queries' types and source
docs, and writes one `scores` row per (query, metric). No LLM is called and
answer() is never called again: the same stored run always gives the same rows.

Metrics per output (with hit_ks [1, 4]):
    empty_answer     1 if the answer is blank
    refusal          1 if the answer contains a configured refusal phrase (a heuristic)
    latency_ms       copied from outputs.latency_ms
    hit@1, hit@4     1 if a source doc's chunk is in the first k chunks
    in_context       1 if a source doc's chunk is anywhere in the retrieved chunks
    reciprocal_rank  1 / rank of the first source chunk, 0 if none
    doc_recall       share of the query's source docs that were retrieved
value NULL means "not applicable" (failed output, unanswerable query, fewer
than k chunks); the reasoning says which. It is never stored as 0 or 1.

The stored version is computed from the settings ("checks-1-<hash>"), so a
changed phrase list or k list can never be mixed up with an older one.

CLI:
    python -m evalharness.checks --run-id 20261015T152638Z-v0-c7c96975
Exit codes: 0 scored (or already scored, unchanged) and reported; 1 refused.
"""

import argparse
import hashlib
import json
import sys
from contextlib import closing

from evalharness import store
from evalharness.metrics import (doc_recall, first_source_rank, hit_at_k, mean_summary,
                                 percentile_summary, rate_summary, reciprocal_rank)
from evalharness.runner import DEFAULT_HARNESS_CONFIG, load_harness_config
from evalharness.schemas import QUERY_TYPES

CHECKS_CODE_VERSION = 1   # bump when the scoring logic (not the settings) changes
CHECKS_KEYS = {"hit_ks", "refusal_phrases", "report_confidence"}
ANSWERABLE_TYPES = tuple(t for t in QUERY_TYPES if t != "unanswerable")

# Curly quotes as LLMs often write them, mapped to plain ASCII.
QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


class ScoringRefused(Exception):
    """The run cannot be scored (unknown, not completed, held-out, or a version conflict)."""


# ---------------------------------------------------------------- config

def _is_int(value) -> bool:
    """True for a real int (bool is a subclass of int in Python, so exclude it)."""
    return isinstance(value, int) and not isinstance(value, bool)


def normalize_text(text: str) -> str:
    """Lowercase, curly quotes to straight, every whitespace run to one space, stripped."""
    return " ".join(text.translate(QUOTE_MAP).lower().split())


def load_checks_config(harness_config: dict) -> dict:
    """Check the `checks` block of the harness config and return it with its "version".

    version = "checks-<CHECKS_CODE_VERSION>-<first 8 hex of sha256(hit_ks + refusal_phrases)>".
    report_confidence is not in the hash: it changes the report, not the stored values.
    """
    checks = harness_config.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("harness config: 'checks' must be a mapping")
    if set(checks) != CHECKS_KEYS:
        raise ValueError(f"harness config: 'checks' keys are {sorted(checks)}, "
                         f"expected {sorted(CHECKS_KEYS)}")

    hit_ks = checks["hit_ks"]
    if (not isinstance(hit_ks, list) or not hit_ks
            or not all(_is_int(k) and k >= 1 for k in hit_ks)):
        raise ValueError("harness config: 'checks.hit_ks' must be a non-empty list of integers >= 1")
    if hit_ks != sorted(set(hit_ks)):
        raise ValueError("harness config: 'checks.hit_ks' must be sorted with no duplicates")

    phrases = checks["refusal_phrases"]
    if not isinstance(phrases, list) or not phrases:
        raise ValueError("harness config: 'checks.refusal_phrases' must be a non-empty list")
    for phrase in phrases:
        # Phrases are matched against normalize_text(answer), so they must already be normalised.
        if not isinstance(phrase, str) or not phrase.strip() or normalize_text(phrase) != phrase:
            raise ValueError(f"harness config: refusal phrase {phrase!r} must be non-empty, "
                             "lowercase, straight quotes, single spaces")
    if len(phrases) != len(set(phrases)):
        raise ValueError("harness config: 'checks.refusal_phrases' has duplicates")

    confidence = checks["report_confidence"]
    if not isinstance(confidence, float) or not 0 < confidence < 1:
        raise ValueError("harness config: 'checks.report_confidence' must be a number "
                         "between 0 and 1, e.g. 0.95")

    settings = json.dumps({"hit_ks": hit_ks, "refusal_phrases": phrases}, sort_keys=True)
    digest = hashlib.sha256(settings.encode("utf-8")).hexdigest()[:8]
    return {**checks, "version": f"checks-{CHECKS_CODE_VERSION}-{digest}"}


# ---------------------------------------------------------------- per-output checks

def is_empty_answer(answer: str) -> bool:
    """True if the answer has no visible characters."""
    return answer.strip() == ""


def find_refusal_phrase(answer: str, phrases: list[str]) -> str | None:
    """The first configured phrase found in the normalised answer, or None."""
    text = normalize_text(answer)
    for phrase in phrases:
        if phrase in text:
            return phrase
    return None


def metric_names(checks_config: dict) -> list[str]:
    """Every metric stored per output, in a fixed order."""
    return (["empty_answer", "refusal", "latency_ms"]
            + [f"hit@{k}" for k in checks_config["hit_ks"]]
            + ["in_context", "reciprocal_rank", "doc_recall"])


def _retrieval_rows(chunks: list[dict], source_doc_ids: list[str], hit_ks: list[int]) -> list[dict]:
    """hit@k, in_context, reciprocal_rank and doc_recall rows for an answerable query."""
    chunk_doc_ids = [chunk["doc_id"] for chunk in chunks]
    n_chunks = len(chunk_doc_ids)
    rank = first_source_rank(chunk_doc_ids, source_doc_ids)
    if rank is None:
        where = f"no source doc in {n_chunks} retrieved chunks"
    else:
        where = f"first source chunk at rank {rank} ({chunks[rank - 1]['chunk_id']})"

    rows = []
    for k in hit_ks:
        if k > n_chunks:
            rows.append({"metric": f"hit@{k}", "value": None,
                         "reasoning": f"only {n_chunks} chunks retrieved, k={k}"})
        else:
            rows.append({"metric": f"hit@{k}", "value": float(hit_at_k(chunk_doc_ids,
                                                                       source_doc_ids, k)),
                         "reasoning": where})
    rows.append({"metric": "in_context", "value": float(rank is not None), "reasoning": where})
    rows.append({"metric": "reciprocal_rank",
                 "value": reciprocal_rank(chunk_doc_ids, source_doc_ids),
                 "reasoning": f"rank {rank} -> 1/{rank}" if rank else "no source doc retrieved"})
    found = sorted(set(source_doc_ids) & set(chunk_doc_ids))
    rows.append({"metric": "doc_recall", "value": doc_recall(chunk_doc_ids, source_doc_ids),
                 "reasoning": f"{len(found)} of {len(set(source_doc_ids))} source docs retrieved: "
                              f"{', '.join(found) or 'none'}"})
    return rows


def score_output(output: dict, query: dict, checks_config: dict) -> list[dict]:
    """All metric rows for one output: [{"metric", "value", "reasoning"}, ...].

    `output` is a get_outputs() row; `query` needs "source_doc_ids". One row per
    metric in metric_names(), always, so coverage is easy to check with SQL.
    """
    names = metric_names(checks_config)
    if output["error"] is not None:
        # A failed output has no answer and no chunks: every metric is not applicable.
        reason = f"output failed: {output['error']}"
        return [{"metric": name, "value": None, "reasoning": reason} for name in names]

    answer = output["answer"]
    rows = []
    if is_empty_answer(answer):
        rows.append({"metric": "empty_answer", "value": 1.0, "reasoning": "answer is blank"})
        rows.append({"metric": "refusal", "value": 0.0,
                     "reasoning": "answer is blank (counted under empty_answer)"})
    else:
        rows.append({"metric": "empty_answer", "value": 0.0,
                     "reasoning": f"answer has {len(answer)} chars"})
        phrase = find_refusal_phrase(answer, checks_config["refusal_phrases"])
        rows.append({"metric": "refusal", "value": float(phrase is not None),
                     "reasoning": f"matched {phrase!r}" if phrase else "no refusal phrase"})
    rows.append({"metric": "latency_ms", "value": float(output["latency_ms"]),
                 "reasoning": "from outputs.latency_ms"})

    if not query["source_doc_ids"]:
        reason = "unanswerable: no source docs"
        rows += [{"metric": name, "value": None, "reasoning": reason}
                 for name in names if name not in ("empty_answer", "refusal", "latency_ms")]
    else:
        rows += _retrieval_rows(output["retrieved_chunks"], query["source_doc_ids"],
                                checks_config["hit_ks"])
    return rows


# ---------------------------------------------------------------- scoring a run

def get_scorable_run(conn, run_id: str, *, allow_heldout: bool = False) -> dict:
    """The runs row, if this run may be scored; else raise ScoringRefused.

    Shared by the checks and the LLM judges: the run must exist, be
    'completed', and not be held-out unless allow_heldout (Step 12 only).
    """
    run = store.get_run(conn, run_id)
    if run is None:
        raise ScoringRefused(f"no run {run_id!r} in the results database")
    if run["status"] != "completed":
        raise ScoringRefused(f"run {run_id} has status {run['status']!r}; only completed runs "
                             "are scored")
    if run["split"] == "heldout" and not allow_heldout:
        raise ScoringRefused("run is on the held-out split, which is used once, in Step 12; "
                             "pass --allow-heldout only then")
    return run


def score_run(conn, run_id: str, checks_config: dict, *, allow_heldout: bool = False) -> str:
    """Score every output of a completed run; return "written" or "unchanged".

    Rows are written in one transaction. If rows for this version already
    exist they must equal the new ones (scored_at aside), otherwise the scoring
    changed without a version change and we refuse instead of overwriting.
    """
    get_scorable_run(conn, run_id, allow_heldout=allow_heldout)

    version = checks_config["version"]
    rows = []
    for output in store.get_outputs_with_queries(conn, run_id):
        for row in score_output(output, output, checks_config):
            rows.append({"query_id": output["query_id"], **row})

    stored = store.get_scores(conn, run_id, version)
    if not stored:
        store.write_scores(conn, run_id, version, rows)
        return "written"

    def as_map(score_rows: list[dict]) -> dict:
        """{(query_id, metric): (value, reasoning)}; scored_at is left out on purpose."""
        return {(r["query_id"], r["metric"]): (r["value"], r["reasoning"]) for r in score_rows}
    if as_map(stored) == as_map(rows):
        return "unchanged"
    raise ScoringRefused(f"run {run_id} already has {version} scores that differ from a fresh "
                         "scoring; the scoring logic changed, so bump CHECKS_CODE_VERSION")


# ---------------------------------------------------------------- report

def _latency_source(run: dict) -> str:
    """Where the latencies came from, per model, from runs.llm_calls_json."""
    summary = json.loads(run["llm_calls_json"] or "{}")
    if not summary:
        return "no LLM calls recorded for this run"
    parts = [f"{model}: {c['misses']} real calls, {c['hits']} cache hits, {c['errors']} errors"
             for model, c in summary.items()]
    text = "; ".join(parts)
    if any(c["hits"] for c in summary.values()):
        text += (" (cache hits return the model latency of the original call, so these "
                 "latencies repeat an earlier run)")
    return text


def build_report(conn, run_id: str, version: str, confidence: float) -> dict:
    """Summaries read back from `scores` (never recomputed in memory), by group.

    Groups: "all", "answerable" (every type but unanswerable) and each query type.
    """
    run = store.get_run(conn, run_id)
    outputs = store.get_outputs_with_queries(conn, run_id)
    values: dict[str, dict[str, float | None]] = {}
    for row in store.get_scores(conn, run_id, version):
        values.setdefault(row["query_id"], {})[row["metric"]] = row["value"]
    metrics = sorted({m for per_query in values.values() for m in per_query})
    hit_metrics = sorted((m for m in metrics if m.startswith("hit@")),
                         key=lambda m: int(m.split("@")[1]))

    group_types = {"all": QUERY_TYPES, "answerable": ANSWERABLE_TYPES,
                   **{t: (t,) for t in QUERY_TYPES}}
    groups = {}
    for name, types in group_types.items():
        members = [o for o in outputs if o["type"] in types]

        def column(metric: str) -> list[float | None]:
            return [values.get(o["query_id"], {}).get(metric) for o in members]

        groups[name] = {
            "outputs": len(members),
            "ok": sum(o["error"] is None for o in members),
            "failed": sum(o["error"] is not None for o in members),
            "empty_answer": rate_summary(column("empty_answer"), confidence),
            "refusal": rate_summary(column("refusal"), confidence),
            "latency_ms": percentile_summary(column("latency_ms")),
            **{m: rate_summary(column(m), confidence) for m in hit_metrics + ["in_context"]},
            "reciprocal_rank": mean_summary(column("reciprocal_rank")),
            "doc_recall": mean_summary(column("doc_recall")),
        }

    def ids(condition) -> list[str]:
        return [o["query_id"] for o in outputs if condition(o, values.get(o["query_id"], {}))]

    return {
        "run_id": run_id, "split": run["split"], "status": run["status"],
        "config_path": run["config_path"], "version": version, "confidence": confidence,
        "hit_metrics": hit_metrics, "n_scores": sum(len(v) for v in values.values()),
        "latency_source": _latency_source(run),
        "groups": groups,
        "retrieval_misses": ids(lambda o, v: v.get("in_context") == 0),
        "correct_refusals": ids(lambda o, v: o["type"] == "unanswerable" and v.get("refusal") == 1),
        "unanswerable_not_refused": ids(lambda o, v: o["type"] == "unanswerable"
                                        and v.get("refusal") == 0),
        "false_refusals": ids(lambda o, v: o["type"] != "unanswerable" and v.get("refusal") == 1),
        "failed": ids(lambda o, v: o["error"] is not None),
    }


def _fmt_rate(s: dict) -> str:
    """'0.750  9/12  [0.468, 0.911]  n/a 0', or '-' when nothing was applicable."""
    if s["rate"] is None:
        return f"{'-':<30}n/a {s['n_na']}"
    text = f"{s['rate']:.3f}  {s['successes']}/{s['n']}  [{s['ci_low']:.3f}, {s['ci_high']:.3f}]"
    return f"{text:<30}n/a {s['n_na']}"


def _fmt_mean(s: dict) -> str:
    if s["mean"] is None:
        return f"{'-':<30}n/a {s['n_na']}"
    return f"{s['mean']:.3f}  n={s['n']:<22}n/a {s['n_na']}"


def _fmt_list(ids: list[str]) -> str:
    return f"{len(ids)}: {', '.join(ids)}" if ids else "0"


def format_report(report: dict) -> str:
    """Plain-text report; every rate shows successes/n and its Wilson interval."""
    g = report["groups"]
    pct = round(report["confidence"] * 100)
    types = list(QUERY_TYPES)
    lines = [
        f"Run {report['run_id']}  split {report['split']}  status {report['status']}  "
        f"config {report['config_path']}",
        f"Scores version {report['version']}  ({report['n_scores']} rows; rates as "
        f"rate  successes/n  [{pct}% Wilson CI]; n/a = not applicable or failed)",
        "",
        f"{'group':<14}{'outputs':>8}{'ok':>6}{'failed':>8}",
    ]
    for name in ["all", *types]:
        lines.append(f"{name:<14}{g[name]['outputs']:>8}{g[name]['ok']:>6}{g[name]['failed']:>8}")

    def section(title: str, metric: str, group_names: list[str], fmt, labels=None) -> None:
        lines.extend(["", title])
        for name in group_names:
            label = labels.get(name, name) if labels else name
            lines.append(f"  {label:<30}{fmt(g[name][metric])}")

    section("Empty answer", "empty_answer", ["all", *types], _fmt_rate)
    section("Refusal (phrase match, not a judgement)", "refusal",
            ["answerable", *types], _fmt_rate,
            labels={"answerable": "answerable (false refusal)",
                    "unanswerable": "unanswerable (correct)",
                    **{t: f"{t} (false refusal)" for t in ANSWERABLE_TYPES}})
    lines.extend(["", "Latency ms (p50 / p95 / max)"])
    for name in ["all", *types]:
        s = g[name]["latency_ms"]
        if s["n"]:
            lines.append(f"  {name:<30}{s['p50']:.0f} / {s['p95']:.0f} / {s['max']:.0f}  n={s['n']}")
        else:
            lines.append(f"  {name:<30}-  n=0")
    lines.append(f"  source: {report['latency_source']}")

    retrieval_groups = ["answerable", *ANSWERABLE_TYPES]
    for metric in report["hit_metrics"]:
        section(f"Retrieval {metric}", metric, retrieval_groups, _fmt_rate)
    section("Retrieval in_context (a source doc anywhere in the retrieved chunks)", "in_context",
            retrieval_groups, _fmt_rate)
    section("MRR (mean reciprocal rank, only comparable at equal top_k)", "reciprocal_rank",
            retrieval_groups, _fmt_mean)
    section("Doc recall (share of source docs retrieved)", "doc_recall", retrieval_groups,
            _fmt_mean)
    unanswerable_n = g["unanswerable"]["outputs"]
    lines.extend([
        f"  (retrieval metrics skip the {unanswerable_n} unanswerable queries: no source docs)",
        "",
        f"Retrieval misses (in_context = 0): {_fmt_list(report['retrieval_misses'])}",
        f"Correct refusals (unanswerable):   {_fmt_list(report['correct_refusals'])}",
        f"Unanswerable, no refusal phrase:   {_fmt_list(report['unanswerable_not_refused'])}",
        f"False refusals (answerable):       {_fmt_list(report['false_refusals'])}",
        f"Failed outputs:                    {_fmt_list(report['failed'])}",
        "",
        "Note: per-type n is small (12-16 on dev); differences inside the CIs are not meaningful.",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evalharness.checks",
                                     description="Score a stored run with deterministic checks "
                                                 "and retrieval metrics, then print a report.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--harness-config", default=DEFAULT_HARNESS_CONFIG)
    parser.add_argument("--allow-heldout", action="store_true",
                        help="allow scoring a held-out run (Step 12 only)")
    args = parser.parse_args(argv)

    try:
        harness = load_harness_config(args.harness_config)
        checks_config = load_checks_config(harness)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    with closing(store.connect(harness["db_path"])) as conn:
        try:
            outcome = score_run(conn, args.run_id, checks_config, allow_heldout=args.allow_heldout)
        except ScoringRefused as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        n_rows = len(store.get_scores(conn, args.run_id, checks_config["version"]))
        report = build_report(conn, args.run_id, checks_config["version"],
                              checks_config["report_confidence"])

    n_outputs = report["groups"]["all"]["outputs"]
    print(f"scored run {args.run_id} with {checks_config['version']}: {outcome} | "
          f"{n_outputs} outputs x {len(metric_names(checks_config))} metrics = {n_rows} rows\n")
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
