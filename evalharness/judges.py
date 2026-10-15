"""LLM judges for stored runs (Step 7: faithfulness).

The faithfulness judge asks: is every claim in the answer supported by the
chunks the generator actually saw? For each stored output it makes two cached
calls to the judge model (configs/judges.yaml, a different and larger model
than the generator):
  1. decompose: split the answer into atomic, self-contained claims
     (it sees the question and the answer, never the context),
  2. verify:    label each numbered claim against the stored chunks as
     supported / unsupported / contradicted, reasoning first.
Then, with no LLM:
    faithfulness = supported claims / total claims   (1.0 when there are no claims)
    ungrounded   = 1 if faithfulness < threshold or any claim is contradicted

Each output gets three `scores` rows (faithfulness, ungrounded, n_claims) and
one `judge_claims` row per claim, under a judge version like
"faith-v1-3fa1c27e". A failed output, a blank answer or a judge failure is
stored as value NULL with the reason, never as grounded or ungrounded.

The judge is NOT calibrated until Step 9: its ungrounded rate is the judge's
opinion, and every report says so.

CLI:
    python -m evalharness.judges faithfulness --run-id 20261015T152638Z-v0-c7c96975 [--query-ids q001,q053]
Exit codes: 0 complete; 1 refused (bad config, digest mismatch, run not
scorable); 2 partial (--query-ids) or aborted.
"""

import argparse
import hashlib
import json
import logging
import string
import sys
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import openai
import yaml
from tqdm import tqdm

from evalharness import store
from evalharness.checks import ANSWERABLE_TYPES, ScoringRefused, get_scorable_run, load_checks_config
from evalharness.metrics import mean_summary, rate_summary
from evalharness.runner import (DEFAULT_HARNESS_CONFIG, _format_calls, llm_calls_max_id,
                                llm_calls_summary, load_harness_config)
from evalharness.schemas import QUERY_TYPES, ClaimList, ClaimVerdict, ClaimVerdicts
from evalharness.testset import resolve_path
from llm.client import LLMOutputError, generate, get_model_digest
from llm.client import load_config as load_llm_config

FAITHFULNESS_METRICS = ("faithfulness", "ungrounded", "n_claims")
REASONING_EFFORTS = ("low", "medium", "high")   # gpt-oss thinking levels (ollama show)

# Every key of the `faithfulness` block in configs/judges.yaml.
FAITHFULNESS_KEYS = {
    "prompt_version", "model", "model_digest", "temperature", "seed", "reasoning_effort",
    "max_prompt_tokens", "threshold", "max_consecutive_failures", "context_chunk_template",
    "context_separator", "decompose_template", "verify_template",
}
# The exact {placeholders} each template must use (doubled braces {{ }} are literal text).
TEMPLATE_FIELDS = {
    "decompose_template": {"question", "answer"},
    "verify_template": {"context", "claims"},
    "context_chunk_template": {"rank", "text"},
}

# Reasoning prefixes of NULL rows; the report counts items by them.
NOT_JUDGED = "not judged: "
JUDGE_FAILED = "judge failed: "


class JudgeError(Exception):
    """A judgement could not be made for one item (bad verdicts, token guard, LLM failure)."""


class ModelDigestError(Exception):
    """The judge's pinned model_digest differs from what the server reports."""


@dataclass
class Judgement:
    """The judge's result for one answer."""
    claims: list[str]
    verdicts: list[ClaimVerdict]   # same length and order as claims
    faithfulness: float            # supported / len(claims); 1.0 when claims == []
    ungrounded: int                # 0 / 1
    reasoning: str                 # one-line summary, stored in scores.reasoning


# ---------------------------------------------------------------- config

def _is_int(value) -> bool:
    """True for a real int (bool is a subclass of int in Python, so exclude it)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _template_fields(template: str) -> set[str]:
    """The {placeholder} names in a str.format template."""
    return {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}


def _load_block(path: str | Path, name: str, keys: set[str]) -> tuple[dict, Path]:
    """The `name` block of a judges YAML file, after checking it has exactly `keys`."""
    path = resolve_path(path)
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    block = config.get(name) if isinstance(config, dict) else None
    if not isinstance(block, dict):
        raise ValueError(f"{path}: '{name}' must be a mapping")
    missing = keys - set(block)
    extra = set(block) - keys
    if missing or extra:
        raise ValueError(f"{path}: {name} keys missing {sorted(missing)}, "
                         f"unknown {sorted(extra)}")
    return block, path


def _check_common(block: dict, bad) -> None:
    """Checks every judge block shares: version, model identity and call settings.

    `bad(key, rule)` builds the ValueError that names the problem.
    """
    if not _is_int(block["prompt_version"]) or block["prompt_version"] < 1:
        raise bad("prompt_version", "must be an integer >= 1")
    for key in ("model", "model_digest"):
        if not isinstance(block[key], str) or not block[key].strip():
            raise bad(key, "must be a non-empty string")
    if isinstance(block["temperature"], bool) or block["temperature"] != 0:
        raise bad("temperature", "must be 0 (CLAUDE.md: judge temperature 0)")
    if not _is_int(block["seed"]):
        raise bad("seed", "must be an integer")
    if block["reasoning_effort"] not in REASONING_EFFORTS:
        raise bad("reasoning_effort", f"must be one of {REASONING_EFFORTS}")
    for key in ("max_prompt_tokens", "max_consecutive_failures"):
        if not _is_int(block[key]) or block[key] < 1:
            raise bad(key, "must be an integer >= 1")


def _check_templates(block: dict, template_fields: dict[str, set[str]], bad) -> None:
    """Each template must be a string using exactly its placeholder names."""
    for key, fields in template_fields.items():
        if not isinstance(block[key], str) or _template_fields(block[key]) != fields:
            raise bad(key, f"must be a string using exactly the placeholders {sorted(fields)}")


def _version(prefix: str, block: dict, schemas: list) -> str:
    """"<prefix>-v<prompt_version>-<first 8 hex of sha256(every other setting + schemas)>"."""
    settings = {key: value for key, value in block.items() if key != "prompt_version"}
    settings["schemas"] = [schema.model_json_schema() for schema in schemas]
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-v{block['prompt_version']}-{digest}"


def load_judges_config(path: str | Path) -> dict:
    """Read configs/judges.yaml, check the `faithfulness` block, return it with its "version".

    version = "faith-v<prompt_version>-<first 8 hex of sha256(every other setting
    + the two reply schemas)>", so a changed prompt, model, digest, seed, effort
    or threshold can never be mixed up with an older judgement, even if
    prompt_version was not bumped. ValueError names the problem.
    """
    block, path = _load_block(path, "faithfulness", FAITHFULNESS_KEYS)

    def bad(key: str, rule: str) -> ValueError:
        return ValueError(f"{path}: 'faithfulness.{key}' {rule}")

    _check_common(block, bad)
    if not isinstance(block["context_separator"], str):
        raise bad("context_separator", "must be a non-empty string")
    threshold = block["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
        raise bad("threshold", "must be a number in (0, 1]")
    _check_templates(block, TEMPLATE_FIELDS, bad)
    return {**block, "version": _version("faith", block, [ClaimList, ClaimVerdicts])}


# ---------------------------------------------------------------- prompts and calls

def format_context(chunks: list[dict], chunk_template: str, separator: str) -> str:
    """The retrieved chunks in stored rank order, e.g. "[1] <text>\\n\\n[2] <text>".

    Chunk text goes in as a format value, so braces inside it are safe.
    """
    return separator.join(chunk_template.format(rank=rank, text=chunk["text"])
                          for rank, chunk in enumerate(chunks, start=1))


def format_claims(claims: list[str]) -> str:
    """"1. <claim>\\n2. <claim>": the numbers the verdicts refer to."""
    return "\n".join(f"{index}. {claim}" for index, claim in enumerate(claims, start=1))


def _call(prompt: str, schema, cfg: dict):
    """One judge call through llm.client.generate (cached, retried, logged); returns the parsed reply.

    Any LLM failure, and a prompt at or over max_prompt_tokens (or with no
    reported prompt_tokens, so truncation cannot be ruled out), becomes a
    JudgeError for this item.
    """
    try:
        result = generate(prompt, cfg["model"], cfg["temperature"], schema,
                          model_digest=cfg["model_digest"], seed=cfg["seed"],
                          reasoning_effort=cfg["reasoning_effort"])
    except (LLMOutputError, openai.OpenAIError) as error:
        raise JudgeError(f"{type(error).__name__}: {error}") from error
    if result.prompt_tokens is None:
        raise JudgeError("server reported no prompt_tokens, so truncation cannot be ruled out")
    if result.prompt_tokens >= cfg["max_prompt_tokens"]:
        raise JudgeError(f"prompt_tokens={result.prompt_tokens} reached "
                         f"max_prompt_tokens={cfg['max_prompt_tokens']}")
    return result.parsed


def decompose(question: str, answer: str, cfg: dict) -> list[str]:
    """Call 1: the answer as a list of atomic, self-contained claims ([] for a pure refusal)."""
    prompt = cfg["decompose_template"].format(question=question, answer=answer)
    return _call(prompt, ClaimList, cfg).claims


def verify(claims: list[str], chunks: list[dict], cfg: dict) -> list[ClaimVerdict]:
    """Call 2: one verdict per claim against the stored chunks, sorted by claim_index.

    Raises JudgeError unless the claim_index values are exactly 1..len(claims),
    each once (a missing, extra or repeated verdict cannot be scored).
    """
    context = format_context(chunks, cfg["context_chunk_template"], cfg["context_separator"])
    prompt = cfg["verify_template"].format(context=context, claims=format_claims(claims))
    verdicts = _call(prompt, ClaimVerdicts, cfg).verdicts
    indices = sorted(v.claim_index for v in verdicts)
    if indices != list(range(1, len(claims) + 1)):
        raise JudgeError(f"verdict indices {indices} do not match claims 1..{len(claims)}")
    return sorted(verdicts, key=lambda v: v.claim_index)


# ---------------------------------------------------------------- scoring (no LLM)

def _claim_numbers(verdicts: list[ClaimVerdict], label: str) -> list[int]:
    return [v.claim_index for v in verdicts if v.label == label]


def score_claims(verdicts: list[ClaimVerdict], threshold: float) -> tuple[float, int, str]:
    """(faithfulness, ungrounded, reasoning) from the verdicts. Pure.

    [] -> (1.0, 0, "no factual claims"). Example reasoning:
    "2/3 claims supported (0.67 < 0.8); unsupported: claim 3".
    """
    if not verdicts:
        return 1.0, 0, "no factual claims"
    n = len(verdicts)
    supported = len(_claim_numbers(verdicts, "supported"))
    unsupported = _claim_numbers(verdicts, "unsupported")
    contradicted = _claim_numbers(verdicts, "contradicted")
    faithfulness = supported / n
    ungrounded = int(faithfulness < threshold or bool(contradicted))

    comparison = "<" if faithfulness < threshold else ">="
    parts = [f"{supported}/{n} claims supported ({faithfulness:.2f} {comparison} {threshold})"]
    if contradicted:
        parts.append("contradicted: claim " + ", ".join(map(str, contradicted)))
    if unsupported:
        parts.append("unsupported: claim " + ", ".join(map(str, unsupported)))
    return faithfulness, ungrounded, "; ".join(parts)


def judge_output(output: dict, cfg: dict) -> Judgement:
    """decompose -> verify -> score_claims for one stored output (verify skipped for 0 claims).

    `output` is a get_outputs_with_queries() row. Raises JudgeError on any failure.
    """
    claims = decompose(output["question"], output["answer"], cfg)
    verdicts = verify(claims, output["retrieved_chunks"], cfg) if claims else []
    faithfulness, ungrounded, reasoning = score_claims(verdicts, cfg["threshold"])
    return Judgement(claims=claims, verdicts=verdicts, faithfulness=faithfulness,
                     ungrounded=ungrounded, reasoning=reasoning)


def _judgement_rows(judgement: Judgement, threshold: float) -> tuple[list[dict], list[dict]]:
    """(the 3 score rows, the claim rows) for a successful judgement."""
    counts = Counter(v.label for v in judgement.verdicts)
    if not judgement.claims:
        why = "no factual claims"
    elif judgement.ungrounded:
        reasons = []
        if judgement.faithfulness < threshold:
            reasons.append(f"faithfulness {judgement.faithfulness:.2f} < {threshold}")
        if counts["contradicted"]:
            reasons.append("contradicted: claim " + ", ".join(
                map(str, _claim_numbers(judgement.verdicts, "contradicted"))))
        why = "; ".join(reasons)
    else:
        why = f"faithfulness {judgement.faithfulness:.2f} >= {threshold}, none contradicted"
    score_rows = [
        {"metric": "faithfulness", "value": judgement.faithfulness,
         "reasoning": judgement.reasoning},
        {"metric": "ungrounded", "value": float(judgement.ungrounded), "reasoning": why},
        {"metric": "n_claims", "value": float(len(judgement.claims)),
         "reasoning": f"{len(judgement.claims)} claims: {counts['supported']} supported, "
                      f"{counts['unsupported']} unsupported, {counts['contradicted']} contradicted"},
    ]
    claim_rows = [{"claim_index": v.claim_index, "claim": judgement.claims[v.claim_index - 1],
                   "label": v.label, "reasoning": v.reasoning} for v in judgement.verdicts]
    return score_rows, claim_rows


def _null_rows(reason: str, metrics: tuple[str, ...] = FAITHFULNESS_METRICS) -> list[dict]:
    """One score row per metric for an item with no judgement (value NULL, reason stored)."""
    return [{"metric": metric, "value": None, "reasoning": reason} for metric in metrics]


# ---------------------------------------------------------------- judging a run

def _stored_reasonings(conn, run_id: str, version: str, metric: str) -> dict[str, str]:
    """{query_id: reasoning} of `metric` for the items already judged under this version."""
    return {row["query_id"]: row["reasoning"] for row in store.get_scores(conn, run_id, version)
            if row["metric"] == metric}


def _judge_items(conn, run_id: str, version: str, outputs: list[dict], judge_one,
                 null_metrics: tuple[str, ...], *, retry_failed: bool,
                 query_ids: list[str] | None, error_max_chars: int,
                 max_consecutive_failures: int) -> dict:
    """The per-item loop every judge shares.

    judge_one(output) -> (score_rows, claim_rows), or raises JudgeError.
    Each item is stored (scores + claims, one transaction) as soon as it is
    judged, so a crash keeps finished items and a rerun resumes from the cache.
    - failed output or blank answer: NULL rows, "not judged: ..." (no LLM call)
    - judge failure: NULL rows, "judge failed: ..."; counted, never dropped
    - retry_failed: this version's "judge failed" items are deleted and judged again
    - query_ids: judge only these outputs (a smoke test on a chosen mix)
    Stops ("aborted") after max_consecutive_failures judge failures in a row.
    """
    if query_ids is not None:
        unknown = sorted(set(query_ids) - {o["query_id"] for o in outputs})
        if unknown:
            raise ScoringRefused(f"query ids not in run {run_id}: {', '.join(unknown)}")

    # The first metric of each judge marks an item as stored (all its rows are written together).
    marker = null_metrics[0]
    existing = _stored_reasonings(conn, run_id, version, marker)
    if retry_failed:
        for query_id, reasoning in existing.items():
            if reasoning.startswith(JUDGE_FAILED):
                store.delete_judgement(conn, run_id, version, query_id)
        existing = _stored_reasonings(conn, run_id, version, marker)

    counts = {"judged": 0, "judge_failed": 0, "not_judged": 0, "skipped_existing": 0}
    todo = [o for o in outputs if query_ids is None or o["query_id"] in query_ids]
    status, streak = None, 0

    def store_null(query_id: str, reason: str) -> None:
        store.write_judgement(conn, run_id, version, query_id,
                              _null_rows(reason[:error_max_chars], null_metrics), [])

    for output in tqdm(todo, desc=f"{version} {run_id}", unit="answer"):
        query_id = output["query_id"]
        if query_id in existing:
            counts["skipped_existing"] += 1
            continue
        if output["error"] is not None:
            store_null(query_id, f"{NOT_JUDGED}output failed: {output['error']}")
            counts["not_judged"] += 1
            continue
        if not output["answer"].strip():
            store_null(query_id, f"{NOT_JUDGED}blank answer")
            counts["not_judged"] += 1
            continue
        try:
            score_rows, claim_rows = judge_one(output)
        except JudgeError as error:
            store_null(query_id, f"{JUDGE_FAILED}{error}")
            counts["judge_failed"] += 1
            streak += 1
            if streak >= max_consecutive_failures:
                status = "aborted"
                break
            continue
        store.write_judgement(conn, run_id, version, query_id, score_rows, claim_rows)
        counts["judged"] += 1
        streak = 0

    if status is None:
        stored = _stored_reasonings(conn, run_id, version, marker)
        status = "complete" if all(o["query_id"] in stored for o in outputs) else "partial"
    return {"status": status, "n_outputs": len(outputs), **counts}


def judge_run(conn, run_id: str, cfg: dict, *, allow_heldout: bool = False,
              retry_failed: bool = False, query_ids: list[str] | None = None,
              error_max_chars: int = 500) -> dict:
    """Judge faithfulness for every output of a completed run not yet judged under this version.

    Rows per output: faithfulness, ungrounded, n_claims (+ one judge_claims row
    per claim). See _judge_items for resume, retry, query_ids and failures.
    """
    get_scorable_run(conn, run_id, allow_heldout=allow_heldout)
    outputs = store.get_outputs_with_queries(conn, run_id)

    def judge_one(output: dict) -> tuple[list[dict], list[dict]]:
        return _judgement_rows(judge_output(output, cfg), cfg["threshold"])

    return _judge_items(conn, run_id, cfg["version"], outputs, judge_one, FAITHFULNESS_METRICS,
                        retry_failed=retry_failed, query_ids=query_ids,
                        error_max_chars=error_max_chars,
                        max_consecutive_failures=cfg["max_consecutive_failures"])


# ---------------------------------------------------------------- report

def build_faithfulness_report(conn, run_id: str, version: str, confidence: float) -> dict:
    """Summaries read back from `scores` and `judge_claims` only, by group.

    Groups: "all", "answerable" (every type but unanswerable) and each query type.
    """
    outputs = store.get_outputs_with_queries(conn, run_id)
    scores: dict[str, dict[str, dict]] = {}
    for row in store.get_scores(conn, run_id, version):
        scores.setdefault(row["query_id"], {})[row["metric"]] = row
    labels: dict[str, Counter] = {}
    for row in store.get_judge_claims(conn, run_id, version):
        labels.setdefault(row["query_id"], Counter())[row["label"]] += 1

    def state(query_id: str) -> str:
        row = scores.get(query_id, {}).get("faithfulness")
        if row is None:
            return "missing"
        if row["reasoning"].startswith(JUDGE_FAILED):
            return "judge_failed"
        if row["reasoning"].startswith(NOT_JUDGED):
            return "not_judged"
        return "judged"

    def value(query_id: str, metric: str):
        row = scores.get(query_id, {}).get(metric)
        return row["value"] if row is not None else None

    group_types = {"all": QUERY_TYPES, "answerable": ANSWERABLE_TYPES,
                   **{t: (t,) for t in QUERY_TYPES}}
    groups = {}
    for name, types in group_types.items():
        ids = [o["query_id"] for o in outputs if o["type"] in types]
        states = Counter(state(q) for q in ids)
        label_totals = sum((labels.get(q, Counter()) for q in ids), Counter())
        groups[name] = {
            "outputs": len(ids),
            **{key: states[key] for key in ("judged", "judge_failed", "not_judged", "missing")},
            "ungrounded": rate_summary([value(q, "ungrounded") for q in ids], confidence),
            "faithfulness": mean_summary([value(q, "faithfulness") for q in ids]),
            "claims": sum(int(value(q, "n_claims") or 0) for q in ids),
            "labels": {label: label_totals[label]
                       for label in ("supported", "unsupported", "contradicted")},
            "zero_claim_answers": sum(value(q, "n_claims") == 0 for q in ids),
        }

    return {
        "run_id": run_id, "version": version, "confidence": confidence, "groups": groups,
        "ungrounded_ids": [o["query_id"] for o in outputs if value(o["query_id"], "ungrounded") == 1],
        "judge_failed_ids": [o["query_id"] for o in outputs if state(o["query_id"]) == "judge_failed"],
        "not_judged_ids": [o["query_id"] for o in outputs if state(o["query_id"]) == "not_judged"],
    }


def _fmt_rate(s: dict) -> str:
    """'0.250  3/12  [0.089, 0.532]', or '-' when nothing was judged."""
    if s["rate"] is None:
        return "-"
    return f"{s['rate']:.3f}  {s['successes']}/{s['n']}  [{s['ci_low']:.3f}, {s['ci_high']:.3f}]"


def _fmt_list(ids: list[str]) -> str:
    return f"{len(ids)}: {', '.join(ids)}" if ids else "0"


def format_faithfulness_report(report: dict) -> str:
    """Plain-text report; every rate shows successes/n and its Wilson interval."""
    g = report["groups"]
    pct = round(report["confidence"] * 100)
    names = ["all", "answerable", *QUERY_TYPES]
    lines = [
        f"Run {report['run_id']}  judge version {report['version']}",
        "",
        f"{'group':<14}{'outputs':>8}{'judged':>8}{'judge failed':>14}{'not judged':>12}"
        f"{'missing':>9}",
    ]
    for name in names:
        x = g[name]
        lines.append(f"{name:<14}{x['outputs']:>8}{x['judged']:>8}{x['judge_failed']:>14}"
                     f"{x['not_judged']:>12}{x['missing']:>9}")
    lines.extend(["", f"Ungrounded rate (uncalibrated judge, Step 9 pending; "
                      f"rate  ungrounded/judged  [{pct}% Wilson CI])"])
    for name in names:
        lines.append(f"  {name:<14}{_fmt_rate(g[name]['ungrounded'])}")
    lines.extend(["", "Mean faithfulness (supported claims / claims; 1.0 for 0 claims)"])
    for name in names:
        s = g[name]["faithfulness"]
        mean = "-" if s["mean"] is None else f"{s['mean']:.3f}"
        lines.append(f"  {name:<14}{mean}  n={s['n']}")
    lines.extend(["", "Claims (supported / unsupported / contradicted; answers with 0 claims)"])
    for name in names:
        x = g[name]
        lines.append(f"  {name:<14}{x['claims']:>4} claims: {x['labels']['supported']} / "
                     f"{x['labels']['unsupported']} / {x['labels']['contradicted']};  "
                     f"0-claim answers {x['zero_claim_answers']}")
    lines.extend([
        "",
        f"Ungrounded:   {_fmt_list(report['ungrounded_ids'])}",
        f"Judge failed: {_fmt_list(report['judge_failed_ids'])}",
        f"Not judged:   {_fmt_list(report['not_judged_ids'])}",
        "",
        "Note: the judge is not calibrated yet (Step 9). Per-type n is small (12-16 on dev);",
        "differences inside the CIs are not meaningful.",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evalharness.judges",
                                     description="Judge a stored run with an LLM judge.")
    sub = parser.add_subparsers(dest="judge", required=True)
    faith = sub.add_parser("faithfulness", help="claim-level faithfulness judge (Step 7)")
    faith.add_argument("--run-id", required=True)
    faith.add_argument("--query-ids", help="comma-separated ids to judge only those (smoke test)")
    faith.add_argument("--retry-failed", action="store_true",
                       help="judge again the items stored as 'judge failed'")
    faith.add_argument("--allow-heldout", action="store_true",
                       help="allow judging a held-out run (Step 12 only)")
    faith.add_argument("--harness-config", default=DEFAULT_HARNESS_CONFIG)
    args = parser.parse_args(argv)
    # Per-call INFO lines from llm.client would bury the progress bar; warnings still show.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    def refuse(problem: str) -> int:
        print(f"error: {problem}", file=sys.stderr)
        return 1

    try:
        harness = load_harness_config(args.harness_config)
        if not isinstance(harness.get("judges_config"), str):
            raise ValueError(f"{args.harness_config}: missing key 'judges_config'")
        cfg = load_judges_config(harness["judges_config"])
        confidence = load_checks_config(harness)["report_confidence"]
    except ValueError as error:
        return refuse(str(error))

    # Model identity: the pinned digest is part of the cache key, so a stale
    # one would mix verdicts from two model builds.
    try:
        live_digest = get_model_digest(cfg["model"])
    except OSError as error:
        return refuse(f"cannot reach the model server to check the judge digest: {error}")
    if live_digest != cfg["model_digest"]:
        return refuse(f"judge model_digest is {cfg['model_digest']!r} but the server reports "
                      f"{live_digest!r} for {cfg['model']}; update configs/judges.yaml if the "
                      "model was re-pulled")

    query_ids = ([q.strip() for q in args.query_ids.split(",") if q.strip()]
                 if args.query_ids else None)
    llm_config = load_llm_config()
    with closing(store.connect(harness["db_path"])) as conn:
        try:
            run = get_scorable_run(conn, args.run_id, allow_heldout=args.allow_heldout)
            generation = json.loads(run["config_json"]).get("generation")
            if isinstance(generation, dict) and generation.get("model") == cfg["model"]:
                return refuse(f"judge model {cfg['model']} is the run's generator model; "
                              "the judge must be a different model (CLAUDE.md)")
            start_id = llm_calls_max_id(llm_config)
            result = judge_run(conn, args.run_id, cfg, allow_heldout=args.allow_heldout,
                               retry_failed=args.retry_failed, query_ids=query_ids,
                               error_max_chars=harness["error_max_chars"])
        except ScoringRefused as error:
            return refuse(str(error))
        calls = llm_calls_summary(llm_config, start_id, llm_calls_max_id(llm_config))
        report = build_faithfulness_report(conn, args.run_id, cfg["version"], confidence)

    print(f"judged run {args.run_id} with {cfg['version']}: {result['status']} | "
          f"{result['n_outputs']} outputs: {result['judged']} judged, "
          f"{result['judge_failed']} judge failed, {result['not_judged']} not judged, "
          f"{result['skipped_existing']} skipped (already judged) | {_format_calls(calls)}\n")
    print(format_faithfulness_report(report))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
