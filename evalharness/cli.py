"""One command for a full evaluation (Step 14).

    python -m evalharness run   --config configs/<name>.yaml --split dev
    python -m evalharness score --run-id <id>

`run` calls the stage entry points that already exist, in order, and passes the
run id from one to the next:

    run (generate answers) -> checks -> faithfulness judge -> relevance judge -> gate

`score` does the same for a run that already exists (everything after `run`),
judging again only the items whose judging failed.

This module only orchestrates. It computes no metric: every number it prints is
returned by a stage or read back from the results database. It never accepts a
run, never changes a threshold or a judge version and never retries a generator
call. Before any LLM call it checks everything that can be checked without one
(preflight), so a stale judge digest does not waste the generation stage.

Exit codes of `run` and `score`:
    0  every stage succeeded and the gate passed (or --no-gate)
    1  every stage succeeded and the gate FAILED (ungrounded rate rose too much)
    2  preflight problem, a stage refused or ended partial / aborted, or the gate
       could not decide; the last lines name the stage and the command to resume
"""

import argparse
import json
import sys
from collections import Counter
from contextlib import closing

from evalharness import checks, judges, regression, runner, store
from evalharness.runner import DEFAULT_HARNESS_CONFIG, load_harness_config

AUTOMATION_KEYS = {"allowed_splits", "retry_failed_judges_in_score"}
SPLITS_NEVER_AUTOMATED = ("heldout",)   # held-out has its own guarded Step 12 commands


# ---------------------------------------------------------------- config

def load_automation_config(harness: dict) -> dict:
    """The checked `automation` block of the harness config; ValueError names the problem."""
    cfg = harness.get("automation")
    if not isinstance(cfg, dict):
        raise ValueError("harness config: 'automation' must be a mapping")
    missing, extra = AUTOMATION_KEYS - set(cfg), set(cfg) - AUTOMATION_KEYS
    if missing or extra:
        raise ValueError(f"harness config: automation keys missing {sorted(missing)}, "
                         f"unknown {sorted(extra)}")
    splits = cfg["allowed_splits"]
    if not isinstance(splits, list) or not splits or not all(isinstance(s, str) for s in splits):
        raise ValueError("harness config: 'automation.allowed_splits' must be a non-empty list")
    for split in SPLITS_NEVER_AUTOMATED:
        if split in splits:
            raise ValueError(f"harness config: '{split}' cannot be in automation.allowed_splits "
                             "(the held-out split is used once, in Step 12)")
    if not isinstance(cfg["retry_failed_judges_in_score"], bool):
        raise ValueError("harness config: 'automation.retry_failed_judges_in_score' must be "
                         "true or false")
    return cfg


# ---------------------------------------------------------------- preflight

def _judge_problems(harness: dict, generator_model: str | None) -> list[str]:
    """Judge configs load, judge != generator, the pinned digest matches the server."""
    problems = []
    cfgs = []
    for loader in (judges.load_judges_config, judges.load_relevance_config):
        try:
            cfgs.append(loader(harness["judges_config"]))
        except (ValueError, KeyError, OSError) as error:
            problems.append(f"judge config: {error}")
    for model, pinned in sorted({(c["model"], c["model_digest"]) for c in cfgs}):
        if generator_model is not None and model == generator_model:
            problems.append(f"judge model {model} is the generator model; the judge must be "
                            "a different model (CLAUDE.md)")
        try:
            live = judges.get_model_digest(model)
        except OSError as error:
            problems.append(f"cannot reach the model server to check the digest of {model}: {error}")
            continue
        if live != pinned:
            problems.append(f"judge model {model}: server digest {live!r} differs from the "
                            f"pinned {pinned!r} in configs/judges.yaml")
    return problems


def _gate_problems(harness: dict, split: str) -> list[str]:
    try:
        regression.load_regression_config(harness)
    except ValueError as error:
        return [str(error)]
    with closing(store.connect(harness["db_path"])) as conn:
        if regression.get_accepted_run(conn, split) is None:
            return [f"no accepted {split} run: record one with `python -m "
                    "evalharness.regression accept --run-id ... --note ...` (or pass --no-gate)"]
    return []


def preflight_run(config_path: str, split: str, harness: dict, auto: dict, *, gate: bool,
                  allow_dirty: bool) -> list[str]:
    """Every problem that can be found without an LLM call, for `run`. Empty list = go."""
    problems = []
    if split not in auto["allowed_splits"]:
        problems.append(f"split {split!r} is not automated (allowed: "
                        f"{', '.join(auto['allowed_splits'])}); the held-out split has its own "
                        "Step 12 commands")
        return problems   # nothing else matters
    generator_model = None
    try:
        system = runner.load_system(harness["system"])
        config = system.load_config(config_path)
        generation = config.get("generation") if isinstance(config, dict) else None
        if isinstance(generation, dict):
            generator_model = generation.get("model")
    except (ValueError, OSError, KeyError) as error:
        problems.append(f"config {config_path}: {error}")
    if not allow_dirty:
        try:
            if runner.git_info()[1]:
                problems.append("uncommitted changes: commit first so git_commit is the code "
                                "that runs (or pass --allow-dirty)")
        except Exception as error:   # git missing or not a repository
            problems.append(f"cannot read the git state: {error}")
    problems += _judge_problems(harness, generator_model)
    if gate:
        problems += _gate_problems(harness, split)
    return problems


def preflight_score(run_id: str, harness: dict, auto: dict, *, gate: bool) -> list[str]:
    """Problems for `score`: the run must exist, be completed and be on an automated split."""
    with closing(store.connect(harness["db_path"])) as conn:
        run = store.get_run(conn, run_id)
    if run is None:
        return [f"no run {run_id!r} in the results database"]
    if run["status"] != "completed":
        return [f"run {run_id} has status {run['status']!r}; only completed runs are scored"]
    if run["split"] not in auto["allowed_splits"]:
        return [f"run {run_id} is on split {run['split']!r}, which is not automated"]
    generation = json.loads(run["config_json"]).get("generation")
    generator_model = generation.get("model") if isinstance(generation, dict) else None
    problems = _judge_problems(harness, generator_model)
    if gate:
        problems += _gate_problems(harness, run["split"])
    return problems


# ---------------------------------------------------------------- stages

def _line(stage: str, status: str, detail: str) -> None:
    print(f"stage {stage:<13} {status:<6} {detail}", flush=True)


def _coverage(harness: dict, run_id: str, judge: str) -> str:
    """'80 judged, 0 judge failed, 0 not judged' for one judge, read from its stored score rows.

    Only this judge's own rows are read (not report.item_table, which needs all three
    scorers): right after the faithfulness stage the relevance rows do not exist yet.
    """
    loader = (judges.load_judges_config if judge == "faithfulness"
              else judges.load_relevance_config)
    version = loader(harness["judges_config"])["version"]
    with closing(store.connect(harness["db_path"])) as conn:
        rows = [r for r in store.get_scores(conn, run_id, version) if r["metric"] == judge]
    counts = Counter(judges._coverage_state(row) for row in rows)
    return (f"{counts['judged']} judged, {counts['judge_failed']} judge failed, "
            f"{counts['not_judged'] + counts['missing']} not judged")


def _stages_after_run(args, harness: dict, auto: dict, run_id: str, *, retry_failed: bool) -> int:
    """checks -> faithfulness -> relevance -> judge version -> gate for an existing run."""
    hc = ["--harness-config", args.harness_config]
    next_step = f"python -m evalharness score --run-id {run_id}"

    def stop(stage: str, code: int, why: str) -> int:
        _line(stage, "STOP", f"{why} (stage exit {code})")
        print(f"pipeline stopped at stage {stage}; run {run_id} is stored. "
              f"Resume with: {next_step}")
        return 2

    code = checks.main(["--run-id", run_id, *hc])
    if code != 0:
        return stop("checks", code, "scoring was refused")
    _line("checks", "ok", f"scored run {run_id}")

    for judge in ("faithfulness", "relevance"):
        argv = [judge, "--run-id", run_id, *hc] + (["--retry-failed"] if retry_failed else [])
        code = judges.main(argv)
        if code != 0:
            return stop(judge, code, f"{judge} judging is incomplete or was refused")
        _line(judge, "ok", _coverage(harness, run_id, judge))

    try:
        versions = {
            "faithfulness": judges.load_judges_config(harness["judges_config"])["version"],
            "relevance": judges.load_relevance_config(harness["judges_config"])["version"],
        }
        with closing(store.connect(harness["db_path"])) as conn:
            store.set_judge_version(conn, run_id, "faithfulness={faithfulness};relevance={relevance}"
                                    .format(**versions))
    except ValueError as error:
        return stop("judge_version", 1, str(error))

    if args.no_gate:
        _line("gate", "skip", "gate skipped (--no-gate); no verdict was produced")
        return 0
    # regression.main takes --harness-config BEFORE its subcommand (unlike the judge stages).
    gate_argv = [*hc, "check", "--run-id", run_id]
    if args.against:
        gate_argv += ["--against", args.against]
    if args.markdown:
        gate_argv += ["--markdown", args.markdown] + (["--force"] if args.force else [])
    if args.no_store:
        gate_argv.append("--no-store")
    code = regression.main(gate_argv)
    if code == 0:
        _line("gate", "PASS", "see the gate report above")
        return 0
    if code == 1:
        _line("gate", "FAIL", "ungrounded rate rose above the gate; see the gate report above")
        return 1
    return stop("gate", code, "the gate could not decide")


def run_pipeline(args) -> int:
    """run -> checks -> faithfulness -> relevance -> gate. Returns the exit code."""
    try:
        harness = load_harness_config(args.harness_config)
        auto = load_automation_config(harness)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    problems = preflight_run(args.config, args.split, harness, auto, gate=not args.no_gate,
                             allow_dirty=args.allow_dirty)
    if problems:
        for problem in problems:
            print(f"preflight: {problem}", file=sys.stderr)
        print("nothing was started (no LLM call made).", file=sys.stderr)
        return 2
    _line("preflight", "ok", "no problems found; no LLM call made")

    argv = ["--config", args.config, "--split", args.split, "--harness-config", args.harness_config]
    if args.allow_dirty:
        argv.append("--allow-dirty")
    code, run_id = runner.run(argv)
    if code != 0 or run_id is None:
        _line("run", "STOP", f"the run was refused, aborted or invalid (stage exit {code})"
              + (f"; partial run {run_id} is stored" if run_id else ""))
        return 2
    with closing(store.connect(harness["db_path"])) as conn:
        row = store.get_run(conn, run_id)
    _line("run", "ok", f"run {run_id}: {row['status']}, {row['n_ok']} ok, {row['n_failed']} failed")
    return _stages_after_run(args, harness, auto, run_id, retry_failed=False)


def score_pipeline(args) -> int:
    """checks -> faithfulness -> relevance -> gate for an existing completed run."""
    try:
        harness = load_harness_config(args.harness_config)
        auto = load_automation_config(harness)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    problems = preflight_score(args.run_id, harness, auto, gate=not args.no_gate)
    if problems:
        for problem in problems:
            print(f"preflight: {problem}", file=sys.stderr)
        print("nothing was started (no LLM call made).", file=sys.stderr)
        return 2
    _line("preflight", "ok", "no problems found; no LLM call made")
    return _stages_after_run(args, harness, auto, args.run_id,
                             retry_failed=auto["retry_failed_judges_in_score"])


# ---------------------------------------------------------------- CLI

def _add_gate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-gate", action="store_true",
                        help="skip the regression gate (exit 0 means 'stages ok', not 'PASS')")
    parser.add_argument("--no-store", action="store_true",
                        help="the gate does not write reviews / run_reports")
    parser.add_argument("--against", help="reference run id for the gate (default: accepted run)")
    parser.add_argument("--markdown", help="the gate writes its report as markdown to this path")
    parser.add_argument("--force", action="store_true", help="overwrite an existing --markdown file")
    parser.add_argument("--harness-config", default=DEFAULT_HARNESS_CONFIG)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evalharness",
                                     description="Run a full RAG evaluation from one command.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="generate answers, score, judge and gate one split")
    run.add_argument("--config", required=True, help="pipeline config, e.g. configs/v0.yaml")
    run.add_argument("--split", default="dev", help="split to run (default: dev)")
    run.add_argument("--allow-dirty", action="store_true",
                     help="run with uncommitted changes (stored as git_dirty = 1)")
    _add_gate_args(run)

    score = sub.add_parser("score", help="checks, judges and gate for an existing run (resume)")
    score.add_argument("--run-id", required=True)
    _add_gate_args(score)

    args = parser.parse_args(argv)
    return run_pipeline(args) if args.command == "run" else score_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
