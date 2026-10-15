"""Held-out confirmation for Step 12: a guard that makes "once" real, and a report.

The 20 held-out queries were never used for tuning. A number measured on them
is an honest estimate only if they are used once: run, dislike the number,
tweak, run again, and the held-out set has become a second dev set. So this
module enforces the rules written in configs/heldout.yaml (the
pre-registration, committed before any held-out run):

- only the registered configs may run on held-out, and only if the file bytes
  still match the pinned SHA-256 (no tweak after the pre-registration);
- each config may run at most `max_runs_per_config` times, counted in the
  `runs` table, ANY status counting as used (a broken run is not a free retry);
- the three scorer versions must be the ones used on dev (no judge tuning).

CLI:
    python -m evalharness.heldout check --config configs/v1_prompt_citations.yaml
Exit codes: 0 allowed; 1 refused.

Import note: runner.py imports this module, and report.py / experiments.py
import runner.py. To avoid a cycle, this module imports only store, checks and
testset at the top; report and experiments are imported inside functions.
"""

import argparse
import hashlib
import re
import sys
from contextlib import closing
from pathlib import Path

import yaml

from evalharness import store
from evalharness.testset import resolve_path

DEFAULT_HARNESS_CONFIG = "configs/harness.yaml"
SCORERS = ("checks", "faithfulness", "relevance")
HELDOUT_KEYS = {"selection_rule", "baseline_config", "chosen_config", "allowed_configs",
                "config_sha256", "max_runs_per_config", "pinned_versions", "dev_runs",
                "significance_level", "output", "amendments"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class HeldoutRefused(Exception):
    """A held-out run or report is not allowed under the pre-registration."""


# ---------------------------------------------------------------- config

def load_heldout_config(harness: dict) -> dict:
    """Read and check configs/heldout.yaml (path = harness["heldout_config"]).

    ValueError names the missing or bad key. A missing `heldout_config` key is
    an error too: the guard fails closed instead of silently not guarding.
    """
    path_name = harness.get("heldout_config")
    if not isinstance(path_name, str) or not path_name.strip():
        raise ValueError("harness config: 'heldout_config' must be the path of the "
                         "pre-registration file (configs/heldout.yaml)")
    path = resolve_path(path_name)
    if not path.exists():
        raise ValueError(f"{path_name}: pre-registration file does not exist")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path_name}: top level must be a mapping")
    missing, extra = HELDOUT_KEYS - set(cfg), set(cfg) - HELDOUT_KEYS
    if missing or extra:
        raise ValueError(f"{path_name}: keys missing {sorted(missing)}, unknown {sorted(extra)}")

    if not isinstance(cfg["selection_rule"], str) or not cfg["selection_rule"].strip():
        raise ValueError(f"{path_name}: 'selection_rule' must be a non-empty string")
    for key in ("baseline_config", "chosen_config"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{path_name}: '{key}' must be a non-empty string")
    allowed = cfg["allowed_configs"]
    if (not isinstance(allowed, list) or not allowed
            or not all(isinstance(p, str) and p for p in allowed)):
        raise ValueError(f"{path_name}: 'allowed_configs' must be a non-empty list of paths")
    for key in ("baseline_config", "chosen_config"):
        if cfg[key] not in allowed:
            raise ValueError(f"{path_name}: '{key}' {cfg[key]!r} is not in 'allowed_configs'")
    sha = cfg["config_sha256"]
    if not isinstance(sha, dict) or set(sha) != set(allowed):
        raise ValueError(f"{path_name}: 'config_sha256' needs exactly one entry per allowed config")
    for config_path, digest in sha.items():
        if not isinstance(digest, str) or not SHA256_PATTERN.match(digest):
            raise ValueError(f"{path_name}: 'config_sha256' for {config_path} must be 64 "
                             "lowercase hex characters")
    runs = cfg["max_runs_per_config"]
    if not isinstance(runs, int) or isinstance(runs, bool) or runs < 1:
        raise ValueError(f"{path_name}: 'max_runs_per_config' must be an integer >= 1")
    for key in ("pinned_versions", "dev_runs"):
        names = SCORERS if key == "pinned_versions" else ("baseline", "chosen")
        block = cfg[key]
        if not isinstance(block, dict) or set(block) != set(names):
            raise ValueError(f"{path_name}: '{key}' must have exactly the keys {list(names)}")
        if not all(isinstance(v, str) and v.strip() for v in block.values()):
            raise ValueError(f"{path_name}: every '{key}' value must be a non-empty string")
    level = cfg["significance_level"]
    if (not isinstance(level, (int, float)) or isinstance(level, bool)
            or not 0 < level < 1):
        raise ValueError(f"{path_name}: 'significance_level' must be a number between 0 and 1")
    output = cfg["output"]
    if (not isinstance(output, dict) or set(output) != {"markdown_path"}
            or not isinstance(output["markdown_path"], str) or not output["markdown_path"].strip()):
        raise ValueError(f"{path_name}: 'output' must be a mapping with 'markdown_path'")
    if not isinstance(cfg["amendments"], list):
        raise ValueError(f"{path_name}: 'amendments' must be a list")
    return cfg


# ---------------------------------------------------------------- the guard

def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(resolve_path(path).read_bytes()).hexdigest()


def _same_path(a: str, b: str) -> bool:
    """True when two config paths point at the same file ('configs/v0.yaml' == absolute form)."""
    return resolve_path(a).resolve() == resolve_path(b).resolve()


def _registered_path(config_path: str, heldout_cfg: dict) -> str | None:
    """The allowed_configs entry that `config_path` names, or None."""
    for allowed in heldout_cfg["allowed_configs"]:
        if _same_path(config_path, allowed):
            return allowed
    return None


def count_heldout_runs(conn, registered_path: str) -> int:
    """Held-out rows in `runs` for this config, any status (a broken run still counts as used)."""
    rows = conn.execute("SELECT config_path FROM runs WHERE split = 'heldout'").fetchall()
    return sum(1 for row in rows
               if row["config_path"] and _same_path(row["config_path"], registered_path))


def current_scorer_versions(harness: dict) -> dict[str, str]:
    """The scorer versions of the current configs (imported late, see the module docstring)."""
    from evalharness.report import current_versions
    return current_versions(harness)


def check_heldout_allowed(conn, config_path: str, heldout_cfg: dict, harness: dict) -> dict:
    """Raise HeldoutRefused unless a held-out run of this config is allowed.

    Returns {"used": int, "max": int, "config_sha256": str} for printing.
    """
    registered = _registered_path(config_path, heldout_cfg)
    if registered is None:
        raise HeldoutRefused(f"{config_path} is not registered for the held-out run; "
                             f"allowed: {', '.join(heldout_cfg['allowed_configs'])}")
    digest = _file_sha256(registered)
    pinned = heldout_cfg["config_sha256"][registered]
    if digest != pinned:
        raise HeldoutRefused(f"{registered} changed since the pre-registration "
                             f"(sha256 {digest[:12]}..., pinned {pinned[:12]}...)")
    used, limit = count_heldout_runs(conn, registered), heldout_cfg["max_runs_per_config"]
    if used >= limit:
        raise HeldoutRefused(f"{registered}: {used} of {limit} held-out runs already used "
                             "(any status counts); a further run is Gaurav's decision, "
                             "recorded as an amendment in the pre-registration file")
    versions = current_scorer_versions(harness)
    if versions != heldout_cfg["pinned_versions"]:
        raise HeldoutRefused(f"scorer versions {versions} differ from the pinned "
                             f"{heldout_cfg['pinned_versions']}; judges and checks must be "
                             "the ones used on dev")
    return {"used": used, "max": limit, "config_sha256": digest}


# ---------------------------------------------------------------- the report

VERDICT_WORDING = {
    "supported_better": "supported better: fewer ungrounded answers than v0 on the held-out "
                        "queries, and the exact paired p is below the pre-registered level",
    "supported_worse": "supported worse: more ungrounded answers than v0 on the held-out "
                       "queries, and the exact paired p is below the pre-registered level",
    "not_distinguishable": "not distinguishable from v0 on the held-out queries; 20 queries "
                           "cannot detect a change of this size",
}


def verdict_label(improved: int, regressed: int, p_value: float | None, level: float) -> str:
    """The pre-registered verdict, from counts only.

    improved / regressed are the paired flips (v0 ungrounded -> chosen grounded, and the
    reverse). Only a p below `level` can give better or worse; a lower point estimate
    with p >= level is never called better. No flips (p is None) is not distinguishable.
    """
    if p_value is None or p_value >= level:
        return "not_distinguishable"
    return "supported_better" if improved > regressed else "supported_worse"


def _check_run(conn, run_id: str, expected_config: str, what: str, *, split: str) -> dict:
    """The runs row of `run_id`, after the checks that make it a valid registered run."""
    from evalharness.checks import get_scorable_run
    run = get_scorable_run(conn, run_id, allow_heldout=True)
    if run["split"] != split:
        raise HeldoutRefused(f"{what} run {run_id} is on split {run['split']!r}, expected {split!r}")
    if run["git_dirty"]:
        raise HeldoutRefused(f"{what} run {run_id} was made on a dirty working tree")
    if not run["config_path"] or not _same_path(run["config_path"], expected_config):
        raise HeldoutRefused(f"{what} run {run_id} used config {run['config_path']!r}, "
                             f"the registered one is {expected_config!r}")
    return run


def _answer_rows(conn, run_id: str, items: list[dict], reasoning: dict[str, str]) -> dict:
    """Ungrounded answers and false refusals of one run, with question, answer, chunks, reasoning."""
    outputs = {o["query_id"]: o for o in store.get_outputs_with_queries(conn, run_id)}

    def row(item: dict) -> dict:
        out = outputs[item["query_id"]]
        return {"query_id": item["query_id"], "type": item["type"], "question": out["question"],
                "answer": out["answer"],
                "chunk_ids": [c["chunk_id"] for c in out["retrieved_chunks"] or []],
                "in_context": item["in_context"], "irrelevant": item["irrelevant"],
                "reasoning": reasoning.get(item["query_id"])}

    return {
        "ungrounded": [row(i) for i in items if i["ungrounded"] == 1],
        "false_refusals": [row(i) for i in items
                           if i["answerable"] and i["irrelevant"] == 1 and i["refusal"] == 1],
    }


def build_confirmation(conn, baseline_run_id: str, run_id: str, heldout_cfg: dict,
                       harness: dict) -> dict:
    """The held-out confirmation report, computed only from stored rows (no LLM call).

    Raises HeldoutRefused, ScoringRefused or ExperimentRefused unless both runs are
    completed, clean, held-out, on the registered configs, scored with the pinned
    versions and fully judged.
    """
    from evalharness import experiments, report as rep
    from evalharness.checks import load_checks_config
    from evalharness.metrics import rate_summary

    if baseline_run_id == run_id:
        raise HeldoutRefused("the baseline and the chosen run are the same run")
    versions = rep.current_versions(harness)
    if versions != heldout_cfg["pinned_versions"]:
        raise HeldoutRefused(f"scorer versions {versions} differ from the pinned "
                             f"{heldout_cfg['pinned_versions']}")
    confidence = load_checks_config(harness)["report_confidence"]
    report_cfg = {**rep.load_report_config(harness), "allow_incomplete": False}

    registered = {"baseline": heldout_cfg["baseline_config"], "chosen": heldout_cfg["chosen_config"]}
    run_ids = {"baseline": baseline_run_id, "chosen": run_id}
    runs = {name: _check_run(conn, run_ids[name], registered[name], name, split="heldout")
            for name in run_ids}
    dev_ids = heldout_cfg["dev_runs"]
    for name in dev_ids:   # the stored dev runs shown beside the held-out numbers
        _check_run(conn, dev_ids[name], registered[name], f"dev {name}", split="dev")

    # Step 9 calibration is stored against the dev run the judges were compared on;
    # a held-out run has no calibration rows of its own.
    calibration = {judge: rep._calibration_status(conn, dev_ids["baseline"], judge, versions[judge])
                   for judge in ("faithfulness", "relevance")}
    judge_label = {j: ("calibrated judge" if c["status"] == "measured" else "uncalibrated judge")
                   for j, c in calibration.items()}

    per_run, listed, items_by_run = {}, {}, {}
    for name, rid in run_ids.items():
        built = rep.build_baseline_report(conn, rid, versions, confidence, report_cfg,
                                          allow_heldout=True)
        built["kind"] = "heldout_run"
        built["calibration"], built["judge_label"] = calibration, judge_label
        items = rep.item_table(conn, rid, versions)
        reasoning = {r["query_id"]: r["reasoning"]
                     for r in store.get_scores(conn, rid, versions["faithfulness"])
                     if r["metric"] == "ungrounded"}
        per_run[name], items_by_run[name] = built, items
        listed[name] = _answer_rows(conn, rid, items, reasoning)

    exp_cfg = experiments.load_experiments_config(harness)
    comparison = experiments.compare_runs(
        conn, baseline_run_id, run_id, versions,
        {"n_bootstrap": exp_cfg["n_bootstrap"], "bootstrap_seed": exp_cfg["bootstrap_seed"],
         "confidence": confidence}, allow_heldout=True)
    comparison["judge_label"] = judge_label

    dev, inside = {}, {}
    for name in dev_ids:
        dev_items = rep.item_table(conn, dev_ids[name], versions)
        dev[name] = {"run_id": dev_ids[name],
                     "ungrounded": rate_summary([i["ungrounded"] for i in dev_items], confidence)}
        held = per_run[name]["ungrounded"]["all"]
        low, high = dev[name]["ungrounded"]["ci_low"], dev[name]["ungrounded"]["ci_high"]
        inside[name] = (None if held["rate"] is None or low is None
                        else low <= held["rate"] <= high)

    counts = comparison["flips"]["counts"]
    label = verdict_label(counts["improved"], counts["regressed"], comparison["p_value"],
                          heldout_cfg["significance_level"])
    n = per_run["chosen"]["metadata"]["n_outputs"]
    return {
        "kind": "heldout",
        "registration": {
            "selection_rule": heldout_cfg["selection_rule"],
            "baseline_config": registered["baseline"], "chosen_config": registered["chosen"],
            "config_sha256": heldout_cfg["config_sha256"],
            "pinned_versions": heldout_cfg["pinned_versions"],
            "max_runs_per_config": heldout_cfg["max_runs_per_config"],
            "amendments": heldout_cfg["amendments"],
            "significance_level": heldout_cfg["significance_level"],
        },
        "confidence": confidence, "n_queries": n,
        "run_ids": run_ids, "git_commit": {k: r["git_commit"] for k, r in runs.items()},
        "runs": per_run, "comparison": comparison, "dev": dev, "inside_dev_ci": inside,
        "verdict": {"label": label, "wording": VERDICT_WORDING[label],
                    "improved": counts["improved"], "regressed": counts["regressed"],
                    "p_value": comparison["p_value"],
                    "significance_level": heldout_cfg["significance_level"]},
        "judge_label": judge_label, "calibration": calibration,
        "listed_answers": listed,
        "limitations": [
            f"n = {n} held-out queries: one answer is {100 / n:.0f} percentage points, so this "
            "set cannot show a change of a few points.",
            "Per-type n is 3-10: no per-type claims are made.",
            f"Judge numbers come from an {judge_label['faithfulness']} (faithfulness) and an "
            f"{judge_label['relevance']} (relevance): they are the judge's measurement, not "
            "verified rates.",
            "The chosen config was the best of 12 dev looks, so its dev rate is optimistic "
            "(winner's curse); a held-out rate nearer the baseline is the normal case.",
            "Each answer is one sample from a cloud model that is not fully deterministic.",
            "CV hypothesis 25% -> 12% is a hypothesis, not a target; the CV follows the "
            "measurement.",
        ],
    }


# ---------------------------------------------------------------- formatting

def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.2f}%"


def _rate(s: dict) -> str:
    """'8.75% [4.30%, 16.98%] (7/80)', or '-' when n = 0."""
    if s["rate"] is None:
        return "-"
    return f"{_pct(s['rate'])} [{_pct(s['ci_low'])}, {_pct(s['ci_high'])}] ({s['successes']}/{s['n']})"


def format_confirmation(report: dict) -> str:
    """Terminal text, verdict first."""
    v, reg = report["verdict"], report["registration"]
    base, chosen = report["runs"]["baseline"], report["runs"]["chosen"]
    pct = round(report["confidence"] * 100)
    label = report["judge_label"]["faithfulness"]
    p = "no query flipped" if v["p_value"] is None else f"exact paired p = {v['p_value']:.4f}"
    lines = [
        f"Step 12 held-out confirmation ({report['n_queries']} unseen queries, {label})",
        f"verdict: {v['label']}: {v['wording']}",
        f"  improved {v['improved']}, regressed {v['regressed']}, {p} "
        f"(significance level {v['significance_level']})",
        "",
        f"ungrounded rate, {pct}% Wilson interval:",
        f"  v0 on held-out        {_rate(base['ungrounded']['all'])}   {report['run_ids']['baseline']}",
        f"  chosen on held-out    {_rate(chosen['ungrounded']['all'])}   {report['run_ids']['chosen']}",
        f"  v0 on dev             {_rate(report['dev']['baseline']['ungrounded'])}",
        f"  chosen on dev         {_rate(report['dev']['chosen']['ungrounded'])}",
        f"  held-out rate inside the dev interval: v0 {report['inside_dev_ci']['baseline']}, "
        f"chosen {report['inside_dev_ci']['chosen']}",
        "",
        f"bad answer (ungrounded or irrelevant): v0 {_rate(base['bad_answer']['all'])}; "
        f"chosen {_rate(chosen['bad_answer']['all'])}",
        f"false refusals: v0 {len(report['listed_answers']['baseline']['false_refusals'])}, "
        f"chosen {len(report['listed_answers']['chosen']['false_refusals'])}",
        f"failed outputs: v0 {base['metadata']['n_failed']}, chosen {chosen['metadata']['n_failed']}",
        f"registered: {reg['chosen_config']} against {reg['baseline_config']}; scorers "
        + ", ".join(f"{k} {v_}" for k, v_ in sorted(reg["pinned_versions"].items())),
        "",
        "limitations:",
        *[f"  - {text}" for text in report["limitations"]],
    ]
    return "\n".join(lines)


def format_confirmation_markdown(report: dict) -> str:
    """experiments/heldout_confirmation.md"""
    v, c, reg = report["verdict"], report["comparison"], report["registration"]
    base, chosen = report["runs"]["baseline"], report["runs"]["chosen"]
    label = report["judge_label"]["faithfulness"]
    p = "no query flipped" if v["p_value"] is None else f"p = {v['p_value']:.4f}"
    fd = c["faithfulness_diff"]
    faith = ("-" if fd["mean"] is None else
             f"{fd['mean']:+.3f} [{fd['ci_low']:+.3f}, {fd['ci_high']:+.3f}] (n={fd['n']})")
    lines = [
        "# Held-out confirmation (Step 12)",
        "",
        f"Chosen config `{reg['chosen_config']}` against `{reg['baseline_config']}`, each run once on "
        f"the {report['n_queries']} held-out queries. Judge numbers come from an **{label}** "
        f"(faithfulness) and an **{report['judge_label']['relevance']}** (relevance).",
        "",
        f"## Verdict: `{v['label']}`",
        "",
        f"{v['wording']}.",
        "",
        f"Improved {v['improved']}, regressed {v['regressed']}, exact paired {p} "
        f"(pre-registered level {v['significance_level']}).",
        "",
        "## Ungrounded rate",
        "",
        f"Intervals are {round(report['confidence'] * 100)}% Wilson intervals. One answer is "
        f"{100 / report['n_queries']:.0f} percentage points.",
        "",
        "| Config | Held-out (unseen) | Dev | Held-out rate inside dev interval |",
        "|---|---|---|---|",
        f"| v0 `{report['run_ids']['baseline']}` | {_rate(base['ungrounded']['all'])} "
        f"| {_rate(report['dev']['baseline']['ungrounded'])} | {report['inside_dev_ci']['baseline']} |",
        f"| chosen `{report['run_ids']['chosen']}` | {_rate(chosen['ungrounded']['all'])} "
        f"| {_rate(report['dev']['chosen']['ungrounded'])} | {report['inside_dev_ci']['chosen']} |",
        "",
        f"Mean faithfulness difference (chosen - v0): {faith}.",
        "",
        "## Other held-out numbers",
        "",
        "| Metric | v0 | chosen |",
        "|---|---|---|",
        f"| Bad answer | {_rate(base['bad_answer']['all'])} | {_rate(chosen['bad_answer']['all'])} |",
        f"| Irrelevant | {_rate(base['irrelevant'])} | {_rate(chosen['irrelevant'])} |",
        f"| False refusals | {len(report['listed_answers']['baseline']['false_refusals'])} "
        f"| {len(report['listed_answers']['chosen']['false_refusals'])} |",
        f"| hit@4 (answerable) | {_rate(base['retrieval']['hit@4'])} "
        f"| {_rate(chosen['retrieval']['hit@4'])} |",
        f"| Failed outputs | {base['metadata']['n_failed']} | {chosen['metadata']['n_failed']} |",
        "",
        "## Registration",
        "",
        f"- Selection rule: {reg['selection_rule'].strip()}",
        "- Scorer versions: " + ", ".join(f"{k} `{x}`" for k, x in sorted(reg["pinned_versions"].items())),
        "- Config SHA-256: " + ", ".join(f"`{k}` `{x}`" for k, x in sorted(reg["config_sha256"].items())),
        f"- Amendments: {len(reg['amendments'])}",
        "",
        "## Limitations",
        "",
        *[f"- {text}" for text in report["limitations"]],
        "",
        "## Answers to read",
        "",
        "Every ungrounded answer and false refusal, for a human check (a human-review note, "
        "not part of any calibration statistic).",
    ]
    for name, title in (("baseline", "v0"), ("chosen", "chosen config")):
        for kind, heading in (("ungrounded", "ungrounded"), ("false_refusals", "false refusal")):
            for item in report["listed_answers"][name][kind]:
                lines += [
                    "",
                    f"**{item['query_id']}** ({item['type']}), {title}, {heading}",
                    f"- Question: {item['question']}",
                    f"- Answer: {item['answer']}",
                    f"- Retrieved chunks: {', '.join(item['chunk_ids']) or '-'}",
                    f"- Judge reasoning: {item['reasoning']}",
                ]
    if not any(report["listed_answers"][n][k] for n in report["listed_answers"]
               for k in ("ungrounded", "false_refusals")):
        lines += ["", "None: no ungrounded answer and no false refusal in either run."]
    return "\n".join(lines) + "\n"


def log_rows(report: dict, created_at: str) -> list[str]:
    """Two experiments/log.md rows (v0 and chosen, both on held-out); numbers from the report."""
    v, label = report["verdict"], report["judge_label"]["faithfulness"]
    reg = report["registration"]
    rows = []
    for name, config, variable, note in (
        ("baseline", reg["baseline_config"], "none (held-out baseline)",
         "held-out, 20 unseen queries"),
        ("chosen", reg["chosen_config"], "held-out confirmation",
         f"held-out verdict {v['label']}; improved {v['improved']}, regressed {v['regressed']}"
         + ("" if v["p_value"] is None else f", p = {v['p_value']:.4f}")),
    ):
        rate = _rate(report["runs"][name]["ungrounded"]["all"])
        rows.append(f"| {created_at[:10]} | `{config}` | {variable} | `{report['run_ids'][name]}` "
                    f"| {rate}, {label} | {note} |")
    return rows


# ---------------------------------------------------------------- CLI

def _cmd_check(args, harness: dict, heldout_cfg: dict) -> int:
    with closing(store.connect(harness["db_path"])) as conn:
        try:
            info = check_heldout_allowed(conn, args.config, heldout_cfg, harness)
        except (HeldoutRefused, ValueError) as error:
            print(f"refused: {error}", file=sys.stderr)
            return 1
    print(f"allowed: {args.config} (held-out runs used {info['used']} of {info['max']}, "
          f"sha256 {info['config_sha256']})")
    print("pinned versions: " + ", ".join(f"{k} {v}" for k, v
                                           in sorted(heldout_cfg["pinned_versions"].items())))
    return 0


def _cmd_confirm(args, harness: dict, heldout_cfg: dict) -> int:
    def refuse(problem: str) -> int:
        print(f"error: {problem}", file=sys.stderr)
        return 1

    from evalharness.checks import ScoringRefused
    from evalharness.experiments import ExperimentRefused

    markdown_path = resolve_path(args.markdown) if args.markdown else None
    log_path = resolve_path(args.log) if args.log else None
    if markdown_path is not None and markdown_path.exists() and not args.force:
        return refuse(f"{markdown_path} exists; pass --force to overwrite it")
    if log_path is not None:
        if not log_path.exists():
            return refuse(f"{log_path} does not exist")
        text = log_path.read_text(encoding="utf-8")
        for run_id in (args.baseline_run, args.run):
            if f"`{run_id}`" in text:
                return refuse(f"{log_path} already has a row for run {run_id}")

    with closing(store.connect(harness["db_path"])) as conn:
        try:   # everything is computed before anything is written
            report = build_confirmation(conn, args.baseline_run, args.run, heldout_cfg, harness)
        except (HeldoutRefused, ScoringRefused, ExperimentRefused, ValueError) as error:
            return refuse(str(error))
        created_at = store._now()
        report_id = store.write_run_report(conn, {
            "run_id": args.run, "kind": "heldout", "versions": heldout_cfg["pinned_versions"],
            "report": report, "created_at": created_at})

    print(format_confirmation(report))
    print(f"\nstored as run_reports id {report_id}")
    if markdown_path is not None:
        markdown_path.write_text(format_confirmation_markdown(report), encoding="utf-8")
        print(f"wrote {markdown_path}")
    if log_path is not None:
        text = log_path.read_text(encoding="utf-8")
        log_path.write_text(text + ("" if text.endswith("\n") else "\n")
                            + "\n".join(log_rows(report, created_at)) + "\n", encoding="utf-8")
        print(f"appended 2 rows to {log_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evalharness.heldout",
                                     description="Held-out guard and confirmation report.")
    parser.add_argument("--harness-config", default=DEFAULT_HARNESS_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="read-only: would a held-out run of this config be allowed?")
    check.add_argument("--config", required=True)
    confirm = sub.add_parser("confirm", help="held-out report for the baseline and chosen runs")
    confirm.add_argument("--baseline-run", required=True, help="held-out run id of the baseline")
    confirm.add_argument("--run", required=True, help="held-out run id of the chosen config")
    confirm.add_argument("--markdown", help="write the report as markdown to this path")
    confirm.add_argument("--force", action="store_true", help="overwrite an existing --markdown file")
    confirm.add_argument("--log", help="append two rows to this experiment log")
    args = parser.parse_args(argv)

    from evalharness.runner import load_harness_config   # late: runner imports this module
    try:
        harness = load_harness_config(args.harness_config)
        heldout_cfg = load_heldout_config(harness)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return (_cmd_check if args.command == "check" else _cmd_confirm)(args, harness, heldout_cfg)


if __name__ == "__main__":
    sys.exit(main())
