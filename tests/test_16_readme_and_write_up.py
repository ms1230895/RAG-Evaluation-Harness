"""Step 16 tests: generated README results block (evalharness/writeup.py) and static checks
of the real README.md, .env.example and the documented commands.

Offline. Stored reports are built on the `experiment_db` fixture with the real
`report.build_baseline_report` / `experiments.compare_runs` and written with
`store.write_run_report`. The real database, the real README block and any LLM are never used.
"""

import ast
import copy
import importlib.util
import json
import re
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest
import yaml

from evalharness import store, writeup
from evalharness.experiments import compare_runs
from evalharness.metrics import rate_summary
from evalharness.report import build_baseline_report
from evalharness.writeup import (END_MARKER, START_MARKER, WriteupRefused, build_results,
                                 check_readme, format_results_markdown, load_writeup_config,
                                 replace_block)

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
BOOT = {"n_bootstrap": 200, "bootstrap_seed": 7, "confidence": 0.95, "report_confidence": 0.95}
REPORT_CFG = {"primary_metric": "ungrounded", "include_bad_answer_rate": True,
              "allow_incomplete": True}
HARNESS = {"writeup": {"readme_path": "README.md"}}


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Spec: this step makes no LLM call; any call raises."""
    def boom(*args, **kwargs):
        raise AssertionError("an LLM call was made")
    monkeypatch.setattr("llm.client.generate", boom)


# ---------------------------------------------------------------- fixtures

class Env:
    """Hand-built stored reports on the experiment_db fixture; `write(...)` stores them."""


@pytest.fixture
def wenv(experiment_db):
    """Reports (not yet stored): baseline (run-baseline-001: ungrounded 4/9), experiment
    run-exp-001 (1/9; improved 3, regressed 1, p 0.625), run-sim-001 (q002 ungrounded: 1/10),
    a twin run on the baseline's own config_path, and a hand-built held-out report whose
    chosen config is run-exp-001's."""
    db, conn = experiment_db, experiment_db.conn
    env = Env()
    env.db = db
    env.baseline = build_baseline_report(conn, db.run_id, db.versions, 0.95, REPORT_CFG)
    env.exp1 = compare_runs(conn, db.run_id, db.exp_run_id, db.versions, BOOT)
    db.add_simple_run("run-sim-001", ["q002"])
    env.exp2 = compare_runs(conn, db.run_id, "run-sim-001", db.versions, BOOT)
    db.add_simple_run("run-twin-001", [])
    with conn:
        conn.execute("UPDATE runs SET config_path = ? WHERE run_id = 'run-twin-001'",
                     (env.baseline["metadata"]["config_path"],))
    env.twin = compare_runs(conn, db.run_id, "run-twin-001", db.versions, BOOT)
    env.heldout = {
        "kind": "heldout",
        "registration": {"chosen_config": env.exp1["config_path"]},
        "runs": {"baseline": {"ungrounded": {"all": rate_summary([1] + [0] * 9, 0.95)}},
                 "chosen": {"ungrounded": {"all": rate_summary([0] * 10, 0.95)}}},
        "run_ids": {"baseline": "run-held-a", "chosen": "run-held-b"},
        "verdict": {"label": "not_distinguishable", "wording": "WORDING-FIXTURE",
                    "improved": 1, "regressed": 0, "p_value": 1.0},
        "judge_label": {"faithfulness": "uncalibrated judge", "relevance": "uncalibrated judge"},
    }

    def put(kind, run_id, report):
        store.write_run_report(conn, {"run_id": run_id, "kind": kind, "versions": db.versions,
                                      "report": report, "created_at": "2026-10-16T00:00:00Z"})

    def write(baseline=True, experiments=True, heldout=True):
        if baseline:
            put("baseline", db.run_id, env.baseline)
        if experiments:   # run order: exp1, twin, exp2 (the twin must be left out of the table)
            for report in (env.exp1, env.twin, env.exp2):
                put("experiment", report["run_id"], report)
        if heldout:
            put("heldout", db.exp_run_id, env.heldout)   # FK: run_id must be a stored run

    env.put, env.write = put, write
    return env


@pytest.fixture
def built(wenv):
    wenv.write()
    results = build_results(wenv.db.conn, HARNESS)
    return wenv, results, format_results_markdown(results)


def sample_readme(body="OLD-BLOCK"):
    return f"# Title\n\nintro text\n{START_MARKER}\n{body}\n{END_MARKER}\n\n## Tail\nend\n"


@pytest.fixture
def cli(wenv, tmp_path):
    """Tmp harness YAML + README pointing at the fixture DB; run(*args) -> (exit code, stdout)."""
    wenv.write()
    harness = yaml.safe_load((ROOT / "configs" / "harness.yaml").read_text(encoding="utf-8"))
    harness["db_path"] = str(wenv.db.path)
    harness["writeup"] = {"readme_path": str(tmp_path / "README.md")}
    harness_path = tmp_path / "harness.yaml"
    harness_path.write_text(yaml.safe_dump(harness), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(sample_readme(), encoding="utf-8")

    def run(*args, capsys=None):
        return writeup.main(["--harness-config", str(harness_path), *args])

    wenv.readme, wenv.harness_path, wenv.run = readme, harness_path, run
    return wenv


# ---------------------------------------------------------------- config

def test_load_writeup_config_valid_block_returns_it():
    """Spec Config: a valid `writeup` block loads; the real configs/harness.yaml has one."""
    assert load_writeup_config(HARNESS) == {"readme_path": "README.md"}
    real = yaml.safe_load((ROOT / "configs" / "harness.yaml").read_text(encoding="utf-8"))
    assert load_writeup_config(real)["readme_path"] == "README.md"


@pytest.mark.parametrize("harness", [
    {},                                                        # block missing
    {"writeup": {}},                                           # key missing
    {"writeup": {"readme_path": "R.md", "extra": 1}},          # unknown key
    {"writeup": {"readme_path": 5}},                           # not a string
])
def test_load_writeup_config_bad_block_raises_value_error(harness):
    """Spec Config: missing / unknown key and non-string path raise ValueError."""
    with pytest.raises(ValueError):
        load_writeup_config(harness)


# ---------------------------------------------------------------- store helper

def test_list_run_reports_returns_one_kind_oldest_first_parsed(wenv):
    """Spec Interfaces: store.list_run_reports(conn, kind) filters by kind, oldest first, parsed JSON."""
    wenv.write()
    rows = store.list_run_reports(wenv.db.conn, "experiment")
    assert [r["report"]["run_id"] for r in rows] == ["run-exp-001", "run-twin-001", "run-sim-001"]
    assert all(r["kind"] == "experiment" and isinstance(r["versions"], dict) for r in rows)
    assert [r["kind"] for r in store.list_run_reports(wenv.db.conn, "baseline")] == ["baseline"]
    assert store.list_run_reports(wenv.db.conn, "regression") == []


# ---------------------------------------------------------------- build

def test_build_results_returns_experiments_in_run_order_without_the_twin(built):
    """Spec Build: experiments in run order, the baseline-config twin excluded (kept as a note)."""
    env, results, _ = built
    assert [r["run_id"] for r in results["experiments"]] == ["run-exp-001", "run-sim-001"]
    assert [r["run_id"] for r in results["twin"]] == ["run-twin-001"]
    assert results["baseline"]["metadata"]["run_id"] == "run-baseline-001"
    assert results["heldout"]["run_ids"] == {"baseline": "run-held-a", "chosen": "run-held-b"}


def test_build_results_reads_calibration_status_per_judge(wenv):
    """Spec Build: calibration per judge, live from calibration_reports (check n 20, TN 20 = inconclusive)."""
    wenv.write()
    before = build_results(wenv.db.conn, HARNESS)["calibration"]
    assert before["faithfulness"]["status"] == "not calibrated"
    wenv.db.add_calibration_report("faithfulness", wenv.db.versions["faithfulness"], 0, 0, 0, 20)
    after = build_results(wenv.db.conn, HARNESS)["calibration"]
    assert after["faithfulness"]["status"] == "inconclusive"
    assert (after["faithfulness"]["n"], after["faithfulness"]["tn"]) == (20, 20)
    assert after["relevance"]["status"] == "not calibrated"


@pytest.mark.parametrize("kwargs", [
    {"baseline": False},      # no baseline report
    {"experiments": False},   # no experiment rows
    {"heldout": False},       # no held-out report
])
def test_build_results_missing_report_raises_writeup_refused(wenv, kwargs):
    """Spec Build: missing baseline, experiments or held-out each raise WriteupRefused."""
    wenv.write(**kwargs)
    with pytest.raises(WriteupRefused):
        build_results(wenv.db.conn, HARNESS)


def test_build_results_with_only_twin_rows_raises_writeup_refused(wenv):
    """Spec Build: the twin is not an experiment, so twin-only means no experiments."""
    wenv.write(experiments=False)
    wenv.put("experiment", "run-twin-001", wenv.twin)
    with pytest.raises(WriteupRefused):
        build_results(wenv.db.conn, HARNESS)


# ---------------------------------------------------------------- format

def test_format_results_markdown_baseline_and_experiment_use_hand_values(built):
    """Spec Format: baseline 4/9 and experiment 1/9 via the Step 10 formatter; flips 3/1, p 0.625."""
    _, results, block = built
    # Wilson 95% for 4/9: centre (0.4444 + 1.9208/9) / 1.4268 = 0.4611, upper 0.7333 (independent Wilson calc)
    assert "44.44% [18.88%, 73.33%] (4/9)" in block
    assert "11.11%" in block and "(1/9)" in block          # experiment 1/9
    row = next(l for l in block.splitlines() if "`run-exp-001`" in l and l.startswith("| generation"))
    assert "3 / 1" in row and "0.6250" in row
    assert "improved 3, regressed 1, exact paired p = 0.6250" in block
    assert "`run-twin-001`" in block and not any(
        l.startswith("|") and "`run-twin-001`" in l for l in block.splitlines())


def test_format_results_markdown_headline_shows_heldout_verdict(built):
    """Spec Format: headline has dev and held-out rates and the verdict word."""
    _, _, block = built
    assert "(1/10)" in block and "(0/10)" in block         # held-out v0 and chosen
    assert "`not_distinguishable`" in block
    assert "exact paired p = 1.0000" in block


def test_format_results_markdown_carries_uncalibrated_judge_label(built):
    """Spec Format: the judge label stays beside the numbers while no calibration is measured."""
    _, _, block = built
    assert "uncalibrated judge" in block
    assert "not calibrated" in block


def test_format_results_markdown_every_k_over_n_comes_from_a_stored_report(built):
    """Spec Format: every k/n in the block exists in the stored reports (no invented number)."""
    env, results, block = built
    allowed = set()

    def walk(node):
        if isinstance(node, dict):
            if "successes" in node and "n" in node:
                allowed.add((node["successes"], node["n"]))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(results)
    found = {(int(a), int(b)) for a, b in re.findall(r"(?<![\w.])(\d+)/(\d+)(?![\w.])", block)}
    assert found, "no k/n in the block"
    assert found <= allowed, f"k/n not in any stored report: {found - allowed}"


def test_format_results_markdown_empty_rate_prints_n_equals_zero_not_zero_percent(built):
    """Spec Format: an n = 0 rate prints 'n = 0', never 0%."""
    _, results, _ = built
    changed = copy.deepcopy(results)
    changed["experiments"][1]["experiment"]["ungrounded"]["all"] = rate_summary([], 0.95)
    block = format_results_markdown(changed)
    row = next(l for l in block.splitlines() if "`run-sim-001`" in l and l.startswith("| "))
    assert "| n = 0 |" in row
    assert "0.00%" not in row


def test_format_results_markdown_lists_stored_regression_checks(built):
    """Spec Format: stored Step 13 checks appear with gate, ungrounded change and flag rate."""
    _, results, _ = built
    results = dict(results)
    results["regression"] = [{
        "run_id": "run-reg-new", "against_run_id": "run-reg-old", "flag_version": "flags-test",
        "gate": {"verdict": "PASS", "delta_points": 1.25},
        "flag_reason_counts": {"ungrounded": 2, "low_relevance": 1},
        "flag_rate": rate_summary([1, 1, 1] + [0] * 7, 0.95),
    }]
    block = format_results_markdown(results)
    row = next(l for l in block.splitlines() if "`run-reg-new`" in l)
    assert "`run-reg-old`" in row and "PASS" in row and "+1.25 pts" in row
    assert "(3/10)" in row and "ungrounded 2" in row and "low_relevance 1" in row


# ---------------------------------------------------------------- replace / check

def test_replace_block_keeps_outside_text_byte_identical_and_is_idempotent():
    """Spec Replace: only the text between the markers changes; a second call changes nothing."""
    text = sample_readme("OLD")
    once = replace_block(text, "NEW-BODY\nline 2")
    assert once.startswith(text[:text.index(START_MARKER) + len(START_MARKER)])
    assert once.endswith(text[text.index(END_MARKER):])
    assert "OLD" not in once and "NEW-BODY\nline 2" in once
    assert replace_block(once, "NEW-BODY\nline 2") == once


@pytest.mark.parametrize("text", [
    "no markers at all",
    f"{START_MARKER}\nbody only",
    f"x\n{END_MARKER}\nend only",
    sample_readme() + f"{START_MARKER}\nagain\n{END_MARKER}\n",   # duplicate pair
])
def test_replace_block_bad_markers_raise_writeup_refused(text):
    """Spec Replace: zero or duplicate markers raise WriteupRefused."""
    with pytest.raises(WriteupRefused):
        replace_block(text, "BODY")


def test_check_readme_fresh_block_matches_and_edited_digit_is_reported(built):
    """Spec Check: [] for a fresh block; a message after a digit in the block is edited."""
    _, _, block = built
    fresh = replace_block(sample_readme(), block)
    assert check_readme(fresh, block) == []
    tampered = fresh.replace("(4/9)", "(5/9)", 1)
    assert tampered != fresh
    problems = check_readme(tampered, block)
    assert len(problems) >= 1 and all(isinstance(p, str) and p for p in problems)


# ---------------------------------------------------------------- CLI

def test_main_update_then_check_exits_zero_and_second_update_changes_nothing(cli):
    """Spec CLI: update then check exits 0; update twice leaves the README unchanged."""
    assert cli.run("update") == 0
    after_first = cli.readme.read_text(encoding="utf-8")
    assert "44.44% [18.88%, 73.33%] (4/9)" in after_first and "OLD-BLOCK" not in after_first
    assert after_first.startswith("# Title\n\nintro text\n") and after_first.endswith("## Tail\nend\n")
    assert cli.run("check") == 0
    assert cli.run("update") == 0
    assert cli.readme.read_text(encoding="utf-8") == after_first


def test_main_check_exits_one_when_readme_block_differs(cli):
    """Spec CLI: check exits 1 when the block differs from the database."""
    assert cli.run("check") == 1                       # the sample block is "OLD-BLOCK"
    cli.run("update")
    edited = cli.readme.read_text(encoding="utf-8").replace("(1/9)", "(2/9)", 1)
    cli.readme.write_text(edited, encoding="utf-8")
    assert cli.run("check") == 1


def test_main_check_exits_two_when_refused(cli, tmp_path):
    """Spec CLI: exit 2 for no markers, a missing README and a missing stored report."""
    cli.readme.write_text("# no markers\n", encoding="utf-8")
    assert cli.run("check") == 2
    assert cli.run("check", "--readme", str(tmp_path / "missing.md")) == 2
    with closing(store.connect(cli.db.path)) as conn, conn:
        conn.execute("DELETE FROM run_reports WHERE kind = 'heldout'")
    cli.readme.write_text(sample_readme(), encoding="utf-8")
    assert cli.run("check") == 2


def test_main_check_exits_two_when_database_file_is_missing(cli, tmp_path):
    """Spec CLI: a fresh clone has no database, so check and render refuse (exit 2, no file made)."""
    harness = yaml.safe_load(cli.harness_path.read_text(encoding="utf-8"))
    harness["db_path"] = str(tmp_path / "nope.sqlite")
    cli.harness_path.write_text(yaml.safe_dump(harness), encoding="utf-8")
    assert cli.run("check") == 2
    assert cli.run("render") == 2
    assert not (tmp_path / "nope.sqlite").exists()


def test_main_render_and_check_write_nothing(cli, capsys):
    """Spec Tests: render and check leave the database bytes and the README unchanged."""
    db_bytes = cli.db.path.read_bytes()
    readme_bytes = cli.readme.read_bytes()
    assert cli.run("render") == 0
    assert "44.44% [18.88%, 73.33%] (4/9)" in capsys.readouterr().out
    cli.run("check")
    assert cli.db.path.read_bytes() == db_bytes
    assert cli.readme.read_bytes() == readme_bytes


def test_main_update_leaves_database_row_counts_unchanged(cli):
    """Spec Data changes: update writes only the README, no table changes."""
    tables = ("runs", "outputs", "scores", "reviews", "run_reports", "judge_claims")

    def counts():
        with closing(store.connect(cli.db.path)) as conn:
            return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    before = counts()
    assert cli.run("update") == 0
    assert counts() == before


# ---------------------------------------------------------------- real README (static, read-only)

def readme_text():
    return README.read_text(encoding="utf-8")


def test_readme_has_each_marker_exactly_once_in_order():
    """Spec Real README: both markers exist exactly once, start before end."""
    text = readme_text()
    assert text.count(START_MARKER) == 1 and text.count(END_MARKER) == 1
    assert text.index(START_MARKER) < text.index(END_MARKER)


@pytest.mark.parametrize("word", ["Results", "Architecture", "Metric definitions",
                                  "Experiments", "Judge calibration", "Limitations"])
def test_readme_has_required_heading(word):
    """Spec outline: each required section has a heading."""
    headings = [l for l in readme_text().splitlines() if l.startswith("#")]
    assert any(word.lower() in h.lower() for h in headings), f"no heading with '{word}'"


def documented_modules():
    return sorted(set(re.findall(r"python -m ([A-Za-z_][\w.]*)", readme_text())))


def test_readme_python_dash_m_modules_all_exist():
    """Spec Real README: every `python -m <module>` named is a real module."""
    modules = documented_modules()
    assert "evalharness.writeup" in modules
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    assert missing == []


def test_readme_named_files_exist():
    """Spec Real README: every configs/*.yaml, experiments/*.md and .env.example named exists."""
    text = readme_text()
    named = set(re.findall(r"configs/[\w./-]+\.yaml", text)) | set(
        re.findall(r"experiments/[\w./-]+\.md", text))
    assert named, "README names no config or experiment file"
    assert [p for p in sorted(named) if not (ROOT / p).exists()] == []
    assert ".env.example" in text and (ROOT / ".env.example").exists()


def test_readme_relative_links_resolve():
    """Spec Real README: every relative markdown link points at an existing path."""
    bad = []
    for target in re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", readme_text()):
        if re.match(r"^(https?:|mailto:|#)", target):
            continue
        if not (ROOT / target.split("#")[0]).exists():
            bad.append(target)
    assert bad == []


def test_readme_states_judge_status_and_verdict():
    """Spec Real README: 'uncalibrated' and 'not_distinguishable' appear."""
    text = readme_text()
    assert "uncalibrated" in text and "not_distinguishable" in text


def test_readme_contains_no_secret_looking_strings():
    """Spec Rules: no sk-, api_key=, Bearer in the README."""
    text = readme_text()
    assert not re.search(r"sk-|api_key\s*=|Bearer ", text)


def test_readme_cv_figures_appear_only_in_hypothesis_sentences():
    """Spec Rules: 25% / 12% / 10-15% only inside a sentence that says 'hypothes' (6.25% is not 25%)."""
    pattern = re.compile(r"(?<![\d.])(25|12)%|10-15%")
    offenders = []
    for paragraph in re.split(r"\n\s*\n", readme_text()):
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            if pattern.search(sentence) and "hypothes" not in sentence.lower():
                offenders.append(sentence.strip()[:120])
    assert offenders == []


# ---------------------------------------------------------------- .env.example

def test_env_example_has_only_the_dummy_llm_api_key():
    """Spec .env.example: exactly LLM_API_KEY, a dummy value, no real-looking secret."""
    lines = [l.strip() for l in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    assert len(lines) == 1
    key, _, value = lines[0].partition("=")
    assert key.strip() == "LLM_API_KEY"
    assert value.strip() and not re.search(r"sk-|Bearer", value)


def test_gitignore_ignores_env_but_not_env_example():
    """Spec Definition of done: .env stays gitignored, .env.example is tracked."""
    def ignored(name):
        return subprocess.run(["git", "check-ignore", "-q", name], cwd=ROOT).returncode == 0
    assert ignored(".env") is True
    assert ignored(".env.example") is False


# ---------------------------------------------------------------- documented commands

def test_documented_evalharness_commands_answer_help_with_exit_zero():
    """Spec Tests: --help of each documented `python -m evalharness...` command exits 0."""
    modules = [m for m in documented_modules() if m.split(".")[0] == "evalharness"]
    assert "evalharness.writeup" in modules
    failures = []
    for module in modules:
        done = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT,
                              capture_output=True, text=True, timeout=120)
        if done.returncode != 0:
            failures.append((module, done.returncode, done.stderr[-200:]))
    assert failures == []


# ---------------------------------------------------------------- layering

def imported_roots(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_writeup_module_imports_nothing_from_rag_llm_or_streamlit():
    """Spec Interfaces: writeup.py imports nothing from rag/, llm/ or streamlit."""
    assert not imported_roots(ROOT / "evalharness" / "writeup.py") & {"rag", "llm", "streamlit"}


def test_rag_package_still_imports_nothing_from_evalharness():
    """Spec Layering: rag/ never imports evalharness/."""
    offenders = [p.name for p in (ROOT / "rag").glob("*.py") if "evalharness" in imported_roots(p)]
    assert offenders == []
