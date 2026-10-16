"""Step 14 tests: one-command pipeline (evalharness/cli.py, __main__.py, runner.run, store).

Offline. The stage entry points (runner.run, checks.main, judges.main, regression.main) are
replaced by fakes that record their argv and return scripted exit codes; the judge configs,
the digest check, git and the accepted-run lookup are faked too. The DB is a tmp file.
Expected values come from the spec's exit-code table and stage order.
"""
import ast
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evalharness import checks, cli, judges, regression, runner, store
from evalharness.testset import save_test_set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "run-14-001"
GEN_MODEL = "gen-model:cloud"
JUDGE_MODEL = "judge-model:cloud"
PINNED = "pinned-digest"
STAGES = ["run", "checks", "faithfulness", "relevance", "gate"]


# ---------------------------------------------------------------- importable fakes (system block)

def fake_answer(question, config):
    """Importable fake answer(): a valid contract dict."""
    return {"answer": f"Answer to: {question}",
            "chunks": [{"chunk_id": "leave_policy#000", "doc_id": "leave_policy",
                        "text": "chunk text", "score": 0.9}],
            "latency_ms": 12.5}


def fake_load_config(path):
    """Importable fake load_config(): reads the YAML it is given (missing file -> OSError)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- fixtures

@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Any LLM call from this step is a bug: both import sites raise."""
    def boom(*args, **kwargs):
        raise AssertionError("Step 14 must not call the LLM")
    monkeypatch.setattr("llm.client.generate", boom)
    monkeypatch.setattr("evalharness.judges.generate", boom)


def _add_run(conn, run_id, *, split="dev", status="completed"):
    store.start_run(conn, run_id=run_id, split=split,
                    config={"name": "fakecfg", "generation": {"model": GEN_MODEL}},
                    config_path="configs/fake.yaml", git_commit="abc1234", git_dirty=False,
                    test_set_sha256="ab" * 32, system_info={}, n_queries=3, llm_calls_start_id=0)
    if status != "running":
        store.finish_run(conn, run_id, status=status, n_ok=3, n_failed=0, llm_calls_summary={})


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fake pipeline world: `.calls` (list of (stage, argv)), `.codes` (exit code per stage),
    knobs `.dirty`, `.judge_model`, `.live_digest`, `.digest_error`, `.accepted`, `.versions`,
    `.run_result`; `.harness` / `.auto` dicts, `.harness_path`, `.config_path`, `.db_path`,
    `.argv_run(*flags)`, `.argv_score(run_id, *flags)`, `.add_run(...)`, `.judge_version()`.
    """
    db_path = tmp_path / "results.sqlite"
    config_path = tmp_path / "fake_pipeline.yaml"
    config_path.write_text(yaml.safe_dump({"name": "fakecfg", "generation": {"model": GEN_MODEL}}),
                           encoding="utf-8")
    auto = {"allowed_splits": ["dev"], "retry_failed_judges_in_score": True}
    harness = {
        "db_path": str(db_path), "test_set_config": str(tmp_path / "ts.yaml"),
        "max_consecutive_failures": 5, "error_max_chars": 500,
        "system": {"answer": f"{__name__}:fake_answer", "load_config": f"{__name__}:fake_load_config",
                   "describe": None},
        "judges_config": str(tmp_path / "judges-unused.yaml"),
        "automation": auto,
    }
    harness_path = tmp_path / "harness_14.yaml"
    harness_path.write_text(yaml.safe_dump(harness), encoding="utf-8")

    e = SimpleNamespace(
        calls=[], codes={s: 0 for s in STAGES}, dirty=False, judge_model=JUDGE_MODEL,
        live_digest=PINNED, digest_error=None, accepted={"run_id": "acc-1"},
        versions={"faithfulness": "faith-t14", "relevance": "rel-t14"},
        run_result=(0, RUN_ID), harness=harness, auto=auto, harness_path=str(harness_path),
        config_path=str(config_path), db_path=db_path, tmp_path=tmp_path)

    def fake_run(argv):
        e.calls.append(("run", list(argv)))
        return e.run_result

    def fake_checks(argv):
        e.calls.append(("checks", list(argv)))
        return e.codes["checks"]

    def fake_judges(argv):
        e.calls.append((argv[0], list(argv)))
        return e.codes[argv[0]]

    def fake_gate(argv):
        e.calls.append(("gate", list(argv)))
        return e.codes["gate"]

    def judge_cfg(kind):
        return lambda path: {"model": e.judge_model, "model_digest": PINNED,
                             "version": e.versions[kind]}

    def fake_digest(model):
        if e.digest_error is not None:
            raise e.digest_error
        return e.live_digest

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(runner, "git_info", lambda: ("abc1234", e.dirty))
    monkeypatch.setattr(checks, "main", fake_checks)
    monkeypatch.setattr(judges, "main", fake_judges)
    monkeypatch.setattr(judges, "load_judges_config", judge_cfg("faithfulness"))
    monkeypatch.setattr(judges, "load_relevance_config", judge_cfg("relevance"))
    monkeypatch.setattr(judges, "get_model_digest", fake_digest)
    monkeypatch.setattr(regression, "main", fake_gate)
    monkeypatch.setattr(regression, "load_regression_config", lambda h: {})
    monkeypatch.setattr(regression, "get_accepted_run", lambda conn, split: e.accepted)

    conn = store.connect(db_path)
    _add_run(conn, RUN_ID)
    e.conn = conn

    def add_run(run_id, **kw):
        _add_run(conn, run_id, **kw)

    def judge_version(run_id=RUN_ID):
        with closing(sqlite3.connect(db_path)) as c:
            return c.execute("SELECT judge_version FROM runs WHERE run_id = ?",
                             (run_id,)).fetchone()[0]

    e.add_run = add_run
    e.judge_version = judge_version
    e.argv_run = lambda *flags, split="dev": ["run", "--config", e.config_path, "--split", split,
                                             "--harness-config", e.harness_path, *flags]
    e.argv_score = lambda run_id=RUN_ID, *flags: ["score", "--run-id", run_id,
                                                  "--harness-config", e.harness_path, *flags]
    e.stages = lambda: [name for name, _ in e.calls]
    e.argv_of = lambda stage: next(a for n, a in e.calls if n == stage)
    yield e
    conn.close()


def preflight(env, **over):
    kw = {"config": env.config_path, "split": "dev", "gate": True, "allow_dirty": False, **over}
    return cli.preflight_run(kw["config"], kw["split"], env.harness, env.auto, gate=kw["gate"],
                             allow_dirty=kw["allow_dirty"])


# ---------------------------------------------------------------- automation config

def test_load_automation_config_valid_block_returns_it(env):
    """Spec Config: a valid automation block loads unchanged."""
    assert cli.load_automation_config(env.harness) == {
        "allowed_splits": ["dev"], "retry_failed_judges_in_score": True}


@pytest.mark.parametrize("block", [
    None,                                                                   # missing block
    {"allowed_splits": ["dev"]},                                            # missing key
    {"allowed_splits": "dev", "retry_failed_judges_in_score": True},        # bad type
    {"allowed_splits": ["dev"], "retry_failed_judges_in_score": "yes"},     # bad type
    {"allowed_splits": ["dev", "heldout"], "retry_failed_judges_in_score": True},  # never heldout
])
def test_load_automation_config_bad_block_raises_value_error(env, block):
    """Spec Config: missing key, bad types and heldout in allowed_splits are refused."""
    harness = dict(env.harness)
    if block is None:
        harness.pop("automation")
    else:
        harness["automation"] = block
    with pytest.raises(ValueError):
        cli.load_automation_config(harness)


def test_real_harness_config_has_valid_automation_block():
    """Spec Config: configs/harness.yaml holds a valid block that excludes heldout."""
    with open(PROJECT_ROOT / "configs" / "harness.yaml", encoding="utf-8") as f:
        harness = yaml.safe_load(f)
    auto = cli.load_automation_config(harness)
    assert "heldout" not in auto["allowed_splits"]


# ---------------------------------------------------------------- preflight

def test_preflight_run_clean_setup_returns_no_problems(env):
    """Spec Preflight: nothing wrong -> empty list."""
    assert preflight(env) == []


@pytest.mark.parametrize("case, mutate", [
    ("split_not_allowed", lambda e: {"split": "heldout"}),
    ("dirty_tree", lambda e: setattr(e, "dirty", True) or {}),
    ("unknown_config", lambda e: {"config": str(e.tmp_path / "nope.yaml")}),
    ("judge_equals_generator", lambda e: setattr(e, "judge_model", GEN_MODEL) or {}),
    ("digest_mismatch", lambda e: setattr(e, "live_digest", "other-digest") or {}),
    ("server_unreachable", lambda e: setattr(e, "digest_error", OSError("refused")) or {}),
    ("no_accepted_run", lambda e: setattr(e, "accepted", None) or {}),
])
def test_preflight_run_each_problem_alone_is_reported_once(env, case, mutate):
    """Spec Preflight: each problem is found alone, exactly one message."""
    over = mutate(env)
    assert len(preflight(env, **over)) == 1, case


def test_preflight_run_dirty_tree_allowed_with_allow_dirty(env):
    """Spec: --allow-dirty waives the dirty-tree problem."""
    env.dirty = True
    assert preflight(env, allow_dirty=True) == []


def test_preflight_run_no_accepted_run_is_fine_with_no_gate(env):
    """Spec Decision 2: no accepted run only blocks when the gate is on."""
    env.accepted = None
    assert preflight(env, gate=False) == []


def test_run_several_problems_are_all_listed_and_no_stage_runs(env, capsys):
    """Spec Preflight: several problems all listed; nothing started; exit 2."""
    env.dirty = True
    env.live_digest = "other-digest"
    env.accepted = None
    assert len(preflight(env)) == 3
    assert cli.main(env.argv_run()) == 2
    err = capsys.readouterr().err
    assert err.count("preflight:") == 3
    assert env.calls == []


def test_run_heldout_split_exits_2_and_writes_no_run(env):
    """Spec DoD: run --split heldout exits 2 from the preflight with nothing written."""
    before = env.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert cli.main(env.argv_run(split="heldout")) == 2
    assert env.calls == []
    assert env.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == before


# ---------------------------------------------------------------- order, run id, exit table

def test_run_all_stages_ok_calls_in_order_passes_run_id_and_exits_0(env, capsys):
    """Spec: order run, checks, faithfulness, relevance, gate; run id reaches every later stage."""
    assert cli.main(env.argv_run()) == 0
    assert env.stages() == STAGES
    for stage in STAGES[1:]:
        argv = env.argv_of(stage)
        assert argv[argv.index("--run-id") + 1] == RUN_ID
    out = capsys.readouterr().out
    assert RUN_ID in out
    for stage in ("faithfulness", "relevance"):   # counts come from stored rows; fake judges store none
        assert f"stage {stage}" in out


def test_run_passes_harness_config_to_every_stage(env):
    """Spec: --harness-config reaches every stage."""
    cli.main(env.argv_run())
    for stage in STAGES:
        argv = env.argv_of(stage)
        assert argv[argv.index("--harness-config") + 1] == env.harness_path, stage


def test_run_judge_stages_get_their_subcommand_name(env):
    """Spec: judges.main is called with faithfulness, then relevance."""
    cli.main(env.argv_run())
    assert env.argv_of("faithfulness")[0] == "faithfulness"
    assert env.argv_of("relevance")[0] == "relevance"
    assert env.argv_of("gate")[0] == "check"


@pytest.mark.parametrize("stage, code, expected_exit, called", [
    ("checks", 1, 2, ["run", "checks"]),
    ("faithfulness", 1, 2, ["run", "checks", "faithfulness"]),
    ("faithfulness", 2, 2, ["run", "checks", "faithfulness"]),
    ("relevance", 2, 2, ["run", "checks", "faithfulness", "relevance"]),
    ("gate", 1, 1, STAGES),
    ("gate", 2, 2, STAGES),
])
def test_run_stage_exit_code_follows_table_and_stops_later_stages(env, capsys, stage, code,
                                                                  expected_exit, called):
    """Spec Stop rule + exit table: non-success stops the pipeline; gate FAIL is 1, others 2."""
    env.codes[stage] = code
    assert cli.main(env.argv_run()) == expected_exit
    assert env.stages() == called
    assert stage in capsys.readouterr().out


@pytest.mark.parametrize("result", [(1, None), (2, RUN_ID)])
def test_run_runner_refusal_or_abort_exits_2_and_stops(env, result):
    """Spec Stop rule: runner refused (1, None) or aborted / invalid (2, id) -> exit 2, no later stage."""
    env.run_result = result
    assert cli.main(env.argv_run()) == 2
    assert env.stages() == ["run"]


def test_run_partial_stage_prints_resume_command(env, capsys):
    """Spec Rules: a partial stage names the exact command to resume."""
    env.codes["relevance"] = 2
    cli.main(env.argv_run())
    assert f"python -m evalharness score --run-id {RUN_ID}" in capsys.readouterr().out


def test_run_no_gate_skips_gate_exits_0_and_says_so(env, capsys):
    """Spec Exit codes: --no-gate returns 0, gate not called, 'gate skipped' printed."""
    env.codes["gate"] = 1
    assert cli.main(env.argv_run("--no-gate")) == 0
    assert env.stages() == STAGES[:-1]
    assert "gate skipped" in capsys.readouterr().out


# ---------------------------------------------------------------- flag routing

def test_run_flags_reach_only_their_stage(env, tmp_path):
    """Spec Flags: --allow-dirty to runner only; --against/--markdown/--force/--no-store to gate only."""
    md = str(tmp_path / "gate.md")
    cli.main(env.argv_run("--allow-dirty", "--against", "acc-9", "--markdown", md, "--force",
                          "--no-store"))
    assert "--allow-dirty" in env.argv_of("run")
    gate = env.argv_of("gate")
    assert gate[gate.index("--against") + 1] == "acc-9"
    assert gate[gate.index("--markdown") + 1] == md
    assert "--force" in gate and "--no-store" in gate
    gate_only = {"--against", "acc-9", "--markdown", md, "--force", "--no-store"}
    for stage in ("run", "checks", "faithfulness", "relevance"):
        assert not gate_only & set(env.argv_of(stage)), stage
    for stage in ("checks", "faithfulness", "relevance", "gate"):
        assert "--allow-dirty" not in env.argv_of(stage), stage


def test_run_without_allow_dirty_does_not_pass_it_to_runner(env):
    """Spec Rules: --allow-dirty is passed through only when given."""
    cli.main(env.argv_run())
    assert "--allow-dirty" not in env.argv_of("run")


@pytest.mark.parametrize("argv_name", ["argv_run", "argv_score"])
def test_main_allow_heldout_flag_does_not_exist(env, argv_name):
    """Spec Flags: --allow-heldout is rejected by argparse on run and score."""
    argv = getattr(env, argv_name)("--allow-heldout")
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    assert env.calls == []


# ---------------------------------------------------------------- score

def test_score_completed_run_calls_stages_after_run_and_passes_retry_failed(env):
    """Spec score: checks, judges (with --retry-failed), gate for an existing run; no runner call."""
    assert cli.main(env.argv_score()) == 0
    assert env.stages() == STAGES[1:]
    assert "--retry-failed" in env.argv_of("faithfulness")
    assert "--retry-failed" in env.argv_of("relevance")


def test_score_retry_failed_not_passed_when_config_false(env):
    """Spec Config: retry_failed_judges_in_score false -> no --retry-failed."""
    env.harness["automation"]["retry_failed_judges_in_score"] = False
    Path(env.harness_path).write_text(yaml.safe_dump(env.harness), encoding="utf-8")
    cli.main(env.argv_score())
    assert "--retry-failed" not in env.argv_of("faithfulness")
    assert "--retry-failed" not in env.argv_of("relevance")


@pytest.mark.parametrize("run_id, kw", [
    ("no-such-run", {}),
    ("run-held", {"split": "heldout"}),
    ("run-aborted", {"status": "aborted"}),
])
def test_score_refuses_unknown_heldout_or_not_completed_run_with_exit_2(env, run_id, kw):
    """Spec score: unknown id, held-out run and non-completed run exit 2 with no stage called."""
    if kw:
        env.add_run(run_id, **kw)
    assert cli.main(env.argv_score(run_id)) == 2
    assert env.calls == []


def test_score_stop_rule_second_judge_partial_exits_2(env):
    """Spec score: same stop rule as run; gate not called after a partial judge."""
    env.codes["relevance"] = 2
    assert cli.main(env.argv_score()) == 2
    assert env.stages() == ["checks", "faithfulness", "relevance"]


# ---------------------------------------------------------------- judge_version

def test_run_fills_judge_version_after_both_judges_succeed(env):
    """Spec Data: judge_version = faithfulness=<v>;relevance=<v>."""
    assert env.judge_version() is None
    cli.main(env.argv_run())
    assert env.judge_version() == "faithfulness=faith-t14;relevance=rel-t14"


@pytest.mark.parametrize("failing", ["faithfulness", "relevance"])
def test_run_leaves_judge_version_null_when_a_judge_is_partial(env, failing):
    """Spec judge_version: not filled when a judge stage is partial."""
    env.codes[failing] = 2
    cli.main(env.argv_run())
    assert env.judge_version() is None


def test_score_with_different_judge_version_keeps_recorded_value(env):
    """Spec: a recorded judge_version is never overwritten by another value."""
    cli.main(env.argv_score())
    env.versions["relevance"] = "rel-t14-changed"
    assert cli.main(env.argv_score()) != 0
    assert env.judge_version() == "faithfulness=faith-t14;relevance=rel-t14"


def test_set_judge_version_sets_null_and_same_value_is_noop(env):
    """Spec store: fills a NULL value; the same value again changes nothing."""
    store.set_judge_version(env.conn, RUN_ID, "faithfulness=a;relevance=b")
    store.set_judge_version(env.conn, RUN_ID, "faithfulness=a;relevance=b")
    assert env.judge_version() == "faithfulness=a;relevance=b"


def test_set_judge_version_different_value_raises_and_keeps_old(env):
    """Spec store: a different value raises ValueError."""
    store.set_judge_version(env.conn, RUN_ID, "v1")
    with pytest.raises(ValueError):
        store.set_judge_version(env.conn, RUN_ID, "v2")
    assert env.judge_version() == "v1"


def test_set_judge_version_unknown_run_raises(env):
    """Spec store: unknown run id raises ValueError."""
    with pytest.raises(ValueError):
        store.set_judge_version(env.conn, "no-such-run", "v1")


# ---------------------------------------------------------------- runner refactor

@pytest.fixture
def runner_env(tmp_path, testset_config, ready_queries, llm_config, monkeypatch):
    """Real runner.run with a fake system, tmp DB, clean git; returns (argv, db_path)."""
    save_test_set(ready_queries, testset_config["path"])
    monkeypatch.setattr(runner, "git_info", lambda: ("a" * 40, False))
    monkeypatch.setattr(runner, "load_llm_config", lambda *a, **k: llm_config)
    db_path = tmp_path / "runner_results.sqlite"
    pipeline = tmp_path / "pipe.yaml"
    pipeline.write_text(yaml.safe_dump({"name": "fakecfg", "top_k": 3}), encoding="utf-8")
    ts_cfg = tmp_path / "ts_cfg.yaml"
    ts_cfg.write_text(yaml.safe_dump(testset_config), encoding="utf-8")
    harness = tmp_path / "harness_runner.yaml"
    harness.write_text(yaml.safe_dump({
        "db_path": str(db_path), "test_set_config": str(ts_cfg),
        "max_consecutive_failures": 2, "error_max_chars": 100,
        "system": {"answer": f"{__name__}:fake_answer",
                   "load_config": f"{__name__}:fake_load_config", "describe": None},
    }), encoding="utf-8")

    def argv(*flags):
        return ["--config", str(pipeline), "--split", "dev", "--harness-config", str(harness),
                *flags]
    return SimpleNamespace(argv=argv, db_path=db_path)


def _run_ids(db_path):
    with closing(sqlite3.connect(db_path)) as conn:
        return [r[0] for r in conn.execute("SELECT run_id FROM runs")]


def test_runner_run_completed_returns_zero_and_the_stored_run_id(runner_env):
    """Spec runner: run() -> (0, run_id) for a completed run; the id is the stored one."""
    code, run_id = runner.run(runner_env.argv())
    assert code == 0
    assert _run_ids(runner_env.db_path) == [run_id]


def test_runner_run_refusal_returns_one_and_none_and_main_keeps_exit_code(runner_env, monkeypatch):
    """Spec runner: refusal -> (1, None) with nothing written; main() still exits 1."""
    monkeypatch.setattr(runner, "git_info", lambda: ("a" * 40, True))
    assert runner.run(runner_env.argv()) == (1, None)
    assert runner.main(runner_env.argv()) == 1
    assert not runner_env.db_path.exists() or _run_ids(runner_env.db_path) == []


def test_runner_main_completed_run_still_returns_zero(runner_env):
    """Spec runner: main(argv) == run(argv)[0]; a completed run exits 0 as before."""
    assert runner.main(runner_env.argv()) == 0


# ---------------------------------------------------------------- module entry and layering

def test_module_entry_help_exits_0_without_running_a_stage():
    """Spec Module entry: python -m evalharness --help exits 0."""
    result = subprocess.run([sys.executable, "-m", "evalharness", "--help"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0
    assert "run" in result.stdout and "score" in result.stdout


def test_cli_py_imports_nothing_from_rag_or_llm_directly():
    """Spec Layering: cli.py reaches the system only through the harness config."""
    tree = ast.parse((PROJECT_ROOT / "evalharness" / "cli.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert not imported & {"rag", "llm"}


def test_coverage_reads_only_the_named_judges_rows(env):
    """Regression (found by the first real run): right after the faithfulness stage the
    relevance rows do not exist yet, so the status line must not need all three scorers."""
    rows = [("q001", "faithfulness", 1.0, "faithfulness 1.00 >= 0.8"),
            ("q002", "faithfulness", None, "judge failed: boom"),
            ("q003", "faithfulness", None, "not judged: failed output")]
    with env.conn:
        for query_id in ("q001", "q002", "q003"):
            env.conn.execute("INSERT OR IGNORE INTO queries (id, question, type, reference_answer,"
                             " source_doc_ids, split, reviewed) VALUES (?, 'q', 'lookup', 'r', ?,"
                             " 'dev', 1)", (query_id, '["leave_policy"]'))
        for query_id, metric, value, reasoning in rows:
            env.conn.execute(
                "INSERT INTO scores (run_id, query_id, metric, version, value, reasoning, scored_at)"
                " VALUES (?, ?, ?, ?, ?, ?, '2026-10-16T00:00:00+00:00')",
                (RUN_ID, query_id, metric, env.versions["faithfulness"], value, reasoning))
    text = cli._coverage(env.harness, RUN_ID, "faithfulness")
    assert text == "1 judged, 1 judge failed, 1 not judged"
    # No relevance rows at all: still answers instead of raising.
    assert cli._coverage(env.harness, RUN_ID, "relevance") == "0 judged, 0 judge failed, 0 not judged"

