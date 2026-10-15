"""Step 12: held-out guard (pre-registration) and held-out confirmation report.

Spec: .claude/specs/12-held-out-confirmation.md. Offline: tmp DBs, tmp config files,
fake system. Never touches data/results.sqlite, the LLM cache or a real model.
Report tests patch `evalharness.report.current_versions` (name where build_confirmation
uses it) so the fixture DB can carry short fake scorer versions.
"""

import ast
import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evalharness import heldout, runner, store
from evalharness.checks import ScoringRefused
from evalharness.experiments import ExperimentRefused
from evalharness.heldout import HeldoutRefused
from evalharness.metrics import wilson_interval
from evalharness.schemas import Query
from evalharness.testset import save_test_set

PINNED = {"checks": "checks-t", "faithfulness": "faith-t", "relevance": "rel-t"}
BASE_CFG = "configs/v0.yaml"
CHOSEN_CFG = "configs/v1_prompt_citations.yaml"
TEST_SET_SHA = "ab" * 32
REFUSALS = (HeldoutRefused, ScoringRefused, ExperimentRefused)


def sha_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def heldout_dict(baseline, chosen, **override):
    """A valid pre-registration dict for two config paths (sha entries are fake hex unless given)."""
    cfg = {
        "selection_rule": "test rule",
        "baseline_config": baseline, "chosen_config": chosen,
        "allowed_configs": [baseline, chosen],
        "config_sha256": {baseline: "1" * 64, chosen: "2" * 64},
        "max_runs_per_config": 1,
        "pinned_versions": dict(PINNED),
        "dev_runs": {"baseline": "dev-base", "chosen": "dev-chosen"},
        "significance_level": 0.05,
        "output": {"markdown_path": "out.md"},
        "amendments": [],
    }
    cfg.update(override)
    return cfg


def add_run_row(conn, run_id, *, split="heldout", config_path, status="completed"):
    """A bare runs row (enough for the guard, which only reads split / config_path)."""
    store.start_run(conn, run_id=run_id, split=split, config={"name": run_id},
                    config_path=config_path, git_commit="c" * 40, git_dirty=False,
                    test_set_sha256=TEST_SET_SHA, system_info={}, n_queries=1,
                    llm_calls_start_id=0)
    if status != "running":
        store.finish_run(conn, run_id, status=status, n_ok=1, n_failed=0, llm_calls_summary={})


def write_harness_yaml(tmp_path, db_path, heldout_path):
    """Copy of the real harness config (read-only) with tmp db_path and heldout_config."""
    real = runner.load_harness_config()
    real["db_path"] = str(db_path)
    real["heldout_config"] = str(heldout_path)
    path = tmp_path / "harness_t12.yaml"
    path.write_text(yaml.safe_dump(real), encoding="utf-8")
    return path


# ================================================================ config validation

@pytest.fixture
def cfg_files(tmp_path):
    """(write(cfg_dict) -> harness dict) helper that stores a pre-registration YAML in tmp_path."""
    def write(cfg):
        path = tmp_path / "heldout_t12.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return {"heldout_config": str(path)}
    return write


def test_load_heldout_config_valid_file_loads_unchanged(cfg_files):
    """Spec Tests/Config: a valid heldout.yaml loads."""
    cfg = heldout_dict(BASE_CFG, CHOSEN_CFG)
    assert heldout.load_heldout_config(cfg_files(cfg)) == cfg


def _drop_key(c):
    del c["pinned_versions"]


def _zero_runs(c):
    c["max_runs_per_config"] = 0


def _baseline_not_allowed(c):
    c["allowed_configs"] = [c["chosen_config"]]
    c["config_sha256"].pop(c["baseline_config"])


def _missing_sha(c):
    c["config_sha256"].pop(c["chosen_config"])


@pytest.mark.parametrize("mutate, key", [
    (_drop_key, "pinned_versions"),
    (_zero_runs, "max_runs_per_config"),
    (_baseline_not_allowed, "baseline_config"),
    (_missing_sha, "config_sha256"),
])
def test_load_heldout_config_bad_file_raises_value_error_naming_key(cfg_files, mutate, key):
    """Spec Tests/Config: missing key, max < 1, baseline not allowed, missing sha -> ValueError."""
    cfg = heldout_dict(BASE_CFG, CHOSEN_CFG)
    mutate(cfg)
    with pytest.raises(ValueError, match=key):
        heldout.load_heldout_config(cfg_files(cfg))


def test_load_heldout_config_harness_without_heldout_config_key_raises_value_error():
    """Spec Interfaces (runner fails closed): no heldout_config key -> ValueError, not silent."""
    with pytest.raises(ValueError, match="heldout_config"):
        heldout.load_heldout_config({})


# ================================================================ guard

@pytest.fixture
def guard(tmp_path, monkeypatch):
    """Two real tmp config files, a pre-registration over them, an empty results DB."""
    base, chosen = tmp_path / "v0.yaml", tmp_path / "chosen.yaml"
    base.write_text("name: v0\n", encoding="utf-8")
    chosen.write_text("name: chosen\n", encoding="utf-8")
    cfg = heldout_dict(str(base), str(chosen),
                       config_sha256={str(base): sha_of(base), str(chosen): sha_of(chosen)})
    monkeypatch.setattr(heldout, "current_scorer_versions", lambda harness: dict(PINNED))
    conn = store.connect(tmp_path / "guard.sqlite")
    with closing(conn):
        yield SimpleNamespace(conn=conn, cfg=cfg, base=str(base), chosen=str(chosen))


def test_check_heldout_allowed_no_prior_runs_returns_used_max_and_sha(guard):
    """Spec Interfaces: allowed with 0 prior rows -> {used 0, max 1, config_sha256 of the file bytes}."""
    info = heldout.check_heldout_allowed(guard.conn, guard.chosen, guard.cfg, {})
    assert info == {"used": 0, "max": 1, "config_sha256": sha_of(guard.chosen)}


def test_check_heldout_allowed_unregistered_config_is_refused(guard, tmp_path):
    """Spec Guard: config_path not in allowed_configs is refused."""
    other = tmp_path / "other.yaml"
    other.write_text("name: other\n", encoding="utf-8")
    with pytest.raises(HeldoutRefused):
        heldout.check_heldout_allowed(guard.conn, str(other), guard.cfg, {})


def test_check_heldout_allowed_changed_config_bytes_is_refused(guard):
    """Spec Guard: sha256 of the file differs from the pinned value -> refused."""
    Path(guard.chosen).write_text("name: chosen\ntweak: 1\n", encoding="utf-8")
    with pytest.raises(HeldoutRefused):
        heldout.check_heldout_allowed(guard.conn, guard.chosen, guard.cfg, {})


@pytest.mark.parametrize("status", ["completed", "aborted", "invalid", "interrupted", "running"])
def test_check_heldout_allowed_prior_heldout_row_of_any_status_refuses_second_run(guard, status):
    """Spec Guard: a held-out row of ANY status counts as used (max_runs_per_config 1)."""
    add_run_row(guard.conn, "h-1", config_path=guard.chosen, status=status)
    assert heldout.count_heldout_runs(guard.conn, guard.chosen) == 1
    with pytest.raises(HeldoutRefused):
        heldout.check_heldout_allowed(guard.conn, guard.chosen, guard.cfg, {})


def test_check_heldout_allowed_other_config_and_dev_rows_do_not_count(guard):
    """Spec Guard: a held-out row of a different config and a dev row of this config are ignored."""
    add_run_row(guard.conn, "h-other", config_path=guard.base)
    add_run_row(guard.conn, "d-same", split="dev", config_path=guard.chosen)
    info = heldout.check_heldout_allowed(guard.conn, guard.chosen, guard.cfg, {})
    assert info["used"] == 0


def test_check_heldout_allowed_limit_two_allows_second_and_refuses_third(guard):
    """Spec Guard: used < max_runs_per_config is allowed; used >= max refused."""
    guard.cfg["max_runs_per_config"] = 2
    add_run_row(guard.conn, "h-1", config_path=guard.chosen)
    assert heldout.check_heldout_allowed(guard.conn, guard.chosen, guard.cfg, {})["used"] == 1
    add_run_row(guard.conn, "h-2", config_path=guard.chosen)
    with pytest.raises(HeldoutRefused):
        heldout.check_heldout_allowed(guard.conn, guard.chosen, guard.cfg, {})


def test_check_heldout_allowed_changed_scorer_version_is_refused(guard, monkeypatch):
    """Spec Guard: current scorer versions != pinned_versions -> refused (no judge tuning)."""
    monkeypatch.setattr(heldout, "current_scorer_versions",
                        lambda harness: {**PINNED, "faithfulness": "faith-v3-changed"})
    with pytest.raises(HeldoutRefused):
        heldout.check_heldout_allowed(guard.conn, guard.chosen, guard.cfg, {})


def test_main_check_prints_allowed_then_refused_after_a_used_run(guard, tmp_path, capsys):
    """Spec CLI: check exits 0 'allowed' before the run and 1 'refused' after it; never starts a run."""
    heldout_yaml = tmp_path / "heldout_cli.yaml"
    heldout_yaml.write_text(yaml.safe_dump(guard.cfg), encoding="utf-8")
    harness = write_harness_yaml(tmp_path, tmp_path / "guard.sqlite", heldout_yaml)
    argv = ["--harness-config", str(harness), "check", "--config", guard.chosen]
    assert heldout.main(argv) == 0
    assert "allowed" in capsys.readouterr().out
    add_run_row(guard.conn, "h-1", config_path=guard.chosen)
    assert heldout.main(argv) == 1
    assert "refused" in capsys.readouterr().err
    runs = guard.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert runs == 1, "check must not create runs"


# ================================================================ runner

FAKES = "tests.test_05_runner_and_store:"


class RunnerEnv:
    """Tmp harness for runner.main(): fake system, 8 test queries (3 held-out), pre-registration."""

    def __init__(self, tmp_path, testset_config):
        self.db_path = tmp_path / "t12_results.sqlite"
        self.pipeline = tmp_path / "fake_pipeline.yaml"
        self.pipeline.write_text(yaml.safe_dump({"name": "fakecfg", "top_k": 3}), encoding="utf-8")
        (tmp_path / "ts.yaml").write_text(yaml.safe_dump(testset_config), encoding="utf-8")
        self.heldout = tmp_path / "heldout_t12.yaml"
        self.write_registration(self.pipeline)
        self.harness = tmp_path / "harness_t12.yaml"
        self.write_harness(heldout_config=str(self.heldout))

    def write_registration(self, config_path):
        p = str(config_path)
        cfg = heldout_dict(p, p, allowed_configs=[p], config_sha256={p: sha_of(config_path)})
        self.heldout.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def write_harness(self, **extra):
        cfg = {"db_path": str(self.db_path), "test_set_config": str(self.heldout.parent / "ts.yaml"),
               "max_consecutive_failures": 2, "error_max_chars": 100,
               "system": {"answer": FAKES + "fake_answer", "load_config": FAKES + "fake_load_config",
                          "describe": None}, **extra}
        self.harness.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def argv(self, split, *flags):
        return ["--config", str(self.pipeline), "--split", split,
                "--harness-config", str(self.harness), *flags]

    def rows(self, sql="SELECT * FROM runs"):
        if not self.db_path.exists():
            return []
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(sql)]


@pytest.fixture
def renv(tmp_path, testset_config, ready_queries, llm_config, monkeypatch):
    save_test_set(ready_queries, testset_config["path"])
    monkeypatch.setattr(runner, "git_info", lambda: ("a" * 40, False))
    monkeypatch.setattr(runner, "load_llm_config", lambda *a, **k: llm_config)
    monkeypatch.setattr(runner.heldout, "current_scorer_versions", lambda harness: dict(PINNED))
    env = RunnerEnv(tmp_path, testset_config)
    env.heldout_ids = sorted(q.id for q in ready_queries if q.split == "heldout")
    return env


def test_main_heldout_allowed_run_writes_one_row_with_exactly_the_heldout_queries(renv):
    """Spec Runner: allowed run -> one runs row split heldout, outputs for exactly the held-out ids."""
    assert runner.main(renv.argv("heldout", "--allow-heldout")) == 0
    runs = renv.rows()
    assert len(runs) == 1 and runs[0]["split"] == "heldout"
    ids = sorted(r["query_id"] for r in renv.rows("SELECT query_id FROM outputs"))
    assert ids == renv.heldout_ids and len(ids) == 3


def test_main_heldout_second_run_of_same_config_exits_one_and_writes_nothing(renv, capsys):
    """Spec Runner: a second held-out invocation of the same config is refused, no new rows."""
    assert runner.main(renv.argv("heldout", "--allow-heldout")) == 0
    n_outputs = len(renv.rows("SELECT * FROM outputs"))
    assert runner.main(renv.argv("heldout", "--allow-heldout")) == 1
    assert len(renv.rows()) == 1
    assert len(renv.rows("SELECT * FROM outputs")) == n_outputs
    assert capsys.readouterr().err


def test_main_heldout_with_allow_dirty_exits_one_and_writes_nothing(renv):
    """Spec Runner: --allow-dirty is refused on the held-out split (even on a clean tree)."""
    assert runner.main(renv.argv("heldout", "--allow-heldout", "--allow-dirty")) == 1
    assert renv.rows() == []


def test_main_heldout_without_pre_registration_key_fails_closed(renv):
    """Spec Runner: no valid heldout_config in the harness -> held-out run refused, nothing written."""
    renv.write_harness()   # no heldout_config key
    assert runner.main(renv.argv("heldout", "--allow-heldout")) == 1
    assert renv.rows() == []


def test_main_heldout_changed_config_bytes_exit_one_and_dev_runs_unaffected(renv):
    """Spec Runner: a tweaked config is refused on held-out; the same config still runs on dev."""
    renv.pipeline.write_text("name: fakecfg\ntop_k: 3\n# tweaked after pre-registration\n",
                             encoding="utf-8")
    assert runner.main(renv.argv("heldout", "--allow-heldout")) == 1
    assert renv.rows() == []
    assert runner.main(renv.argv("dev")) == 0
    assert [r["split"] for r in renv.rows()] == ["dev"]


# ================================================================ verdict rule

@pytest.mark.parametrize("improved, regressed, p, expected", [
    (6, 0, 0.03125, "supported_better"),
    (0, 6, 0.03125, "supported_worse"),
    (3, 0, 0.25, "not_distinguishable"),      # lower point estimate, p >= level: never "better"
    (5, 1, 0.05, "not_distinguishable"),      # p == level is not below the level
    (0, 0, None, "not_distinguishable"),      # no flips: p undefined
    (0, 4, 0.125, "not_distinguishable"),     # worse point estimate, p >= level: never "worse"
])
def test_verdict_label_follows_the_pre_registered_rule(improved, regressed, p, expected):
    """Spec Decision 3: better/worse only when p < level, direction from the flips."""
    assert heldout.verdict_label(improved, regressed, p, 0.05) == expected


# ================================================================ report fixture

H_IDS = [f"q1{n:02d}" for n in range(1, 11)]
D_IDS = [f"q2{n:02d}" for n in range(1, 11)]
TYPES = ["lookup"] * 4 + ["multi_chunk"] * 2 + ["paraphrased"] * 2 + ["unanswerable"] * 2


@pytest.fixture
def make_env(tmp_path, make_query, monkeypatch):
    """Factory: tmp DB with 10 held-out + 10 dev queries and four completed, clean, scored runs.

    Hand layout: query index 0-3 lookup, 4-5 multi_chunk, 6-7 paraphrased, 8-9 unanswerable
    (n = 10, so one answer = 10 points). Ungrounded sets are given per run as id tuples.
    """
    import evalharness.report as report_mod

    monkeypatch.setattr(report_mod, "current_versions", lambda harness: dict(PINNED))
    harness = runner.load_harness_config()   # real blocks (checks, report, experiments), read-only
    conn = store.connect(tmp_path / "t12.sqlite")

    def build(*, held_base=("q101", "q102", "q103"), held_chosen=(),
              dev_base=("q201", "q202"), dev_chosen=()):
        queries = []
        for ids, split in ((H_IDS, "heldout"), (D_IDS, "dev")):
            for qid, qtype in zip(ids, TYPES):
                sources = [] if qtype == "unanswerable" else ["leave_policy"]
                queries.append(Query.model_validate(make_query(
                    id=qid, question=f"Question {qid}?", type=qtype, reference_answer=f"Ref {qid}.",
                    source_doc_ids=sources, split=split, reviewed=True)))
        store.sync_queries(conn, queries)

        def add_run(run_id, ungrounded, *, split="heldout", config_path=CHOSEN_CFG, dirty=False,
                    sha=TEST_SET_SHA, status="completed"):
            ids = H_IDS if split == "heldout" else D_IDS
            store.start_run(conn, run_id=run_id, split=split, config={"name": run_id},
                            config_path=config_path, git_commit="c" * 40, git_dirty=dirty,
                            test_set_sha256=sha, system_info={}, n_queries=10, llm_calls_start_id=0)
            chunks = [{"chunk_id": "leave_policy#000", "doc_id": "leave_policy",
                       "text": "chunk", "score": 0.5}]
            checks, faith, rel = [], [], []
            for qid, qtype in zip(ids, TYPES):
                store.write_output(conn, run_id=run_id, query_id=qid, answer=f"Answer {run_id} {qid}.",
                                   chunks=chunks, latency_ms=10.0, error=None)
                ans = qtype != "unanswerable"
                na = "not applicable"
                checks += [
                    {"query_id": qid, "metric": "empty_answer", "value": 0.0, "reasoning": "r"},
                    {"query_id": qid, "metric": "refusal", "value": 0.0, "reasoning": "r"},
                    {"query_id": qid, "metric": "latency_ms", "value": 10.0, "reasoning": "r"},
                    {"query_id": qid, "metric": "in_context", "value": 1.0 if ans else None,
                     "reasoning": "r" if ans else na},
                    {"query_id": qid, "metric": "hit@4", "value": 1.0 if ans else None,
                     "reasoning": "r" if ans else na},
                    {"query_id": qid, "metric": "reciprocal_rank", "value": 1.0 if ans else None,
                     "reasoning": "r" if ans else na},
                ]
                bad = qid in ungrounded
                faith += [
                    {"query_id": qid, "metric": "faithfulness", "value": 0.5 if bad else 1.0,
                     "reasoning": f"faith {qid}"},
                    {"query_id": qid, "metric": "ungrounded", "value": 1.0 if bad else 0.0,
                     "reasoning": f"JUDGE-REASON-{run_id}-{qid}"},
                    {"query_id": qid, "metric": "n_claims", "value": 2.0, "reasoning": "r"},
                ]
                rel += [{"query_id": qid, "metric": "relevance", "value": 1.0, "reasoning": "r"},
                        {"query_id": qid, "metric": "irrelevant", "value": 0.0, "reasoning": "r"}]
            if status != "running":
                store.finish_run(conn, run_id, status=status, n_ok=10, n_failed=0,
                                 llm_calls_summary={})
            store.write_scores(conn, run_id, PINNED["checks"], checks)
            store.write_scores(conn, run_id, PINNED["faithfulness"], faith)
            store.write_scores(conn, run_id, PINNED["relevance"], rel)

        add_run("dev-base", dev_base, split="dev", config_path=BASE_CFG)
        add_run("dev-chosen", dev_chosen, split="dev", config_path=CHOSEN_CFG)
        add_run("held-base", held_base, config_path=BASE_CFG)
        add_run("held-chosen", held_chosen, config_path=CHOSEN_CFG)

        def fail_judge(run_id, qid):
            """Store a faithfulness judge failure the way Step 7 does: NULL values, reason text."""
            with conn:
                conn.execute("UPDATE scores SET value = NULL, reasoning = ? WHERE run_id = ?"
                             " AND query_id = ? AND version = ?",
                             ("judge failed: LLMOutputError: bad json", run_id, qid,
                              PINNED["faithfulness"]))

        cfg = heldout_dict(BASE_CFG, CHOSEN_CFG)
        heldout_yaml = tmp_path / "heldout_report.yaml"
        heldout_yaml.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return SimpleNamespace(
            conn=conn, db_path=tmp_path / "t12.sqlite", cfg=cfg, harness=harness,
            add_run=add_run, fail_judge=fail_judge, tmp_path=tmp_path,
            harness_yaml=write_harness_yaml(tmp_path, tmp_path / "t12.sqlite", heldout_yaml),
            report=lambda base="held-base", run="held-chosen": heldout.build_confirmation(
                conn, base, run, cfg, harness),
            argv=lambda *a: ["--harness-config", str(write_harness_yaml(
                tmp_path, tmp_path / "t12.sqlite", heldout_yaml)), *a])

    yield build
    conn.close()


def llm_guard(monkeypatch):
    """Fail the test if confirm makes any LLM call (spec: confirm is read-only on scores)."""
    def boom(*a, **k):
        raise AssertionError("confirm must not call the LLM")
    monkeypatch.setattr("llm.client.generate", boom)
    monkeypatch.setattr("evalharness.judges.generate", boom)


# ================================================================ report refusals

def test_build_confirmation_valid_runs_succeed_control(make_env):
    """Control for the refusal tests: the untouched fixture builds a heldout report."""
    env = make_env()
    report = env.report()
    assert report["kind"] == "heldout"
    assert report["run_ids"] == {"baseline": "held-base", "chosen": "held-chosen"}


@pytest.mark.parametrize("case", ["unknown_run", "not_completed", "dev_split", "other_test_set",
                                  "unjudged", "version_mismatch", "dirty", "wrong_config"])
def test_build_confirmation_invalid_input_is_refused(make_env, case):
    """Spec build_confirmation: each listed defect is refused, not reported around."""
    env = make_env()
    chosen = "held-chosen"
    if case == "unknown_run":
        chosen = "no-such-run"
    elif case == "not_completed":
        env.add_run("x", (), status="running")
        chosen = "x"
    elif case == "dev_split":
        chosen = "dev-chosen"
    elif case == "other_test_set":
        env.add_run("x", (), sha="cd" * 32)
        chosen = "x"
    elif case == "unjudged":
        env.fail_judge("held-chosen", "q105")
    elif case == "version_mismatch":
        env.cfg["pinned_versions"]["relevance"] = "rel-other"
    elif case == "dirty":
        env.add_run("x", (), dirty=True)
        chosen = "x"
    elif case == "wrong_config":
        env.add_run("x", (), config_path="configs/v1_rewrite.yaml")
        chosen = "x"
    with pytest.raises(REFUSALS):
        env.report(run=chosen)


# ================================================================ report numbers

def test_build_confirmation_rates_flips_and_p_match_hand_values(make_env):
    """Spec Tests/Numbers: 3/10 vs 0/10, 3 improved 0 regressed, exact p = 2 * (1/2)^3 = 0.25."""
    report = make_env().report()
    base = report["runs"]["baseline"]["ungrounded"]["all"]
    chosen = report["runs"]["chosen"]["ungrounded"]["all"]
    assert (base["successes"], base["n"]) == (3, 10)
    assert (chosen["successes"], chosen["n"]) == (0, 10)
    # Wilson 95%: 3/10 -> [0.1078, 0.6032]; 0/10 -> [0, z^2/(n+z^2)] = [0, 0.2775]
    assert (base["ci_low"], base["ci_high"]) == pytest.approx((0.1078, 0.6032), abs=1e-3)
    assert chosen["ci_low"] == pytest.approx(0.0, abs=1e-9)
    assert chosen["ci_high"] == pytest.approx(0.2775, abs=1e-3)
    assert (base["ci_low"], base["ci_high"]) == pytest.approx(wilson_interval(3, 10, 0.95))
    comp = report["comparison"]
    assert comp["flips"]["counts"]["improved"] == 3
    assert comp["flips"]["counts"]["regressed"] == 0
    assert comp["p_value"] == pytest.approx(0.25)
    assert report["verdict"]["label"] == "not_distinguishable"
    assert report["n_queries"] == 10


def test_build_confirmation_no_flips_gives_not_distinguishable_without_p(make_env):
    """Spec Tests/Numbers: b = c = 0 gives p None or 1.0 (mcnemar_exact) and not_distinguishable."""
    report = make_env(held_base=("q101",), held_chosen=("q101",)).report()
    assert report["verdict"]["label"] == "not_distinguishable"
    assert report["comparison"]["p_value"] in (None, 1.0)
    assert report["comparison"]["flips"]["counts"]["improved"] == 0


def test_build_confirmation_six_improved_gives_supported_better(make_env):
    """Spec verdict rule end to end: 6 improved, 0 regressed -> p = 2 * (1/2)^6 = 0.03125 < 0.05."""
    ids = ("q101", "q102", "q103", "q104", "q105", "q106")
    report = make_env(held_base=ids).report()
    assert report["comparison"]["p_value"] == pytest.approx(0.03125)
    assert report["verdict"]["label"] == "supported_better"


def test_build_confirmation_per_type_counts_sum_to_overall(make_env):
    """Spec Tests/Numbers: per-type k and n sum to the overall k and n (4+2+2+2 = 10)."""
    report = make_env(held_base=("q101", "q105", "q107")).report()
    groups = report["runs"]["baseline"]["ungrounded"]
    types = ("lookup", "multi_chunk", "paraphrased", "unanswerable")
    assert sum(groups[t]["successes"] for t in types) == groups["all"]["successes"] == 3
    assert sum(groups[t]["n"] for t in types) == groups["all"]["n"] == 10


def test_build_confirmation_inside_dev_ci_flags_true_and_false(make_env):
    """Spec Report 4: held-out rate vs dev Wilson CI is a plain fact.

    Dev v0 2/10 -> CI [0.0567, 0.5098]; held-out v0 3/10 = 0.30 inside -> True.
    Dev chosen 5/10 -> CI [0.2366, 0.7634]; held-out chosen 0/10 = 0 outside -> False.
    """
    env = make_env(dev_base=("q201", "q202"), dev_chosen=("q201", "q202", "q203", "q204", "q205"))
    report = env.report()
    assert report["inside_dev_ci"] == {"baseline": True, "chosen": False}
    assert report["dev"]["chosen"]["ungrounded"]["successes"] == 5
    assert report["dev"]["baseline"]["ungrounded"]["ci_low"] == pytest.approx(0.0567, abs=1e-3)


def test_build_confirmation_rate_at_lower_bound_zero_is_inside_zero_dev_ci(make_env):
    """Spec Report 4 boundary: 0/10 on held-out vs 0/10 on dev (CI [0, 0.2775]) is inside."""
    report = make_env(held_chosen=(), dev_chosen=()).report()
    assert report["inside_dev_ci"]["chosen"] is True


def test_build_confirmation_lists_every_ungrounded_answer_with_reasoning(make_env):
    """Spec Report 6: ungrounded held-out answers listed with question, answer, chunk ids, reasoning."""
    report = make_env().report()
    listed = report["listed_answers"]["baseline"]["ungrounded"]
    assert sorted(i["query_id"] for i in listed) == ["q101", "q102", "q103"]
    first = listed[0]
    assert first["question"] == "Question q101?"
    assert first["chunk_ids"] == ["leave_policy#000"]
    assert first["reasoning"] == "JUDGE-REASON-held-base-q101"
    assert report["listed_answers"]["chosen"]["ungrounded"] == []


# ================================================================ honesty

def test_format_markdown_states_uncalibrated_judge_n_one_answer_and_cv_line(make_env):
    """Spec Tests/Honesty: uncalibrated judge, held-out n, one-answer points, CV hypothesis line."""
    report = make_env().report()
    md = heldout.format_confirmation_markdown(report)
    assert "uncalibrated judge" in md
    assert re.search(r"one answer is 10 percentage points", md, re.IGNORECASE)   # 100 / n = 10
    assert re.search(r"10 held-out queries", md)
    assert "25%" in md and "12%" in md and "hypothesis" in md.lower()
    assert "not_distinguishable" in md
    assert "uncalibrated judge" in heldout.format_confirmation(report)


def test_format_markdown_every_fraction_and_percent_traces_back_to_report(make_env):
    """Spec Tests/Honesty: each k/n and decimal percent token in the markdown exists in the report."""
    report = make_env(held_chosen=("q102",), dev_chosen=("q203",)).report()
    md = heldout.format_confirmation_markdown(report)
    pairs, floats = set(), []

    def walk(node):
        if isinstance(node, dict):
            if "successes" in node and "n" in node:
                pairs.add((node["successes"], node["n"]))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, float):
            floats.append(node)

    walk(json.loads(json.dumps(report)))
    for k, n in re.findall(r"\((\d+)/(\d+)\)", md):
        assert (int(k), int(n)) in pairs, f"{k}/{n} not in report"
    for token in re.findall(r"(\d+\.\d+)%", md):
        assert any(abs(100 * x - float(token)) < 0.006 for x in floats), f"{token}% not in report"


# ================================================================ storage / CLI confirm

def test_main_confirm_writes_one_heldout_report_row_and_no_llm_call(make_env, monkeypatch, capsys):
    """Spec Storage: confirm stores exactly one run_reports row, kind heldout, and calls no LLM."""
    llm_guard(monkeypatch)
    env = make_env()
    assert heldout.main(env.argv("confirm", "--baseline-run", "held-base", "--run", "held-chosen")) == 0
    out = capsys.readouterr().out
    assert "3/10" in out and "0/10" in out
    rows = store.get_run_reports(env.conn, "held-chosen")
    assert [r["kind"] for r in rows] == ["heldout"]
    assert rows[0]["report"]["verdict"]["label"] == "not_distinguishable"
    assert env.conn.execute("SELECT COUNT(*) FROM run_reports").fetchone()[0] == 1


def test_main_confirm_refusal_exits_one_and_writes_nothing(make_env, capsys):
    """Spec CLI: a refused confirm exits 1 with no run_reports row and no markdown file."""
    env = make_env()
    md = env.tmp_path / "out.md"
    argv = env.argv("confirm", "--baseline-run", "held-base", "--run", "dev-chosen",
                    "--markdown", str(md))
    assert heldout.main(argv) == 1
    assert env.conn.execute("SELECT COUNT(*) FROM run_reports").fetchone()[0] == 0
    assert not md.exists()
    assert capsys.readouterr().err


def test_main_confirm_markdown_refuses_overwrite_without_force(make_env):
    """Spec Storage: --markdown refuses to overwrite an existing file; --force overwrites."""
    env = make_env()
    md = env.tmp_path / "out.md"
    md.write_text("OLD", encoding="utf-8")
    args = ["confirm", "--baseline-run", "held-base", "--run", "held-chosen", "--markdown", str(md)]
    assert heldout.main(env.argv(*args)) == 1
    assert md.read_text(encoding="utf-8") == "OLD"
    assert env.conn.execute("SELECT COUNT(*) FROM run_reports").fetchone()[0] == 0
    assert heldout.main(env.argv(*args, "--force")) == 0
    assert "uncalibrated judge" in md.read_text(encoding="utf-8")


def test_main_confirm_log_appends_two_rows_then_refuses_duplicate(make_env):
    """Spec Storage: --log appends one row per run; a second confirm for the same runs is refused."""
    env = make_env()
    log = env.tmp_path / "log.md"
    header = "| date | config | variable | run | result | note |\n|---|---|---|---|---|---|\n"
    log.write_text(header, encoding="utf-8")
    args = ["confirm", "--baseline-run", "held-base", "--run", "held-chosen", "--log", str(log)]
    assert heldout.main(env.argv(*args)) == 0
    text = log.read_text(encoding="utf-8")
    assert text.count("held-base") == 1 and text.count("held-chosen") == 1
    assert heldout.main(env.argv(*args)) == 1
    assert log.read_text(encoding="utf-8") == text
    assert env.conn.execute("SELECT COUNT(*) FROM run_reports").fetchone()[0] == 1


def test_log_rows_returns_one_row_per_run_with_run_id_and_rate(make_env):
    """Spec Interfaces: log_rows gives one row for each of the two held-out runs."""
    report = make_env().report()
    rows = heldout.log_rows(report, "2026-10-16T12:00:00Z")
    assert len(rows) == 2
    assert "held-base" in rows[0] and "3/10" in rows[0]
    assert "held-chosen" in rows[1] and "0/10" in rows[1]


# ================================================================ layering

def test_heldout_module_imports_nothing_from_rag_or_llm():
    """Spec Layering: evalharness/heldout.py has no import of rag or llm (AST check, any depth)."""
    source = (Path(__file__).resolve().parents[1] / "evalharness" / "heldout.py").read_text("utf-8")
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert not roots & {"rag", "llm"}
