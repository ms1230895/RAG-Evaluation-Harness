"""Tests for Step 02: llm/client.py (cached, retrying, logged LLM wrapper).

Written from .claude/specs/02-llm-wrapper.md. Nothing here reaches Ollama or
the network: load_config is patched to a tmp_path cache, and either _call_api
(cache / schema / logging tests) or _send_request + _get_client (retry tests,
so the real tenacity loop runs) is replaced by a fake.
"""
import ast
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import httpx2
import openai
import pytest
from pydantic import BaseModel

import llm.client as client_module
from llm.client import LLMOutputError, LLMResult, generate, make_cache_key

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-oss:120b-cloud"


class Claims(BaseModel):
    claims: list[str]


class OtherClaims(BaseModel):
    claims: list[str]
    confidence: float = 0.0


# ---------------------------------------------------------------- helpers

class FakeApi:
    """Stand-in for llm.client._call_api: returns a fixed reply and counts calls."""

    def __init__(self, text="hello", prompt_tokens=11, completion_tokens=7):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.calls = []

    def __call__(self, prompt, model, temperature, schema, config):
        self.calls.append(
            {"prompt": prompt, "model": model, "temperature": temperature, "schema": schema}
        )
        return self.text, self.prompt_tokens, self.completion_tokens


class FakeSend:
    """Stand-in for llm.client._send_request: raises queued errors, then succeeds."""

    def __init__(self, errors=None, text="ok", prompt_tokens=5, completion_tokens=3):
        self.errors = list(errors or [])
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.attempts = 0

    def __call__(self, client, prompt, model, temperature, schema):
        self.attempts += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.text, self.prompt_tokens, self.completion_tokens


def _request():
    return httpx2.Request("POST", "http://test")


def _connection_error():
    return openai.APIConnectionError(request=_request())


def _status_error(cls, status):
    response = httpx2.Response(status, request=_request())
    return cls("error", response=response, body=None)


def _query(config, sql, params=()):
    """Run a read-only query against the test cache file; returns all rows."""
    with closing(sqlite3.connect(config["cache_path"])) as conn:
        return conn.execute(sql, params).fetchall()


@pytest.fixture
def config(llm_config, monkeypatch):
    """Patch llm.client.load_config to return the tmp_path config."""
    monkeypatch.setattr(client_module, "load_config", lambda *a, **k: llm_config)
    return llm_config


@pytest.fixture
def fake_api(config, monkeypatch):
    """Patch llm.client._call_api with a counting fake."""
    fake = FakeApi()
    monkeypatch.setattr(client_module, "_call_api", fake)
    return fake


@pytest.fixture
def fake_send(config, monkeypatch):
    """Patch _get_client and _send_request so the real tenacity loop runs offline."""
    monkeypatch.setattr(client_module, "_get_client", lambda *a, **k: object())

    def install(**kwargs):
        fake = FakeSend(**kwargs)
        monkeypatch.setattr(client_module, "_send_request", fake)
        return fake

    return install


# ------------------------------------------------------------- cache key

class TestCacheKey:
    def test_cache_key_is_stable(self):
        """Spec: same four inputs give the same 64-char hex SHA-256 key."""
        k1 = make_cache_key("Say hi", MODEL, 0.0, None)
        k2 = make_cache_key("Say hi", MODEL, 0.0, None)
        assert k1 == k2, "Same inputs must give the same key"
        assert re.fullmatch(r"[0-9a-f]{64}", k1), f"Key is not 64 hex chars: {k1!r}"

    def test_cache_key_stable_with_schema(self):
        """Spec: schema is part of the key and the key is still stable."""
        k1 = make_cache_key("Say hi", MODEL, 0.0, Claims)
        k2 = make_cache_key("Say hi", MODEL, 0.0, Claims)
        assert k1 == k2

    @pytest.mark.parametrize(
        "changed",
        [
            ("Say hello", MODEL, 0.0, None),          # prompt
            ("Say hi", "gemma4:31b-cloud", 0.0, None),  # model
            ("Say hi", MODEL, 0.2, None),             # temperature
            ("Say hi", MODEL, 0.0, Claims),           # schema added
        ],
        ids=["prompt", "model", "temperature", "schema"],
    )
    def test_cache_key_changes_with_each_input(self, changed):
        """Spec: changing model, prompt, temperature or schema each gives a different key."""
        base = make_cache_key("Say hi", MODEL, 0.0, None)
        other = make_cache_key(*changed)
        assert other != base, "Changing one input must change the key"

    def test_cache_key_changes_when_schema_fields_change(self):
        """Spec: schema is hashed as its JSON Schema, so a changed field misses."""
        k1 = make_cache_key("Say hi", MODEL, 0.0, Claims)
        k2 = make_cache_key("Say hi", MODEL, 0.0, OtherClaims)
        assert k1 != k2, "Different schema fields must give different keys"


# ---------------------------------------------------------------- caching

class TestCaching:
    def test_second_identical_call_hits_cache(self, fake_api):
        """Spec: two identical generate() calls hit the API once; second is cached=True."""
        first = generate("Say hi", MODEL, 0.0)
        second = generate("Say hi", MODEL, 0.0)
        assert len(fake_api.calls) == 1, "API must be called only once"
        assert first.cached is False
        assert second.cached is True
        assert second.text == first.text == "hello"

    def test_first_call_result_fields_come_from_api(self, fake_api):
        """Spec: LLMResult carries text, model, token counts, latency_ms and cached=False."""
        result = generate("Say hi", MODEL, 0.0)
        assert isinstance(result, LLMResult)
        assert result.text == "hello"
        assert result.parsed is None, "parsed must be None when no schema is given"
        assert result.model == MODEL
        assert result.prompt_tokens == 11
        assert result.completion_tokens == 7
        assert isinstance(result.latency_ms, float) and result.latency_ms >= 0
        assert result.cached is False

    def test_cache_hit_latency_is_non_negative(self, fake_api):
        """Spec: latency_ms is the wall-clock time of THIS call, also on a hit."""
        generate("Say hi", MODEL, 0.0)
        hit = generate("Say hi", MODEL, 0.0)
        assert isinstance(hit.latency_ms, float) and hit.latency_ms >= 0

    def test_different_temperature_misses_cache(self, fake_api):
        """Spec: same prompt at 0.0 and 0.2 calls the API twice."""
        generate("Say hi", MODEL, 0.0)
        second = generate("Say hi", MODEL, 0.2)
        assert len(fake_api.calls) == 2
        assert second.cached is False

    def test_different_model_misses_cache(self, fake_api):
        """Spec: cache key covers the model, so another model is a miss."""
        generate("Say hi", MODEL, 0.0)
        generate("Say hi", "gemma4:31b-cloud", 0.0)
        assert len(fake_api.calls) == 2

    def test_different_prompt_misses_cache(self, fake_api):
        """Spec: cache key covers the prompt, so another prompt is a miss."""
        generate("Say hi", MODEL, 0.0)
        generate("Say bye", MODEL, 0.0)
        assert len(fake_api.calls) == 2

    def test_different_schema_misses_cache(self, config, monkeypatch):
        """Spec: cache key covers the schema, so a call with a schema misses a schemaless entry."""
        fake = FakeApi(text='{"claims": ["a"]}')
        monkeypatch.setattr(client_module, "_call_api", fake)
        generate("Split it", MODEL, 0.0)
        generate("Split it", MODEL, 0.0, schema=Claims)
        assert len(fake.calls) == 2

    def test_miss_stores_full_row_in_llm_cache(self, fake_api, config):
        """Spec: reply is stored in llm_cache with request fields, raw text, tokens and latency."""
        generate("Say hi", MODEL, 0.3)
        key = make_cache_key("Say hi", MODEL, 0.3, None)
        rows = _query(
            config,
            "SELECT model, temperature, schema_name, prompt, response_text, "
            "prompt_tokens, completion_tokens, latency_ms, created_at "
            "FROM llm_cache WHERE cache_key = ?",
            (key,),
        )
        assert len(rows) == 1, "Expected exactly one cache row for the key"
        model, temp, schema_name, prompt, text, p_tok, c_tok, latency, created = rows[0]
        assert model == MODEL
        assert temp == pytest.approx(0.3)
        assert schema_name is None, "schema_name must be NULL when no schema is given"
        assert prompt == "Say hi"
        assert text == "hello"
        assert (p_tok, c_tok) == (11, 7)
        assert latency >= 0
        assert created, "created_at must be filled"
        assert config["cache_path"].exists()

    def test_cache_row_stores_schema_class_name(self, config, monkeypatch):
        """Spec: llm_cache.schema_name holds the pydantic class name."""
        monkeypatch.setattr(client_module, "_call_api", FakeApi(text='{"claims": ["a"]}'))
        generate("Split it", MODEL, 0.0, schema=Claims)
        key = make_cache_key("Split it", MODEL, 0.0, Claims)
        rows = _query(config, "SELECT schema_name FROM llm_cache WHERE cache_key = ?", (key,))
        assert rows == [("Claims",)]

    def test_cache_stores_raw_reply_text(self, config, monkeypatch):
        """Spec: the cache stores the raw reply, including a code fence."""
        raw = '```json\n{"claims": ["a"]}\n```'
        monkeypatch.setattr(client_module, "_call_api", FakeApi(text=raw))
        generate("Split it", MODEL, 0.0, schema=Claims)
        key = make_cache_key("Split it", MODEL, 0.0, Claims)
        rows = _query(config, "SELECT response_text FROM llm_cache WHERE cache_key = ?", (key,))
        assert rows == [(raw,)]

    def test_cache_hit_with_schema_still_returns_parsed(self, config, monkeypatch):
        """Spec: a structured reply served from the cache is still validated into parsed."""
        fake = FakeApi(text='{"claims": ["a", "b"]}')
        monkeypatch.setattr(client_module, "_call_api", fake)
        generate("Split it", MODEL, 0.0, schema=Claims)
        hit = generate("Split it", MODEL, 0.0, schema=Claims)
        assert hit.cached is True
        assert isinstance(hit.parsed, Claims)
        assert hit.parsed.claims == ["a", "b"]
        assert len(fake.calls) == 1

    def test_generate_passes_inputs_to_api_layer(self, fake_api):
        """Spec: generate() sends the prompt, model, temperature and schema to the API layer."""
        generate("Say hi", MODEL, 0.2, schema=None)
        call = fake_api.calls[0]
        assert call["prompt"] == "Say hi"
        assert call["model"] == MODEL
        assert call["temperature"] == pytest.approx(0.2)
        assert call["schema"] is None


# --------------------------------------------------------- structured output

class TestSchemaParsing:
    def test_schema_reply_is_parsed(self, config, monkeypatch):
        """Spec: valid JSON reply with a schema gives result.parsed as a schema instance."""
        monkeypatch.setattr(client_module, "_call_api", FakeApi(text='{"claims": ["a", "b"]}'))
        result = generate("Split it", MODEL, 0.0, schema=Claims)
        assert isinstance(result.parsed, Claims)
        assert result.parsed.claims == ["a", "b"]
        assert result.text == '{"claims": ["a", "b"]}', "text must stay the raw reply"

    def test_code_fenced_json_is_parsed(self, config, monkeypatch):
        """Spec: one Markdown code fence around the whole reply is stripped before validating."""
        monkeypatch.setattr(
            client_module, "_call_api", FakeApi(text='```json\n{"claims": ["a"]}\n```')
        )
        result = generate("Split it", MODEL, 0.0, schema=Claims)
        assert result.parsed.claims == ["a"]

    def test_plain_fence_without_language_is_parsed(self, config, monkeypatch):
        """Spec: a bare ``` fence (no language tag) wrapping the reply is also stripped."""
        monkeypatch.setattr(
            client_module, "_call_api", FakeApi(text='```\n{"claims": ["a"]}\n```')
        )
        result = generate("Split it", MODEL, 0.0, schema=Claims)
        assert result.parsed.claims == ["a"]

    def test_invalid_schema_reply_raises_and_is_not_cached(self, config, monkeypatch):
        """Spec: non-JSON reply raises LLMOutputError and nothing is cached for that key."""
        monkeypatch.setattr(client_module, "_call_api", FakeApi(text="not json"))
        with pytest.raises(LLMOutputError):
            generate("Split it", MODEL, 0.0, schema=Claims)
        key = make_cache_key("Split it", MODEL, 0.0, Claims)
        rows = _query(config, "SELECT cache_key FROM llm_cache WHERE cache_key = ?", (key,))
        assert rows == [], "A reply that fails validation must not be cached"

    def test_wrong_shape_json_raises_and_is_not_cached(self, config, monkeypatch):
        """Spec: valid JSON of the wrong shape raises LLMOutputError and is not cached."""
        monkeypatch.setattr(client_module, "_call_api", FakeApi(text='{"items": ["a"]}'))
        with pytest.raises(LLMOutputError):
            generate("Split it", MODEL, 0.0, schema=Claims)
        assert _query(config, "SELECT cache_key FROM llm_cache") == []

    def test_wrong_type_json_raises(self, config, monkeypatch):
        """Spec: JSON with the right key but wrong value type fails pydantic validation."""
        monkeypatch.setattr(client_module, "_call_api", FakeApi(text='{"claims": "oops"}'))
        with pytest.raises(LLMOutputError):
            generate("Split it", MODEL, 0.0, schema=Claims)

    def test_invalid_reply_is_not_served_from_cache_next_time(self, config, monkeypatch):
        """Spec: an invalid reply is never cached, so the next identical call hits the API again."""
        fake = FakeApi(text="not json")
        monkeypatch.setattr(client_module, "_call_api", fake)
        for _ in range(2):
            with pytest.raises(LLMOutputError):
                generate("Split it", MODEL, 0.0, schema=Claims)
        assert len(fake.calls) == 2

    def test_no_schema_does_not_validate_text(self, fake_api):
        """Spec: without a schema the reply is returned as-is and parsed is None."""
        fake_api.text = "not json"
        result = generate("Say hi", MODEL, 0.0)
        assert result.text == "not json"
        assert result.parsed is None


# --------------------------------------------------------------- retries

class TestRetries:
    def test_transient_error_is_retried(self, config, fake_send):
        """Spec: APIConnectionError twice then success: call succeeds after 3 attempts."""
        send = fake_send(errors=[_connection_error(), _connection_error()], text="done")
        result = generate("Say hi", MODEL, 0.0)
        assert send.attempts == 3
        assert result.text == "done"
        assert result.cached is False

    def test_gives_up_after_max_attempts(self, config, fake_send):
        """Spec: always failing -> error propagates after exactly max_attempts tries."""
        config["max_attempts"] = 4
        send = fake_send(errors=[_connection_error() for _ in range(10)])
        with pytest.raises(openai.APIConnectionError):
            generate("Say hi", MODEL, 0.0)
        assert send.attempts == 4, "Must try exactly max_attempts times"

    def test_max_attempts_one_means_no_retry(self, config, fake_send):
        """Spec: max_attempts counts the first try, so 1 means no retry."""
        config["max_attempts"] = 1
        send = fake_send(errors=[_connection_error()])
        with pytest.raises(openai.APIConnectionError):
            generate("Say hi", MODEL, 0.0)
        assert send.attempts == 1

    @pytest.mark.parametrize(
        "make_error",
        [
            lambda: openai.APIConnectionError(request=_request()),
            lambda: openai.APITimeoutError(request=_request()),
            lambda: _status_error(openai.RateLimitError, 429),
            lambda: _status_error(openai.InternalServerError, 500),
        ],
        ids=["connection", "timeout", "rate_limit", "server_error"],
    )
    def test_each_transient_error_type_is_retried(self, config, fake_send, make_error):
        """Spec: APIConnectionError, APITimeoutError, RateLimitError, InternalServerError are retried."""
        send = fake_send(errors=[make_error()], text="done")
        result = generate("Say hi", MODEL, 0.0)
        assert send.attempts == 2
        assert result.text == "done"

    @pytest.mark.parametrize(
        "make_error",
        [
            lambda: _status_error(openai.BadRequestError, 400),
            lambda: _status_error(openai.NotFoundError, 404),
        ],
        ids=["bad_request", "not_found"],
    )
    def test_permanent_errors_are_not_retried(self, config, fake_send, make_error):
        """Spec: BadRequestError and NotFoundError are never retried."""
        error = make_error()
        send = fake_send(errors=[error])
        with pytest.raises(type(error)):
            generate("Say hi", MODEL, 0.0)
        assert send.attempts == 1, "Permanent errors must not be retried"

    def test_failed_call_is_not_cached(self, config, fake_send):
        """Spec: a failed call is never cached."""
        config["max_attempts"] = 2
        fake_send(errors=[_connection_error() for _ in range(5)])
        with pytest.raises(openai.APIConnectionError):
            generate("Say hi", MODEL, 0.0)
        assert _query(config, "SELECT cache_key FROM llm_cache") == []

    def test_schema_failure_is_not_retried(self, config, monkeypatch):
        """Spec: LLMOutputError is never retried: the API layer is hit once per generate()."""
        fake = FakeApi(text="not json")
        monkeypatch.setattr(client_module, "_call_api", fake)
        with pytest.raises(LLMOutputError):
            generate("Split it", MODEL, 0.0, schema=Claims)
        assert len(fake.calls) == 1

    def test_retried_success_is_cached(self, config, fake_send):
        """Spec: after a retried success the reply is cached, so the next call needs no request."""
        send = fake_send(errors=[_connection_error()], text="done")
        generate("Say hi", MODEL, 0.0)
        second = generate("Say hi", MODEL, 0.0)
        assert second.cached is True
        assert send.attempts == 2, "1 failure + 1 success, nothing for the cached call"


# --------------------------------------------------------------- logging

class TestLogging:
    def test_every_call_is_logged(self, fake_api, config):
        """Spec: one miss then one hit give 2 llm_calls rows, cached 0 then 1, tokens from the fake."""
        generate("Say hi", MODEL, 0.0)
        generate("Say hi", MODEL, 0.0)
        rows = _query(
            config,
            "SELECT cached, prompt_tokens, completion_tokens, model FROM llm_calls ORDER BY id",
        )
        assert len(rows) == 2, "Expected one llm_calls row per generate() call"
        assert [r[0] for r in rows] == [0, 1]
        # Token counts are those the fake returned (11 prompt, 7 completion).
        assert rows[0][1:3] == (11, 7)
        assert rows[1][1:3] == (11, 7)
        assert all(r[3] == MODEL for r in rows)

    def test_logged_rows_link_to_cache_key_and_have_latency(self, fake_api, config):
        """Spec: llm_calls rows carry the cache_key, an ISO timestamp and a non-negative latency."""
        generate("Say hi", MODEL, 0.0)
        generate("Say hi", MODEL, 0.0)
        key = make_cache_key("Say hi", MODEL, 0.0, None)
        rows = _query(config, "SELECT cache_key, called_at, latency_ms FROM llm_calls ORDER BY id")
        assert len(rows) == 2
        for cache_key, called_at, latency in rows:
            assert cache_key == key
            assert called_at, "called_at must be filled"
            assert latency >= 0

    def test_distinct_requests_log_separate_cache_rows(self, fake_api, config):
        """Spec: llm_cache has one row per distinct request."""
        generate("Say hi", MODEL, 0.0)
        generate("Say bye", MODEL, 0.0)
        generate("Say hi", MODEL, 0.0)
        assert len(_query(config, "SELECT cache_key FROM llm_cache")) == 2
        assert len(_query(config, "SELECT id FROM llm_calls")) == 3

    def test_tables_are_created_on_first_use(self, fake_api, config):
        """Spec: llm_cache and llm_calls are created on first use in the configured path."""
        assert not config["cache_path"].exists()
        generate("Say hi", MODEL, 0.0)
        tables = {r[0] for r in _query(config, "SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"llm_cache", "llm_calls"} <= tables

    def test_cache_path_from_config_is_used(self, fake_api, config, tmp_path):
        """Spec: the cache file location comes from config cache_path (tmp_path here)."""
        generate("Say hi", MODEL, 0.0)
        assert config["cache_path"].parent == tmp_path
        assert config["cache_path"].exists()


# ----------------------------------------------------------- architecture

def test_rag_does_not_import_evalharness():
    """Spec: no file under rag/ has an import of evalharness (architecture guard)."""
    for path in (PROJECT_ROOT / "rag").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(n.split(".")[0] == "evalharness" for n in names), (
                f"{path} imports from evalharness"
            )
