# tests/test_02_llm_wrapper.py
# Step 02: llm/client.py (cached, retried, logged LLM calls).
# Written from .claude/specs/02-llm-wrapper.md. No test reaches Ollama or the network.
import ast
import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

# httpx2 is not a direct dependency: it arrives through openai==3.31, which
# builds its requests and responses with httpx2 (not httpx).
import httpx2
import openai
import pytest
import yaml
from pydantic import BaseModel

from llm import client
from llm.client import (
    REQUIRED_CONFIG_KEYS,
    ApiReply,
    CallStats,
    LLMOutputError,
    generate,
    load_config,
    make_cache_key,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL = "fake-model:cloud"
PROMPT = "Say hi in 3 words"
FAKE_REQUEST = httpx2.Request("POST", "http://test")


class Claims(BaseModel):
    """Small schema used by the structured-output tests."""
    claims: list[str]


class OtherClaims(BaseModel):
    """Same field as Claims plus one more, so its JSON Schema differs."""
    claims: list[str]
    confidence: float


# ---------------------------------------------------------------- helpers

@pytest.fixture(autouse=True)
def block_real_client(monkeypatch):
    """Safety net: no test may build the real openai client (and so reach any endpoint)."""
    def refuse(*args, **kwargs):
        """Fail loudly if the real client is requested."""
        raise AssertionError("test tried to build the real openai client")
    monkeypatch.setattr(client, "_get_client", refuse)


def make_reply(text="hello", prompt_tokens=10, completion_tokens=5,
               finish_reason="stop", model_latency_ms=1234.0):
    """Build an ApiReply with sensible defaults (a normal, finished reply)."""
    return ApiReply(text=text, prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    finish_reason=finish_reason, model_latency_ms=model_latency_ms)


def fetch(config, sql, params=()):
    """Run one read query on the test cache DB and return rows as dicts."""
    with closing(sqlite3.connect(config["cache_path"])) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def call_rows(config):
    """Return every llm_calls row of the test DB, oldest first."""
    return fetch(config, "SELECT * FROM llm_calls ORDER BY id")


def cache_rows(config):
    """Return every llm_cache row of the test DB."""
    return fetch(config, "SELECT * FROM llm_cache")


def count_cache_rows(config, key):
    """Count llm_cache rows stored under one cache key."""
    rows = fetch(config, "SELECT COUNT(*) AS n FROM llm_cache WHERE cache_key = ?", (key,))
    return rows[0]["n"]


def key_for(config, prompt=PROMPT, model=MODEL, temperature=0.0, schema=None, model_digest=None):
    """Compute the cache key a generate() call with these inputs should use."""
    return make_cache_key(prompt, model, temperature, schema, config["base_url"], model_digest)


def status_error(error_class, code, message="boom"):
    """Build an openai status error (429, 500, 400, ...) the way openai==3.31 needs it."""
    response = httpx2.Response(code, request=FAKE_REQUEST)
    return error_class(message, response=response, body=None)


def connection_error():
    """Build a transient openai.APIConnectionError."""
    return openai.APIConnectionError(request=FAKE_REQUEST)


class FakeSend:
    """Stand-in for llm.client._send_request: plays back outcomes, counts calls.

    Each outcome is an exception (raised) or an ApiReply (returned). The last
    outcome is repeated once the list is used up.
    """

    def __init__(self, outcomes):
        """Store the outcomes to play back and start the call counter."""
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, client_obj, prompt, model, temperature, schema,
                 seed=None, reasoning_effort=None):
        """Count the call, then raise or return the next outcome."""
        self.calls += 1
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def install_fake_send(monkeypatch, outcomes):
    """Fake _get_client and _send_request so the REAL tenacity retry loop runs."""
    send = FakeSend(outcomes)
    send.client_args = []

    def fake_get_client(base_url, timeout_s):
        """Record the client arguments and return a dummy object."""
        send.client_args.append((base_url, timeout_s))
        return object()

    monkeypatch.setattr(client, "_get_client", fake_get_client)
    monkeypatch.setattr(client, "_send_request", send)
    return send


class FakeOpenAIClient:
    """Fake client object: chat.completions.create(**kwargs) records kwargs, returns a canned response."""

    def __init__(self, content="hi", finish_reason="stop", usage=(7, 3)):
        """Build the canned response (usage=None means the server reported no usage)."""
        self.create_kwargs = []
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message, finish_reason=finish_reason)
        usage_obj = (None if usage is None
                     else SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]))
        self._response = SimpleNamespace(choices=[choice], usage=usage_obj)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        """Record the request kwargs and return the canned response."""
        self.create_kwargs.append(kwargs)
        return self._response


def write_config_file(tmp_path, config, drop=None, **overrides):
    """Write config (minus `drop`, plus `overrides`) as YAML and return the file path."""
    data = {key: (str(value) if isinstance(value, Path) else value)
            for key, value in config.items()}
    data.update(overrides)
    if drop is not None:
        del data[drop]
    path = tmp_path / "llm_test.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------- cache key

class TestCacheKey:
    BASE = dict(prompt="Say hi", model="m:cloud", temperature=0.0, schema=None,
                base_url="http://fake-llm.test/v1", model_digest=None,
                seed=None, reasoning_effort=None)

    def test_cache_key_is_stable(self):
        """Spec: same inputs give the same 64-char hex SHA-256 key."""
        first = make_cache_key(**self.BASE)
        second = make_cache_key(**self.BASE)
        assert first == second, "Same inputs must give the same key"
        assert re.fullmatch(r"[0-9a-f]{64}", first), "Key must be 64 lowercase hex characters"

    def test_cache_key_matches_spec_formula(self):
        """Spec: key = sha256(json.dumps({base_url, model, model_digest, prompt, reasoning_effort, seed, temperature, schema}, sort_keys=True))."""
        expected_input = json.dumps(
            {"base_url": "http://fake-llm.test/v1", "model": "m:cloud", "model_digest": None,
             "prompt": "Say hi", "reasoning_effort": None, "seed": None,
             "temperature": 0.0, "schema": None},
            sort_keys=True,
        )
        expected = hashlib.sha256(expected_input.encode("utf-8")).hexdigest()
        assert make_cache_key(**self.BASE) == expected

    def test_cache_key_digest_defaults_to_none(self):
        """Spec: model_digest defaults to None, which equals passing None."""
        args = {k: v for k, v in self.BASE.items() if k != "model_digest"}
        assert make_cache_key(**args) == make_cache_key(**self.BASE)

    @pytest.mark.parametrize("field, new_value", [
        ("base_url", "http://other-host.test/v1"),
        ("model", "other-model:cloud"),
        ("model_digest", "abc123"),
        ("prompt", "Say bye"),
        ("temperature", 0.1),
        ("schema", Claims),
    ])
    def test_cache_key_changes_with_each_input(self, field, new_value):
        """Spec: changing any one of the six inputs gives a different key."""
        changed = dict(self.BASE, **{field: new_value})
        assert make_cache_key(**changed) != make_cache_key(**self.BASE), (
            f"Changing {field} must change the cache key"
        )

    def test_cache_key_changes_when_schema_fields_change(self):
        """Spec: the schema is keyed by its JSON Schema, so a changed field misses."""
        with_claims = make_cache_key(**dict(self.BASE, schema=Claims))
        with_other = make_cache_key(**dict(self.BASE, schema=OtherClaims))
        assert with_claims != with_other


# ---------------------------------------------------------------- caching

class TestCaching:
    def test_second_identical_call_hits_cache(self, fake_api, llm_config):
        """Spec: second identical call does not call the API and returns cached=True."""
        first = generate(PROMPT, MODEL, 0.0, config=llm_config)
        second = generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert len(fake_api.calls) == 1, "API must be called once, not twice"
        assert first.cached is False
        assert second.cached is True
        assert second.text == first.text

    def test_cache_hit_reports_original_model_latency(self, fake_api, llm_config):
        """Spec: a hit returns the original model_latency_ms; latency_ms is separate wall time."""
        miss = generate(PROMPT, MODEL, 0.0, config=llm_config)
        hit = generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert miss.model_latency_ms == pytest.approx(1234.0)
        assert hit.model_latency_ms == pytest.approx(1234.0)
        # latency_ms is wall time: assert only type and sign, never a value.
        assert isinstance(hit.latency_ms, float) and hit.latency_ms >= 0
        assert isinstance(miss.latency_ms, float) and miss.latency_ms >= 0

    def test_different_temperature_misses_cache(self, fake_api, llm_config):
        """Spec: same prompt at temperature 0.0 and 0.2 calls the API twice."""
        generate(PROMPT, MODEL, 0.0, config=llm_config)
        generate(PROMPT, MODEL, 0.2, config=llm_config)
        assert len(fake_api.calls) == 2

    def test_different_model_misses_cache(self, fake_api, llm_config):
        """Spec: the model is part of the cache key."""
        generate(PROMPT, "model-a:cloud", 0.0, config=llm_config)
        generate(PROMPT, "model-b:cloud", 0.0, config=llm_config)
        assert len(fake_api.calls) == 2

    def test_different_prompt_misses_cache(self, fake_api, llm_config):
        """Spec: the prompt is part of the cache key."""
        generate("first prompt", MODEL, 0.0, config=llm_config)
        generate("second prompt", MODEL, 0.0, config=llm_config)
        assert len(fake_api.calls) == 2

    def test_different_base_url_misses_cache(self, fake_api, llm_config):
        """Spec: the same model served from another endpoint never reuses replies."""
        other = dict(llm_config, base_url="http://other-host.test/v1")
        generate(PROMPT, MODEL, 0.0, config=llm_config)
        generate(PROMPT, MODEL, 0.0, config=other)
        assert len(fake_api.calls) == 2

    def test_different_schema_misses_cache(self, fake_api, llm_config):
        """Spec: the schema is part of the cache key."""
        fake_api.reply = make_reply(text='{"claims": ["a"]}')
        generate(PROMPT, MODEL, 0.0, config=llm_config)
        generate(PROMPT, MODEL, 0.0, schema=Claims, config=llm_config)
        assert len(fake_api.calls) == 2

    def test_cache_persists_across_calls_with_new_config_dict(self, fake_api, llm_config):
        """Spec: the cache lives in the SQLite file, so an equal config dict still hits."""
        generate(PROMPT, MODEL, 0.0, config=llm_config)
        generate(PROMPT, MODEL, 0.0, config=dict(llm_config))
        assert len(fake_api.calls) == 1

    def test_model_digest_is_stored_and_misses_cache(self, fake_api, llm_config):
        """Spec: same request with digest 'a' then 'b' calls the API twice; both tables store the digest."""
        first = generate(PROMPT, MODEL, 0.0, config=llm_config, model_digest="a")
        second = generate(PROMPT, MODEL, 0.0, config=llm_config, model_digest="b")
        assert len(fake_api.calls) == 2, "A new digest must miss the cache"
        assert (first.model_digest, second.model_digest) == ("a", "b")
        assert sorted(r["model_digest"] for r in cache_rows(llm_config)) == ["a", "b"]
        assert [r["model_digest"] for r in call_rows(llm_config)] == ["a", "b"]

    def test_same_model_digest_hits_cache(self, fake_api, llm_config):
        """Spec: an identical digest is part of an identical request, so it hits."""
        generate(PROMPT, MODEL, 0.0, config=llm_config, model_digest="a")
        hit = generate(PROMPT, MODEL, 0.0, config=llm_config, model_digest="a")
        assert len(fake_api.calls) == 1
        assert hit.cached is True and hit.model_digest == "a"

    def test_no_digest_is_stored_as_null_and_none(self, fake_api, llm_config):
        """Spec: model_digest is None (NULL in both tables) until Step 5 supplies it."""
        result = generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert result.model_digest is None
        assert cache_rows(llm_config)[0]["model_digest"] is None
        assert call_rows(llm_config)[0]["model_digest"] is None

    def test_generate_without_config_uses_load_config(self, fake_api, llm_config, monkeypatch):
        """Spec: `config` defaults to load_config()."""
        monkeypatch.setattr(client, "load_config", lambda: llm_config)
        generate(PROMPT, MODEL, 0.0)
        assert llm_config["cache_path"].exists(), "Cache DB must be created at the loaded cache_path"
        assert len(fake_api.calls) == 1


# ---------------------------------------------------------------- request + result fields

class TestResultAndRequest:
    def test_generate_passes_request_fields_to_call_api(self, fake_api, llm_config):
        """Spec: the prompt, model, temperature and schema reach _call_api unchanged."""
        fake_api.reply = make_reply(text='{"claims": []}')
        generate("my prompt", "my-model:cloud", 0.3, schema=Claims, config=llm_config)
        call = fake_api.calls[0]
        assert call["prompt"] == "my prompt"
        assert call["model"] == "my-model:cloud"
        assert call["temperature"] == pytest.approx(0.3)
        assert call["schema"] is Claims

    def test_miss_result_fields_copy_the_reply(self, fake_api, llm_config):
        """Spec: LLMResult carries text, model, tokens, finish_reason, attempts and no parsed object."""
        result = generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert result.text == "hello"
        assert result.parsed is None, "parsed is None when no schema is given"
        assert result.model == MODEL
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.finish_reason == "stop"
        assert result.cached is False
        assert result.attempts == 1, "One HTTP attempt on a clean miss (fake adds 1)"

    def test_cache_hit_has_zero_attempts_and_copies_tokens(self, fake_api, llm_config):
        """Spec: a hit makes 0 HTTP attempts and copies the original token counts and finish reason."""
        generate(PROMPT, MODEL, 0.0, config=llm_config)
        hit = generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert hit.attempts == 0
        assert hit.prompt_tokens == 10 and hit.completion_tokens == 5
        assert hit.finish_reason == "stop"
        assert hit.model == MODEL

    def test_cache_row_stores_request_and_reply_fields(self, fake_api, llm_config):
        """Spec: llm_cache stores base_url, model, temperature, schema_name, prompt, reply, tokens, latency."""
        fake_api.reply = make_reply(text='{"claims": ["a"]}')
        generate(PROMPT, MODEL, 0.25, schema=Claims, config=llm_config)
        rows = cache_rows(llm_config)
        assert len(rows) == 1
        row = rows[0]
        assert row["cache_key"] == key_for(llm_config, temperature=0.25, schema=Claims)
        assert row["base_url"] == llm_config["base_url"]
        assert row["model"] == MODEL
        assert row["temperature"] == pytest.approx(0.25)
        assert row["schema_name"] == "Claims"
        assert row["prompt"] == PROMPT
        assert row["response_text"] == '{"claims": ["a"]}'
        assert row["prompt_tokens"] == 10 and row["completion_tokens"] == 5
        assert row["finish_reason"] == "stop"
        assert row["model_latency_ms"] == pytest.approx(1234.0)
        assert row["created_at"], "created_at timestamp must be set"

    def test_cache_row_without_schema_has_null_schema_name(self, fake_api, llm_config):
        """Spec: schema_name is the pydantic class name, or NULL."""
        generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert cache_rows(llm_config)[0]["schema_name"] is None

    def test_tables_have_the_columns_listed_in_the_spec(self, fake_api, llm_config):
        """Spec 'Data and schema changes': exact column names of llm_cache and llm_calls."""
        generate(PROMPT, MODEL, 0.0, config=llm_config)
        cache_cols = {r["name"] for r in fetch(llm_config, "PRAGMA table_info(llm_cache)")}
        call_cols = {r["name"] for r in fetch(llm_config, "PRAGMA table_info(llm_calls)")}
        assert cache_cols == {
            "cache_key", "base_url", "model", "model_digest", "temperature", "seed",
            "reasoning_effort", "schema_name",
            "prompt", "response_text", "prompt_tokens", "completion_tokens", "finish_reason",
            "model_latency_ms", "created_at",
        }
        assert call_cols == {
            "id", "called_at", "cache_key", "model", "model_digest", "seed", "reasoning_effort",
            "cached", "attempts",
            "prompt_tokens", "completion_tokens", "finish_reason", "latency_ms",
            "model_latency_ms", "error",
        }


# ---------------------------------------------------------------- structured output

class TestStructuredOutput:
    def test_schema_reply_is_parsed(self, fake_api, llm_config):
        """Spec: with a schema, result.parsed is a validated instance of it."""
        fake_api.reply = make_reply(text='{"claims": ["a", "b"]}')
        result = generate(PROMPT, MODEL, 0.0, schema=Claims, config=llm_config)
        assert isinstance(result.parsed, Claims)
        assert result.parsed.claims == ["a", "b"]

    def test_cache_hit_with_schema_is_parsed(self, fake_api, llm_config):
        """Spec: a cache hit with a schema also returns a validated parsed object."""
        fake_api.reply = make_reply(text='{"claims": ["a"]}')
        generate(PROMPT, MODEL, 0.0, schema=Claims, config=llm_config)
        hit = generate(PROMPT, MODEL, 0.0, schema=Claims, config=llm_config)
        assert hit.cached is True
        assert isinstance(hit.parsed, Claims) and hit.parsed.claims == ["a"]

    def test_code_fenced_json_is_parsed(self, fake_api, llm_config):
        """Spec: one Markdown fence wrapping the whole reply is stripped before validation."""
        fenced = '```json\n{"claims": ["a"]}\n```'
        fake_api.reply = make_reply(text=fenced)
        result = generate(PROMPT, MODEL, 0.0, schema=Claims, config=llm_config)
        assert result.parsed.claims == ["a"]

    def test_code_fenced_reply_is_cached_raw(self, fake_api, llm_config):
        """Spec: the cache stores the raw reply, fence included; result.text is raw too."""
        fenced = '```json\n{"claims": ["a"]}\n```'
        fake_api.reply = make_reply(text=fenced)
        result = generate(PROMPT, MODEL, 0.0, schema=Claims, config=llm_config)
        assert result.text == fenced
        assert cache_rows(llm_config)[0]["response_text"] == fenced

    def test_text_before_a_fence_is_not_accepted(self, fake_api, llm_config):
        """Spec: only a fence wrapping the WHOLE reply is stripped; other malformed replies raise."""
        fake_api.reply = make_reply(text='Here you go:\n```json\n{"claims": ["a"]}\n```')
        with pytest.raises(LLMOutputError):
            generate(PROMPT, MODEL, 0.0, schema=Claims, config=llm_config)

    def test_invalid_schema_reply_raises_and_is_not_cached(self, fake_api, llm_config):
        """Spec: a reply that is not JSON raises LLMOutputError and gets no llm_cache row."""
        fake_api.reply = make_reply(text="not json")
        with pytest.raises(LLMOutputError):
            generate(PROMPT, MODEL, 0.0, schema=Claims, config=llm_config)
        assert count_cache_rows(llm_config, key_for(llm_config, schema=Claims)) == 0

    def test_wrong_shaped_json_raises_and_is_not_cached(self, fake_api, llm_config):
        """Spec: valid JSON that fails pydantic validation raises LLMOutputError, not cached."""
        fake_api.reply = make_reply(text='{"wrong_field": 1}')
        with pytest.raises(LLMOutputError):
            generate(PROMPT, MODEL, 0.0, schema=Claims, config=llm_config)
        assert len(cache_rows(llm_config)) == 0

    def test_invalid_schema_reply_is_not_retried(self, fake_api, llm_config):
        """Spec: LLMOutputError is never retried; the API is called once."""
        fake_api.reply = make_reply(text="not json")
        with pytest.raises(LLMOutputError):
            generate(PROMPT, MODEL, 0.0, schema=Claims, config=llm_config)
        assert len(fake_api.calls) == 1


# ---------------------------------------------------------------- output checks

class TestOutputChecks:
    def test_empty_reply_raises_and_is_not_cached(self, fake_api, llm_config):
        """Spec: an empty reply raises LLMOutputError and gets no llm_cache row."""
        fake_api.reply = make_reply(text="")
        with pytest.raises(LLMOutputError):
            generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert count_cache_rows(llm_config, key_for(llm_config)) == 0
        assert len(fake_api.calls) == 1, "An empty reply is not retried"

    def test_cut_off_reply_raises_and_is_not_cached(self, fake_api, llm_config):
        """Spec: finish_reason 'length' raises LLMOutputError and is not cached."""
        fake_api.reply = make_reply(text="partial answer", finish_reason="length")
        with pytest.raises(LLMOutputError):
            generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert count_cache_rows(llm_config, key_for(llm_config)) == 0

    @pytest.mark.parametrize("finish_reason", ["content_filter", None])
    def test_abnormal_finish_reason_raises(self, fake_api, llm_config, finish_reason):
        """Spec: only finish_reason 'stop' is valid; 'content_filter' and None raise and are not cached."""
        fake_api.reply = make_reply(text="some text", finish_reason=finish_reason)
        with pytest.raises(LLMOutputError):
            generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert count_cache_rows(llm_config, key_for(llm_config)) == 0

    def test_failed_reply_is_not_served_on_next_call(self, fake_api, llm_config):
        """Spec: a rejected reply is never cached, so the next identical call hits the API again."""
        fake_api.reply = make_reply(text="", finish_reason="stop")
        with pytest.raises(LLMOutputError):
            generate(PROMPT, MODEL, 0.0, config=llm_config)
        fake_api.reply = make_reply(text="good now")
        result = generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert result.cached is False and result.text == "good now"
        assert len(fake_api.calls) == 2

    def test_cache_hit_parse_failure_is_logged(self, fake_api, llm_config):
        """Spec: a cached reply edited to invalid JSON raises on the hit and logs one cached=1 row with error."""
        fake_api.reply = make_reply(text='{"claims": ["a"]}')
        generate(PROMPT, MODEL, 0.0, schema=Claims, config=llm_config)
        key = key_for(llm_config, schema=Claims)
        with closing(sqlite3.connect(llm_config["cache_path"])) as conn:
            conn.execute("UPDATE llm_cache SET response_text = ? WHERE cache_key = ?",
                         ("not json", key))
            conn.commit()

        with pytest.raises(LLMOutputError):
            generate(PROMPT, MODEL, 0.0, schema=Claims, config=llm_config)

        hit_rows = fetch(llm_config, "SELECT * FROM llm_calls WHERE cached = ?", (1,))
        assert len(hit_rows) == 1, "Exactly one cached=1 row for the failed hit"
        assert hit_rows[0]["error"], "The failed hit must carry an error"
        assert hit_rows[0]["error"].startswith("LLMOutputError")
        assert hit_rows[0]["attempts"] == 0
        assert len(call_rows(llm_config)) == 2, "One row for the miss, one for the failed hit"
        assert len(fake_api.calls) == 1, "The failed hit must not call the API"

    def test_missing_usage_is_stored_as_null(self, fake_api, llm_config):
        """Spec: unknown token counts are None / NULL, never 0."""
        fake_api.reply = make_reply(prompt_tokens=None, completion_tokens=None)
        result = generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert result.prompt_tokens is None and result.completion_tokens is None
        row = call_rows(llm_config)[0]
        assert row["prompt_tokens"] is None and row["completion_tokens"] is None
        cached = cache_rows(llm_config)[0]
        assert cached["prompt_tokens"] is None and cached["completion_tokens"] is None

    def test_missing_usage_stays_none_on_cache_hit(self, fake_api, llm_config):
        """Spec: a hit copies the original counts, so unknown stays None (not 0)."""
        fake_api.reply = make_reply(prompt_tokens=None, completion_tokens=None)
        generate(PROMPT, MODEL, 0.0, config=llm_config)
        hit = generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert hit.prompt_tokens is None and hit.completion_tokens is None
        assert call_rows(llm_config)[1]["prompt_tokens"] is None


# ---------------------------------------------------------------- retries (real tenacity loop)

TRANSIENT_ERRORS = [
    pytest.param(connection_error, id="APIConnectionError"),
    pytest.param(lambda: openai.APITimeoutError(request=FAKE_REQUEST), id="APITimeoutError"),
    pytest.param(lambda: status_error(openai.RateLimitError, 429), id="RateLimitError"),
    pytest.param(lambda: status_error(openai.InternalServerError, 500), id="InternalServerError"),
]

PERMANENT_ERRORS = [
    pytest.param(lambda: status_error(openai.BadRequestError, 400), id="BadRequestError"),
    pytest.param(lambda: status_error(openai.NotFoundError, 404), id="NotFoundError"),
]


class TestRetries:
    def test_transient_error_is_retried(self, monkeypatch, llm_config):
        """Spec: two APIConnectionErrors then success: the call succeeds after 3 attempts."""
        send = install_fake_send(
            monkeypatch, [connection_error(), connection_error(), make_reply(text="ok")]
        )
        result = generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert result.text == "ok"
        assert send.calls == 3, "Expected 2 failures plus 1 success"
        assert result.attempts == 3

    @pytest.mark.parametrize("make_error", TRANSIENT_ERRORS)
    def test_each_transient_error_type_is_retried(self, monkeypatch, llm_config, make_error):
        """Spec: APIConnectionError, APITimeoutError, RateLimitError and InternalServerError are retried."""
        send = install_fake_send(monkeypatch, [make_error(), make_reply(text="ok")])
        result = generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert result.text == "ok"
        assert send.calls == 2

    @pytest.mark.parametrize("make_error", PERMANENT_ERRORS)
    def test_permanent_error_is_not_retried(self, monkeypatch, llm_config, make_error):
        """Spec: BadRequestError and NotFoundError are never retried; they propagate after one try."""
        error = make_error()
        send = install_fake_send(monkeypatch, [error, make_reply()])
        with pytest.raises(type(error)):
            generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert send.calls == 1, "A permanent error must not be retried"

    def test_gives_up_after_max_attempts(self, monkeypatch, llm_config):
        """Spec: an always-failing call raises the original error after exactly max_attempts tries."""
        llm_config["max_attempts"] = 3
        send = install_fake_send(monkeypatch, [connection_error()])
        with pytest.raises(openai.APIConnectionError):
            generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert send.calls == 3

    def test_max_attempts_one_means_no_retry(self, monkeypatch, llm_config):
        """Spec: max_attempts is the total tries including the first; 1 means a single try."""
        llm_config["max_attempts"] = 1
        send = install_fake_send(monkeypatch, [connection_error()])
        with pytest.raises(openai.APIConnectionError):
            generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert send.calls == 1

    def test_failed_after_retries_is_not_cached(self, monkeypatch, llm_config):
        """Spec: a failed call is never cached."""
        llm_config["max_attempts"] = 2
        install_fake_send(monkeypatch, [connection_error()])
        with pytest.raises(openai.APIConnectionError):
            generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert len(cache_rows(llm_config)) == 0

    def test_client_is_built_from_config(self, monkeypatch, llm_config):
        """Spec: the client uses base_url and timeout_s from the config."""
        send = install_fake_send(monkeypatch, [make_reply()])
        generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert send.client_args, "_get_client must be called on a miss"
        assert send.client_args[0] == (llm_config["base_url"], llm_config["timeout_s"])

    def test_cache_hit_does_not_send_a_request(self, monkeypatch, llm_config):
        """Spec: a cache hit makes no HTTP call."""
        send = install_fake_send(monkeypatch, [make_reply()])
        generate(PROMPT, MODEL, 0.0, config=llm_config)
        generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert send.calls == 1

    def test_attempts_are_logged(self, monkeypatch, llm_config):
        """Spec: retries log attempts=3; a cache hit logs 0; giving up logs max_attempts."""
        llm_config["max_attempts"] = 4
        # Section 1: 2 transient errors then success -> attempts = 3.
        install_fake_send(
            monkeypatch, [connection_error(), connection_error(), make_reply(text="ok")]
        )
        generate("retry prompt", MODEL, 0.0, config=llm_config)
        # Section 2: the same request again is a cache hit -> attempts = 0.
        generate("retry prompt", MODEL, 0.0, config=llm_config)
        # Section 3: always failing -> attempts = max_attempts (4).
        install_fake_send(monkeypatch, [connection_error()])
        with pytest.raises(openai.APIConnectionError):
            generate("failing prompt", MODEL, 0.0, config=llm_config)

        rows = call_rows(llm_config)
        assert [r["attempts"] for r in rows] == [3, 0, 4]
        assert rows[2]["error"].startswith("APIConnectionError")


# ---------------------------------------------------------------- _send_request (fake client object)

class TestSendRequest:
    def test_send_request_builds_the_request(self):
        """Spec: request has the given model and temperature, one user message with the prompt, no response_format without a schema."""
        fake = FakeOpenAIClient()
        client._send_request(fake, "my prompt", "my-model:cloud", 0.3, None)
        assert len(fake.create_kwargs) == 1, "Exactly one HTTP call"
        kwargs = fake.create_kwargs[0]
        assert kwargs["model"] == "my-model:cloud"
        assert kwargs["temperature"] == pytest.approx(0.3)
        assert kwargs["messages"] == [{"role": "user", "content": "my prompt"}]
        assert kwargs.get("response_format") is None

    def test_send_request_with_schema_sends_json_schema_format(self):
        """Spec: response_format of type json_schema is sent only when a schema is given."""
        fake = FakeOpenAIClient(content='{"claims": []}')
        client._send_request(fake, "my prompt", MODEL, 0.0, Claims)
        kwargs = fake.create_kwargs[0]
        assert kwargs["response_format"]["type"] == "json_schema"
        assert len(kwargs["messages"]) == 1 and kwargs["messages"][0]["role"] == "user"

    def test_send_request_maps_the_response(self):
        """Spec: text, token counts and finish_reason are copied from the response."""
        fake = FakeOpenAIClient(content="hello", finish_reason="length", usage=(7, 3))
        reply = client._send_request(fake, "p", MODEL, 0.0, None)
        assert isinstance(reply, ApiReply)
        assert reply.text == "hello"
        assert reply.prompt_tokens == 7
        assert reply.completion_tokens == 3
        assert reply.finish_reason == "length"
        assert isinstance(reply.model_latency_ms, float) and reply.model_latency_ms >= 0

    def test_send_request_missing_usage_gives_none_counts(self):
        """Spec: usage=None gives None token counts (not 0)."""
        fake = FakeOpenAIClient(usage=None)
        reply = client._send_request(fake, "p", MODEL, 0.0, None)
        assert reply.prompt_tokens is None
        assert reply.completion_tokens is None

    def test_send_request_none_content_gives_empty_string(self):
        """Spec: content=None gives an empty text."""
        fake = FakeOpenAIClient(content=None)
        reply = client._send_request(fake, "p", MODEL, 0.0, None)
        assert reply.text == ""

    def test_send_request_none_finish_reason_is_copied(self):
        """Spec: finish_reason is copied as is, including None."""
        fake = FakeOpenAIClient(finish_reason=None)
        reply = client._send_request(fake, "p", MODEL, 0.0, None)
        assert reply.finish_reason is None


# ---------------------------------------------------------------- _call_api + stats

class TestCallApi:
    def test_call_stats_starts_at_zero(self):
        """Spec: CallStats.attempts defaults to 0."""
        assert CallStats().attempts == 0

    def test_call_api_counts_attempts_in_stats(self, monkeypatch, llm_config):
        """Spec: _call_api adds 1 to stats.attempts per HTTP attempt (1 failure + 1 success = 2)."""
        install_fake_send(monkeypatch, [connection_error(), make_reply()])
        stats = CallStats()
        reply = client._call_api(PROMPT, MODEL, 0.0, None, llm_config, stats)
        assert isinstance(reply, ApiReply)
        assert stats.attempts == 2

    def test_call_api_keeps_attempts_when_it_fails(self, monkeypatch, llm_config):
        """Spec: stats.attempts is kept even if the call fails (max_attempts=3 gives 3)."""
        llm_config["max_attempts"] = 3
        install_fake_send(monkeypatch, [connection_error()])
        stats = CallStats()
        with pytest.raises(openai.APIConnectionError):
            client._call_api(PROMPT, MODEL, 0.0, None, llm_config, stats)
        assert stats.attempts == 3


# ---------------------------------------------------------------- logging

class TestLogging:
    def test_every_call_is_logged(self, fake_api, llm_config):
        """Spec: one miss and one hit leave 2 llm_calls rows (cached 0 then 1), error NULL, tokens copied."""
        generate(PROMPT, MODEL, 0.0, config=llm_config)
        generate(PROMPT, MODEL, 0.0, config=llm_config)
        rows = call_rows(llm_config)
        assert len(rows) == 2, "Expected one llm_calls row per generate() call"
        assert [r["cached"] for r in rows] == [0, 1]
        assert [r["error"] for r in rows] == [None, None]
        for row in rows:
            assert row["prompt_tokens"] == 10 and row["completion_tokens"] == 5
            assert row["model"] == MODEL
            assert row["cache_key"] == key_for(llm_config)
            assert row["finish_reason"] == "stop"
            assert row["model_latency_ms"] == pytest.approx(1234.0)
            assert row["latency_ms"] >= 0
            assert row["called_at"], "called_at timestamp must be set"
        assert rows[0]["attempts"] == 1 and rows[1]["attempts"] == 0

    def test_failed_call_is_logged_with_error_invalid_schema(self, fake_api, llm_config):
        """Spec: an invalid-schema reply leaves one row; error starts with the exception name; tokens kept."""
        fake_api.reply = make_reply(text="not json")
        with pytest.raises(LLMOutputError):
            generate(PROMPT, MODEL, 0.0, schema=Claims, config=llm_config)
        rows = call_rows(llm_config)
        assert len(rows) == 1
        assert rows[0]["error"].startswith("LLMOutputError")
        assert rows[0]["prompt_tokens"] == 10 and rows[0]["completion_tokens"] == 5
        assert rows[0]["cached"] == 0

    def test_failed_call_is_logged_with_error_permanent_api_error(self, fake_api, llm_config):
        """Spec: a permanent API error leaves one row; error starts with the type; tokens, finish_reason, model latency NULL."""
        fake_api.error = status_error(openai.BadRequestError, 400, "bad request")
        with pytest.raises(openai.BadRequestError):
            generate(PROMPT, MODEL, 0.0, config=llm_config)
        rows = call_rows(llm_config)
        assert len(rows) == 1
        row = rows[0]
        assert row["error"].startswith("BadRequestError")
        assert row["prompt_tokens"] is None and row["completion_tokens"] is None
        assert row["finish_reason"] is None
        assert row["model_latency_ms"] is None
        assert row["cached"] == 0
        assert row["attempts"] == 1

    def test_permanent_api_error_is_not_cached(self, fake_api, llm_config):
        """Spec: a failed call is never cached."""
        fake_api.error = status_error(openai.NotFoundError, 404)
        with pytest.raises(openai.NotFoundError):
            generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert len(cache_rows(llm_config)) == 0

    def test_error_message_is_truncated_to_200_chars(self, fake_api, llm_config):
        """Spec: error is '<ErrorType>: <message, max 200 chars>'."""
        fake_api.error = status_error(openai.BadRequestError, 400, "x" * 500)
        with pytest.raises(openai.BadRequestError):
            generate(PROMPT, MODEL, 0.0, config=llm_config)
        error = call_rows(llm_config)[0]["error"]
        assert error.startswith("BadRequestError: ")
        message = error[len("BadRequestError: "):]
        assert 0 < len(message) <= 200, f"Message part has {len(message)} chars"

    def test_each_failure_adds_exactly_one_row(self, fake_api, llm_config):
        """Spec: every generate() call, failed or not, leaves exactly one llm_calls row."""
        fake_api.reply = make_reply(text="")
        for _ in range(2):
            with pytest.raises(LLMOutputError):
                generate(PROMPT, MODEL, 0.0, config=llm_config)
        assert len(call_rows(llm_config)) == 2

    def test_empty_reply_failure_is_logged(self, fake_api, llm_config):
        """Spec: an empty reply is logged with an LLMOutputError error and the reply's tokens."""
        fake_api.reply = make_reply(text="")
        with pytest.raises(LLMOutputError):
            generate(PROMPT, MODEL, 0.0, config=llm_config)
        row = call_rows(llm_config)[0]
        assert row["error"].startswith("LLMOutputError")
        assert row["prompt_tokens"] == 10

    def test_abnormal_finish_reason_failure_is_logged(self, fake_api, llm_config):
        """Spec: a cut-off reply is logged as an error, never as a normal-looking row."""
        fake_api.reply = make_reply(text="partial", finish_reason="length")
        with pytest.raises(LLMOutputError):
            generate(PROMPT, MODEL, 0.0, config=llm_config)
        row = call_rows(llm_config)[0]
        assert row["error"].startswith("LLMOutputError")
        assert row["finish_reason"] == "length"


# ---------------------------------------------------------------- config

class TestLoadConfig:
    def test_load_config_reads_a_valid_file(self, tmp_path, llm_config):
        """Spec: load_config returns all required keys from the YAML file."""
        assert set(llm_config) == set(REQUIRED_CONFIG_KEYS), (
            "llm_config fixture must have exactly the required keys"
        )
        path = write_config_file(tmp_path, llm_config)
        loaded = load_config(path)
        for key in REQUIRED_CONFIG_KEYS:
            assert key in loaded, f"{key} missing from loaded config"
        assert loaded["base_url"] == llm_config["base_url"]
        assert loaded["max_attempts"] == llm_config["max_attempts"]

    @pytest.mark.parametrize("missing_key", REQUIRED_CONFIG_KEYS)
    def test_load_config_rejects_missing_keys(self, tmp_path, llm_config, missing_key):
        """Spec: a YAML file without a required key makes load_config raise ValueError naming that key."""
        path = write_config_file(tmp_path, llm_config, drop=missing_key)
        with pytest.raises(ValueError, match=re.escape(missing_key)) as excinfo:
            load_config(path)
        assert path.name in str(excinfo.value), "Error should name the config file"

    def test_load_config_lists_all_missing_keys(self, tmp_path, llm_config):
        """Spec: ValueError lists every missing key, not just the first."""
        data = {k: v for k, v in llm_config.items() if k not in REQUIRED_CONFIG_KEYS[:2]}
        path = tmp_path / "partial.yaml"
        path.write_text(yaml.safe_dump({k: str(v) for k, v in data.items()}), encoding="utf-8")
        with pytest.raises(ValueError) as excinfo:
            load_config(path)
        for key in REQUIRED_CONFIG_KEYS[:2]:
            assert key in str(excinfo.value), f"{key} should be listed as missing"

    @pytest.mark.parametrize("key, bad_value", [
        ("max_attempts", 0),
        ("max_attempts", -1),
        ("timeout_s", 0),
        ("timeout_s", -5),
        ("retry_wait_min_s", -1),
        ("retry_wait_max_s", -1),
    ])
    def test_load_config_rejects_bad_values(self, tmp_path, llm_config, key, bad_value):
        """Spec: max_attempts < 1, timeout_s <= 0 and negative retry waits raise ValueError."""
        path = write_config_file(tmp_path, llm_config, **{key: bad_value})
        with pytest.raises(ValueError):
            load_config(path)

    def test_load_config_accepts_boundary_values(self, tmp_path, llm_config):
        """Spec: max_attempts 1 and zero retry waits are valid (they are the limits)."""
        path = write_config_file(tmp_path, llm_config, max_attempts=1,
                                 retry_wait_min_s=0, retry_wait_max_s=0)
        loaded = load_config(path)
        assert loaded["max_attempts"] == 1
        assert loaded["retry_wait_min_s"] == 0

    def test_load_config_resolves_relative_cache_path_against_repo_root(self, tmp_path, llm_config):
        """Spec: a relative cache_path is resolved against the repo root."""
        path = write_config_file(tmp_path, llm_config, cache_path="some_dir/test_cache.sqlite")
        loaded = load_config(path)
        assert Path(loaded["cache_path"]) == PROJECT_ROOT / "some_dir" / "test_cache.sqlite"

    def test_load_config_keeps_absolute_cache_path(self, tmp_path, llm_config):
        """Spec: only a RELATIVE cache_path is resolved; an absolute one is kept."""
        absolute = tmp_path / "abs_cache.sqlite"
        path = write_config_file(tmp_path, llm_config, cache_path=str(absolute))
        assert Path(load_config(path)["cache_path"]) == absolute

    def test_real_config_file_has_all_required_keys(self):
        """Spec 'Config changes': reads the real configs/llm.yaml (read-only) and checks the six keys."""
        loaded = load_config()
        for key in REQUIRED_CONFIG_KEYS:
            assert key in loaded, f"configs/llm.yaml is missing {key}"

    def test_required_config_keys_are_the_six_spec_keys(self):
        """Spec: REQUIRED_CONFIG_KEYS is a public tuple of six keys."""
        assert isinstance(REQUIRED_CONFIG_KEYS, tuple)
        assert len(REQUIRED_CONFIG_KEYS) == 6


# ---------------------------------------------------------------- Step 3 additions: seed + reasoning_effort
# From .claude/specs/03-rag-pipeline-v0.md ("Step 2 additions" and "Change to llm/client.py").

class TestSeedAndReasoningEffortKey:
    BASE = dict(prompt="Say hi", model="m:cloud", temperature=0.0, schema=None,
                base_url="http://fake-llm.test/v1")

    @pytest.mark.parametrize("field, value", [
        ("seed", 42),
        ("seed", 0),
        ("reasoning_effort", "low"),
    ])
    def test_cache_key_changes_when_field_is_set(self, field, value):
        """Spec: seed and reasoning_effort are part of the key (seed=0 is a value, not 'unset')."""
        unset = make_cache_key(**self.BASE)
        with_value = make_cache_key(**self.BASE, **{field: value})
        assert with_value != unset, f"{field}={value!r} must change the cache key"

    def test_cache_key_changes_between_two_seeds(self):
        """Spec: seed 1 and seed 2 give different keys."""
        assert make_cache_key(**self.BASE, seed=1) != make_cache_key(**self.BASE, seed=2)

    def test_cache_key_changes_between_two_reasoning_efforts(self):
        """Spec: reasoning_effort 'low' and 'high' give different keys."""
        low = make_cache_key(**self.BASE, reasoning_effort="low")
        high = make_cache_key(**self.BASE, reasoning_effort="high")
        assert low != high

    def test_cache_key_same_seed_and_effort_is_stable(self):
        """Spec: identical inputs, including seed and reasoning_effort, give the same key."""
        first = make_cache_key(**self.BASE, seed=42, reasoning_effort="low")
        second = make_cache_key(**self.BASE, seed=42, reasoning_effort="low")
        assert first == second

    def test_cache_key_defaults_equal_explicit_none(self):
        """Spec: seed and reasoning_effort default to None, which equals passing None."""
        assert make_cache_key(**self.BASE) == make_cache_key(
            **self.BASE, seed=None, reasoning_effort=None)

    def test_cache_key_matches_spec_formula_with_values(self):
        """Spec: the key JSON has 8 fields and includes seed and reasoning_effort values."""
        expected_input = json.dumps(
            {"base_url": "http://fake-llm.test/v1", "model": "m:cloud", "model_digest": None,
             "prompt": "Say hi", "reasoning_effort": "low", "seed": 42,
             "temperature": 0.0, "schema": None},
            sort_keys=True,
        )
        expected = hashlib.sha256(expected_input.encode("utf-8")).hexdigest()
        assert make_cache_key(**self.BASE, seed=42, reasoning_effort="low") == expected


class TestSeedAndReasoningEffortCaching:
    def test_different_seed_misses_cache(self, fake_api, llm_config):
        """Spec: the same request with seed 1 then seed 2 calls the API twice."""
        generate(PROMPT, MODEL, 0.0, config=llm_config, seed=1)
        generate(PROMPT, MODEL, 0.0, config=llm_config, seed=2)
        assert len(fake_api.calls) == 2, "A new seed must miss the cache"

    def test_same_seed_hits_cache(self, fake_api, llm_config):
        """Spec: an identical seed is part of an identical request, so the second call hits."""
        generate(PROMPT, MODEL, 0.0, config=llm_config, seed=1)
        hit = generate(PROMPT, MODEL, 0.0, config=llm_config, seed=1)
        assert len(fake_api.calls) == 1
        assert hit.cached is True

    def test_seed_none_and_seed_zero_are_different_requests(self, fake_api, llm_config):
        """Spec: None means 'not sent'; 0 is a real seed, so they must not share a cache row."""
        generate(PROMPT, MODEL, 0.0, config=llm_config)
        generate(PROMPT, MODEL, 0.0, config=llm_config, seed=0)
        assert len(fake_api.calls) == 2

    def test_different_reasoning_effort_misses_cache(self, fake_api, llm_config):
        """Spec: the same request with reasoning_effort 'low' then 'high' calls the API twice."""
        generate(PROMPT, MODEL, 0.0, config=llm_config, reasoning_effort="low")
        generate(PROMPT, MODEL, 0.0, config=llm_config, reasoning_effort="high")
        assert len(fake_api.calls) == 2, "A new reasoning_effort must miss the cache"

    def test_same_reasoning_effort_hits_cache(self, fake_api, llm_config):
        """Spec: an identical reasoning_effort hits the cache."""
        generate(PROMPT, MODEL, 0.0, config=llm_config, reasoning_effort="low")
        hit = generate(PROMPT, MODEL, 0.0, config=llm_config, reasoning_effort="low")
        assert len(fake_api.calls) == 1
        assert hit.cached is True

    def test_reasoning_effort_none_and_low_are_different_requests(self, fake_api, llm_config):
        """Spec: None (not sent) and 'low' are different requests."""
        generate(PROMPT, MODEL, 0.0, config=llm_config)
        generate(PROMPT, MODEL, 0.0, config=llm_config, reasoning_effort="low")
        assert len(fake_api.calls) == 2


class TestSeedAndReasoningEffortStored:
    def test_cache_row_stores_seed_and_reasoning_effort(self, fake_api, llm_config):
        """Spec: llm_cache stores seed and reasoning_effort, and the row sits under the 8-field key."""
        generate(PROMPT, MODEL, 0.0, config=llm_config, seed=42, reasoning_effort="low")
        rows = cache_rows(llm_config)
        assert len(rows) == 1
        assert rows[0]["seed"] == 42
        assert rows[0]["reasoning_effort"] == "low"
        expected_key = make_cache_key(PROMPT, MODEL, 0.0, None, llm_config["base_url"],
                                      seed=42, reasoning_effort="low")
        assert rows[0]["cache_key"] == expected_key

    def test_calls_row_stores_seed_and_reasoning_effort(self, fake_api, llm_config):
        """Spec: llm_calls stores seed and reasoning_effort on a miss and on a hit."""
        generate(PROMPT, MODEL, 0.0, config=llm_config, seed=42, reasoning_effort="low")
        generate(PROMPT, MODEL, 0.0, config=llm_config, seed=42, reasoning_effort="low")
        rows = call_rows(llm_config)
        assert [r["cached"] for r in rows] == [0, 1]
        assert [r["seed"] for r in rows] == [42, 42]
        assert [r["reasoning_effort"] for r in rows] == ["low", "low"]

    def test_unset_seed_and_effort_are_stored_as_null(self, fake_api, llm_config):
        """Spec: seed and reasoning_effort are NULL in both tables when not sent."""
        generate(PROMPT, MODEL, 0.0, config=llm_config)
        cache_row = cache_rows(llm_config)[0]
        call_row = call_rows(llm_config)[0]
        assert cache_row["seed"] is None and cache_row["reasoning_effort"] is None
        assert call_row["seed"] is None and call_row["reasoning_effort"] is None

    def test_seed_zero_is_stored_as_zero_not_null(self, fake_api, llm_config):
        """Spec: seed=0 is a real value and must not be stored as NULL."""
        generate(PROMPT, MODEL, 0.0, config=llm_config, seed=0)
        assert cache_rows(llm_config)[0]["seed"] == 0
        assert call_rows(llm_config)[0]["seed"] == 0

    def test_failed_call_logs_seed_and_reasoning_effort(self, fake_api, llm_config):
        """Spec: a failed call still logs its request parameters in llm_calls."""
        fake_api.reply = make_reply(text="")
        with pytest.raises(LLMOutputError):
            generate(PROMPT, MODEL, 0.0, config=llm_config, seed=7, reasoning_effort="high")
        row = call_rows(llm_config)[0]
        assert row["seed"] == 7 and row["reasoning_effort"] == "high"
        assert row["error"].startswith("LLMOutputError")


class TestSeedAndReasoningEffortRequest:
    def test_generate_passes_seed_and_effort_to_call_api(self, fake_api, llm_config):
        """Spec: generate() passes seed and reasoning_effort on to _call_api."""
        generate(PROMPT, MODEL, 0.0, config=llm_config, seed=42, reasoning_effort="low")
        call = fake_api.calls[0]
        assert call["seed"] == 42
        assert call["reasoning_effort"] == "low"

    def test_generate_defaults_pass_none_to_call_api(self, fake_api, llm_config):
        """Spec: seed and reasoning_effort default to None (keyword-only, optional)."""
        generate(PROMPT, MODEL, 0.0, config=llm_config)
        call = fake_api.calls[0]
        assert call["seed"] is None
        assert call["reasoning_effort"] is None

    def test_send_request_sends_seed_and_effort_when_set(self):
        """Spec: seed and reasoning_effort are put in the request when not None."""
        fake = FakeOpenAIClient()
        client._send_request(fake, "p", MODEL, 0.0, None, seed=42, reasoning_effort="low")
        kwargs = fake.create_kwargs[0]
        assert kwargs["seed"] == 42
        assert kwargs["reasoning_effort"] == "low"

    def test_send_request_omits_seed_and_effort_when_none(self):
        """Spec: with None, neither field is in the request (not sent as null)."""
        fake = FakeOpenAIClient()
        client._send_request(fake, "p", MODEL, 0.0, None)
        kwargs = fake.create_kwargs[0]
        assert "seed" not in kwargs, "seed=None must not be sent"
        assert "reasoning_effort" not in kwargs, "reasoning_effort=None must not be sent"

    def test_send_request_sends_seed_zero(self):
        """Spec: only None means 'not sent'; seed=0 must be sent."""
        fake = FakeOpenAIClient()
        client._send_request(fake, "p", MODEL, 0.0, None, seed=0)
        assert fake.create_kwargs[0]["seed"] == 0

    def test_send_request_sends_only_the_one_that_is_set(self):
        """Spec: each field is sent independently of the other."""
        fake = FakeOpenAIClient()
        client._send_request(fake, "p", MODEL, 0.0, None, reasoning_effort="high")
        client._send_request(fake, "p", MODEL, 0.0, None, seed=5)
        first, second = fake.create_kwargs
        assert first["reasoning_effort"] == "high" and "seed" not in first
        assert second["seed"] == 5 and "reasoning_effort" not in second

    def test_call_api_passes_seed_and_effort_to_send_request(self, monkeypatch, llm_config):
        """Spec: _call_api takes seed and reasoning_effort (keyword, default None) and forwards both."""
        seen = []

        def recording_send(client_obj, prompt, model, temperature, schema,
                           seed=None, reasoning_effort=None):
            """Record the forwarded seed and reasoning_effort, then return a reply."""
            seen.append((seed, reasoning_effort))
            return make_reply()

        monkeypatch.setattr(client, "_get_client", lambda base_url, timeout_s: object())
        monkeypatch.setattr(client, "_send_request", recording_send)
        client._call_api(PROMPT, MODEL, 0.0, None, llm_config, CallStats(),
                         seed=9, reasoning_effort="low")
        client._call_api(PROMPT, MODEL, 0.0, None, llm_config, CallStats())
        assert seen == [(9, "low"), (None, None)]


# ---------------------------------------------------------------- layering

def test_rag_does_not_import_evalharness():
    """Spec/CLAUDE.md rule: no file under rag/ imports from evalharness (architecture guard)."""
    rag_files = list((PROJECT_ROOT / "rag").rglob("*.py"))
    assert rag_files, "Expected Python files under rag/"
    for path in rag_files:
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
