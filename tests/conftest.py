# tests/conftest.py
# Shared fixtures for the whole test suite.
import os

# Backstop, set BEFORE importing anything from rag/ or llm/:
# hide API keys and force Hugging Face libraries offline, so an accidental
# real call fails fast instead of spending credits or downloading a model.
# This is a safety net, not a guarantee (code that calls load_dotenv() can
# reload keys), so tests must still never reach a real client.
for _name in list(os.environ):
    if _name.endswith("_API_KEY"):
        del os.environ[_name]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pytest


@pytest.fixture
def db_path(tmp_path):
    """Path for a throwaway SQLite database, never the real results database."""
    return tmp_path / "test_results.db"


@pytest.fixture
def llm_config(tmp_path):
    """Config dict for llm.client with the cache in tmp_path and zero retry waits.

    Keys match configs/llm.yaml. Tests may change values (e.g. max_attempts)
    on the returned dict before calling generate().
    """
    return {
        "base_url": "http://fake-llm.test/v1",
        "timeout_s": 5,
        "max_attempts": 5,
        "retry_wait_min_s": 0,
        "retry_wait_max_s": 0,
        "cache_path": tmp_path / "llm_cache_test.sqlite",
    }


class FakeApi:
    """Stand-in for llm.client._call_api: returns a canned ApiReply and records calls.

    Set `reply` to change what is returned, or `error` to make every call raise.
    Like the real retry loop it adds 1 to stats.attempts per call, even when it raises.
    """

    def __init__(self, reply):
        self.reply = reply
        self.error = None
        self.calls = []

    def __call__(self, prompt, model, temperature, schema, config, stats):
        stats.attempts += 1
        self.calls.append(
            {"prompt": prompt, "model": model, "temperature": temperature,
             "schema": schema, "config": config}
        )
        if self.error is not None:
            raise self.error
        return self.reply


@pytest.fixture
def fake_api(monkeypatch):
    """Replace llm.client._call_api with a FakeApi (no HTTP, no retries)."""
    from llm import client

    api = FakeApi(client.ApiReply(
        text="hello", prompt_tokens=10, completion_tokens=5,
        finish_reason="stop", model_latency_ms=1234.0,
    ))
    monkeypatch.setattr(client, "_call_api", api)
    return api
