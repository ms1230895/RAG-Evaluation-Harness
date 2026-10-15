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
        "timeout_s": 5,
        "max_attempts": 5,
        "retry_wait_min_s": 0,
        "retry_wait_max_s": 0,
        "cache_path": tmp_path / "llm_cache_test.sqlite",
    }
