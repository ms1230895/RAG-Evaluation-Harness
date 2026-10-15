# Test suite notes

## Fixtures (`tests/conftest.py`)
- Module top: removes `*_API_KEY` env vars, sets `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` (backstop only).
- `db_path`: throwaway SQLite path in `tmp_path` (never the real results DB).
- `llm_config`: config dict for `llm.client` (keys as `configs/llm.yaml`), `cache_path` in `tmp_path`, retry waits 0.

## Test files
- `test_setup.py` (Step 0): every dependency imports.
- `test_corpus.py` (Step 1): HR corpus loads, IDs unique, no garbage text.
- `test_02_llm_wrapper.py` (Step 2): `llm/client.py` cache key, cache hit/miss, schema parsing (incl. code fences), tenacity retries, `llm_cache` / `llm_calls` rows, rag/ layering guard.

## Patterns that work
- LLM wrapper: patch `llm.client.load_config` (tmp_path cache), then patch `llm.client._call_api` for cache/schema/logging tests, or `llm.client._send_request` + `_get_client` so the real tenacity loop runs.
- Transient errors: `openai.APIConnectionError(request=httpx2.Request("POST", "http://test"))`; status errors need `response=httpx2.Response(code, request=...)`, `body=None`.
- Inspect the cache DB with `contextlib.closing(sqlite3.connect(path))` and `?` placeholders.

## Pitfalls found
- openai v3.31 uses `httpx2`, not `httpx`.
- Spec leaves open which exception surfaces after retries are exhausted; tests assume the original `openai` error (reraise).
