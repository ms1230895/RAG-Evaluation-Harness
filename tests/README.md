# Test suite notes

## Fixtures (`tests/conftest.py`)
- Module top: removes `*_API_KEY` env vars, sets `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` (backstop only).
- `db_path`: throwaway SQLite path in `tmp_path` (never the real results DB).
- `llm_config`: config dict for `llm.client` (all six keys of `configs/llm.yaml`, fake `base_url`), `cache_path` in `tmp_path`, retry waits 0.
- `fake_api`: replaces `llm.client._call_api` with a `FakeApi` (set `.reply` or `.error`, read `.calls`). Its signature has the `stats` argument and it does `stats.attempts += 1` per call, even when it raises, like the real retry loop.

## Test files
- `test_setup.py` (Step 0): every dependency imports.
- `test_corpus.py` (Step 1): HR corpus loads, IDs unique, no garbage text.
- `test_02_llm_wrapper.py` (Step 2, rewritten after the 2nd review-updated spec): six-input cache key (+ spec formula), cache hit/miss per input incl. `model_digest`, `attempts`, schema parsing (code fences), output checks (empty, `finish_reason != "stop"`, bad JSON, cache-hit parse failure logged), tenacity retries (4 transient types retried, permanent not), `_send_request` with a fake client object, `_call_api` + `CallStats`, `llm_cache` / `llm_calls` columns and rows, `load_config` (keys looped from `REQUIRED_CONFIG_KEYS`, bad values), rag/ layering guard. An autouse fixture blocks the real `_get_client`.

## Patterns that work
- LLM wrapper: pass `config=llm_config` to `generate()`, use `fake_api` for cache/schema/logging tests, or `install_fake_send(monkeypatch, outcomes)` in the test file (fakes `_send_request` + `_get_client`) so the real tenacity loop runs. `_send_request` itself is tested with `FakeOpenAIClient` (records `chat.completions.create(**kwargs)`).
- Transient errors: `openai.APIConnectionError(request=httpx2.Request("POST", "http://test"))`; status errors need `response=httpx2.Response(code, request=...)`, `body=None`.
- Inspect the cache DB with `contextlib.closing(sqlite3.connect(path))` and `?` placeholders.

## Pitfalls found
- `httpx2` is imported in the Step 2 tests only for building openai errors; it comes through `openai==3.31`.
- Config tests loop over `REQUIRED_CONFIG_KEYS`; never hand-copy key names or the endpoint.
- openai v3.31 uses `httpx2`, not `httpx`.
- After retries are exhausted the original `openai` error is re-raised (`reraise=True`), as the spec says.
