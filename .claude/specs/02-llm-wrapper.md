# Spec: Step 02 — LLM Wrapper

## Overview
This step builds `llm/client.py`: the single function through which
every LLM call in the project goes, `generate(prompt, model, temperature,
schema=None)`. It talks to Ollama's OpenAI-compatible endpoint with the
`openai` SDK, retries transient failures with `tenacity`, caches every
response in SQLite keyed by a hash of (model, prompt, temperature, schema),
and logs tokens and latency for every call. It comes before the pipeline
(Step 3) and the judges (Steps 7-8) because both need LLM calls. If the
caching and logging are right from the start, every later run is cheap to
repeat, gives the same answer for the same input, and has a record of what
it cost. Models decided on 2026-10-15: generator `gemma4:31b-cloud`, judge
`gpt-oss:120b-cloud` (the larger model), both Ollama Free-plan cloud models
reached through the local Ollama server. (`deepseek-v4.1-flash:cloud` was the
first judge choice, but Ollama returned `402 Payment Required`: it is not in
the Free plan.)

## Concepts to learn
- **LLM wrapper:** one small function that hides the HTTP details of calling
  a model. Every caller (generator, faithfulness judge, relevance judge) gets
  retries, caching and logging for free, and switching provider later means
  changing one file.
- **OpenAI-compatible endpoint:** many providers accept the same request
  format as OpenAI's Chat Completions API. Ollama serves one at
  `http://localhost:11434/v1`. So we use the `openai` SDK with
  `base_url="http://localhost:11434/v1"`. A model tag ending in `:cloud`
  (e.g. `gpt-oss:120b-cloud`) is forwarded by the local Ollama server to
  Ollama's cloud. That needs `ollama signin`, but no API key in our code.
  The SDK still insists on a non-empty `api_key`, so we pass a dummy value
  (`ollama`) from `.env`.
- **Temperature:** controls randomness when the model picks the next token.
  0 means (nearly) always the most likely token, so the same input gives the
  same output. CLAUDE.md fixes the judge at 0 and the generator at a low
  value.
- **Caching LLM calls:** the first time we send a request, we store the
  response. Next time the exact same request arrives, we return the stored
  response without calling the model. Why it matters here:
  1. Cost and speed: a 100-query run with 3 LLM calls per query is 300
     calls. Re-running the harness after fixing a bug in metrics code should
     cost nothing.
  2. Repeatability: cloud models are not perfectly deterministic, even at
     temperature 0. With a cache, re-scoring the same run gives exactly the
     same numbers.
  3. Trade-off: the cache freezes one sample. To get a fresh answer, change
     the input (for example bump the judge prompt version, which is part of
     the prompt) or delete the cache file.
- **Cache key = hash of the request:** we build a canonical string from
  everything that changes the output, then hash it with SHA-256:
  ```
  key_input = json.dumps({"base_url": "http://localhost:11434/v1",
                          "model": "gpt-oss:120b-cloud",
                          "model_digest": null,
                          "prompt": "Say hi",
                          "temperature": 0.0,
                          "schema": null}, sort_keys=True)
  cache_key = sha256(key_input).hexdigest()   # 64 hex characters
  ```
  `sort_keys=True` makes the string identical no matter the dict order.
  Changing any one field (even temperature 0.0 to 0.1) gives a completely
  different key, so it is a cache miss. The schema is included as its JSON
  Schema text, so changing a field in a pydantic model also misses. The
  endpoint is included so the same model tag served somewhere else never
  reuses old replies. `model_digest` (the model build, e.g. an Ollama digest)
  is included so an updated model never reuses replies from the old build; it
  is None until Step 5 supplies it.
- **Structured output:** judges must return data we can parse, e.g.
  `{"claims": ["...", "..."]}`, not free prose. We describe the shape with a
  pydantic model, send its JSON Schema with the request
  (`response_format={"type": "json_schema", ...}`), and then validate the
  reply with pydantic. Validation is the safety net: if the model returns
  broken or wrong-shaped JSON, we raise a clear error instead of silently
  storing garbage, and the bad reply is not cached.
- **Retries with exponential backoff:** network calls fail now and then
  (timeouts, rate limits, 5xx server errors). These are *transient*, and
  trying again later usually works. We wait longer after each failure:
  `wait = min(wait_max, wait_min × 2^(attempt-1))`.
  Example with `wait_min=2`, `wait_max=30`, `max_attempts=5`: waits of 2, 4,
  8, 16 s between the 5 attempts, then give up and raise. *Permanent* errors
  (bad request, unknown model) are not retried, because retrying cannot fix
  them.
- **Tokens:** tokens are the units models read and write; providers bill
  and rate-limit by them. `usage.prompt_tokens` is input,
  `usage.completion_tokens` is output (for thinking models like gpt-oss this
  includes the hidden reasoning tokens). If the server reports no usage we
  store NULL, not 0: unknown is not zero. A cache-hit row copies the original
  counts, so cost totals must filter on `cached = 0`.
- **Two latencies, kept apart:**
  - `model_latency_ms`: the time of the one successful HTTP request, i.e.
    real model time. It is stored in the cache, so a cache hit reports the
    original model time. Use this for "how fast is the model".
  - `latency_ms`: the wall-clock time of the whole `generate()` call
    (config load, cache lookup, retry waits, validation). On a cache hit it
    is about 2 ms. Use this only for "how long did the harness take".
  Example: a miss that failed once and succeeded on try 2 might show
  `model_latency_ms=1100` but `latency_ms=3200` (it includes a 2 s backoff
  wait). Mixing these up would make re-runs look 1000x faster than the model.
- **Finish reason:** each reply says why the model stopped: `"stop"` (it
  finished normally), `"length"` (it hit an output limit and was cut off),
  `"content_filter"`, or nothing at all. Only `"stop"` is a valid answer;
  anything else raises and is never cached.

## Depends on
- Step 0: venv with `openai`, `pydantic`, `tenacity`, `pyyaml`,
  `python-dotenv` installed; `.gitignore` already ignores `*.sqlite`.
- Step 1: nothing used directly.
- Local setup (not code): Ollama running on `localhost:11434`, signed in
  with `ollama signin`, and `gemma4:31b-cloud` plus `gpt-oss:120b-cloud`
  pulled (both confirmed in `ollama list` on 2026-10-15).

## Interfaces
`llm/client.py` (import as `from llm.client import generate`)
```python
@dataclass
class LLMResult:
    text: str                       # raw reply text from the model
    parsed: BaseModel | None        # validated pydantic object when schema is given, else None
    model: str
    prompt_tokens: int | None       # None if the server reported no usage
    completion_tokens: int | None
    finish_reason: str | None       # e.g. "stop"
    latency_ms: float               # wall-clock time of THIS generate() call (tiny on a cache hit)
    model_latency_ms: float         # time of the original successful HTTP request
    cached: bool                    # True if served from the cache
    attempts: int                   # HTTP attempts this call made (0 on a cache hit)
    model_digest: str | None        # model build the caller passed (None until Step 5)

def generate(prompt: str, model: str, temperature: float,
             schema: type[BaseModel] | None = None, *,
             config: dict | None = None,
             model_digest: str | None = None) -> LLMResult:
    """Send one user message to the model and return its reply.
    Checks the cache first; on a miss, calls the API with retries, checks the
    reply (non-empty, finish_reason == "stop", valid against `schema` if
    given), stores it, and logs the call. Every call leaves exactly one
    llm_calls row: a failed call (hit or miss) is logged with its error, then
    re-raised. `config` defaults to load_config()."""

def make_cache_key(prompt: str, model: str, temperature: float,
                   schema: type[BaseModel] | None, base_url: str,
                   model_digest: str | None = None) -> str:
    """SHA-256 hex digest of the canonical JSON of the six inputs."""

def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read configs/llm.yaml. Raises ValueError naming the file and any missing
    keys (REQUIRED_CONFIG_KEYS), or an unusable value: max_attempts must be an
    int >= 1, timeout_s a number > 0, retry waits numbers >= 0. A relative
    cache_path is resolved against the repo root."""

REQUIRED_CONFIG_KEYS: tuple[str, ...]   # the six keys below

class LLMOutputError(Exception):
    """The model replied, but the reply was empty, did not finish normally
    (finish_reason other than "stop"), or did not match the schema."""
```
Example:
```python
from pydantic import BaseModel
from llm.client import generate

r = generate("In one word, what colour is the sky?", model="gpt-oss:120b-cloud", temperature=0.0)
r.text, r.cached, r.model_latency_ms     # ("Blue", False, 2140.5)
r = generate("In one word, what colour is the sky?", model="gpt-oss:120b-cloud", temperature=0.0)
r.cached, r.latency_ms, r.model_latency_ms   # (True, 0.9, 2140.5)

class Claims(BaseModel):
    claims: list[str]
r = generate("Split into claims: Interns get 1 CL per month.", "gpt-oss:120b-cloud", 0.0, schema=Claims)
r.parsed.claims                       # ["Interns get 1 casual leave per completed month."]
```
(The values above are illustrative, not measured.)

CLI demo (used for the Done-when check):
```
python -m llm.client --model gpt-oss:120b-cloud --prompt "Say hi in 3 words"
```
`main()` calls `generate()` twice with the same input. For each call it
prints `cached`, `latency_ms`, `model_latency_ms`, prompt and completion
tokens, `finish_reason` and the reply. The first call is a miss and the
second a hit. If the cache already holds the prompt, both are hits; the
output says so. With `--json-demo`, it uses a small built-in `DemoClaims`
schema and prints the validated `parsed` object.

Internal helpers (monkeypatched in tests so no real API is called):
```python
@dataclass
class ApiReply:                 # what one successful HTTP request returned
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    model_latency_ms: float

def _get_client(base_url: str, timeout_s: float) -> openai.OpenAI:
    """openai client with max_retries=0; LLM_API_KEY read from .env."""

def _send_request(client, prompt, model, temperature, schema) -> ApiReply:
    """Exactly one HTTP call; times only the request (model_latency_ms)."""

@dataclass
class CallStats:                # filled in by the retry loop
    attempts: int = 0           # +1 per HTTP attempt, kept even if the call fails

def _call_api(prompt, model, temperature, schema, config, stats: CallStats) -> ApiReply:
    """_send_request wrapped in the tenacity retry loop built from config;
    increments stats.attempts once per attempt."""
```
Output checks (empty reply, `finish_reason != "stop"`, schema validation)
run in `generate()` after `_call_api` returns, so they apply to fakes too
and are never retried.

This step does not touch the `answer()` contract.

## Data and schema changes
New SQLite file `data/llm_cache.sqlite` (path from config; gitignored by
`*.sqlite`, and its `-journal`/`-wal`/`-shm` side files are gitignored too).
It is kept separate from the results database (Step 5) so the cache can be
deleted without losing any run data. Two tables:

`llm_cache` (one row per distinct request; only validated replies):
| column | type | meaning |
|---|---|---|
| cache_key | TEXT PRIMARY KEY | SHA-256 of (base_url, model, model_digest, prompt, temperature, schema) |
| base_url | TEXT | endpoint the reply came from |
| model | TEXT | model tag |
| model_digest | TEXT | model build; NULL until Step 5 supplies it |
| temperature | REAL | |
| schema_name | TEXT | pydantic class name, or NULL |
| prompt | TEXT | full prompt, kept for debugging |
| response_text | TEXT | raw reply (fence included, if any) |
| prompt_tokens | INTEGER | from the API's usage; NULL if not reported |
| completion_tokens | INTEGER | from the API's usage; NULL if not reported |
| finish_reason | TEXT | e.g. "stop" |
| model_latency_ms | REAL | time of the original successful HTTP request |
| created_at | TEXT | ISO-8601 UTC timestamp |

`llm_calls` (one row per `generate()` call: hit, miss, or failure):
| column | type | meaning |
|---|---|---|
| id | INTEGER PRIMARY KEY | |
| called_at | TEXT | ISO-8601 UTC timestamp |
| cache_key | TEXT | links to `llm_cache` |
| model | TEXT | |
| model_digest | TEXT | NULL until Step 5 supplies it |
| cached | INTEGER | 1 = cache hit, 0 = real API call |
| attempts | INTEGER | HTTP attempts made (0 on a cache hit; >1 means retries) |
| prompt_tokens | INTEGER | NULL if unknown (no usage, or the API call failed) |
| completion_tokens | INTEGER | NULL if unknown |
| finish_reason | TEXT | NULL if the API call failed or the server sent none |
| latency_ms | REAL | wall-clock time of this generate() call |
| model_latency_ms | REAL | time of the successful HTTP request; NULL if it failed |
| error | TEXT | NULL on success; else "<ErrorType>: <message, max 200 chars>" |

Tables are created with `CREATE TABLE IF NOT EXISTS` on first use. Only
successful, checked replies go into `llm_cache`. The table layout changed
twice during the Step 2 code reviews; each time the cache file (2 rows) was
deleted and rebuilt on 2026-10-15.

## Config changes
New file `configs/llm.yaml`, the endpoint, retry and cache settings for
`llm/client.py`. Model names and temperatures are NOT here; callers pass them
(the generator's come from `configs/v0.yaml` in Step 3, the judge's from the
judge config in Step 7).
```yaml
base_url: http://localhost:11434/v1   # Ollama's OpenAI-compatible endpoint; part of the cache key
timeout_s: 120          # max seconds to wait for one API response
max_attempts: 5         # total tries per call, including the first
retry_wait_min_s: 2     # first backoff wait; doubles each retry
retry_wait_max_s: 30    # cap on a single backoff wait
cache_path: data/llm_cache.sqlite   # relative to the repo root
```
All six keys are required; `load_config` raises `ValueError` listing any
that are missing. Values are starting defaults, not tuned. `timeout_s` is
generous because thinking models can take tens of seconds. `base_url` is
not a secret, so it lives here, not in `.env`.

`.env` (local only, never committed):
```
LLM_API_KEY=ollama        # dummy; Ollama ignores it, the SDK requires one
```

## Files to create
- `llm/__init__.py`
- `llm/client.py`
- `configs/llm.yaml`
- `tests/test_02_llm_wrapper.py`
- `tests/conftest.py` (shared fixtures)

## Files to change
- `.env`: only `LLM_API_KEY` (dummy). Local only; it is gitignored.
- `.gitignore`: add `*.sqlite-journal`, `*.sqlite-wal`, `*.sqlite-shm`,
  `*.db-journal`.
- `CLAUDE.md`: Step 2 entry in the Progress section. Because of Decision 1,
  also: the Repo layout gains `llm/client.py` (and drops `llm.py` from
  `evalharness/`), the Step 2 heading names `llm/client.py`, the caching
  convention says all LLM calls go through `llm/client.py`, and the
  Architecture section notes the shared `llm/` package.
- `.claude/agents/rag-eval-integrity-reviewer.md`: project context matches
  this design (`llm/client.py`, openai SDK, Ollama cloud models).

## New dependencies
No new dependencies. `openai`, `pydantic`, `tenacity`, `pyyaml` and
`python-dotenv` are already pinned in `requirements.txt`.

## Tests
`tests/test_02_llm_wrapper.py`. Every test passes a config dict (via the
`config=` argument of `generate()`, or by monkeypatching
`llm.client.load_config`) whose `cache_path` is a `tmp_path` SQLite file,
whose `base_url` is a fake URL, and whose retry waits are 0. Cache, output
check and logging tests monkeypatch `llm.client._call_api` with a fake that
returns an `ApiReply` and counts calls. Retry tests monkeypatch
`llm.client._send_request` (one HTTP call) and `llm.client._get_client`, so
the real tenacity loop runs. Build
`openai.APIConnectionError(request=httpx2.Request("POST", "http://test"))`
for transient errors (comment the import: `httpx2` arrives through
`openai==3.31`). Fakes of `_call_api` must accept the `stats` argument and do
`stats.attempts += 1`. `_send_request` tests use a fake client object whose
`chat.completions.create(**kwargs)` records the kwargs and returns a canned
response. Config tests take key names from `llm.client.REQUIRED_CONFIG_KEYS`
(no hand-copied lists, no literal endpoint value). Every helper function in
the test file has a one-line docstring. No test reaches Ollama or the
network.
- `test_cache_key_is_stable`: same six inputs give the same 64-char hex
  key.
- `test_cache_key_changes_with_each_input`: changing base_url, model,
  model_digest, prompt, temperature or schema each gives a different key.
- `test_model_digest_is_stored_and_misses_cache`: the same request with
  digest "a" then "b" calls the fake twice; both tables store the digest.
- `test_second_identical_call_hits_cache`: two identical `generate()` calls.
  The fake is called once, the second result has `cached=True`, and the text
  is identical.
- `test_cache_hit_reports_original_model_latency`: the fake reply has
  `model_latency_ms=1234.0`. Both the miss and the hit return
  `model_latency_ms == 1234.0`; the hit's `latency_ms` is separate.
- `test_different_temperature_misses_cache`: same prompt at 0.0 and 0.2
  calls the fake twice.
- `test_schema_reply_is_parsed`: the fake returns valid JSON. `result.parsed`
  is an instance of the schema with the right values.
- `test_code_fenced_json_is_parsed`: the fake returns
  `"```json\n{\"claims\": [\"a\"]}\n```"`. `result.parsed.claims == ["a"]`.
- `test_invalid_schema_reply_raises_and_is_not_cached`: the fake returns
  `"not json"`. `LLMOutputError` is raised and `llm_cache` has no row for
  that key.
- `test_empty_reply_raises_and_is_not_cached`: the fake returns `""`.
- `test_cut_off_reply_raises_and_is_not_cached`: the fake returns
  `finish_reason="length"`.
- `test_abnormal_finish_reason_raises`: `"content_filter"` and `None` each
  raise `LLMOutputError` and are not cached.
- `test_cache_hit_parse_failure_is_logged`: a cached reply edited to invalid
  JSON raises `LLMOutputError` on the hit and leaves one `llm_calls` row with
  `cached = 1` and `error` set.
- `test_missing_usage_is_stored_as_null`: the fake returns `None` token
  counts; `LLMResult` and the `llm_calls` row hold `None` / NULL, not 0.
- `test_transient_error_is_retried`: `_send_request` raises
  `openai.APIConnectionError` twice, then succeeds. The call succeeds after 3
  attempts.
- `test_gives_up_after_max_attempts`: it always raises. The original error
  propagates after exactly `max_attempts` tries.
- `test_attempts_are_logged`: 2 transient errors then success logs
  `attempts = 3`; a cache hit logs `attempts = 0`; giving up logs
  `attempts = max_attempts`.
- `test_send_request_builds_the_request`: with a fake client, the request has
  the given model and temperature, exactly one user message with the prompt,
  and `response_format` (type `json_schema`) only when a schema is given.
- `test_send_request_maps_the_response`: `usage=None` gives `None` token
  counts, `content=None` gives `""`, and `finish_reason` is copied.
- `test_every_call_is_logged`: after one miss and one hit, `llm_calls` has 2
  rows with `cached` = 0 and 1, `error` NULL, and token counts copied from
  the fake.
- `test_failed_call_is_logged_with_error`: an invalid-schema reply and a
  permanent API error each leave one `llm_calls` row whose `error` starts
  with the exception type name; token counts are kept when the reply
  arrived and NULL when the API call itself failed.
- `test_load_config_rejects_missing_keys`: for each key in
  `REQUIRED_CONFIG_KEYS`, a YAML file without it makes `load_config` raise
  `ValueError` naming that key.
- `test_load_config_rejects_bad_values`: `max_attempts: 0`,
  `timeout_s: 0` and `retry_wait_min_s: -1` each raise `ValueError`.
- `test_rag_does_not_import_evalharness`: no file under `rag/` contains
  `evalharness` in an import line (architecture guard).

## Rules for implementation
- Plain Python and only libraries approved in CLAUDE.md
- No LangChain; no RAGAS or DeepEval (except the optional final cross-check)
- `rag/` must never import from `evalharness/`; the harness only calls `answer()`
- All tunable values come from YAML config — no hardcoded magic numbers
- SQLite via the sqlite3 module with parameterised queries only
- All LLM calls go through `llm/client.py` so they are cached
- API keys only from `.env`, never in code, logs or commits
- The held-out split is never used before Step 12
- Never invent, estimate or adjust metric numbers; report only what a
  stored run produced
- Human labels (calibration and review) come only from Gaurav
- Code is simple, readable and commented

Step-specific:
- Use the `openai` SDK (`client.chat.completions.create`) with `base_url`
  from `configs/llm.yaml` and `api_key` from `.env` via python-dotenv. No
  `ChatOllama`, no LangChain, no `ollama` Python package.
- The prompt is sent as a single user message. No hidden system prompt, so
  the cache key really covers everything sent.
- Cache key inputs are exactly base_url, model, prompt, temperature and the
  schema's JSON Schema (`schema.model_json_schema()`), serialised with
  `json.dumps(..., sort_keys=True)` and hashed with SHA-256. If a later step
  adds another request parameter that changes the output (e.g. `seed`,
  `reasoning_effort`), it must be added to the key and the stored row too.
- Retry only transient errors: `APIConnectionError`, `APITimeoutError`,
  `RateLimitError`, `InternalServerError`. Never retry `BadRequestError`,
  `NotFoundError` or `LLMOutputError`. The SDK's own retries are off
  (`max_retries=0`).
- Never cache a failed call, an empty reply, a reply that did not finish
  normally (`finish_reason != "stop"`) or a reply that fails schema
  validation.
- Every `generate()` call leaves exactly one `llm_calls` row, including
  failures on a miss or a hit (with `error` set), and records how many HTTP
  attempts it made. Never turn an error into a normal-looking row.
- `model_digest` is part of the cache key and both tables. This step only
  passes it through; Step 5 decides how to obtain it.
- Unknown token counts are NULL, never 0.
- `model_latency_ms` times only the successful HTTP request. `latency_ms` is
  wall time. Later steps must report model speed from `model_latency_ms`,
  never from `latency_ms`.
- Always validate structured replies with pydantic, even if the server
  claims to enforce the schema. If Ollama rejects `response_format` of type
  `json_schema` for a model, fall back to `{"type": "json_object"}` plus the
  schema in the prompt, and say so in the Step 2 summary. Do not silently
  change behaviour.
- Before validating, strip ONE Markdown code fence that wraps the whole
  reply (```` ```json ... ``` ````). On 2026-10-15 `gpt-oss:120b-cloud`
  accepted `response_format` of type `json_schema` but still returned fenced
  JSON, so the server does not enforce the schema for this model. The cache
  stores the raw reply; any other malformed reply still raises
  `LLMOutputError`.
- Never log or print the API key or the `.env` contents.
- Token counts and latencies printed in the Step 2 summary must be copied
  from the actual `llm_calls` rows, not estimated.

## Carry-forward to later steps (from the 2026-10-15 code review)
These came out of the Step 2 review and belong to later steps. Each later
spec must address the ones for its step.
- **Step 3:** decide `seed` and `reasoning_effort` (thinking level) for the
  generator and judge; take them from config, send them, and add them to the
  cache key and stored row. *Decision for Gaurav.*
- **Step 3:** context window. The `/v1` endpoint cannot set `num_ctx`.
  Decide how the context limit is guaranteed, add a per-model
  `context_limit` to config, and flag answers whose `prompt_tokens` come
  close to it (`truncation_suspected`) instead of scoring them silently.
- **Step 3:** `answer()["latency_ms"]` must come from `model_latency_ms`
  (plus retrieval time), never from the wall-time `latency_ms`.
- **Step 5:** obtain each model's identity (Ollama digest where available),
  pass it to `generate(..., model_digest=...)` so it enters the cache key and
  rows, and record it in run metadata with per-run cache-hit counts. Add a
  run-level abort after N consecutive failed calls.
- **Steps 5 and 7:** record per-item failures in `scores` and report
  "scored N, failed M", broken down by query type (timeouts hit the hardest
  items first, so failures are not random); never score a failed item as 0
  or 1.
- **Step 6 and later:** cost and token totals filter `llm_calls` on
  `cached = 0`; latency figures filter on `cached = 0 AND error IS NULL`, use
  `model_latency_ms`, and exclude the first (cold) call per model or warm up
  first. `attempts > 1` shows retries.
- **Step 7:** judge schemas are strict: `extra="forbid"`, `Literal` labels,
  bounded scores (`Field(ge=0, le=1)`), no defaults on label or score fields.
- **Step 14:** wall-clock budget for CI runs (`timeout-minutes:`); never
  upload `data/*.sqlite` as a CI artifact (the cache stores full prompts).
- **Before any real API key is used:** allow-list `base_url` hosts in
  `_get_client` (today the key is a dummy and the endpoint is localhost).

## Decisions for Gaurav
All resolved on 2026-10-15:
1. ~~Where the generator in `rag/` gets its LLM calls, given "all LLM calls
   go through `evalharness/llm.py`" and "`rag/` must never import from
   `evalharness/`".~~ **Resolved (option B):** the wrapper lives in a neutral
   top-level package, `llm/client.py`, which knows nothing about RAG or
   evaluation. Both `rag/` and `evalharness/` import it, so every call is
   cached and logged, and `rag/` still never imports `evalharness/`.
2. ~~Real API calls for the Done-when check.~~ **Resolved:** approved. About 2
   uncached calls (one plain, one structured). The first structured call to
   `deepseek-v4.1-flash:cloud` failed with `402 Payment Required` (not in the
   Ollama Free plan), so on 2026-10-15 the roles changed: generator
   `gemma4:31b-cloud`, judge `gpt-oss:120b-cloud`.

## Definition of done
- [ ] Done when (CLAUDE.md): an identical second call returns instantly from
      the cache. Run
      `python -m llm.client --model gpt-oss:120b-cloud --prompt "Say hi in 3 words"`.
      Call 1 shows `cached=False` and call 2 shows `cached=True`, with the
      same text, a much lower `latency_ms`, and the same `model_latency_ms`.
- [ ] Structured output works against the real judge model: one call to
      `gpt-oss:120b-cloud` with a small pydantic schema returns a
      validated `parsed` object:
      `python -m llm.client --model gpt-oss:120b-cloud --json-demo --prompt "..."`.
- [ ] `rag/` does not import `evalharness/`: `grep -rn "evalharness" rag/`
      prints nothing.
- [ ] Calls are logged:
      `sqlite3 data/llm_cache.sqlite "SELECT model, cached, attempts, prompt_tokens, completion_tokens, finish_reason, latency_ms, model_latency_ms, error FROM llm_calls"`
      shows one row per call above.
- [ ] `data/llm_cache.sqlite` and `.env` are not tracked:
      `git status --porcelain` and `git ls-files` show neither.
- [ ] `pytest tests/test_02_llm_wrapper.py` passes with no network access.
- [ ] `pytest` passes (whole suite).
- [ ] The Progress section of CLAUDE.md has a Step 2 entry, with token and
      latency figures copied from `llm_calls` and the Step 2 test count.
- [ ] All changes are committed on `feature/02-llm-wrapper`.
