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
  key_input = json.dumps({"model": "gpt-oss:120b-cloud",
                          "prompt": "Say hi",
                          "temperature": 0.0,
                          "schema": null}, sort_keys=True)
  cache_key = sha256(key_input).hexdigest()   # 64 hex characters
  ```
  `sort_keys=True` makes the string identical no matter the dict order.
  Changing any one field (even temperature 0.0 to 0.1) gives a completely
  different key, so it is a cache miss. The schema is included as its JSON
  Schema text, so changing a field in a pydantic model also misses.
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
- **Tokens and latency:** tokens are the units models read and write;
  providers bill and rate-limit by them. `usage.prompt_tokens` is input,
  `usage.completion_tokens` is output (for thinking models like gpt-oss this
  includes the hidden reasoning tokens). Latency is the wall-clock time of the
  call in milliseconds. We log both for every call, including cache hits, so
  later we can report what a run cost and how much the cache saved.

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
    text: str                  # raw reply text from the model
    parsed: BaseModel | None   # validated pydantic object when schema is given, else None
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float          # wall-clock time of THIS call (tiny on a cache hit)
    cached: bool               # True if served from the cache

def generate(prompt: str, model: str, temperature: float,
             schema: type[BaseModel] | None = None) -> LLMResult:
    """Send one user message to the model and return its reply.
    Checks the cache first; on a miss, calls the API with retries,
    validates against `schema` if given, stores the reply, and logs the call.
    Raises LLMOutputError if the reply does not match `schema`."""

def make_cache_key(prompt: str, model: str, temperature: float,
                   schema: type[BaseModel] | None) -> str:
    """SHA-256 hex digest of the canonical JSON of the four inputs."""

class LLMOutputError(Exception):
    """The model replied, but the reply did not validate against the schema."""
```
Example:
```python
from pydantic import BaseModel
from llm.client import generate

r = generate("In one word, what colour is the sky?", model="gpt-oss:120b-cloud", temperature=0.0)
r.text, r.cached, r.latency_ms        # ("Blue", False, 2140.5)
r = generate("In one word, what colour is the sky?", model="gpt-oss:120b-cloud", temperature=0.0)
r.cached, r.latency_ms                # (True, 0.9)

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
Calls `generate()` twice with the same input. For each call it prints the
reply, `cached`, `latency_ms`, prompt and completion tokens. The first call
is a miss and the second a hit. If the cache already holds the prompt, both
are hits; the output says so.

With `--json-demo`, it makes one structured call with a small built-in
`Claims` schema and prints the validated `parsed` object (twice: miss, then
hit).

Internal helpers (monkeypatched in tests so no real API is called):
```python
def load_config(path=CONFIG_PATH) -> dict:
    """Read configs/llm.yaml; cache_path is resolved against the repo root."""

def _send_request(client, prompt, model, temperature, schema) -> tuple[str, int, int]:
    """Exactly one HTTP call. Returns (text, prompt_tokens, completion_tokens)."""

def _call_api(prompt, model, temperature, schema, config) -> tuple[str, int, int]:
    """_send_request wrapped in the tenacity retry loop built from config."""
```
The `OpenAI` client is created with `max_retries=0`, so the SDK's own
built-in retries do not stack on top of tenacity.

This step does not touch the `answer()` contract.

## Data and schema changes
New SQLite file `data/llm_cache.sqlite` (path from config; already
gitignored by `*.sqlite`). It is kept separate from the results database
(Step 5) so the cache can be deleted without losing any run data. Two tables:

`llm_cache` (one row per distinct request):
| column | type | meaning |
|---|---|---|
| cache_key | TEXT PRIMARY KEY | SHA-256 of (model, prompt, temperature, schema) |
| model | TEXT | model tag |
| temperature | REAL | |
| schema_name | TEXT | pydantic class name, or NULL |
| prompt | TEXT | full prompt, kept for debugging |
| response_text | TEXT | raw reply |
| prompt_tokens | INTEGER | from the API's usage |
| completion_tokens | INTEGER | from the API's usage |
| latency_ms | REAL | latency of the original (uncached) call |
| created_at | TEXT | ISO-8601 UTC timestamp |

`llm_calls` (one row per `generate()` call, hit or miss: the token and
latency log):
| column | type | meaning |
|---|---|---|
| id | INTEGER PRIMARY KEY | |
| called_at | TEXT | ISO-8601 UTC timestamp |
| cache_key | TEXT | links to `llm_cache` |
| model | TEXT | |
| cached | INTEGER | 1 = cache hit, 0 = real API call |
| prompt_tokens | INTEGER | |
| completion_tokens | INTEGER | |
| latency_ms | REAL | wall-clock time of this call |

Tables are created with `CREATE TABLE IF NOT EXISTS` on first use. Only
successful, validated replies go into `llm_cache`.

## Config changes
New file `configs/llm.yaml`, the connection, retry and cache settings for
`llm/client.py`. Model names and temperatures are NOT here; callers
pass them (the generator's come from `configs/v0.yaml` in Step 3, the
judge's from the judge config in Step 7).
```yaml
timeout_s: 120          # max seconds to wait for one API response
max_attempts: 5         # total tries per call, including the first
retry_wait_min_s: 2     # first backoff wait; doubles each retry
retry_wait_max_s: 30    # cap on a single backoff wait
cache_path: data/llm_cache.sqlite   # relative to the repo root
```
Values are starting defaults, not tuned. `timeout_s` is generous because
thinking models can take tens of seconds.

`.env` (local only, never committed):
```
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama        # dummy; Ollama ignores it, the SDK requires one
```

## Files to create
- `llm/__init__.py`
- `llm/client.py`
- `configs/llm.yaml`
- `tests/test_02_llm_wrapper.py`

## Files to change
- `.env`: fill in `LLM_BASE_URL` and `LLM_API_KEY` as above, and update the
  comment. Local only; it is gitignored.
- `CLAUDE.md`: Step 2 entry in the Progress section. Because of Decision 1,
  also: the Repo layout gains `llm/client.py` (and drops `llm.py` from
  `evalharness/`), the Step 2 heading names `llm/client.py`, the caching
  convention says all LLM calls go through `llm/client.py`, and the
  Architecture section notes the shared `llm/` package.

## New dependencies
No new dependencies. `openai`, `pydantic`, `tenacity`, `pyyaml` and
`python-dotenv` are already pinned in `requirements.txt`.

## Tests
`tests/test_02_llm_wrapper.py`. Every test monkeypatches
`llm.client.load_config` to return a config whose `cache_path` is a
`tmp_path` SQLite file and whose retry waits are 0. Cache, schema and logging
tests monkeypatch `llm.client._call_api` with a fake that counts calls. Retry
tests monkeypatch `llm.client._send_request` (one HTTP call) and
`llm.client._get_client`, so the real tenacity loop runs. Build
`openai.APIConnectionError(request=httpx2.Request("POST", "http://test"))`
for transient errors. No test reaches Ollama or the network.
- `test_cache_key_is_stable`: same four inputs give the same 64-char hex
  key.
- `test_cache_key_changes_with_each_input`: changing model, prompt,
  temperature or schema each gives a different key.
- `test_second_identical_call_hits_cache`: two identical `generate()` calls.
  The fake is called once, the second result has `cached=True`, and the text
  is identical.
- `test_different_temperature_misses_cache`: same prompt at 0.0 and 0.2
  calls the fake twice.
- `test_schema_reply_is_parsed`: the fake returns valid JSON. `result.parsed`
  is an instance of the schema with the right values.
- `test_code_fenced_json_is_parsed`: the fake returns
  `"```json\n{\"claims\": [\"a\"]}\n```"`. `result.parsed.claims == ["a"]`.
- `test_invalid_schema_reply_raises_and_is_not_cached`: the fake returns
  `"not json"`. `LLMOutputError` is raised and `llm_cache` has no row for
  that key.
- `test_transient_error_is_retried`: the API layer raises
  `openai.APIConnectionError` twice, then succeeds. The call succeeds after
  3 attempts (backoff waits patched to 0).
- `test_gives_up_after_max_attempts`: it always raises. The error propagates
  after exactly `max_attempts` tries.
- `test_every_call_is_logged`: after one miss and one hit, `llm_calls` has 2
  rows with `cached` = 0 and 1, and token counts copied from the fake.
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
  and `api_key` read from `.env` via python-dotenv. No `ChatOllama`, no
  LangChain, no `ollama` Python package.
- The prompt is sent as a single user message. No hidden system prompt, so
  the cache key really covers everything sent.
- Cache key inputs are exactly model, prompt, temperature and the schema's
  JSON Schema (`schema.model_json_schema()`), serialised with
  `json.dumps(..., sort_keys=True)` and hashed with SHA-256. If a later step
  adds another request parameter that changes the output, it must be added
  to the key too.
- Retry only transient errors: `APIConnectionError`, `APITimeoutError`,
  `RateLimitError`, `InternalServerError`. Never retry `BadRequestError`,
  `NotFoundError` or `LLMOutputError`.
- Never cache a failed call or a reply that fails schema validation.
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
- If a reply has empty `content` (for example a thinking model spent all its
  output on reasoning), raise an error. Do not cache or return an empty
  string as a valid answer.
- Never log or print the API key or the `.env` contents.
- Token counts and latencies printed in the Step 2 summary must be copied
  from the actual `llm_calls` rows, not estimated.

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
      same text and a much lower latency.
- [ ] Structured output works against the real judge model: one call to
      `gpt-oss:120b-cloud` with a small pydantic schema returns a
      validated `parsed` object:
      `python -m llm.client --model gpt-oss:120b-cloud --json-demo --prompt "..."`.
- [ ] `rag/` does not import `evalharness/`: `grep -rn "evalharness" rag/`
      prints nothing.
- [ ] Calls are logged:
      `sqlite3 data/llm_cache.sqlite "SELECT model, cached, prompt_tokens, completion_tokens, latency_ms FROM llm_calls"`
      shows one row per call above.
- [ ] `data/llm_cache.sqlite` and `.env` are not tracked:
      `git status --porcelain` and `git ls-files` show neither.
- [ ] `pytest tests/test_02_llm_wrapper.py` passes with no network access.
- [ ] `pytest` passes (whole suite).
- [ ] The Progress section of CLAUDE.md has a Step 2 entry, with token and
      latency figures copied from `llm_calls`.
- [ ] All changes are committed on `feature/02-llm-wrapper`.
