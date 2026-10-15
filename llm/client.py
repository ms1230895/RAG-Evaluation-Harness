"""The one place every LLM call in this project goes through.

generate(prompt, model, temperature, schema=None) sends one user message to a
model and returns its reply. Around that single call it adds:

- caching: the reply is stored in SQLite under a hash of (endpoint, model,
  prompt, temperature, schema); an identical request later is answered from
  the cache, so re-runs are free and give exactly the same output,
- retries: transient failures (connection errors, timeouts, rate limits, 5xx)
  are retried with exponential backoff; permanent ones are raised at once,
- output checks: an empty reply, a reply that did not finish normally
  (finish_reason other than "stop", e.g. cut off by a length limit), or (when
  a pydantic schema is given) a reply that does not validate raises
  LLMOutputError and is never cached,
- logging: every call (cache hit, miss or failure) is written to the llm_calls
  table with its token counts, finish reason, latencies, number of API
  attempts and any error.

Two latencies are kept apart on purpose:
- model_latency_ms: time of the one successful HTTP request, i.e. real model
  time. Stored in the cache, so a cache hit reports the original model time.
- latency_ms: wall-clock time of this generate() call (cache lookup, retry
  waits and all). On a cache hit it is tiny. Use it for "how long did the
  harness take", never for "how fast is the model".

The models are reached through Ollama's OpenAI-compatible endpoint with the
openai SDK. The endpoint, retry and cache settings come from configs/llm.yaml;
the (dummy) API key comes from .env. This module knows nothing about RAG or
evaluation, so both rag/ (the generator) and evalharness/ (the judges) can
import it.

Quick check from the command line:
    python -m llm.client --model gpt-oss:120b-cloud --prompt "Say hi in 3 words"
"""

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import openai
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from tenacity import (RetryCallState, Retrying, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "llm.yaml"

# Keys configs/llm.yaml must define; checked once in load_config.
REQUIRED_CONFIG_KEYS = ("base_url", "timeout_s", "max_attempts",
                        "retry_wait_min_s", "retry_wait_max_s", "cache_path")

# Errors worth retrying: trying again later can succeed.
# (APITimeoutError is a subclass of APIConnectionError; listed for readability.)
# Anything else, e.g. BadRequestError or NotFoundError (unknown model), is
# permanent and raised immediately.
TRANSIENT_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)

# How much of a bad reply or error message to keep in errors and the call log.
ERROR_PREVIEW_CHARS = 200

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    text: str                       # raw reply text from the model
    parsed: BaseModel | None        # validated pydantic object when a schema is given, else None
    model: str
    prompt_tokens: int | None       # None if the server did not report usage
    completion_tokens: int | None   # includes hidden reasoning tokens for thinking models
    finish_reason: str | None       # always "stop" here; anything else raises
    latency_ms: float               # wall-clock time of THIS generate() call (tiny on a cache hit)
    model_latency_ms: float         # time of the original successful HTTP request (real model time)
    cached: bool                    # True if served from the cache
    attempts: int                   # API attempts this call made (0 on a cache hit)
    model_digest: str | None        # model build the caller said it used (None until Step 5)


@dataclass
class CallStats:
    """Filled in by the retry loop, so the attempt count survives even when the call fails."""
    attempts: int = 0


@dataclass
class ApiReply:
    """What one successful HTTP request returned."""
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    model_latency_ms: float


class LLMOutputError(Exception):
    """The model replied, but the reply was empty, did not finish normally
    (finish_reason other than "stop"), or did not match the schema."""


# ---------------------------------------------------------------- helpers

def _now() -> str:
    """Current UTC time as an ISO-8601 string, for created_at / called_at columns."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _elapsed_ms(start: float) -> float:
    """Milliseconds since `start` (a time.perf_counter() value)."""
    return (time.perf_counter() - start) * 1000


# ---------------------------------------------------------------- config

def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read configs/llm.yaml, check every required key is there, and resolve
    a relative cache_path against the repo root."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"{path} is missing keys: {missing}")
    # Catch unusable values here, not later as an obscure tenacity/openai error.
    if not isinstance(config["max_attempts"], int) or config["max_attempts"] < 1:
        raise ValueError(f"{path}: max_attempts must be an integer >= 1")
    if not isinstance(config["timeout_s"], (int, float)) or config["timeout_s"] <= 0:
        raise ValueError(f"{path}: timeout_s must be a number > 0")
    for key in ("retry_wait_min_s", "retry_wait_max_s"):
        if not isinstance(config[key], (int, float)) or config[key] < 0:
            raise ValueError(f"{path}: {key} must be a number >= 0")
    cache_path = Path(config["cache_path"])
    if not cache_path.is_absolute():
        cache_path = REPO_ROOT / cache_path
    config["cache_path"] = cache_path
    return config


# ---------------------------------------------------------------- cache key

def make_cache_key(prompt: str, model: str, temperature: float,
                   schema: type[BaseModel] | None, base_url: str,
                   model_digest: str | None = None) -> str:
    """SHA-256 hex digest of the canonical JSON of everything that changes the output.

    sort_keys=True makes the string identical whatever the dict order, and
    float() makes temperature 0 and 0.0 the same key. The schema is included
    as its JSON Schema, so changing a field in the pydantic model is a miss.
    base_url is included so the same model tag served by a different endpoint
    never reuses replies from the old one. model_digest (the model build, e.g.
    an Ollama digest; None until Step 5 supplies it) is included so a re-pulled
    or updated model never reuses replies from the old build.
    """
    key_input = json.dumps(
        {
            "base_url": base_url,
            "model": model,
            "model_digest": model_digest,
            "prompt": prompt,
            "temperature": float(temperature),
            "schema": schema.model_json_schema() if schema is not None else None,
        },
        sort_keys=True,
    )
    return hashlib.sha256(key_input.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- SQLite

def _create_tables(conn: sqlite3.Connection) -> None:
    """Create both cache tables if they do not exist yet."""
    with conn:
        # One row per distinct request: the stored, validated reply.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key         TEXT PRIMARY KEY,
                base_url          TEXT,
                model             TEXT,
                model_digest      TEXT,
                temperature       REAL,
                schema_name       TEXT,
                prompt            TEXT,
                response_text     TEXT,
                prompt_tokens     INTEGER,
                completion_tokens INTEGER,
                finish_reason     TEXT,
                model_latency_ms  REAL,
                created_at        TEXT
            )""")
        # One row per generate() call (hit, miss or failure): the call log.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                id                INTEGER PRIMARY KEY,
                called_at         TEXT,
                cache_key         TEXT,
                model             TEXT,
                model_digest      TEXT,
                cached            INTEGER,
                attempts          INTEGER,
                prompt_tokens     INTEGER,
                completion_tokens INTEGER,
                finish_reason     TEXT,
                latency_ms        REAL,
                model_latency_ms  REAL,
                error             TEXT
            )""")


def _connect(cache_path: Path) -> sqlite3.Connection:
    """Open the cache database, creating the file and both tables if needed.
    The caller closes the connection (use contextlib.closing)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cache_path)
    try:
        _create_tables(conn)
    except sqlite3.Error:
        conn.close()   # do not leak the connection if the file is locked or corrupt
        raise
    return conn


def _read_cache(conn: sqlite3.Connection, key: str) -> ApiReply | None:
    """Return the stored reply for `key`, or None on a miss."""
    row = conn.execute(
        "SELECT response_text, prompt_tokens, completion_tokens, finish_reason, model_latency_ms"
        " FROM llm_cache WHERE cache_key = ?",
        (key,),
    ).fetchone()
    if row is None:
        return None
    return ApiReply(text=row[0], prompt_tokens=row[1], completion_tokens=row[2],
                    finish_reason=row[3], model_latency_ms=row[4])


def _write_cache(conn: sqlite3.Connection, *, key: str, base_url: str, model: str,
                 model_digest: str | None, temperature: float, schema_name: str | None,
                 prompt: str, reply: ApiReply) -> None:
    """Store one validated reply. Only called after all output checks passed."""
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache (cache_key, base_url, model, model_digest,"
            " temperature, schema_name, prompt, response_text, prompt_tokens,"
            " completion_tokens, finish_reason, model_latency_ms, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key, base_url, model, model_digest, float(temperature), schema_name, prompt,
             reply.text,
             reply.prompt_tokens, reply.completion_tokens, reply.finish_reason,
             reply.model_latency_ms, _now()),
        )


def _log_call(conn: sqlite3.Connection, *, key: str, model: str, model_digest: str | None,
              cached: bool, attempts: int, reply: ApiReply | None, latency_ms: float,
              error: str | None) -> None:
    """Append one row to llm_calls. `reply` is None when the API call itself failed."""
    with conn:
        conn.execute(
            "INSERT INTO llm_calls (called_at, cache_key, model, model_digest, cached, attempts,"
            " prompt_tokens, completion_tokens, finish_reason, latency_ms, model_latency_ms, error)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_now(), key, model, model_digest, int(cached), attempts,
             reply.prompt_tokens if reply else None,
             reply.completion_tokens if reply else None,
             reply.finish_reason if reply else None,
             latency_ms,
             reply.model_latency_ms if reply else None,
             error),
        )


# ---------------------------------------------------------------- API call

def _get_client(base_url: str, timeout_s: float) -> openai.OpenAI:
    """Build the openai client. The API key comes from .env and is never printed or logged."""
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set. Add it to .env (a dummy value works for Ollama).")
    # max_retries=0: the SDK has its own built-in retries; we turn them off so
    # tenacity is the only retry loop and the attempt count is what we configured.
    return openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)


def _send_request(client: openai.OpenAI, prompt: str, model: str, temperature: float,
                  schema: type[BaseModel] | None) -> ApiReply:
    """Exactly one HTTP call. Only the request itself is timed (model_latency_ms)."""
    request = {
        "model": model,
        # A single user message and no hidden system prompt, so the cache key
        # covers everything we send.
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if schema is not None:
        # Ask the server to constrain the reply to this JSON Schema.
        # We still validate with pydantic afterwards (see _check_reply).
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }

    start = time.perf_counter()
    response = client.chat.completions.create(**request)
    model_latency_ms = _elapsed_ms(start)

    choice = response.choices[0]
    usage = response.usage
    return ApiReply(
        text=choice.message.content or "",
        # None (not 0) when the server reports no usage: unknown is not zero.
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
        finish_reason=choice.finish_reason,
        model_latency_ms=model_latency_ms,
    )


def _log_retry(retry_state: RetryCallState) -> None:
    """tenacity hook: called before each backoff wait."""
    error = retry_state.outcome.exception()
    logger.warning("LLM call failed (attempt %d, %s); retrying",
                   retry_state.attempt_number, type(error).__name__)


def _call_api(prompt: str, model: str, temperature: float,
              schema: type[BaseModel] | None, config: dict, stats: CallStats) -> ApiReply:
    """_send_request wrapped in a retry loop built from config.
    stats.attempts counts every HTTP attempt, including failed ones.

    Waits between tries: retry_wait_min_s * 2^(attempt-1), capped at
    retry_wait_max_s. With min=2, max=30, max_attempts=5: 2, 4, 8, 16 s.
    reraise=True: after the last try the original error is raised, not a
    tenacity RetryError.
    """
    client = _get_client(config["base_url"], config["timeout_s"])
    retrying = Retrying(
        stop=stop_after_attempt(config["max_attempts"]),
        # retry_wait_min_s is used twice on purpose: as the multiplier it makes
        # the first wait equal to it (min * 2^0), and as `min` it is the floor.
        wait=wait_exponential(
            multiplier=config["retry_wait_min_s"],
            min=config["retry_wait_min_s"],
            max=config["retry_wait_max_s"],
        ),
        retry=retry_if_exception_type(TRANSIENT_ERRORS),
        before_sleep=_log_retry,
        reraise=True,
    )

    def one_attempt() -> ApiReply:
        stats.attempts += 1
        return _send_request(client, prompt, model, temperature, schema)

    return retrying(one_attempt)


def _strip_code_fence(text: str) -> str:
    """Remove one Markdown code fence around a reply, e.g. ```json ... ```.

    Some models wrap JSON in a fence even when response_format asks for a
    schema (seen with gpt-oss:120b-cloud through Ollama on 2026-10-15). Only a
    fence around the WHOLE reply is removed; anything else is left unchanged
    and must still validate.
    """
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```") and "\n" in stripped:
        first_newline = stripped.index("\n")   # drops the opening ``` or ```json line
        return stripped[first_newline + 1:-3].strip()
    return text


def _parse(text: str, schema: type[BaseModel]) -> BaseModel:
    """Validate a JSON reply against the schema; covers both bad JSON and wrong shape."""
    try:
        return schema.model_validate_json(_strip_code_fence(text))
    except ValidationError as e:
        raise LLMOutputError(
            f"reply does not match {schema.__name__}: {text[:ERROR_PREVIEW_CHARS]!r}"
        ) from e


def _check_reply(reply: ApiReply, model: str, schema: type[BaseModel] | None) -> BaseModel | None:
    """Reject replies that must never be cached or scored; return the parsed object if any."""
    if not reply.text.strip():
        # e.g. a thinking model spent its whole output on reasoning
        raise LLMOutputError(f"{model} returned an empty reply")
    if reply.finish_reason != "stop":
        # "length": hit an output limit and cut off mid-way; "content_filter",
        # None, etc.: did not finish normally. Never treat these as answers.
        raise LLMOutputError(f"{model} reply did not finish normally "
                             f"(finish_reason={reply.finish_reason!r})")
    return _parse(reply.text, schema) if schema is not None else None


# ---------------------------------------------------------------- public API

def generate(prompt: str, model: str, temperature: float,
             schema: type[BaseModel] | None = None, *,
             config: dict | None = None, model_digest: str | None = None) -> LLMResult:
    """Send one user message to the model and return its reply.

    Checks the cache first; on a miss, calls the API with retries, checks the
    reply (non-empty, finish_reason "stop", valid against `schema` if given),
    stores it, and logs the call. Every call leaves exactly one llm_calls row:
    a failed call (hit or miss) is logged with its error, then re-raised.
    `config` defaults to configs/llm.yaml. `model_digest` identifies the model
    build (Step 5 supplies it); it is part of the cache key and stored rows.
    Raises LLMOutputError if the reply is empty, did not finish normally, or
    does not match `schema`.
    """
    start = time.perf_counter()
    if config is None:
        config = load_config()
    key = make_cache_key(prompt, model, temperature, schema, config["base_url"], model_digest)
    stats = CallStats()
    reply = None
    cached = False

    with closing(_connect(config["cache_path"])) as conn:
        try:
            reply = _read_cache(conn, key)
            cached = reply is not None
            if cached:
                # Cache hit: no API call. The stored reply already passed every check.
                parsed = _parse(reply.text, schema) if schema is not None else None
            else:
                # Cache miss: real API call. Check the reply BEFORE writing, so a
                # bad reply raises here and never reaches the cache.
                reply = _call_api(prompt, model, temperature, schema, config, stats)
                parsed = _check_reply(reply, model, schema)
        except Exception as e:
            # Log the failure (one row per call, hit or miss), then re-raise.
            _log_call(conn, key=key, model=model, model_digest=model_digest, cached=cached,
                      attempts=stats.attempts, reply=reply, latency_ms=_elapsed_ms(start),
                      error=f"{type(e).__name__}: {str(e)[:ERROR_PREVIEW_CHARS]}")
            raise

        if not cached:
            _write_cache(conn, key=key, base_url=config["base_url"], model=model,
                         model_digest=model_digest, temperature=temperature,
                         schema_name=schema.__name__ if schema is not None else None,
                         prompt=prompt, reply=reply)
        latency_ms = _elapsed_ms(start)
        _log_call(conn, key=key, model=model, model_digest=model_digest, cached=cached,
                  attempts=stats.attempts, reply=reply, latency_ms=latency_ms, error=None)

    logger.info("model=%s cached=%s attempts=%d prompt_tokens=%s completion_tokens=%s "
                "finish_reason=%s model_latency_ms=%.1f latency_ms=%.1f",
                model, cached, stats.attempts, reply.prompt_tokens, reply.completion_tokens,
                reply.finish_reason, reply.model_latency_ms, latency_ms)
    return LLMResult(text=reply.text, parsed=parsed, model=model,
                     prompt_tokens=reply.prompt_tokens,
                     completion_tokens=reply.completion_tokens,
                     finish_reason=reply.finish_reason, latency_ms=latency_ms,
                     model_latency_ms=reply.model_latency_ms, cached=cached,
                     attempts=stats.attempts, model_digest=model_digest)


# ---------------------------------------------------------------- CLI demo

class DemoClaims(BaseModel):
    """Tiny schema for the --json-demo check (the real judge schemas come in Step 7)."""
    claims: list[str]


def _print_result(label: str, result: LLMResult) -> None:
    """Print one generate() result for the CLI demo."""
    print(f"{label}: cached={result.cached}  attempts={result.attempts}  "
          f"latency_ms={result.latency_ms:.1f}  "
          f"model_latency_ms={result.model_latency_ms:.1f}  "
          f"prompt_tokens={result.prompt_tokens}  completion_tokens={result.completion_tokens}  "
          f"finish_reason={result.finish_reason}")
    shown = result.parsed if result.parsed is not None else result.text
    print(f"  reply: {shown!r}")


def main() -> None:
    """Call a model twice with the same input to show the cache working."""
    parser = argparse.ArgumentParser(description="Call a model twice to show the cache working.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json-demo", action="store_true",
                        help="ask for structured output using a small built-in DemoClaims schema")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    schema = DemoClaims if args.json_demo else None
    first = generate(args.prompt, args.model, args.temperature, schema)
    second = generate(args.prompt, args.model, args.temperature, schema)
    _print_result("Call 1", first)
    _print_result("Call 2", second)
    print(f"Same text on both calls: {first.text == second.text}")
    if first.cached:
        print("Note: call 1 was already in the cache from an earlier run, so both are hits.")


if __name__ == "__main__":
    main()
