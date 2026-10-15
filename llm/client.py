"""The one place every LLM call in this project goes through.

generate(prompt, model, temperature, schema=None) sends one user message to a
model and returns its reply. Around that single call it adds:

- caching: the reply is stored in SQLite under a hash of (model, prompt,
  temperature, schema); an identical request later is answered from the cache,
  so re-runs are free and give exactly the same output,
- retries: transient failures (connection errors, timeouts, rate limits, 5xx)
  are retried with exponential backoff; permanent ones are raised at once,
- structured output: if a pydantic schema is given, the reply is validated
  against it, and a reply that does not validate is never cached,
- logging: every call (cache hit or miss) is written to the llm_calls table
  with its token counts and latency.

The models are reached through Ollama's OpenAI-compatible endpoint with the
openai SDK. Connection settings come from .env, retry and cache settings from
configs/llm.yaml. This module knows nothing about RAG or evaluation, so both
rag/ (the generator) and evalharness/ (the judges) can import it.

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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import openai
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "llm.yaml"

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

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    text: str                  # raw reply text from the model
    parsed: BaseModel | None   # validated pydantic object when a schema is given, else None
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float          # wall-clock time of THIS call (tiny on a cache hit)
    cached: bool               # True if served from the cache


class LLMOutputError(Exception):
    """The model replied, but the reply was empty or did not match the schema."""


# ---------------------------------------------------------------- config

def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read configs/llm.yaml. A relative cache_path is resolved against the repo root."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    cache_path = Path(config["cache_path"])
    if not cache_path.is_absolute():
        cache_path = REPO_ROOT / cache_path
    config["cache_path"] = cache_path
    return config


# ---------------------------------------------------------------- cache key

def make_cache_key(prompt: str, model: str, temperature: float,
                   schema: type[BaseModel] | None) -> str:
    """SHA-256 hex digest of the canonical JSON of everything that changes the output.

    sort_keys=True makes the string identical whatever the dict order, and
    float() makes temperature 0 and 0.0 the same key. The schema is included
    as its JSON Schema, so changing a field in the pydantic model is a miss.
    """
    key_input = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "temperature": float(temperature),
            "schema": schema.model_json_schema() if schema else None,
        },
        sort_keys=True,
    )
    return hashlib.sha256(key_input.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- SQLite

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(cache_path: Path) -> sqlite3.Connection:
    """Open the cache database, creating the file and both tables if needed."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cache_path)
    with conn:
        # One row per distinct request: the stored reply.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key         TEXT PRIMARY KEY,
                model             TEXT,
                temperature       REAL,
                schema_name       TEXT,
                prompt            TEXT,
                response_text     TEXT,
                prompt_tokens     INTEGER,
                completion_tokens INTEGER,
                latency_ms        REAL,
                created_at        TEXT
            )""")
        # One row per generate() call, hit or miss: the token and latency log.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                id                INTEGER PRIMARY KEY,
                called_at         TEXT,
                cache_key         TEXT,
                model             TEXT,
                cached            INTEGER,
                prompt_tokens     INTEGER,
                completion_tokens INTEGER,
                latency_ms        REAL
            )""")
    return conn


def _read_cache(conn: sqlite3.Connection, key: str) -> tuple | None:
    """Return (response_text, prompt_tokens, completion_tokens) or None on a miss."""
    return conn.execute(
        "SELECT response_text, prompt_tokens, completion_tokens FROM llm_cache WHERE cache_key = ?",
        (key,),
    ).fetchone()


def _write_cache(conn, key, model, temperature, schema_name, prompt, text,
                 prompt_tokens, completion_tokens, latency_ms) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key, model, float(temperature), schema_name, prompt, text,
             prompt_tokens, completion_tokens, latency_ms, _now()),
        )


def _log_call(conn, key, model, cached, prompt_tokens, completion_tokens, latency_ms) -> None:
    with conn:
        conn.execute(
            "INSERT INTO llm_calls (called_at, cache_key, model, cached, prompt_tokens,"
            " completion_tokens, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_now(), key, model, int(cached), prompt_tokens, completion_tokens, latency_ms),
        )


# ---------------------------------------------------------------- API call

def _get_client(timeout_s: float) -> openai.OpenAI:
    """Build the openai client from .env. Never prints or logs the key."""
    load_dotenv(REPO_ROOT / ".env")
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    for name, value in (("LLM_BASE_URL", base_url), ("LLM_API_KEY", api_key)):
        if not value:
            raise RuntimeError(f"{name} is not set. Add it to .env (see configs/llm.yaml).")
    # max_retries=0: the SDK has its own built-in retries; we turn them off so
    # tenacity is the only retry loop and the attempt count is what we configured.
    return openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)


def _send_request(client: openai.OpenAI, prompt: str, model: str, temperature: float,
                  schema: type[BaseModel] | None) -> tuple[str, int, int]:
    """Exactly one HTTP call. Returns (text, prompt_tokens, completion_tokens)."""
    request = {
        "model": model,
        # A single user message and no hidden system prompt, so the cache key
        # covers everything we send.
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if schema is not None:
        # Ask the server to constrain the reply to this JSON Schema.
        # We still validate with pydantic afterwards (see _parse).
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }
    response = client.chat.completions.create(**request)

    text = response.choices[0].message.content
    if not text or not text.strip():
        # e.g. a thinking model spent its whole output on reasoning
        raise LLMOutputError(f"{model} returned an empty reply")
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return text, prompt_tokens, completion_tokens


def _log_retry(retry_state) -> None:
    """tenacity hook: called before each backoff wait."""
    error = retry_state.outcome.exception()
    logger.warning("LLM call failed (attempt %d, %s); retrying",
                   retry_state.attempt_number, type(error).__name__)


def _call_api(prompt: str, model: str, temperature: float,
              schema: type[BaseModel] | None, config: dict) -> tuple[str, int, int]:
    """_send_request wrapped in a retry loop built from config.

    Waits between tries: retry_wait_min_s * 2^(attempt-1), capped at
    retry_wait_max_s. With min=2, max=30, max_attempts=5: 2, 4, 8, 16 s.
    reraise=True: after the last try the original error is raised, not a
    tenacity RetryError.
    """
    client = _get_client(config["timeout_s"])
    retrying = Retrying(
        stop=stop_after_attempt(config["max_attempts"]),
        wait=wait_exponential(
            multiplier=config["retry_wait_min_s"],
            min=config["retry_wait_min_s"],
            max=config["retry_wait_max_s"],
        ),
        retry=retry_if_exception_type(TRANSIENT_ERRORS),
        before_sleep=_log_retry,
        reraise=True,
    )
    for attempt in retrying:
        with attempt:
            return _send_request(client, prompt, model, temperature, schema)


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
            f"reply does not match {schema.__name__}: {text[:200]!r}"
        ) from e


# ---------------------------------------------------------------- public API

def generate(prompt: str, model: str, temperature: float,
             schema: type[BaseModel] | None = None) -> LLMResult:
    """Send one user message to the model and return its reply.

    Checks the cache first; on a miss, calls the API with retries, validates
    against `schema` if given, stores the reply, and logs the call.
    Raises LLMOutputError if the reply is empty or does not match `schema`.
    """
    start = time.perf_counter()
    config = load_config()
    key = make_cache_key(prompt, model, temperature, schema)
    conn = _connect(config["cache_path"])
    try:
        row = _read_cache(conn, key)
        if row is not None:
            # Cache hit: no API call. Latency is the time of this lookup.
            text, prompt_tokens, completion_tokens = row
            parsed = _parse(text, schema) if schema else None
            cached = True
        else:
            # Cache miss: real API call. Validate BEFORE writing, so a bad
            # reply raises here and never reaches the cache.
            text, prompt_tokens, completion_tokens = _call_api(
                prompt, model, temperature, schema, config)
            parsed = _parse(text, schema) if schema else None
            cached = False

        latency_ms = (time.perf_counter() - start) * 1000
        if not cached:
            _write_cache(conn, key, model, temperature,
                         schema.__name__ if schema else None, prompt, text,
                         prompt_tokens, completion_tokens, latency_ms)
        _log_call(conn, key, model, cached, prompt_tokens, completion_tokens, latency_ms)
    finally:
        conn.close()

    logger.info("model=%s cached=%s prompt_tokens=%d completion_tokens=%d latency_ms=%.1f",
                model, cached, prompt_tokens, completion_tokens, latency_ms)
    return LLMResult(text=text, parsed=parsed, model=model, prompt_tokens=prompt_tokens,
                     completion_tokens=completion_tokens, latency_ms=latency_ms, cached=cached)


# ---------------------------------------------------------------- CLI demo

class Claims(BaseModel):
    """Tiny schema for the --json-demo check (the real judge schemas come in Step 7)."""
    claims: list[str]


def _print_result(label: str, result: LLMResult) -> None:
    print(f"{label}: cached={result.cached}  latency_ms={result.latency_ms:.1f}  "
          f"prompt_tokens={result.prompt_tokens}  completion_tokens={result.completion_tokens}")
    shown = result.parsed if result.parsed is not None else result.text
    print(f"  reply: {shown!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call a model twice to show the cache working.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json-demo", action="store_true",
                        help="ask for structured output using a small built-in Claims schema")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    schema = Claims if args.json_demo else None
    first = generate(args.prompt, args.model, args.temperature, schema)
    second = generate(args.prompt, args.model, args.temperature, schema)
    _print_result("Call 1", first)
    _print_result("Call 2", second)
    print(f"Same text on both calls: {first.text == second.text}")
    if first.cached:
        print("Note: call 1 was already in the cache from an earlier run, so both are hits.")
