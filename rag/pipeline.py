"""RAG pipeline v0 and THE CONTRACT the evaluation harness calls:

    answer(question: str, config: dict) -> {
        "answer": str,
        "chunks": [{"chunk_id", "doc_id", "text", "score"}],
        "latency_ms": float
    }

Flow: load (or build) the index -> retrieve top_k chunks -> fill the prompt
template -> generate through llm/client.py (cached, retried, logged).
Every knob comes from the pipeline config (configs/v0.yaml).

latency_ms = retrieval time (embed question + search) + the model's own time
(LLMResult.model_latency_ms, which a cache hit reports from the original
call). The wall-clock LLMResult.latency_ms is never used: on a cache hit it is
a few ms and would make the pipeline look far faster than it is. Building the
index is one-off setup and is not counted.

describe_system(config) is a second, optional function next to the
contract: it returns facts a run must record that the config alone does not
pin down (index key and corpus meta, model digests), and refuses a run whose
pinned generator digest no longer matches the server. answer() is unchanged.

rag/ knows nothing about evaluation: it never imports the harness package.

Quick check from the command line:
    python -m rag.pipeline --config configs/v0.yaml --question "How many casual leaves do interns get?"
"""

import argparse
import logging
import time
from pathlib import Path

import yaml

from llm.client import generate, get_model_digest
from rag.index import describe_index, get_index
from rag.retriever import retrieve

logger = logging.getLogger(__name__)

# Sections and keys every pipeline config must define (checked in load_pipeline_config).
REQUIRED_CONFIG_KEYS: dict[str, tuple[str, ...]] = {
    "corpus": ("docs_dir",),
    "chunking": ("chunk_size", "chunk_overlap"),
    "embedding": ("model", "doc_prefix", "query_prefix", "batch_size", "max_input_words"),
    "index": ("dir",),
    "retrieval": ("method", "top_k"),
    "generation": ("model", "model_digest", "temperature", "seed", "reasoning_effort",
                   "max_prompt_tokens", "context_separator", "prompt_template"),
}

# Retrieval methods this version implements (bm25 / hybrid come in Step 11).
SUPPORTED_METHODS = ("vector",)


class ContextLimitError(Exception):
    """The generator's prompt_tokens reached generation.max_prompt_tokens, or the
    server did not report prompt_tokens at all, so a silently truncated prompt
    cannot be ruled out. The answer must not be scored."""


class ModelDigestError(Exception):
    """generation.model_digest is null, or differs from the digest the server
    reports for the generator model, so cached answers could belong to another
    model build."""


# ---------------------------------------------------------------- config

def _is_int(value) -> bool:
    """True for a real int (bool is a subclass of int in Python, so exclude it)."""
    return isinstance(value, int) and not isinstance(value, bool)


def load_pipeline_config(path: str | Path) -> dict:
    """Read a pipeline YAML and check every key and value; raise ValueError naming the problem.

    Paths (corpus.docs_dir, index.dir) are kept as written; they are resolved
    against the repo root where they are used, so the config dict stays
    JSON-serialisable and machine-independent (Step 5 stores it with each run).
    """
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    def fail(problem: str):
        """Raise a ValueError that names the config file."""
        raise ValueError(f"{path}: {problem}")

    if not isinstance(config, dict):
        fail("the top level must be a mapping of sections")
    if "name" not in config:
        fail("missing key 'name'")
    for section, keys in REQUIRED_CONFIG_KEYS.items():
        if not isinstance(config.get(section), dict):
            fail(f"missing section '{section}'")
        for key in keys:
            if key not in config[section]:
                fail(f"missing key '{section}.{key}'")

    chunking, retrieval = config["chunking"], config["retrieval"]
    embedding, generation = config["embedding"], config["generation"]
    if not _is_int(chunking["chunk_size"]) or chunking["chunk_size"] < 1:
        fail("chunking.chunk_size must be an integer >= 1")
    if (not _is_int(chunking["chunk_overlap"])
            or not 0 <= chunking["chunk_overlap"] < chunking["chunk_size"]):
        fail("chunking.chunk_overlap must be an integer with 0 <= overlap < chunk_size")
    if not _is_int(embedding["batch_size"]) or embedding["batch_size"] < 1:
        fail("embedding.batch_size must be an integer >= 1")
    if not _is_int(embedding["max_input_words"]) or embedding["max_input_words"] < 1:
        fail("embedding.max_input_words must be an integer >= 1")
    if chunking["chunk_size"] > embedding["max_input_words"]:
        # Ollama silently truncates embedding inputs that are too long, so the
        # tail of an oversized chunk would never be searchable.
        fail(f"chunking.chunk_size ({chunking['chunk_size']}) is above "
             f"embedding.max_input_words ({embedding['max_input_words']})")
    for section, key in (("embedding", "model"), ("embedding", "doc_prefix"),
                         ("embedding", "query_prefix"), ("generation", "model"),
                         ("generation", "context_separator")):
        if not isinstance(config[section][key], str):
            fail(f"{section}.{key} must be a string")
    for key in ("model_digest", "reasoning_effort"):
        if generation[key] is not None and not isinstance(generation[key], str):
            fail(f"generation.{key} must be a string or null")
    if retrieval["method"] not in SUPPORTED_METHODS:
        fail(f"retrieval.method must be one of {SUPPORTED_METHODS}, got {retrieval['method']!r}")
    if not _is_int(retrieval["top_k"]) or retrieval["top_k"] < 1:
        fail("retrieval.top_k must be an integer >= 1")
    if not isinstance(generation["temperature"], (int, float)) or generation["temperature"] < 0:
        fail("generation.temperature must be a number >= 0")
    if generation["seed"] is not None and not _is_int(generation["seed"]):
        fail("generation.seed must be an integer or null")
    if not _is_int(generation["max_prompt_tokens"]) or generation["max_prompt_tokens"] < 1:
        fail("generation.max_prompt_tokens must be an integer >= 1")

    template = generation["prompt_template"]
    if not isinstance(template, str) or "{context}" not in template or "{question}" not in template:
        fail("generation.prompt_template must contain {context} and {question}")
    try:
        template.format(context="", question="")
    except (KeyError, IndexError, ValueError) as e:
        fail(f"generation.prompt_template has other {{...}} fields or stray braces: {e!r}")
    return config


# ---------------------------------------------------------------- prompt

def build_prompt(question: str, chunks: list[dict], template: str, separator: str) -> str:
    """Fill the template with the chunk texts (rank order, joined by separator) and the question.

    Braces inside chunk text or the question are safe: they are inserted as
    values, not parsed as part of the template.
    """
    context = separator.join(chunk["text"] for chunk in chunks)
    return template.format(context=context, question=question)


# ---------------------------------------------------------------- the contract

def answer(question: str, config: dict) -> dict:
    """THE CONTRACT: answer one question with the pipeline described by `config`.

    Returns {"answer": str, "chunks": [{"chunk_id", "doc_id", "text", "score"}],
    "latency_ms": float}. Chunks are in rank order, as they appear in the prompt.
    Raises ValueError on a blank question and ContextLimitError if the prompt
    reached max_prompt_tokens or the server did not report prompt_tokens. LLM errors (LLMOutputError, openai errors) are
    not caught: a failed call must surface, never become a made-up answer.
    """
    if not question.strip():
        raise ValueError("question is empty")

    index, chunks = get_index(config)   # one-off setup; not part of latency_ms

    start = time.perf_counter()
    top_chunks = retrieve(question, index, chunks, config["retrieval"]["top_k"],
                          config["embedding"])
    retrieval_ms = (time.perf_counter() - start) * 1000

    gen = config["generation"]
    prompt = build_prompt(question, top_chunks, gen["prompt_template"], gen["context_separator"])
    result = generate(prompt, gen["model"], gen["temperature"],
                      model_digest=gen["model_digest"], seed=gen["seed"],
                      reasoning_effort=gen["reasoning_effort"])

    # Note: the reply is already cached and logged (error NULL) by now; the
    # caller (Step 5) must record this exception as the item's failure.
    if result.prompt_tokens is None:
        raise ContextLimitError("server reported no prompt_tokens, so truncation cannot be "
                                "ruled out")
    if result.prompt_tokens >= gen["max_prompt_tokens"]:
        raise ContextLimitError(f"prompt_tokens={result.prompt_tokens} reached "
                                f"max_prompt_tokens={gen['max_prompt_tokens']}")

    return {
        "answer": result.text,
        "chunks": top_chunks,
        "latency_ms": retrieval_ms + result.model_latency_ms,
    }


# ---------------------------------------------------------------- run metadata

def describe_system(config: dict) -> dict:
    """Facts about this pipeline for run metadata that config_json alone does not pin down.

    Returns {"index": describe_index(config), "generator_model",
    "generator_model_digest", "embedding_model", "embedding_model_digest"}.
    Raises ModelDigestError if generation.model_digest is null or not equal to
    the digest Ollama reports now (the pinned digest is part of the LLM cache
    key, so a stale one would mix answers from two model builds). The
    embedding digest is recorded only: index_key does not cover it.
    """
    gen, emb = config["generation"], config["embedding"]
    live = get_model_digest(gen["model"])
    if gen["model_digest"] is None or live is None or gen["model_digest"] != live:
        raise ModelDigestError(
            f"generation.model_digest is {gen['model_digest']!r} but the server reports "
            f"{live!r} for {gen['model']}; update the pipeline config if the model was re-pulled")
    return {
        "index": describe_index(config),
        "generator_model": gen["model"],
        "generator_model_digest": live,
        "embedding_model": emb["model"],
        "embedding_model_digest": get_model_digest(emb["model"]),
    }


# ---------------------------------------------------------------- CLI

def main() -> None:
    """Answer one question and print the answer, the chunks with scores, and latency."""
    parser = argparse.ArgumentParser(description="Answer one question with a pipeline config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    # INFO logs show how the index was obtained (rag.index) and the generator's
    # cached flag and token counts (llm.client).
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # The openai SDK's HTTP client (httpx2 in openai 3.x) logs every request at
    # INFO; keep the output readable.
    logging.getLogger("httpx2").setLevel(logging.WARNING)

    config = load_pipeline_config(args.config)
    out = answer(args.question, config)

    print(f"\nQuestion: {args.question}")
    print(f"\nAnswer:\n{out['answer']}\n")
    print("Chunks (rank, chunk_id, doc_id, score, text):")
    for rank, chunk in enumerate(out["chunks"], start=1):
        preview = " ".join(chunk["text"].split())[:80]
        print(f"  {rank}. {chunk['chunk_id']:<36} {chunk['doc_id']:<30} "
              f"{chunk['score']:.3f}  {preview}")
    print(f"\nlatency_ms: {out['latency_ms']:.1f}")


if __name__ == "__main__":
    main()
