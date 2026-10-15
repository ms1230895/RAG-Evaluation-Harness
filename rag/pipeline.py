"""RAG pipeline v0 and THE CONTRACT the evaluation harness calls:

    answer(question: str, config: dict) -> {
        "answer": str,
        "chunks": [{"chunk_id", "doc_id", "text", "score"}],
        "latency_ms": float
    }

Flow: load (or build) the index -> optionally rewrite the question into a
search query -> retrieve chunks (vector, bm25 or hybrid) -> optionally rerank
the pool with a cross-encoder -> fill the prompt template -> generate through
llm/client.py (cached, retried, logged). Every knob comes from the pipeline
config (configs/v0.yaml); in v0 the rewrite and rerank steps are switched off.

latency_ms = retrieval time (embed question + search + fusion + rerank round
trip) + the rewrite call's and the generator's own time
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
from rag.reranker import rerank
from rag.retriever import retrieve_with_method
from rag.rewriter import rewrite_query

logger = logging.getLogger(__name__)

# Sections and keys every pipeline config must define (checked in load_pipeline_config).
REQUIRED_CONFIG_KEYS: dict[str, tuple[str, ...]] = {
    "corpus": ("docs_dir",),
    "chunking": ("chunk_size", "chunk_overlap"),
    "embedding": ("model", "doc_prefix", "query_prefix", "batch_size", "max_input_words"),
    "index": ("dir",),
    "retrieval": ("method", "top_k", "bm25", "hybrid", "rerank"),
    "query_rewrite": ("enabled", "prompt_template"),
    "generation": ("model", "model_digest", "temperature", "seed", "reasoning_effort",
                   "max_prompt_tokens", "context_separator", "context_chunk_template",
                   "prompt_template"),
}

# Keys inside the nested retrieval blocks (checked in load_pipeline_config).
NESTED_KEYS: dict[str, tuple[str, ...]] = {
    "bm25": ("k1", "b"),
    "hybrid": ("fusion", "rrf_k", "pool_k"),
    "rerank": ("enabled", "pool_k", "model", "revision", "worker_timeout_s"),
}

# Retrieval methods this version implements.
SUPPORTED_METHODS = ("vector", "bm25", "hybrid")
SUPPORTED_FUSIONS = ("rrf",)


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


def _is_number(value) -> bool:
    """True for an int or float that is not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_retrieval_blocks(config: dict, fail) -> None:
    """Check retrieval.bm25 / hybrid / rerank (the Step 11 blocks); `fail` raises ValueError."""
    retrieval = config["retrieval"]
    for block, keys in NESTED_KEYS.items():
        if not isinstance(retrieval[block], dict):
            fail(f"retrieval.{block} must be a mapping")
        for key in keys:
            if key not in retrieval[block]:
                fail(f"missing key 'retrieval.{block}.{key}'")
    bm25, hybrid, rerank_cfg = retrieval["bm25"], retrieval["hybrid"], retrieval["rerank"]
    if not _is_number(bm25["k1"]) or bm25["k1"] <= 0:
        fail("retrieval.bm25.k1 must be a number > 0")
    if not _is_number(bm25["b"]) or not 0 <= bm25["b"] <= 1:
        fail("retrieval.bm25.b must be a number between 0 and 1")
    if hybrid["fusion"] not in SUPPORTED_FUSIONS:
        fail(f"retrieval.hybrid.fusion must be one of {SUPPORTED_FUSIONS}, got {hybrid['fusion']!r}")
    if not _is_int(hybrid["rrf_k"]) or hybrid["rrf_k"] < 1:
        fail("retrieval.hybrid.rrf_k must be an integer >= 1")
    for block, name in ((hybrid, "hybrid"), (rerank_cfg, "rerank")):
        if not _is_int(block["pool_k"]) or block["pool_k"] < retrieval["top_k"]:
            fail(f"retrieval.{name}.pool_k must be an integer >= retrieval.top_k "
                 f"({retrieval['top_k']})")
    if not isinstance(rerank_cfg["enabled"], bool):
        fail("retrieval.rerank.enabled must be true or false")
    if not isinstance(rerank_cfg["model"], str) or not rerank_cfg["model"]:
        fail("retrieval.rerank.model must be a non-empty string")
    if rerank_cfg["revision"] is not None and not isinstance(rerank_cfg["revision"], str):
        fail("retrieval.rerank.revision must be a string or null")
    if not _is_int(rerank_cfg["worker_timeout_s"]) or rerank_cfg["worker_timeout_s"] < 1:
        fail("retrieval.rerank.worker_timeout_s must be an integer >= 1")


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

    _check_retrieval_blocks(config, fail)
    rewrite = config["query_rewrite"]
    if not isinstance(rewrite, dict) or not isinstance(rewrite.get("enabled"), bool):
        fail("query_rewrite.enabled must be true or false")
    rewrite_template = rewrite["prompt_template"]
    if not isinstance(rewrite_template, str) or "{question}" not in rewrite_template:
        fail("query_rewrite.prompt_template must contain {question}")
    try:
        rewrite_template.format(question="")
    except (KeyError, IndexError, ValueError) as e:
        fail(f"query_rewrite.prompt_template has other {{...}} fields or stray braces: {e!r}")
    chunk_template = generation["context_chunk_template"]
    if not isinstance(chunk_template, str) or "{text}" not in chunk_template:
        fail("generation.context_chunk_template must be a string containing {text}")
    try:
        chunk_template.format(rank=1, text="")
    except (KeyError, IndexError, ValueError) as e:
        fail(f"generation.context_chunk_template may only use {{rank}} and {{text}}: {e!r}")

    template = generation["prompt_template"]
    if not isinstance(template, str) or "{context}" not in template or "{question}" not in template:
        fail("generation.prompt_template must contain {context} and {question}")
    try:
        template.format(context="", question="")
    except (KeyError, IndexError, ValueError) as e:
        fail(f"generation.prompt_template has other {{...}} fields or stray braces: {e!r}")
    return config


# ---------------------------------------------------------------- prompt

def build_prompt(question: str, chunks: list[dict], template: str, separator: str,
                 chunk_template: str = "{text}") -> str:
    """Fill the template with the rendered chunks (rank order, joined by separator) and the question.

    Each chunk is rendered by chunk_template, which may use {rank} (1-based) and
    {text}. The default "{text}" gives exactly the v0 prompt. Braces inside
    chunk text or the question are safe: they are inserted as values, not
    parsed as part of a template.
    """
    context = separator.join(chunk_template.format(rank=rank, text=chunk["text"])
                             for rank, chunk in enumerate(chunks, start=1))
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

    retrieval = config["retrieval"]
    extra_ms = 0.0   # model time of the rewrite call, if any
    search_query = question
    if config["query_rewrite"]["enabled"]:
        # Only retrieval sees the rewrite; the generator and reranker get the original question.
        search_query, extra_ms = rewrite_query(question, config)

    start = time.perf_counter()
    top_chunks = retrieve_with_method(search_query, index, chunks, config)
    if retrieval["rerank"]["enabled"]:
        top_chunks = rerank(question, top_chunks, retrieval["top_k"], retrieval["rerank"])
    retrieval_ms = (time.perf_counter() - start) * 1000

    gen = config["generation"]
    prompt = build_prompt(question, top_chunks, gen["prompt_template"], gen["context_separator"],
                          gen["context_chunk_template"])
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
        "latency_ms": retrieval_ms + extra_ms + result.model_latency_ms,
    }


# ---------------------------------------------------------------- run metadata

def describe_system(config: dict) -> dict:
    """Facts about this pipeline for run metadata that config_json alone does not pin down.

    Returns {"index": describe_index(config), "generator_model",
    "generator_model_digest", "embedding_model", "embedding_model_digest"},
    plus "reranker_model" and "reranker_revision" when reranking is enabled.
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
    info = {
        "index": describe_index(config),
        "generator_model": gen["model"],
        "generator_model_digest": live,
        "embedding_model": emb["model"],
        "embedding_model_digest": get_model_digest(emb["model"]),
    }
    rerank_cfg = config["retrieval"]["rerank"]
    if rerank_cfg["enabled"]:   # only recorded when it is part of the run
        info["reranker_model"] = rerank_cfg["model"]
        info["reranker_revision"] = rerank_cfg["revision"]
    return info


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
