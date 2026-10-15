# tests/test_03_rag_pipeline_v0.py
# Step 03: RAG pipeline v0 (chunker, embedder, index, retriever, prompt, answer(), config).
# Written from .claude/specs/03-rag-pipeline-v0.md. No test reaches Ollama or the network:
# embeddings come from `fake_embed` (conftest), generation from `fake_generate` (below),
# and index files go to tmp_path. The Step 2 additions (seed / reasoning_effort) are in
# tests/test_02_llm_wrapper.py; the rag/ layering guard is test_02's
# test_rag_does_not_import_evalharness (not duplicated here).
import copy
import json
import logging
import math
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import faiss
# httpx2 is not a direct dependency: it arrives through openai==3.31, which
# builds its requests and responses with httpx2 (not httpx).
import httpx2
import numpy as np
import openai
import pytest
import yaml

from llm import client as llm_client
from llm.client import LLMOutputError, LLMResult
from rag import chunker as chunker_module
from rag import embedder as embedder_module
from rag import index as index_module
from rag.chunker import chunk_corpus, chunk_document
from rag.embedder import embed_documents, embed_query
from rag.index import (
    build_index,
    describe_index,
    get_index,
    index_key,
    load_index,
    resolve_path,
    save_index,
)
from rag.loader import load_docs
from rag.pipeline import (
    REQUIRED_CONFIG_KEYS,
    ContextLimitError,
    answer,
    build_prompt,
    load_pipeline_config,
)
from rag.retriever import retrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAKE_REQUEST = httpx2.Request("POST", "http://test")

# 3 docs x 3 chunks each (see the pipeline_config fixture), in doc_id order.
TINY_CHUNK_IDS = [f"{doc}#{n:03d}" for doc in ("it_policy", "leave_policy", "payroll_policy")
                  for n in range(3)]


# ---------------------------------------------------------------- helpers

@pytest.fixture(autouse=True)
def block_real_client(monkeypatch):
    """Safety net: no test may build the real openai client (so none can reach any endpoint)."""
    def refuse(*args, **kwargs):
        """Fail loudly if the real client is requested."""
        raise AssertionError("test tried to build the real openai client")
    monkeypatch.setattr(llm_client, "_get_client", refuse)


def make_doc(doc_id, text):
    """Build a doc dict in the shape rag.loader.load_docs() returns."""
    return {"doc_id": doc_id, "title": doc_id.title(), "text": text}


def make_result(text="The answer.", prompt_tokens=100, model_latency_ms=1234.0, latency_ms=2.0):
    """Build an LLMResult like generate() returns (a normal, finished, uncached reply)."""
    return LLMResult(
        text=text, parsed=None, model="fake-gen:cloud", prompt_tokens=prompt_tokens,
        completion_tokens=5, finish_reason="stop", latency_ms=latency_ms,
        model_latency_ms=model_latency_ms, cached=False, attempts=1, model_digest=None,
    )


class FakeGenerate:
    """Stand-in for rag.pipeline.generate: records every call, returns `result` or raises `error`."""

    def __init__(self):
        """Start with a normal result, no error and no recorded calls."""
        self.result = make_result()
        self.error = None
        self.calls = []

    def __call__(self, prompt, model, temperature, schema=None, *, config=None,
                 model_digest=None, seed=None, reasoning_effort=None):
        """Record the arguments, then raise the queued error or return the result."""
        self.calls.append({
            "prompt": prompt, "model": model, "temperature": temperature, "schema": schema,
            "seed": seed, "reasoning_effort": reasoning_effort,
            "model_digest": model_digest,
        })
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def fake_generate(monkeypatch):
    """Replace rag.pipeline.generate (where it is used) with a FakeGenerate."""
    from rag import pipeline

    fake = FakeGenerate()
    monkeypatch.setattr(pipeline, "generate", fake)
    return fake


def write_pipeline_yaml(tmp_path, config, name="pipeline_test.yaml"):
    """Write a config dict as a pipeline YAML file and return its path."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def flat_required_keys():
    """Flatten REQUIRED_CONFIG_KEYS into (section, key) pairs for parametrize."""
    return [(section, key) for section, keys in REQUIRED_CONFIG_KEYS.items() for key in keys]


def words(text):
    """Split text into words the way the spec counts them (runs of non-space characters)."""
    return text.split()


def rebuild_words(chunks, chunk_overlap):
    """Rebuild a document's word list from its chunks: drop the shared overlap from every chunk after the first."""
    rebuilt = []
    for i, chunk in enumerate(chunks):
        chunk_words = words(chunk["text"])
        rebuilt.extend(chunk_words if i == 0 else chunk_words[chunk_overlap:])
    return rebuilt


def status_error(error_class, code, message="boom"):
    """Build an openai status error (400, 429, ...) the way openai==3.31 needs it."""
    return error_class(message, response=httpx2.Response(code, request=FAKE_REQUEST), body=None)


def connection_error():
    """Build a transient openai.APIConnectionError."""
    return openai.APIConnectionError(request=FAKE_REQUEST)


class FakeEmbeddingsClient:
    """Fake client object: embeddings.create(**kwargs) records kwargs and plays back outcomes.

    Each outcome is an exception (raised) or a response (returned); the last one repeats.
    """

    def __init__(self, outcomes):
        """Store the outcomes and start with no recorded requests."""
        self.outcomes = list(outcomes)
        self.create_kwargs = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        """Record the request, then raise or return the next outcome."""
        self.create_kwargs.append(kwargs)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def embeddings_response(*pairs):
    """Build a fake embeddings response from (index, vector) pairs, in the order given."""
    return SimpleNamespace(
        data=[SimpleNamespace(index=i, embedding=vector) for i, vector in pairs])


def install_fake_embeddings_client(monkeypatch, fake, llm_config):
    """Make llm.client._get_client return `fake`; record the (base_url, timeout_s) it was asked for."""
    asked = []

    def fake_get_client(base_url, timeout_s):
        """Record the client arguments and return the fake client."""
        asked.append((base_url, timeout_s))
        return fake

    monkeypatch.setattr(llm_client, "_get_client", fake_get_client)
    return asked


# ---------------------------------------------------------------- chunker

class TestChunker:
    def test_chunk_worked_example(self):
        """Spec worked example: 10 words, size 4, overlap 1 -> d#000 'a b c d', d#001 'd e f g', d#002 'g h i j'."""
        doc = make_doc("d", "a b c d e f g h i j")
        # step = 4 - 1 = 3; window starts at word 0, 3, 6; the window at 6 reaches the end (6 + 4 >= 10).
        chunks = chunk_document(doc, 4, 1)
        assert [c["chunk_id"] for c in chunks] == ["d#000", "d#001", "d#002"]
        assert [c["text"] for c in chunks] == ["a b c d", "d e f g", "g h i j"]
        assert all(c["doc_id"] == "d" for c in chunks)
        assert all(set(c) == {"chunk_id", "doc_id", "text"} for c in chunks), (
            "Chunk keys must be exactly chunk_id, doc_id, text"
        )

    def test_short_doc_is_one_chunk(self):
        """Spec: a doc with fewer words than chunk_size is one chunk holding the whole text."""
        doc = make_doc("d", "a b c")
        chunks = chunk_document(doc, 5, 2)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "a b c"
        assert chunks[0]["chunk_id"] == "d#000"

    def test_doc_with_exactly_chunk_size_words_is_one_chunk(self):
        """Spec boundary: window 0 already reaches the end (0 + 4 >= 4), so no second chunk."""
        chunks = chunk_document(make_doc("d", "a b c d"), 4, 1)
        assert [c["text"] for c in chunks] == ["a b c d"]

    def test_doc_one_word_over_chunk_size_is_two_chunks(self):
        """Spec boundary: 5 words, size 4, overlap 1 -> window 0 'a b c d' does not reach the end; window 3 'd e' does."""
        # step 3: starts 0 (0 + 4 = 4 < 5, continue), 3 (3 + 4 = 7 >= 5, stop) -> 2 chunks.
        chunks = chunk_document(make_doc("d", "a b c d e"), 4, 1)
        assert [c["text"] for c in chunks] == ["a b c d", "d e"]

    def test_no_overlap_splits_into_disjoint_chunks(self):
        """Spec: overlap 0 -> step = size; 8 words with size 4 give exactly 2 chunks, no empty third."""
        chunks = chunk_document(make_doc("d", "a b c d e f g h"), 4, 0)
        assert [c["text"] for c in chunks] == ["a b c d", "e f g h"]

    def test_overlap_two_gives_step_two(self):
        """Spec: step = size - overlap; 10 words, size 4, overlap 2 -> starts 0, 2, 4, 6."""
        chunks = chunk_document(make_doc("d", "a b c d e f g h i j"), 4, 2)
        assert [c["text"] for c in chunks] == ["a b c d", "c d e f", "e f g h", "g h i j"]

    def test_chunk_size_one_gives_one_chunk_per_word_with_padded_ids(self):
        """Spec: chunk_id is <doc_id>#<nnn> with three digits; 12 one-word chunks end at d#011."""
        text = " ".join(f"w{i}" for i in range(12))
        chunks = chunk_document(make_doc("d", text), 1, 0)
        assert [c["chunk_id"] for c in chunks] == [f"d#{i:03d}" for i in range(12)]
        assert chunks[10]["chunk_id"] == "d#010"
        assert chunks[11]["text"] == "w11"

    def test_overlap_words_are_shared(self):
        """Spec: the last chunk_overlap words of chunk i are the first words of chunk i+1."""
        text = " ".join(f"w{i}" for i in range(1, 21))   # w1..w20
        chunks = chunk_document(make_doc("d", text), 6, 2)
        # step 4: starts 0, 4, 8, 12, 16 -> 5 chunks (16 + 6 >= 20 stops).
        assert len(chunks) == 5
        for i in range(len(chunks) - 1):
            assert words(chunks[i]["text"])[-2:] == words(chunks[i + 1]["text"])[:2], (
                f"Chunks {i} and {i + 1} must share 2 words"
            )

    def test_every_word_is_covered(self):
        """Spec: the chunks, with the overlap removed, rebuild the document's word list in order."""
        original = [f"w{i}" for i in range(1, 21)]
        chunks = chunk_document(make_doc("d", " ".join(original)), 6, 2)
        assert rebuild_words(chunks, 2) == original

    def test_chunk_keeps_line_breaks(self):
        """Spec: a chunk is cut from the original text between its first and last word, so '\\n' survives."""
        doc = make_doc("d", "alpha beta\ngamma delta epsilon")
        chunks = chunk_document(doc, 4, 0)
        assert chunks[0]["text"] == "alpha beta\ngamma delta"
        assert chunks[1]["text"] == "epsilon"

    def test_chunk_keeps_markdown_list_layout(self):
        """Spec: Markdown lists survive chunking (line breaks and '- ' markers stay)."""
        doc = make_doc("d", "- one two\n- three four\n- five six")
        chunks = chunk_document(doc, 6, 0)
        # words: '-', 'one', 'two', '-', 'three', 'four' | '-', 'five', 'six'
        assert chunks[0]["text"] == "- one two\n- three four"
        assert chunks[1]["text"] == "- five six"

    def test_non_ascii_text_is_chunked_by_words(self):
        """Edge case: non-ASCII words count as words and are kept intact."""
        chunks = chunk_document(make_doc("d", "café naïve 日本語 text"), 2, 0)
        assert [c["text"] for c in chunks] == ["café naïve", "日本語 text"]

    def test_chunking_is_deterministic(self):
        """Spec: chunk_id is stable for a given doc text and chunk settings."""
        doc = make_doc("d", " ".join(f"w{i}" for i in range(50)))
        assert chunk_document(doc, 7, 3) == chunk_document(doc, 7, 3)

    @pytest.mark.parametrize("chunk_size, chunk_overlap", [
        (4, 4),     # overlap == size
        (4, 5),     # overlap > size
        (0, 0),     # size 0
        (-1, 0),    # negative size
        (4, -1),    # negative overlap
    ])
    def test_bad_chunk_settings_raise(self, chunk_size, chunk_overlap):
        """Spec: ValueError unless chunk_size >= 1 and 0 <= chunk_overlap < chunk_size."""
        with pytest.raises(ValueError):
            chunk_document(make_doc("d", "a b c d e"), chunk_size, chunk_overlap)

    def test_chunk_corpus_keeps_doc_order_and_never_crosses_documents(self):
        """Spec: chunk_corpus = chunk_document over every doc in doc order; no chunk mixes two docs."""
        docs = [make_doc("d1", "a b c d e f"), make_doc("d2", "g h i j k l")]
        chunks = chunk_corpus(docs, 4, 1)
        # d1: step 3, starts 0 and 3 -> 'a b c d', 'd e f'. d2: same -> 'g h i j', 'j k l'.
        assert [(c["chunk_id"], c["text"]) for c in chunks] == [
            ("d1#000", "a b c d"), ("d1#001", "d e f"),
            ("d2#000", "g h i j"), ("d2#001", "j k l"),
        ]
        assert chunks == chunk_document(docs[0], 4, 1) + chunk_document(docs[1], 4, 1)

    def test_real_corpus_chunks(self):
        """Spec: on the real corpus (read-only data/docs via load_docs) every chunk has a known doc_id, non-empty text and a unique chunk_id; two runs are identical."""
        docs = load_docs()
        known_ids = {d["doc_id"] for d in docs}
        chunks = chunk_corpus(docs, 150, 30)
        assert chunks, "Expected chunks from the real corpus"
        assert all(c["doc_id"] in known_ids for c in chunks)
        assert all(c["text"].strip() for c in chunks), "No chunk may be empty"
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "chunk_id must be unique"
        assert {c["doc_id"] for c in chunks} == known_ids, "Every doc must produce at least one chunk"
        assert all(len(words(c["text"])) <= 150 for c in chunks), "A chunk may not exceed chunk_size words"
        assert chunks == chunk_corpus(docs, 150, 30), "Chunking must be deterministic"

    def test_real_corpus_every_word_is_covered(self):
        """Spec: on the real corpus (read-only) the chunks of each doc rebuild its word list."""
        for doc in load_docs():
            doc_chunks = chunk_document(doc, 150, 30)
            assert rebuild_words(doc_chunks, 30) == words(doc["text"]), (
                f"{doc['doc_id']}: chunks do not cover the document"
            )


# ---------------------------------------------------------------- embedder

class TestEmbedder:
    def test_prefixes_are_applied(self, fake_embed, pipeline_config):
        """Spec: documents get doc_prefix, the query gets query_prefix; the configured model is used."""
        cfg = pipeline_config["embedding"]
        embed_documents(["alpha", "beta"], cfg)
        embed_query("gamma", cfg)
        assert fake_embed.texts == ["search_document: alpha", "search_document: beta",
                                    "search_query: gamma"]
        assert {call["model"] for call in fake_embed.calls} == {"fake-embed"}

    def test_vectors_are_normalised_float32(self, fake_embed, pipeline_config):
        """Spec: embed_documents returns a float32 (n, dim) array with every row L2-normalised."""
        vectors = embed_documents(["leave leave salary", "laptop"], pipeline_config["embedding"])
        # The fake returns raw counts, so normalising is the code's job.
        # Row 0 raw = leave 2, salary 1, constant 1 -> norm sqrt(4 + 1 + 1) = sqrt(6); leave -> 2 / sqrt(6) = 0.8165.
        assert vectors.dtype == np.float32
        assert vectors.shape == (2, 9), "8 vocabulary words + 1 constant"
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
        assert vectors[0][0] == pytest.approx(2 / math.sqrt(6), abs=1e-5)

    def test_query_vector_is_one_dimensional_and_normalised(self, fake_embed, pipeline_config):
        """Spec: embed_query returns a float32 array of shape (dim,) with L2 norm 1."""
        vector = embed_query("laptop leave", pipeline_config["embedding"])
        # Raw = leave 1, laptop 1, constant 1 -> each entry 1 / sqrt(3) = 0.57735 (the prefix words are not in the vocabulary).
        assert vector.dtype == np.float32
        assert vector.shape == (9,)
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)
        assert vector[0] == pytest.approx(1 / math.sqrt(3), abs=1e-5)
        assert vector[3] == pytest.approx(1 / math.sqrt(3), abs=1e-5)

    def test_server_vectors_are_normalised_by_our_code(self, monkeypatch, pipeline_config):
        """Spec: normalise in our code; (3, 4) becomes (0.6, 0.8)."""
        monkeypatch.setattr(embedder_module, "_embed_texts",
                            lambda texts, model: [[3.0, 4.0] for _ in texts])
        # |(3, 4)| = 5 -> (0.6, 0.8)
        vectors = embed_documents(["x"], pipeline_config["embedding"])
        assert vectors[0].tolist() == pytest.approx([0.6, 0.8], abs=1e-6)

    def test_batches(self, monkeypatch, pipeline_config):
        """Spec: 5 texts with batch_size=2 -> 3 calls (2, 2, 1 texts), output in input order."""
        batch_sizes = []

        def fake_embed_texts(texts, model):
            """Return [n, 1] for the text 'search_document: n' and record the batch size."""
            batch_sizes.append(len(texts))
            return [[float(t.split(": ")[1]), 1.0] for t in texts]

        monkeypatch.setattr(embedder_module, "_embed_texts", fake_embed_texts)
        cfg = dict(pipeline_config["embedding"], batch_size=2)
        vectors = embed_documents(["1", "2", "3", "4", "5"], cfg)
        assert batch_sizes == [2, 2, 1], "Expected 3 requests of 2, 2 and 1 texts"
        assert vectors.shape == (5, 2)
        for i, n in enumerate([1, 2, 3, 4, 5]):
            # row i = [n, 1] / sqrt(n^2 + 1)
            expected = [n / math.sqrt(n * n + 1), 1 / math.sqrt(n * n + 1)]
            assert vectors[i].tolist() == pytest.approx(expected, abs=1e-5), (
                f"Row {i} is out of order or wrongly normalised"
            )

    def test_batch_size_larger_than_input_is_one_call(self, fake_embed, pipeline_config):
        """Edge case: batch_size 4 with 3 texts -> a single embeddings call."""
        embed_documents(["a", "b", "c"], pipeline_config["embedding"])
        assert len(fake_embed.calls) == 1

    def test_zero_vector_raises(self, monkeypatch, pipeline_config):
        """Spec: both functions raise ValueError if a returned vector has zero length."""
        monkeypatch.setattr(embedder_module, "_embed_texts",
                            lambda texts, model: [[0.0, 0.0] for _ in texts])
        with pytest.raises(ValueError):
            embed_documents(["x"], pipeline_config["embedding"])
        with pytest.raises(ValueError):
            embed_query("x", pipeline_config["embedding"])


# ---------------------------------------------------------------- llm.client.embed

class TestClientEmbed:
    def test_embed_builds_request_and_orders_by_index(self, monkeypatch, llm_config):
        """Spec: one embeddings request with the model and the texts; the result follows the response `index`, not its order."""
        fake = FakeEmbeddingsClient([embeddings_response(
            (2, [2.0, 2.5]), (0, [0.0, 0.5]), (1, [1.0, 1.5]))])
        asked = install_fake_embeddings_client(monkeypatch, fake, llm_config)

        result = llm_client.embed(["a", "b", "c"], "nomic-embed-text", config=llm_config)

        assert len(fake.create_kwargs) == 1, "Exactly one embeddings request"
        assert fake.create_kwargs[0]["model"] == "nomic-embed-text"
        assert fake.create_kwargs[0]["input"] == ["a", "b", "c"]
        assert result == [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]]
        assert asked == [(llm_config["base_url"], llm_config["timeout_s"])], (
            "The client must be built from the llm config"
        )

    def test_embed_retries_transient_errors(self, monkeypatch, llm_config):
        """Spec: connection error twice, then success -> 3 requests and the vectors are returned."""
        ok = embeddings_response((0, [1.0, 2.0]))
        fake = FakeEmbeddingsClient([connection_error(), connection_error(), ok])
        install_fake_embeddings_client(monkeypatch, fake, llm_config)
        result = llm_client.embed(["a"], "nomic-embed-text", config=llm_config)
        assert len(fake.create_kwargs) == 3, "Expected 2 failures plus 1 success"
        assert result == [[1.0, 2.0]]

    def test_embed_gives_up_after_max_attempts(self, monkeypatch, llm_config):
        """Spec: embed shares generate()'s retry policy; an always-failing call stops after max_attempts and re-raises the original error."""
        llm_config["max_attempts"] = 3
        fake = FakeEmbeddingsClient([connection_error()])
        install_fake_embeddings_client(monkeypatch, fake, llm_config)
        with pytest.raises(openai.APIConnectionError):
            llm_client.embed(["a"], "nomic-embed-text", config=llm_config)
        assert len(fake.create_kwargs) == 3

    def test_embed_does_not_retry_permanent_errors(self, monkeypatch, llm_config):
        """Spec: only transient errors are retried; a 400 propagates after one request."""
        fake = FakeEmbeddingsClient([status_error(openai.BadRequestError, 400)])
        install_fake_embeddings_client(monkeypatch, fake, llm_config)
        with pytest.raises(openai.BadRequestError):
            llm_client.embed(["a"], "nomic-embed-text", config=llm_config)
        assert len(fake.create_kwargs) == 1

    def test_embed_is_not_cached_or_logged(self, monkeypatch, llm_config):
        """Spec (Decision 5): embed() is not cached and not logged to llm_calls."""
        fake = FakeEmbeddingsClient([embeddings_response((0, [1.0, 2.0]))])
        install_fake_embeddings_client(monkeypatch, fake, llm_config)
        llm_client.embed(["a"], "nomic-embed-text", config=llm_config)
        llm_client.embed(["a"], "nomic-embed-text", config=llm_config)
        assert len(fake.create_kwargs) == 2, "An identical second call must reach the API again"
        cache_path = llm_config["cache_path"]
        if cache_path.exists():
            with closing(sqlite3.connect(cache_path)) as conn:
                has_table = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?",
                    ("llm_calls",)).fetchone()[0]
                if has_table:
                    rows = conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
                    assert rows == 0, "embed() must not write llm_calls rows"


# ---------------------------------------------------------------- index

HAND_VECTORS = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype="float32")
HAND_CHUNKS = [{"chunk_id": f"d#{i:03d}", "doc_id": "d", "text": f"text {i}"} for i in range(3)]


class TestIndex:
    def test_build_index_is_exact_inner_product(self):
        """Spec: build_index returns a flat inner-product index holding every vector."""
        index = build_index(HAND_VECTORS)
        assert isinstance(index, faiss.IndexFlatIP)
        assert index.ntotal == 3

    def test_search_returns_expected_order_and_cosine(self):
        """Spec: vectors (1,0), (0.6,0.8), (0,1) and query (1,0) -> order 0, 1, 2 with scores 1.0, 0.6, 0.0."""
        index = build_index(HAND_VECTORS)
        scores, ids = index.search(np.array([[1.0, 0.0]], dtype="float32"), 3)
        # dot products: (1,0).(1,0) = 1.0; (1,0).(0.6,0.8) = 0.6; (1,0).(0,1) = 0.0
        assert ids[0].tolist() == [0, 1, 2]
        assert scores[0].tolist() == pytest.approx([1.0, 0.6, 0.0], abs=1e-6)

    def test_retrieve_returns_hand_computed_order_and_scores(self, monkeypatch, pipeline_config):
        """Spec: retrieve returns chunks in rank order with the cosine as `score`."""
        monkeypatch.setattr(embedder_module, "_embed_texts", lambda texts, model: [[1.0, 0.0]])
        index = build_index(HAND_VECTORS)
        results = retrieve("any question", index, HAND_CHUNKS, 3, pipeline_config["embedding"])
        # Same dot products as above: 1.0, 0.6, 0.0
        assert [r["chunk_id"] for r in results] == ["d#000", "d#001", "d#002"]
        assert [r["score"] for r in results] == pytest.approx([1.0, 0.6, 0.0], abs=1e-6)
        top_two = retrieve("any question", index, HAND_CHUNKS, 2, pipeline_config["embedding"])
        assert [r["chunk_id"] for r in top_two] == ["d#000", "d#001"]

    def test_retrieve_ranks_a_middle_vector_first_when_query_points_at_it(self, monkeypatch,
                                                                         pipeline_config):
        """Spec: highest score first, whatever the row order; query (0,1) -> rows 2, 1, 0."""
        monkeypatch.setattr(embedder_module, "_embed_texts", lambda texts, model: [[0.0, 1.0]])
        index = build_index(HAND_VECTORS)
        results = retrieve("q", index, HAND_CHUNKS, 3, pipeline_config["embedding"])
        # dot products with (0,1): row 0 -> 0.0; row 1 -> 0.8; row 2 -> 1.0
        assert [r["chunk_id"] for r in results] == ["d#002", "d#001", "d#000"]
        assert [r["score"] for r in results] == pytest.approx([1.0, 0.8, 0.0], abs=1e-6)

    def test_save_load_roundtrip(self, tmp_path):
        """Spec: after save_index + load_index the search results and the chunks are the same."""
        index = build_index(HAND_VECTORS)
        save_index(index, HAND_CHUNKS, tmp_path / "idx", "abc123", {"chunk_size": 4})
        loaded = load_index(tmp_path / "idx", "abc123")
        assert loaded is not None
        loaded_index, loaded_chunks = loaded
        assert loaded_chunks == HAND_CHUNKS
        assert loaded_index.ntotal == 3
        query = np.array([[1.0, 0.0]], dtype="float32")
        before = index.search(query, 3)
        after = loaded_index.search(query, 3)
        assert after[1].tolist() == before[1].tolist()
        # search() returns (scores, ids), each of shape (1, k): compare the single row.
        assert after[0][0].tolist() == pytest.approx(before[0][0].tolist(), abs=1e-6)

    def test_save_writes_the_file_pair_named_after_the_key(self, tmp_path):
        """Spec: files are indexes/<index_key>.faiss and indexes/<index_key>.chunks.json."""
        save_index(build_index(HAND_VECTORS), HAND_CHUNKS, tmp_path / "idx", "abc123", {})
        names = sorted(p.name for p in (tmp_path / "idx").iterdir())
        assert names == ["abc123.chunks.json", "abc123.faiss"]

    def test_load_missing_returns_none(self, tmp_path):
        """Spec: load_index returns None if the files do not exist."""
        (tmp_path / "idx").mkdir()
        assert load_index(tmp_path / "idx", "nokey") is None

    def test_load_with_one_file_missing_returns_none(self, tmp_path):
        """Spec: None if EITHER file is missing."""
        save_index(build_index(HAND_VECTORS), HAND_CHUNKS, tmp_path / "idx", "abc123", {})
        (tmp_path / "idx" / "abc123.chunks.json").unlink()
        assert load_index(tmp_path / "idx", "abc123") is None
        save_index(build_index(HAND_VECTORS), HAND_CHUNKS, tmp_path / "idx", "abc123", {})
        (tmp_path / "idx" / "abc123.faiss").unlink()
        assert load_index(tmp_path / "idx", "abc123") is None

    def test_load_with_mismatched_counts_raises(self, tmp_path):
        """Spec: ValueError if index.ntotal != len(chunks)."""
        save_index(build_index(HAND_VECTORS), HAND_CHUNKS[:2], tmp_path / "idx", "abc123", {})
        with pytest.raises(ValueError):
            load_index(tmp_path / "idx", "abc123")

    def test_resolve_path_relative_goes_under_repo_root(self):
        """Spec: relative config paths are resolved against the repo root."""
        assert resolve_path("indexes") == PROJECT_ROOT / "indexes"
        assert resolve_path(Path("data") / "docs") == PROJECT_ROOT / "data" / "docs"

    def test_resolve_path_absolute_is_kept(self, tmp_path):
        """Spec: only relative paths are resolved; an absolute path is returned unchanged."""
        assert resolve_path(tmp_path / "idx") == tmp_path / "idx"
        assert resolve_path(str(tmp_path / "idx")) == tmp_path / "idx"


class TestIndexKey:
    DOCS = [make_doc("a", "alpha text"), make_doc("b", "beta text")]

    def test_index_key_is_stable_hex_sha256(self, pipeline_config):
        """Spec: index_key is a SHA-256 hex digest, the same for the same inputs."""
        first = index_key(self.DOCS, pipeline_config)
        assert re.fullmatch(r"[0-9a-f]{64}", first)
        assert first == index_key(copy.deepcopy(self.DOCS), copy.deepcopy(pipeline_config))

    @pytest.mark.parametrize("section, key, new_value", [
        ("chunking", "chunk_size", 9),
        ("chunking", "chunk_overlap", 3),
        ("embedding", "model", "other-embed"),
        ("embedding", "doc_prefix", "passage: "),
    ])
    def test_index_key_changes_with_vector_inputs(self, pipeline_config, section, key, new_value):
        """Spec: the key changes with chunk_size, chunk_overlap, embedding model and doc_prefix."""
        changed = copy.deepcopy(pipeline_config)
        changed[section][key] = new_value
        assert index_key(self.DOCS, changed) != index_key(self.DOCS, pipeline_config), (
            f"{section}.{key} must change the index key"
        )

    def test_index_key_changes_with_doc_text(self, pipeline_config):
        """Spec: the key covers each document's text."""
        changed = [make_doc("a", "alpha text edited"), make_doc("b", "beta text")]
        assert index_key(changed, pipeline_config) != index_key(self.DOCS, pipeline_config)

    def test_index_key_changes_with_doc_id(self, pipeline_config):
        """Spec: the key covers each document's ID."""
        changed = [make_doc("a2", "alpha text"), make_doc("b", "beta text")]
        assert index_key(changed, pipeline_config) != index_key(self.DOCS, pipeline_config)

    def test_index_key_changes_with_a_new_document(self, pipeline_config):
        """Spec: adding a document changes the key."""
        more = self.DOCS + [make_doc("c", "gamma text")]
        assert index_key(more, pipeline_config) != index_key(self.DOCS, pipeline_config)

    @pytest.mark.parametrize("section, key, new_value", [
        ("retrieval", "top_k", 7),
        ("generation", "prompt_template", "Q: {question}\nC: {context}"),
        ("generation", "temperature", 0.9),
        ("generation", "model", "other-gen:cloud"),
        ("generation", "seed", 1),
        ("embedding", "query_prefix", "question: "),
        ("embedding", "batch_size", 1),
    ])
    def test_index_key_ignores_settings_that_do_not_change_vectors(
            self, pipeline_config, section, key, new_value):
        """Spec: the key does NOT change with top_k, prompt_template, temperature or query_prefix (nor other non-vector settings)."""
        changed = copy.deepcopy(pipeline_config)
        changed[section][key] = new_value
        assert index_key(self.DOCS, changed) == index_key(self.DOCS, pipeline_config), (
            f"{section}.{key} must NOT change the index key"
        )

    def test_index_key_ignores_index_dir_and_docs_dir_paths(self, pipeline_config):
        """Spec: the key is built from doc IDs and text, not from where the files live."""
        changed = copy.deepcopy(pipeline_config)
        changed["index"]["dir"] = "elsewhere"
        changed["corpus"]["docs_dir"] = "elsewhere_docs"
        assert index_key(self.DOCS, changed) == index_key(self.DOCS, pipeline_config)


class TestGetIndex:
    def index_files(self, pipeline_config, pattern):
        """List the files matching `pattern` in the config's index dir."""
        index_dir = Path(pipeline_config["index"]["dir"])
        return sorted(p.name for p in index_dir.glob(pattern))

    def test_get_index_builds_once_then_reuses_memory_then_loads_disk(
            self, fake_embed, pipeline_config):
        """Spec: 2nd call reuses memory (no embedding); after clearing memory the index loads from
        disk without re-embedding any chunk, plus exactly one warm-up query (code review item 4)."""
        index1, chunks1 = get_index(pipeline_config)
        # 3 docs x 20 words, size 8, overlap 2 -> 3 chunks per doc -> 9 chunks, 9 texts embedded.
        assert len(fake_embed.texts) == 9, "First call must embed every chunk once"
        assert all(t.startswith("search_document: ") for t in fake_embed.texts)

        index2, chunks2 = get_index(pipeline_config)
        assert len(fake_embed.texts) == 9, "Second call must come from memory"
        assert chunks2 == chunks1

        index_module._LOADED.clear()
        index3, chunks3 = get_index(pipeline_config)
        assert fake_embed.texts[9:] == ["search_query: " + index_module.WARM_UP_TEXT], (
            "A disk load must not re-embed chunks; it embeds exactly one warm-up query")
        assert chunks3 == chunks1
        assert index3.ntotal == 9

    def test_get_index_returns_chunks_in_row_order(self, fake_embed, pipeline_config):
        """Spec: row i of the index is chunk i; chunks are in doc_id order with ids <doc_id>#<nnn>."""
        index, chunks = get_index(pipeline_config)
        assert [c["chunk_id"] for c in chunks] == TINY_CHUNK_IDS
        assert index.ntotal == len(chunks) == 9
        assert all(set(c) == {"chunk_id", "doc_id", "text"} for c in chunks)

    def test_get_index_writes_exactly_one_file_pair_named_after_the_key(
            self, fake_embed, pipeline_config):
        """Spec: one .faiss and one .chunks.json per index, named <index_key>."""
        get_index(pipeline_config)
        key = index_key(load_docs(Path(pipeline_config["corpus"]["docs_dir"])), pipeline_config)
        assert self.index_files(pipeline_config, "*.faiss") == [f"{key}.faiss"]
        assert self.index_files(pipeline_config, "*.chunks.json") == [f"{key}.chunks.json"]

    def test_get_index_chunks_json_has_meta(self, fake_embed, pipeline_config):
        """Spec: the .chunks.json file holds the chunk list and a 'meta' object."""
        get_index(pipeline_config)
        path = Path(pipeline_config["index"]["dir"])
        (chunks_file,) = list(path.glob("*.chunks.json"))
        content = json.loads(chunks_file.read_text(encoding="utf-8"))
        assert "meta" in content, "chunks.json must contain a 'meta' object"

    def test_changing_top_k_reuses_the_same_index(self, fake_embed, pipeline_config):
        """Spec: changing top_k reuses the same index (no chunk re-embedded, no new files;
        the disk load embeds only the one warm-up query)."""
        get_index(pipeline_config)
        index_module._LOADED.clear()
        changed = copy.deepcopy(pipeline_config)
        changed["retrieval"]["top_k"] = 2
        get_index(changed)
        assert fake_embed.texts[9:] == ["search_query: " + index_module.WARM_UP_TEXT]
        assert len(self.index_files(pipeline_config, "*.faiss")) == 1

    def test_changing_chunk_size_builds_a_new_index(self, fake_embed, pipeline_config):
        """Spec: changing chunk size builds a new index and keeps the old one."""
        get_index(pipeline_config)
        changed = copy.deepcopy(pipeline_config)
        changed["chunking"]["chunk_size"] = 10
        get_index(changed)
        assert len(fake_embed.texts) > 9, "New chunk settings must embed again"
        assert len(self.index_files(pipeline_config, "*.faiss")) == 2
        assert len(self.index_files(pipeline_config, "*.chunks.json")) == 2

    def test_edited_document_is_never_served_from_a_stale_index(self, fake_embed, pipeline_config):
        """Spec: a stale index must never be loaded; edited doc text -> new key -> rebuild."""
        get_index(pipeline_config)
        doc_path = Path(pipeline_config["corpus"]["docs_dir"]) / "leave_policy.md"
        doc_path.write_text(doc_path.read_text(encoding="utf-8") + "extra words appended\n",
                            encoding="utf-8")
        index_module._LOADED.clear()
        get_index(pipeline_config)
        assert len(fake_embed.texts) > 9, "Edited document must be embedded again"
        assert len(self.index_files(pipeline_config, "*.faiss")) == 2

    def test_get_index_logs_how_the_index_was_obtained(self, fake_embed, pipeline_config, caplog):
        """Spec: logs 'index <key[:12]> built | loaded from disk | reused from memory: N chunks'."""
        key = index_key(load_docs(Path(pipeline_config["corpus"]["docs_dir"])), pipeline_config)
        with caplog.at_level(logging.INFO, logger="rag.index"):
            get_index(pipeline_config)
            get_index(pipeline_config)
            index_module._LOADED.clear()
            get_index(pipeline_config)
        messages = [r.getMessage() for r in caplog.records if r.name == "rag.index"]
        assert len(messages) == 3, f"Expected one log line per get_index call, got {messages}"
        for message, how in zip(messages, ["built", "reused from memory", "loaded from disk"]):
            assert key[:12] in message
            assert how in message
            assert "9 chunks" in message


class TestIndexKeyReview:
    """Code review changes (2026-10-15): CHUNKER_VERSION and the embedding endpoint are in the key."""

    DOCS = [make_doc("a", "alpha text"), make_doc("b", "beta text")]

    def test_index_key_changes_with_chunker_version(self, monkeypatch, pipeline_config):
        """Spec index_key: CHUNKER_VERSION (rag/chunker.py) is part of the key."""
        before = index_key(self.DOCS, pipeline_config)
        # The key code may read the version from rag.chunker or from its own imported name; patch both.
        monkeypatch.setattr(chunker_module, "CHUNKER_VERSION", "test-version-99")
        monkeypatch.setattr(index_module, "CHUNKER_VERSION", "test-version-99")
        after = index_key(self.DOCS, pipeline_config)
        assert after != before, "A new CHUNKER_VERSION must give a new index key"

    def test_index_key_changes_with_embedding_base_url(self, monkeypatch, pipeline_config):
        """Spec index_key: the embedding endpoint (base_url from configs/llm.yaml) is part of the key."""
        monkeypatch.setattr(index_module, "load_llm_config",
                            lambda *args, **kwargs: {"base_url": "http://endpoint-one.test/v1"})
        first = index_key(self.DOCS, pipeline_config)
        monkeypatch.setattr(index_module, "load_llm_config",
                            lambda *args, **kwargs: {"base_url": "http://endpoint-two.test/v1"})
        second = index_key(self.DOCS, pipeline_config)
        assert first != second, "A different embedding base_url must give a different key"

    def test_index_key_same_base_url_gives_same_key(self, monkeypatch, pipeline_config):
        """Spec index_key: the same base_url (and inputs) gives the same key, so the base_url test above is meaningful."""
        monkeypatch.setattr(index_module, "load_llm_config",
                            lambda *args, **kwargs: {"base_url": "http://endpoint-one.test/v1"})
        assert index_key(self.DOCS, pipeline_config) == index_key(self.DOCS, pipeline_config)


class TestGetIndexReview:
    """Code review changes (2026-10-15): empty corpus error, warm-up query, describe_index."""

    def test_get_index_empty_corpus_raises_value_error_naming_docs_dir(
            self, fake_embed, pipeline_config, tmp_path):
        """Spec get_index: ValueError naming docs_dir if the corpus gives no chunks (empty docs dir)."""
        empty_dir = tmp_path / "empty_docs"
        empty_dir.mkdir()
        pipeline_config["corpus"]["docs_dir"] = str(empty_dir)
        with pytest.raises(ValueError) as excinfo:
            get_index(pipeline_config)
        message = str(excinfo.value)
        assert "docs_dir" in message or str(empty_dir) in message, (
            f"Error must name docs_dir, got: {message}")

    def test_get_index_empty_corpus_embeds_nothing_and_writes_no_files(
            self, fake_embed, pipeline_config, tmp_path):
        """Spec get_index: a corpus with no chunks fails before any embedding and leaves no index files."""
        empty_dir = tmp_path / "empty_docs"
        empty_dir.mkdir()
        pipeline_config["corpus"]["docs_dir"] = str(empty_dir)
        with pytest.raises(ValueError):
            get_index(pipeline_config)
        assert fake_embed.calls == [], "No embedding call for an empty corpus"
        index_dir = Path(pipeline_config["index"]["dir"])
        assert not index_dir.exists() or list(index_dir.iterdir()) == [], (
            "No index files may be written for an empty corpus")

    def test_get_index_no_warm_up_after_a_build(self, fake_embed, pipeline_config):
        """Spec get_index: after a build the model is already warm, so no warm-up query is embedded."""
        get_index(pipeline_config)
        assert len(fake_embed.texts) == 9, "Build embeds exactly the 9 chunks"
        assert not any(t.startswith("search_query: ") for t in fake_embed.texts), (
            "No query (warm-up) embedding after a build")

    def test_get_index_no_warm_up_after_a_memory_reuse(self, fake_embed, pipeline_config):
        """Spec get_index: a reuse from memory embeds nothing (no warm-up)."""
        get_index(pipeline_config)
        calls_after_build = len(fake_embed.calls)
        get_index(pipeline_config)
        assert len(fake_embed.calls) == calls_after_build, "Memory reuse must not embed anything"

    def test_get_index_disk_load_warms_up_with_exactly_one_prefixed_query(
            self, fake_embed, pipeline_config):
        """Spec get_index: after a disk load, exactly one embedding call holding the single text query_prefix + WARM_UP_TEXT, with the embedding model."""
        get_index(pipeline_config)
        calls_after_build = len(fake_embed.calls)
        index_module._LOADED.clear()
        get_index(pipeline_config)
        warm_up_calls = fake_embed.calls[calls_after_build:]
        assert len(warm_up_calls) == 1, "Exactly one warm-up embedding request"
        assert warm_up_calls[0]["texts"] == ["search_query: " + index_module.WARM_UP_TEXT]
        assert warm_up_calls[0]["model"] == "fake-embed"

    def test_get_index_disk_load_warm_up_uses_the_configured_query_prefix(
            self, fake_embed, pipeline_config):
        """Spec get_index: the warm-up text is WARM_UP_TEXT with the config's query_prefix."""
        pipeline_config["embedding"]["query_prefix"] = "question: "
        get_index(pipeline_config)
        index_module._LOADED.clear()
        get_index(pipeline_config)
        assert fake_embed.texts[9:] == ["question: " + index_module.WARM_UP_TEXT]

    def test_get_index_warm_up_is_not_repeated_on_later_memory_reuse(
            self, fake_embed, pipeline_config):
        """Spec get_index: only the disk load warms up; the reuse after it embeds nothing more."""
        get_index(pipeline_config)
        index_module._LOADED.clear()
        get_index(pipeline_config)        # disk load: +1 warm-up
        texts_after_disk_load = len(fake_embed.texts)
        get_index(pipeline_config)        # memory reuse
        assert len(fake_embed.texts) == texts_after_disk_load

    def test_describe_index_returns_index_key_and_meta(self, fake_embed, pipeline_config):
        """Spec describe_index: returns {"index_key": <key>, **meta} where meta comes from the .chunks.json file."""
        info = describe_index(pipeline_config)
        key = index_key(load_docs(Path(pipeline_config["corpus"]["docs_dir"])), pipeline_config)
        assert info["index_key"] == key
        (chunks_file,) = list(Path(pipeline_config["index"]["dir"]).glob("*.chunks.json"))
        meta = json.loads(chunks_file.read_text(encoding="utf-8"))["meta"]
        assert meta, "Meta must not be empty"
        assert info == {"index_key": key, **meta}, "describe_index must be index_key plus the file's meta"

    def test_describe_index_meta_holds_corpus_and_chunk_counts(self, fake_embed, pipeline_config):
        """Spec Data changes: meta holds the key inputs plus doc count and chunk count (3 docs, 9 chunks here)."""
        info = describe_index(pipeline_config)
        values = list(info.values())
        # 3 docs x 20 words, size 8, overlap 2 -> 9 chunks; 3 docs.
        assert 9 in values, f"Chunk count 9 missing from {info}"
        assert 3 in values, f"Doc count 3 missing from {info}"
        assert "fake-embed" in values, f"Embedding model missing from {info}"
        assert "search_document: " in values, f"doc_prefix missing from {info}"

    def test_describe_index_builds_the_index_when_missing(self, fake_embed, pipeline_config):
        """Spec describe_index: the index is built if needed (files appear, 9 chunks embedded)."""
        index_dir = Path(pipeline_config["index"]["dir"])
        assert not index_dir.exists() or not list(index_dir.glob("*.faiss"))
        describe_index(pipeline_config)
        assert len(list(index_dir.glob("*.faiss"))) == 1
        assert len(list(index_dir.glob("*.chunks.json"))) == 1
        assert len(fake_embed.texts) == 9

    def test_describe_index_does_not_rebuild_an_existing_index(self, fake_embed, pipeline_config):
        """Spec describe_index: an index that already exists is not rebuilt (no chunk re-embedded)."""
        get_index(pipeline_config)
        describe_index(pipeline_config)
        assert len(fake_embed.texts) == 9, "describe_index must reuse the existing index"

    def test_describe_index_key_changes_with_chunk_size(self, fake_embed, pipeline_config):
        """Spec describe_index: it describes THIS config's index, so another chunk_size gives another index_key."""
        first = describe_index(pipeline_config)["index_key"]
        changed = copy.deepcopy(pipeline_config)
        changed["chunking"]["chunk_size"] = 10
        assert describe_index(changed)["index_key"] != first


# ---------------------------------------------------------------- retriever

class TestRetriever:
    def test_retrieve_shape(self, fake_embed, pipeline_config):
        """Spec: top_k results, scores descending, keys exactly chunk_id/doc_id/text/score, score a Python float."""
        index, chunks = get_index(pipeline_config)
        results = retrieve("leave", index, chunks, 3, pipeline_config["embedding"])
        assert len(results) == 3
        for result in results:
            assert set(result) == {"chunk_id", "doc_id", "text", "score"}
            assert type(result["score"]) is float, "score must be a plain Python float"
            assert result["text"].strip()
            assert result["chunk_id"] in TINY_CHUNK_IDS
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Highest score first"

    @pytest.mark.parametrize("top_k", [1, 2, 5, 9])
    def test_retrieve_returns_exactly_top_k(self, fake_embed, pipeline_config, top_k):
        """Spec: retrieve returns top_k chunks (here from a 9-chunk index)."""
        index, chunks = get_index(pipeline_config)
        assert len(retrieve("leave", index, chunks, top_k, pipeline_config["embedding"])) == top_k

    def test_retrieve_embeds_the_question_with_the_query_prefix(self, fake_embed, pipeline_config):
        """Spec: the question is embedded once, with query_prefix."""
        index, chunks = get_index(pipeline_config)
        before = len(fake_embed.texts)
        retrieve("leave", index, chunks, 3, pipeline_config["embedding"])
        assert fake_embed.texts[before:] == ["search_query: leave"]

    def test_retrieve_puts_the_best_matching_chunk_first(self, fake_embed, pipeline_config):
        """Spec: the chunk closest to the question ranks first; hand-computed cosine 5/6 for leave_policy#000."""
        index, chunks = get_index(pipeline_config)
        results = retrieve("leave", index, chunks, 3, pipeline_config["embedding"])
        # Query "search_query: leave" -> raw (leave 1, constant 1), norm sqrt(2).
        # leave_policy#000 = '# Leave Policy leave leave leave interns x0' -> leave 4, policy 1, constant 1,
        # norm sqrt(18). cosine = (4*1 + 1*1) / (sqrt(18) * sqrt(2)) = 5 / 6 = 0.8333.
        # Every other chunk scores at most 1 / sqrt(2) = 0.7071 (filler chunks) or 0.2041 (topic chunks).
        assert results[0]["chunk_id"] == "leave_policy#000"
        assert results[0]["doc_id"] == "leave_policy"
        assert results[0]["score"] == pytest.approx(5 / 6, abs=1e-5)

    def test_retrieve_finds_the_topic_doc_for_each_question(self, fake_embed, pipeline_config):
        """Spec: retrieval is by similarity; a salary question puts a payroll chunk first, a laptop question an IT chunk."""
        index, chunks = get_index(pipeline_config)
        salary = retrieve("salary", index, chunks, 1, pipeline_config["embedding"])
        laptop = retrieve("laptop", index, chunks, 1, pipeline_config["embedding"])
        # Same hand calculation as above: the first chunk of the topic doc has the topic word 3-4 times.
        assert salary[0]["doc_id"] == "payroll_policy"
        assert laptop[0]["doc_id"] == "it_policy"


# ---------------------------------------------------------------- prompt

class TestBuildPrompt:
    TEMPLATE = "Ctx:\n{context}\nQ: {question}\nA:"

    def test_build_prompt_joins_chunks_in_rank_order(self):
        """Spec: template.format(context=separator.join(chunk texts in rank order), question=question)."""
        chunks = [{"text": "AAA"}, {"text": "BBB"}, {"text": "CCC"}]
        prompt = build_prompt("why?", chunks, self.TEMPLATE, " | ")
        assert prompt == "Ctx:\nAAA | BBB | CCC\nQ: why?\nA:"

    def test_build_prompt_keeps_the_given_order(self):
        """Spec: chunks appear in the order given (rank order), not sorted by anything else."""
        forward = build_prompt("q", [{"text": "AAA"}, {"text": "BBB"}], self.TEMPLATE, "|")
        backward = build_prompt("q", [{"text": "BBB"}, {"text": "AAA"}], self.TEMPLATE, "|")
        assert forward.index("AAA") < forward.index("BBB")
        assert backward.index("BBB") < backward.index("AAA")

    def test_build_prompt_contains_question_and_every_chunk_text(self):
        """Spec: the question and all chunk texts appear in the prompt."""
        chunks = [{"chunk_id": "d#000", "doc_id": "d", "text": "first text", "score": 0.9},
                  {"chunk_id": "d#001", "doc_id": "d", "text": "second text", "score": 0.5}]
        prompt = build_prompt("How many days?", chunks, self.TEMPLATE, "\n\n---\n\n")
        assert "How many days?" in prompt
        assert "first text\n\n---\n\nsecond text" in prompt

    def test_build_prompt_does_not_add_ids_or_scores(self):
        """Spec: v0 has no doc IDs, scores or titles in the prompt, only the chunk text."""
        chunks = [{"chunk_id": "doc_x#000", "doc_id": "doc_x", "text": "only this", "score": 0.9}]
        prompt = build_prompt("q", chunks, self.TEMPLATE, "|")
        assert prompt == "Ctx:\nonly this\nQ: q\nA:"

    def test_build_prompt_braces_in_text_do_not_crash(self):
        """Spec: chunk text or a question containing '{' and '}' is inserted verbatim, not parsed as a template."""
        chunks = [{"text": "set {x} and {} and {0}"}]
        prompt = build_prompt("what is {y}?", chunks, self.TEMPLATE, "|")
        assert "set {x} and {} and {0}" in prompt
        assert "what is {y}?" in prompt

    def test_build_prompt_single_chunk_has_no_separator(self):
        """Edge case: one chunk -> the separator never appears."""
        prompt = build_prompt("q", [{"text": "solo"}], self.TEMPLATE, "<SEP>")
        assert "<SEP>" not in prompt

    def test_build_prompt_with_no_chunks_has_empty_context(self):
        """Edge case: no chunks -> context is the empty string (join of nothing)."""
        prompt = build_prompt("q", [], self.TEMPLATE, "|")
        assert prompt == "Ctx:\n\nQ: q\nA:"


# ---------------------------------------------------------------- answer() contract

class TestAnswerContract:
    QUESTION = "How many leave days?"

    def test_answer_contract_shape(self, fake_embed, fake_generate, pipeline_config):
        """Spec: keys exactly answer/chunks/latency_ms; chunk keys exactly the four; types str/list/float; len(chunks) == top_k."""
        out = answer(self.QUESTION, pipeline_config)
        assert set(out) == {"answer", "chunks", "latency_ms"}
        assert isinstance(out["answer"], str)
        assert isinstance(out["chunks"], list)
        assert type(out["latency_ms"]) is float
        assert out["latency_ms"] >= 0
        assert len(out["chunks"]) == pipeline_config["retrieval"]["top_k"] == 3
        for chunk in out["chunks"]:
            assert set(chunk) == {"chunk_id", "doc_id", "text", "score"}
            assert isinstance(chunk["chunk_id"], str)
            assert isinstance(chunk["doc_id"], str)
            assert isinstance(chunk["text"], str)
            assert type(chunk["score"]) is float

    @pytest.mark.parametrize("top_k", [1, 2, 5])
    def test_answer_returns_top_k_chunks_from_config(self, fake_embed, fake_generate,
                                                     pipeline_config, top_k):
        """Spec: top_k comes from the config (retrieval.top_k)."""
        pipeline_config["retrieval"]["top_k"] = top_k
        out = answer(self.QUESTION, pipeline_config)
        assert len(out["chunks"]) == top_k

    def test_answer_text_is_the_generator_reply(self, fake_embed, fake_generate, pipeline_config):
        """Spec: `answer` is the text the generator returned."""
        fake_generate.result = make_result(text="Interns get 12 days.")
        assert answer(self.QUESTION, pipeline_config)["answer"] == "Interns get 12 days."

    def test_answer_chunks_are_sorted_by_score_descending(self, fake_embed, fake_generate,
                                                         pipeline_config):
        """Spec: chunks are returned in rank order, highest score first."""
        out = answer("leave", pipeline_config)
        scores = [c["score"] for c in out["chunks"]]
        assert scores == sorted(scores, reverse=True)
        assert out["chunks"][0]["chunk_id"] == "leave_policy#000"

    def test_answer_prompt_has_the_same_chunks_in_the_same_order(self, fake_embed, fake_generate,
                                                                 pipeline_config):
        """Spec: chunks appear in the prompt in the order they are returned; the generator gets the built prompt."""
        out = answer(self.QUESTION, pipeline_config)
        gen = pipeline_config["generation"]
        expected_prompt = build_prompt(self.QUESTION, out["chunks"], gen["prompt_template"],
                                       gen["context_separator"])
        assert len(fake_generate.calls) == 1, "Exactly one generator call per answer()"
        assert fake_generate.calls[0]["prompt"] == expected_prompt
        assert self.QUESTION in expected_prompt

    def test_answer_passes_generation_settings(self, fake_embed, fake_generate, pipeline_config):
        """Spec: the generator receives model, temperature, seed and reasoning_effort from the config."""
        pipeline_config["generation"].update(
            model="other-gen:cloud", temperature=0.2, seed=7, reasoning_effort="low")
        answer(self.QUESTION, pipeline_config)
        call = fake_generate.calls[0]
        assert call["model"] == "other-gen:cloud"
        assert call["temperature"] == pytest.approx(0.2)
        assert call["seed"] == 7
        assert call["reasoning_effort"] == "low"
        assert call["schema"] is None, "The pipeline asks for plain text, no schema"

    def test_answer_passes_null_reasoning_effort_as_none(self, fake_embed, fake_generate,
                                                         pipeline_config):
        """Spec: reasoning_effort null means not sent -> the generator gets None; seed 42 is passed."""
        answer(self.QUESTION, pipeline_config)
        call = fake_generate.calls[0]
        assert call["reasoning_effort"] is None
        assert call["seed"] == 42
        assert call["model"] == "fake-gen:cloud"
        assert call["temperature"] == pytest.approx(0.0)

    def test_latency_uses_model_latency(self, fake_embed, fake_generate, pipeline_config):
        """Spec: latency_ms = retrieval_ms + model_latency_ms; the wall-clock LLMResult.latency_ms is never used."""
        fake_generate.result = make_result(model_latency_ms=1234.0, latency_ms=999999.0)
        out = answer(self.QUESTION, pipeline_config)
        # retrieval_ms (fake embedding + search) is >= 0 and tiny, so 1234.0 <= latency < 999999.0.
        assert 1234.0 <= out["latency_ms"] < 999999.0

    def test_latency_on_cache_hit_still_uses_original_model_latency(self, fake_embed,
                                                                    fake_generate, pipeline_config):
        """Spec: a cached reply (tiny wall latency) still reports the original model time."""
        fake_generate.result = make_result(model_latency_ms=800.0, latency_ms=2.0)
        out = answer(self.QUESTION, pipeline_config)
        assert out["latency_ms"] >= 800.0, "Wall latency 2 ms must not replace the model latency"

    def test_latency_does_not_include_index_building(self, fake_embed, fake_generate,
                                                    pipeline_config):
        """Spec: index building is one-off setup and is not counted; first and second call both stay near the model latency."""
        fake_generate.result = make_result(model_latency_ms=100.0)
        first = answer(self.QUESTION, pipeline_config)
        second = answer(self.QUESTION, pipeline_config)
        # No sleeping or exact timing: only a coarse bound (retrieval with fakes is far below 10 s).
        assert 100.0 <= first["latency_ms"] < 10_000.0
        assert 100.0 <= second["latency_ms"] < 10_000.0

    def test_answer_is_deterministic_for_the_same_input(self, fake_embed, fake_generate,
                                                        pipeline_config):
        """Spec: same question, config and (fake) generator give the same answer and the same chunks with the same scores."""
        first = answer(self.QUESTION, pipeline_config)
        second = answer(self.QUESTION, pipeline_config)
        assert first["answer"] == second["answer"]
        assert first["chunks"] == second["chunks"]

    def test_answer_builds_the_index_once_for_many_questions(self, fake_embed, fake_generate,
                                                             pipeline_config):
        """Spec: the index is built once; later questions only embed the query."""
        answer("leave", pipeline_config)
        texts_after_first = len(fake_embed.texts)   # 9 chunks + 1 query
        answer("salary", pipeline_config)
        assert texts_after_first == 10
        assert len(fake_embed.texts) == 11, "A second question must embed only the question"

    def test_llm_output_error_propagates(self, fake_embed, fake_generate, pipeline_config):
        """Spec: answer() never swallows an LLM error to return a fallback answer."""
        fake_generate.error = LLMOutputError("empty reply")
        with pytest.raises(LLMOutputError, match="empty reply"):
            answer(self.QUESTION, pipeline_config)

    def test_openai_error_propagates(self, fake_embed, fake_generate, pipeline_config):
        """Spec: openai errors from the generator propagate too."""
        fake_generate.error = connection_error()
        with pytest.raises(openai.APIConnectionError):
            answer(self.QUESTION, pipeline_config)

    @pytest.mark.parametrize("question", ["", "   ", "\n\t "])
    def test_empty_question_raises(self, fake_embed, fake_generate, pipeline_config, question):
        """Spec: ValueError on an empty (or blank) question, before any model call."""
        with pytest.raises(ValueError):
            answer(question, pipeline_config)
        assert fake_generate.calls == [], "No generator call for an empty question"
        assert fake_embed.calls == [], "No embedding call for an empty question"

    def test_context_limit_raises_at_the_limit(self, fake_embed, fake_generate, pipeline_config):
        """Spec boundary: prompt_tokens equal to max_prompt_tokens raises ContextLimitError."""
        pipeline_config["generation"]["max_prompt_tokens"] = 500
        fake_generate.result = make_result(prompt_tokens=500)
        with pytest.raises(ContextLimitError):
            answer(self.QUESTION, pipeline_config)

    def test_context_limit_raises_above_the_limit(self, fake_embed, fake_generate, pipeline_config):
        """Spec: prompt_tokens above max_prompt_tokens raises ContextLimitError."""
        pipeline_config["generation"]["max_prompt_tokens"] = 500
        fake_generate.result = make_result(prompt_tokens=501)
        with pytest.raises(ContextLimitError):
            answer(self.QUESTION, pipeline_config)

    def test_context_limit_not_raised_just_below_the_limit(self, fake_embed, fake_generate,
                                                           pipeline_config):
        """Spec boundary: prompt_tokens one below max_prompt_tokens is fine."""
        pipeline_config["generation"]["max_prompt_tokens"] = 500
        fake_generate.result = make_result(prompt_tokens=499)
        out = answer(self.QUESTION, pipeline_config)
        assert out["answer"] == "The answer."

    def test_context_limit_raises_when_prompt_tokens_not_reported(self, fake_embed, fake_generate,
                                                                  pipeline_config):
        """Spec answer(): raises ContextLimitError if the server reports no prompt_tokens (truncation cannot be ruled out)."""
        fake_generate.result = make_result(prompt_tokens=None)
        with pytest.raises(ContextLimitError):
            answer(self.QUESTION, pipeline_config)

    def test_answer_passes_model_digest_from_config(self, fake_embed, fake_generate, pipeline_config):
        """Spec answer(): passes generation.model_digest from the config to generate()."""
        pipeline_config["generation"]["model_digest"] = "sha256:abc123"
        answer(self.QUESTION, pipeline_config)
        assert fake_generate.calls[0]["model_digest"] == "sha256:abc123"

    def test_answer_passes_null_model_digest_as_none(self, fake_embed, fake_generate, pipeline_config):
        """Spec: model_digest may be null in the config -> generate() gets None."""
        answer(self.QUESTION, pipeline_config)
        assert fake_generate.calls[0]["model_digest"] is None

    def test_context_limit_error_is_an_exception_class(self):
        """Spec: ContextLimitError is an Exception subclass."""
        assert issubclass(ContextLimitError, Exception)


# ---------------------------------------------------------------- config

class TestConfig:
    def test_v0_config_loads(self):
        """Spec: configs/v0.yaml (read-only) loads and passes validation; every required section is present."""
        config = load_pipeline_config(PROJECT_ROOT / "configs" / "v0.yaml")
        assert config["name"] == "v0"
        for section, keys in REQUIRED_CONFIG_KEYS.items():
            for key in keys:
                assert key in config[section], f"v0.yaml is missing {section}.{key}"

    def test_v0_config_has_the_approved_values(self):
        """Spec 'Config changes' + Decisions 1-4: the v0 knob values Gaurav approved."""
        config = load_pipeline_config(str(PROJECT_ROOT / "configs" / "v0.yaml"))
        assert config["corpus"]["docs_dir"] == "data/docs"
        assert config["chunking"] == {"chunk_size": 150, "chunk_overlap": 30}
        assert config["embedding"]["model"] == "nomic-embed-text"
        assert config["embedding"]["doc_prefix"] == "search_document: "
        assert config["embedding"]["query_prefix"] == "search_query: "
        assert config["embedding"]["batch_size"] == 32
        assert config["index"]["dir"] == "indexes"
        assert config["retrieval"] == {"method": "vector", "top_k": 4}
        generation = config["generation"]
        assert generation["model"] == "gemma4:31b-cloud"
        assert generation["temperature"] == 0.0
        assert generation["seed"] == 42
        assert generation["reasoning_effort"] is None
        assert generation["max_prompt_tokens"] == 3500
        assert generation["context_separator"] == "\n\n---\n\n"

    def test_v0_config_is_json_serialisable_with_relative_paths(self):
        """Spec: paths stay as written so the config dict is JSON-serialisable (Step 5 stores it as config_json)."""
        config = load_pipeline_config(PROJECT_ROOT / "configs" / "v0.yaml")
        assert json.loads(json.dumps(config)) == config
        assert not Path(config["corpus"]["docs_dir"]).is_absolute()
        assert not Path(config["index"]["dir"]).is_absolute()

    def test_v0_prompt_is_plain(self):
        """Spec: v0 template has {context} and {question}, no refusal, 'only from context' or citation instructions."""
        template = load_pipeline_config(PROJECT_ROOT / "configs" / "v0.yaml")[
            "generation"]["prompt_template"]
        assert "{context}" in template and "{question}" in template
        lowered = template.lower()
        for forbidden in ("only", "refuse", "cite", "citation", "don't know", "do not know"):
            assert forbidden not in lowered, f"v0 prompt must not contain '{forbidden}'"

    def test_fixture_config_survives_a_yaml_round_trip(self, tmp_path, pipeline_config):
        """Helper check: the pipeline_config fixture is a valid config (writes to YAML and loads back unchanged)."""
        path = write_pipeline_yaml(tmp_path, pipeline_config)
        assert load_pipeline_config(path) == pipeline_config

    def test_required_config_keys_cover_every_section(self):
        """Spec: REQUIRED_CONFIG_KEYS maps section -> keys for all the sections in configs/v0.yaml."""
        assert set(REQUIRED_CONFIG_KEYS) == {
            "corpus", "chunking", "embedding", "index", "retrieval", "generation"}
        assert set(REQUIRED_CONFIG_KEYS["generation"]) >= {
            "model", "temperature", "seed", "reasoning_effort", "max_prompt_tokens",
            "context_separator", "prompt_template"}

    @pytest.mark.parametrize("section, key", flat_required_keys())
    def test_missing_key_raises(self, tmp_path, pipeline_config, section, key):
        """Spec: for each key in REQUIRED_CONFIG_KEYS a YAML without it raises ValueError naming the key and the file."""
        del pipeline_config[section][key]
        path = write_pipeline_yaml(tmp_path, pipeline_config)
        with pytest.raises(ValueError, match=re.escape(key)) as excinfo:
            load_pipeline_config(path)
        assert path.name in str(excinfo.value), "Error should name the config file"

    @pytest.mark.parametrize("section", list(REQUIRED_CONFIG_KEYS))
    def test_missing_section_raises(self, tmp_path, pipeline_config, section):
        """Spec: a YAML without a whole required section raises ValueError naming the section."""
        del pipeline_config[section]
        path = write_pipeline_yaml(tmp_path, pipeline_config)
        with pytest.raises(ValueError, match=re.escape(section)):
            load_pipeline_config(path)

    def test_null_reasoning_effort_is_valid(self, tmp_path, pipeline_config):
        """Spec: reasoning_effort is required but may be null."""
        pipeline_config["generation"]["reasoning_effort"] = None
        config = load_pipeline_config(write_pipeline_yaml(tmp_path, pipeline_config))
        assert config["generation"]["reasoning_effort"] is None

    @pytest.mark.parametrize("changes", [
        [("chunking", "chunk_overlap", 8)],                         # overlap == size (8)
        [("chunking", "chunk_overlap", 9)],                         # overlap > size
        [("chunking", "chunk_overlap", -1)],                        # negative overlap
        [("chunking", "chunk_size", 0), ("chunking", "chunk_overlap", 0)],   # size 0
        [("retrieval", "top_k", 0)],
        [("embedding", "batch_size", 0)],
        [("retrieval", "method", "bm25")],
        [("retrieval", "method", "hybrid")],
        [("generation", "prompt_template", "Context: {context}\nAnswer:")],        # no {question}
        [("generation", "prompt_template", "Question: {question}\nAnswer:")],       # no {context}
        [("generation", "prompt_template", "{context} {question} {extra}")],        # unknown field
        [("generation", "prompt_template", "{context} {question} {")],              # stray brace
    ], ids=["overlap_eq_size", "overlap_gt_size", "overlap_negative", "size_zero", "top_k_zero",
            "batch_size_zero", "method_bm25", "method_hybrid", "template_no_question",
            "template_no_context", "template_unknown_field", "template_stray_brace"])
    def test_bad_values_raise(self, tmp_path, pipeline_config, changes):
        """Spec: overlap >= size, size < 1, negative overlap, top_k 0, batch_size 0, method other than vector, or a bad template raise ValueError naming the file."""
        for section, key, value in changes:
            pipeline_config[section][key] = value
        path = write_pipeline_yaml(tmp_path, pipeline_config)
        with pytest.raises(ValueError) as excinfo:
            load_pipeline_config(path)
        assert path.name in str(excinfo.value), "Error should name the config file"

    @pytest.mark.parametrize("changes", [
        [("chunking", "chunk_overlap", 7)],                         # overlap = size - 1 (the largest valid value)
        [("chunking", "chunk_overlap", 0)],
        [("chunking", "chunk_size", 1), ("chunking", "chunk_overlap", 0)],
        [("retrieval", "top_k", 1)],
        [("embedding", "batch_size", 1)],
        [("generation", "reasoning_effort", "low")],
    ], ids=["overlap_size_minus_1", "overlap_zero", "size_one", "top_k_one", "batch_one",
            "effort_low"])
    def test_boundary_values_are_accepted(self, tmp_path, pipeline_config, changes):
        """Spec boundaries: overlap = size-1, overlap 0, size 1, top_k 1 and batch_size 1 are valid."""
        for section, key, value in changes:
            pipeline_config[section][key] = value
        config = load_pipeline_config(write_pipeline_yaml(tmp_path, pipeline_config))
        for section, key, value in changes:
            assert config[section][key] == value

    def test_load_keeps_paths_as_written(self, tmp_path, pipeline_config):
        """Spec: load_pipeline_config does not resolve paths; they stay exactly as written."""
        pipeline_config["corpus"]["docs_dir"] = "data/docs"
        pipeline_config["index"]["dir"] = "indexes"
        config = load_pipeline_config(write_pipeline_yaml(tmp_path, pipeline_config))
        assert config["corpus"]["docs_dir"] == "data/docs"
        assert config["index"]["dir"] == "indexes"

    def test_v0_config_has_review_added_values(self):
        """Spec Config changes: configs/v0.yaml has embedding.max_input_words: 1000 and a
        generation.model_digest (null in Step 3; Step 5 pins the 64-hex Ollama digest)."""
        config = load_pipeline_config(PROJECT_ROOT / "configs" / "v0.yaml")
        assert config["embedding"]["max_input_words"] == 1000
        assert "model_digest" in config["generation"], "model_digest must be present"
        assert re.fullmatch(r"[0-9a-f]{64}", config["generation"]["model_digest"])

    def test_required_config_keys_include_review_additions(self):
        """Spec Config changes: max_input_words and model_digest are required keys (may be null for model_digest)."""
        assert "max_input_words" in REQUIRED_CONFIG_KEYS["embedding"]
        assert "model_digest" in REQUIRED_CONFIG_KEYS["generation"]

    def test_non_mapping_top_level_raises(self, tmp_path):
        """Spec load_pipeline_config: the top level must be a mapping (a YAML list is rejected, naming the file)."""
        path = tmp_path / "list_config.yaml"
        path.write_text(yaml.safe_dump(["corpus", "chunking"]), encoding="utf-8")
        with pytest.raises(ValueError) as excinfo:
            load_pipeline_config(path)
        assert path.name in str(excinfo.value), "Error should name the config file"

    def test_empty_yaml_file_raises(self, tmp_path):
        """Spec load_pipeline_config: an empty YAML (loads as None, not a mapping) raises ValueError."""
        path = tmp_path / "empty_config.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            load_pipeline_config(path)

    @pytest.mark.parametrize("section, key, bad_value", [
        ("embedding", "model", 123),
        ("embedding", "model", ["nomic-embed-text"]),
        ("embedding", "doc_prefix", 5),
        ("embedding", "doc_prefix", ["search_document: "]),
        ("embedding", "query_prefix", 5),
        ("embedding", "query_prefix", {"a": "b"}),
        ("generation", "model", 123),
        ("generation", "model", ["gemma4:31b-cloud"]),
        ("generation", "context_separator", 5),
        ("generation", "context_separator", ["---"]),
    ])
    def test_non_string_text_settings_raise(self, tmp_path, pipeline_config, section, key, bad_value):
        """Spec Config changes: models, prefixes and the separator must be strings."""
        pipeline_config[section][key] = bad_value
        path = write_pipeline_yaml(tmp_path, pipeline_config)
        with pytest.raises(ValueError) as excinfo:
            load_pipeline_config(path)
        assert path.name in str(excinfo.value), "Error should name the config file"
        assert key in str(excinfo.value), f"Error should name the bad key {key}"

    @pytest.mark.parametrize("key, bad_value", [
        ("model_digest", 123),
        ("model_digest", ["sha256:abc"]),
        ("reasoning_effort", 5),
        ("reasoning_effort", ["low"]),
    ])
    def test_non_string_non_null_digest_and_effort_raise(self, tmp_path, pipeline_config, key, bad_value):
        """Spec Config changes: model_digest and reasoning_effort must be strings or null."""
        pipeline_config["generation"][key] = bad_value
        path = write_pipeline_yaml(tmp_path, pipeline_config)
        with pytest.raises(ValueError) as excinfo:
            load_pipeline_config(path)
        assert path.name in str(excinfo.value), "Error should name the config file"
        assert key in str(excinfo.value), f"Error should name the bad key {key}"

    def test_string_model_digest_is_accepted(self, tmp_path, pipeline_config):
        """Spec Config changes: model_digest may be a string (Step 5 fills it)."""
        pipeline_config["generation"]["model_digest"] = "sha256:abc123"
        config = load_pipeline_config(write_pipeline_yaml(tmp_path, pipeline_config))
        assert config["generation"]["model_digest"] == "sha256:abc123"

    def test_null_model_digest_is_accepted(self, tmp_path, pipeline_config):
        """Spec Config changes: model_digest is required but may be null."""
        pipeline_config["generation"]["model_digest"] = None
        config = load_pipeline_config(write_pipeline_yaml(tmp_path, pipeline_config))
        assert config["generation"]["model_digest"] is None

    def test_chunk_size_above_max_input_words_raises(self, tmp_path, pipeline_config):
        """Spec Config changes: chunk_size must not exceed embedding.max_input_words (8 > 7)."""
        pipeline_config["embedding"]["max_input_words"] = 7     # chunk_size is 8
        path = write_pipeline_yaml(tmp_path, pipeline_config)
        with pytest.raises(ValueError) as excinfo:
            load_pipeline_config(path)
        assert path.name in str(excinfo.value), "Error should name the config file"
        assert "max_input_words" in str(excinfo.value)

    def test_chunk_size_equal_to_max_input_words_is_accepted(self, tmp_path, pipeline_config):
        """Spec boundary: chunk_size == max_input_words (8 == 8) is valid."""
        pipeline_config["embedding"]["max_input_words"] = 8     # chunk_size is 8
        config = load_pipeline_config(write_pipeline_yaml(tmp_path, pipeline_config))
        assert config["embedding"]["max_input_words"] == 8

    @pytest.mark.parametrize("bad_value", [0, -1])
    def test_max_input_words_below_one_raises(self, tmp_path, pipeline_config, bad_value):
        """Spec Config changes: embedding.max_input_words < 1 raises ValueError."""
        pipeline_config["embedding"]["max_input_words"] = bad_value
        path = write_pipeline_yaml(tmp_path, pipeline_config)
        with pytest.raises(ValueError) as excinfo:
            load_pipeline_config(path)
        assert path.name in str(excinfo.value), "Error should name the config file"
        assert "max_input_words" in str(excinfo.value)

    def test_max_input_words_of_one_with_chunk_size_one_is_accepted(self, tmp_path, pipeline_config):
        """Spec boundary: max_input_words 1 is valid when chunk_size is 1 (1 <= 1)."""
        pipeline_config["embedding"]["max_input_words"] = 1
        pipeline_config["chunking"].update(chunk_size=1, chunk_overlap=0)
        config = load_pipeline_config(write_pipeline_yaml(tmp_path, pipeline_config))
        assert config["embedding"]["max_input_words"] == 1
