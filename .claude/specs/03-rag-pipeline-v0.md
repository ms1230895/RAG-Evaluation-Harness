# Spec: Step 03 — RAG Pipeline v0

## Overview
This step builds the system under test: a small, config-driven RAG pipeline
in `rag/` that turns a question into an answer plus the chunks it was based
on. It has six parts: chunker, embedder, FAISS index, top-k retriever,
prompt builder and generator. Every knob (chunk size, overlap, top-k,
embedding model, prompt, generator model and temperature) comes from
`configs/v0.yaml`. It exposes exactly one public entry point, the contract
`answer(question, config) -> {"answer", "chunks", "latency_ms"}`, which is
all the evaluation harness (Steps 5 onwards) will ever call. It comes now
because the test set (Step 4) and the runner (Step 5) need something to
evaluate, and the harness can only see retrieved chunks and scores if the
pipeline returns them. v0 is a deliberately plain baseline: no hybrid
search, no reranker, no refusal or citation instructions. Those are
Step 11 experiments, and they only show a measurable gain if v0 has not
already been tuned.

## Concepts to learn
- **Retrieval-Augmented Generation (RAG):** an LLM does not know Kestrel
  Labs' policies. So for each question we first *retrieve* the few most
  relevant passages from our documents, paste them into the prompt, and ask
  the LLM to answer from them. Two places can fail: retrieval (the right
  passage never reached the prompt) and generation (it reached the prompt,
  but the LLM ignored it or added things). Returning the chunks with the
  answer is what lets the harness tell these apart later.
- **Chunking:** documents are too long, and too mixed in topic, to retrieve
  whole. We cut each document into overlapping windows of words.
  - `chunk_size` = words per chunk, `chunk_overlap` = words shared with the
    previous chunk, `step = chunk_size - chunk_overlap`.
  - Overlap means a fact that sits on a boundary still appears whole in at
    least one chunk.
  - Chunks never cross documents, so each chunk has exactly one `doc_id`.
  - Worked example: a 10-word document, `chunk_size=4`, `chunk_overlap=1`,
    so step 3. Windows start at words 0, 3, 6: `[0-3] [3-6] [6-9]`. The
    window starting at 6 reaches the end (6 + 4 >= 10), so we stop: 3
    chunks, and words 3 and 6 appear twice.
  - We count words, not tokens, so no tokenizer is needed. Rough rule:
    1 word is about 1.3 tokens.
  - Each chunk's text is cut from the original document between its first
    and last word, so line breaks and Markdown lists survive.
- **Embedding:** a model turns a piece of text into a fixed-length vector
  (here 768 numbers from `nomic-embed-text`, running locally in Ollama) so
  that texts with similar meaning get vectors pointing in similar
  directions. We embed every chunk once (when building the index), and the
  question once per call.
- **Task prefixes:** `nomic-embed-text` was trained to see
  `search_document: ` before passages and `search_query: ` before
  questions. Leaving them out quietly lowers retrieval quality. They are
  the model's documented usage, not a tuning trick, so v0 uses them. Both
  prefixes live in the config.
- **Cosine similarity:** how close two vectors are in direction:
  `cos(a, b) = (a · b) / (|a| × |b|)`, from -1 (opposite) to 1 (same).
  Example: `a = (1, 0)`, `b = (0.6, 0.8)`: `a · b = 0.6`, `|a| = |b| = 1`,
  so `cos = 0.6`. If we first scale every vector to length 1
  (L2-normalise), cosine is just the dot product, which is what FAISS
  computes fastest.
- **FAISS `IndexFlatIP`:** a "flat" index stores all vectors and compares
  the query with every one of them (exact search, no approximation). With
  about 70 chunks, exact search takes well under a millisecond. "IP" means
  inner product (the dot product), which equals cosine because all vectors
  are normalised. The chunk's `score` in the contract is this cosine value.
- **Top-k:** the `k` chunks with the highest score go into the prompt.
  Larger k means a better chance the right chunk is included (recall), but
  also more off-topic text the LLM might use (noise). Tuning k is a
  Step 11 experiment.
- **Index cache on disk:** embedding every chunk costs one embedding call per
  chunk, so the built index is saved under `indexes/<index_key>.faiss`
  (plus the chunk list as JSON). `index_key` is a SHA-256 of everything
  that changes the vectors: the documents' IDs and text, `chunk_size`,
  `chunk_overlap`, the embedding model and `doc_prefix`. Changing top-k or
  the prompt reuses the same index. Changing chunk size builds a new one.
  This file is effectively the embedding cache.
- **Prompt builder:** fills a template from the config with the question
  and the retrieved chunks (joined by a separator, in rank order). v0's
  template is plain on purpose: no "answer only from the context", no
  refusal instruction, no citations.
- **The contract:** `answer(question, config)` returns
  `{"answer": str, "chunks": [{"chunk_id", "doc_id", "text", "score"}],
  "latency_ms": float}`. The harness depends on nothing else, so it can
  later evaluate any RAG system that implements this one function.
- **Latency, honestly measured:** `latency_ms = retrieval_ms +
  model_latency_ms`. `retrieval_ms` is the measured time to embed the
  question and search the index. `model_latency_ms` comes from
  `llm/client.py` and is the original model time, even on a cache hit. The
  wall-clock `LLMResult.latency_ms` is never used, because on a cache hit it
  is about 2 ms and would make the pipeline look 1000x faster than it is
  (Step 2 carry-forward). Index building is one-off setup and is not
  counted.

## Depends on
- Step 0: `faiss-cpu`, `numpy`, `openai`, `pyyaml` installed; `.gitignore`
  already ignores `*.faiss` and `indexes/`.
- Step 1: `rag/loader.py` → `load_docs(docs_dir)` returns
  `[{"doc_id", "title", "text"}]` sorted by `doc_id` (22 docs, 7,178 words).
- Step 2: `llm/client.py` → `generate(prompt, model, temperature, schema=None,
  *, config=None, model_digest=None) -> LLMResult` (cached, retried,
  logged), `load_config()`, `_get_client()`, `_call_api()` retry loop,
  `TRANSIENT_ERRORS`. `configs/llm.yaml` holds `base_url`.
- Step 2 carry-forward items for this step (all addressed below):
  `seed` and `reasoning_effort`; the context limit; `latency_ms` from
  `model_latency_ms`.
- Local setup (checked 2026-10-15 with `ollama list` / `ollama show`):
  `nomic-embed-text:latest` (768-dim, context length 2048) is pulled
  locally; `gemma4:31b-cloud` has context length 262144 and thinking levels
  `false` / `true`, default `false`.

## Interfaces
This step **creates** the `answer()` contract exactly as written in
CLAUDE.md. It adds no keys and removes none (see Decision 4).

`rag/chunker.py`
```python
def chunk_document(doc: dict, chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Split one doc ({"doc_id", "title", "text"}) into overlapping word windows.
    Returns [{"chunk_id": "<doc_id>#<nnn>", "doc_id", "text"}], nnn = 000, 001, ...
    Raises ValueError unless chunk_size >= 1 and 0 <= chunk_overlap < chunk_size."""

def chunk_corpus(docs: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    """chunk_document over every doc, in doc order."""
```
Example: `chunk_document({"doc_id": "d", "title": "T", "text": "a b c d e f g h i j"}, 4, 1)`
→ chunks `d#000` "a b c d", `d#001` "d e f g", `d#002` "g h i j".

`llm/client.py` (new function, if Decision 5 is A)
```python
def embed(texts: list[str], model: str, *, config: dict | None = None) -> list[list[float]]:
    """One embeddings request to the OpenAI-compatible endpoint, inside the
    same tenacity retry loop as generate(). Returns one vector per input, in
    input order (sorted by the response's `index`). Not cached and not
    logged to llm_calls (see Decision 5)."""
```

`rag/embedder.py`
```python
def embed_documents(texts: list[str], embedding_config: dict) -> np.ndarray:
    """Prefix each text with doc_prefix, embed in batches of batch_size,
    return float32 array (n, dim), every row L2-normalised."""

def embed_query(question: str, embedding_config: dict) -> np.ndarray:
    """Prefix with query_prefix, embed, return float32 array (dim,), L2-normalised."""
```
Both raise `ValueError` if a returned vector has zero length (it cannot be
normalised).

`rag/index.py`
```python
def index_key(docs: list[dict], config: dict) -> str:
    """SHA-256 hex of: each (doc_id, text), chunk_size, chunk_overlap,
    CHUNKER_VERSION (rag/chunker.py), embedding model, doc_prefix, and the
    embedding endpoint (base_url from configs/llm.yaml). Nothing else."""

def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Exact inner-product index over already-normalised vectors."""

def resolve_path(path: str | Path) -> Path:
    """Relative config paths are resolved against the repo root."""

def save_index(index, chunks: list[dict], index_dir: Path, key: str, meta: dict) -> None
def load_index(index_dir: Path, key: str) -> tuple[faiss.Index, list[dict]] | None
    """None if either file is missing; ValueError if index.ntotal != len(chunks)."""

def get_index(config: dict) -> tuple[faiss.Index, list[dict]]:
    """Return (index, chunks) for this config: from memory if already loaded
    in this process (module-level dict `_LOADED`, keyed by index_key), else
    from disk, else build (load_docs -> chunk_corpus -> embed_documents ->
    build_index -> save_index). Logs "index <key[:12]> built | loaded from
    disk | reused from memory: N chunks (X ms)". After a disk load it embeds
    one warm-up query (WARM_UP_TEXT) so Ollama has loaded the embedding model
    before any timed retrieval; after a build the model is already warm.
    Raises ValueError naming docs_dir if the corpus gives no chunks."""

def describe_index(config: dict) -> dict:
    """{"index_key": ..., **meta} for this config's index (built if needed),
    so Step 5 can record which corpus and index a run used (config_json alone
    does not cover the document contents)."""
```

`rag/retriever.py`
```python
def retrieve(question: str, index, chunks: list[dict], top_k: int,
             embedding_config: dict) -> list[dict]:
    """Embed the question, search, return top_k chunks in rank order:
    [{"chunk_id", "doc_id", "text", "score": float}] (score = cosine,
    plain Python float, highest first)."""
```

`rag/pipeline.py`
```python
REQUIRED_CONFIG_KEYS: dict[str, tuple[str, ...]]   # section -> keys, see Config changes

def load_pipeline_config(path: str | Path) -> dict:
    """Read a pipeline YAML (e.g. configs/v0.yaml). Raises ValueError naming
    the file and the problem: missing section or key, chunk_size < 1,
    chunk_overlap < 0 or >= chunk_size, top_k < 1, batch_size < 1,
    retrieval.method other than "vector", prompt_template without
    {context} or {question}, or a template that .format(context="",
    question="") rejects. Paths stay as written (relative strings) so the
    config dict is JSON-serialisable for Step 5's config_json; rag/index.py
    resolves them against the repo root where they are used."""

def build_prompt(question: str, chunks: list[dict], template: str, separator: str) -> str:
    """template.format(context=separator.join(chunk texts in rank order), question=question)."""

def answer(question: str, config: dict) -> dict:
    """THE CONTRACT. Returns {"answer": str,
    "chunks": [{"chunk_id", "doc_id", "text", "score"}], "latency_ms": float}.
    latency_ms = retrieval_ms + LLMResult.model_latency_ms.
    Raises ValueError on an empty question; lets LLM errors
    (LLMOutputError, openai errors) propagate, never returns a made-up
    answer; raises ContextLimitError (Decision 4) if prompt_tokens reaches
    max_prompt_tokens OR the server reports no prompt_tokens (truncation
    cannot be ruled out). Passes generation.model_digest to generate()."""

class ContextLimitError(Exception):
    """prompt_tokens reached generation.max_prompt_tokens, or was not reported."""
```
Example:
```python
from rag.pipeline import load_pipeline_config, answer
config = load_pipeline_config("configs/v0.yaml")
out = answer("How many casual leaves do interns get?", config)
out["answer"]                       # text from gemma4:31b-cloud
[(c["doc_id"], round(c["score"], 3)) for c in out["chunks"]]
out["latency_ms"]
```
(No values are shown because none have been measured yet.)

CLI (used for the Done-when check):
```
python -m rag.pipeline --config configs/v0.yaml --question "How many casual leaves do interns get?"
```
Turns on INFO logging, so the `rag.index` log line shows how the index was
obtained (built / loaded from disk) and its chunk count, and the existing
`llm.client` log line shows the generator's `cached` flag and
`prompt_tokens` (this keeps `answer()`'s return value exactly the
contract). Then prints the answer, one line per chunk (rank, `chunk_id`,
`doc_id`, score to 3 decimals, first 80 characters of text) and
`latency_ms`.

Change to `llm/client.py` `generate()` (Step 2 carry-forward):
```python
def generate(prompt, model, temperature, schema=None, *, config=None,
             model_digest=None, seed: int | None = None,
             reasoning_effort: str | None = None) -> LLMResult
def make_cache_key(prompt, model, temperature, schema, base_url,
                   model_digest=None, seed=None, reasoning_effort=None) -> str
```
`seed` and `reasoning_effort` are sent in the request only when not
`None`, are part of the cache key (always, also when None: the key's JSON
now has 8 fields), and are stored in `llm_cache` and `llm_calls`.
`_send_request` and `_call_api` gain the same two parameters (keyword,
default None). A new helper `_retrying(config) -> tenacity.Retrying` builds
the retry policy once; `_call_api` and `embed` both use it.

## Data and schema changes
- No results-database changes (`evalharness/store.py` does not exist yet;
  it arrives in Step 5).
- `data/llm_cache.sqlite` (Step 2): `llm_cache` and `llm_calls` each gain
  two columns, `seed INTEGER` and `reasoning_effort TEXT` (NULL when not
  sent). `CREATE TABLE IF NOT EXISTS` will not add columns to the existing
  file, so the cache file (only the Step 2 demo rows) is deleted and
  rebuilt, as was done twice in Step 2.
- New gitignored files, one pair per distinct index:
  - `indexes/<index_key>.faiss`: FAISS index written with
    `faiss.write_index`.
  - `indexes/<index_key>.chunks.json`: the chunk list
    `[{"chunk_id", "doc_id", "text"}]`, in the same order as the index rows
    (row i of the index is chunk i), plus a `"meta"` object with the
    inputs of the key (chunk settings, embedding model, doc_prefix, doc
    count, chunk count, dim, created_at).
  Both are covered by the existing `.gitignore` (`indexes/`, `*.faiss`).

## Config changes
New file `configs/v0.yaml`. Values marked (D1) and (D2) wait on
Decisions 1 and 2; those shown are recommendations, not tuned values.
```yaml
# Pipeline v0: a plain, untuned baseline. Each Step 11 experiment copies
# this file and changes ONE value.
name: v0

corpus:
  docs_dir: data/docs                 # relative to the repo root

chunking:
  chunk_size: 150                     # (D1) words per chunk
  chunk_overlap: 30                   # (D1) words shared with the previous chunk

embedding:
  model: nomic-embed-text             # local Ollama model, 768-dim
  doc_prefix: "search_document: "     # nomic's documented passage prefix
  query_prefix: "search_query: "      # nomic's documented question prefix
  batch_size: 32                      # texts per embeddings request
  max_input_words: 1000               # chunk_size must not exceed this (nomic reads at most 2048 tokens)

index:
  dir: indexes                        # gitignored; one file pair per index_key

retrieval:
  method: vector                      # only "vector" in v0; bm25/hybrid come in Step 11
  top_k: 4                            # (D1) chunks put into the prompt

generation:
  model: gemma4:31b-cloud
  model_digest: null                  # model build; null in v0, Step 5 fills it (cache key)
  temperature: 0.0                    # (D1)
  seed: 42                            # (D3) sent to the API, part of the cache key
  reasoning_effort: null              # (D3) null = not sent; gemma4 thinking defaults to off
  max_prompt_tokens: 3500             # (D4) raise ContextLimitError at or above this
  context_separator: "\n\n---\n\n"    # between chunks in the prompt
  prompt_template: |                  # (D2) must contain {context} and {question}
    Use the following context to answer the question.

    Context:
    {context}

    Question: {question}

    Answer:
```
All sections and keys above are required (`REQUIRED_CONFIG_KEYS` in
`rag/pipeline.py`); `reasoning_effort` and `model_digest` may be `null`.
`load_pipeline_config` also checks: the top level is a mapping; models,
prefixes and the separator are strings; `model_digest` /
`reasoning_effort` are strings or null; `chunk_size <= max_input_words`.
(`max_input_words` and `model_digest` were added in the Step 3 code review.) `configs/llm.yaml`
is unchanged: `embed()` and `generate()` both use its `base_url`, timeout
and retry settings.

Size check for the recommended values: about 7,178 / 120 ≈ 60-75 chunks.
4 chunks × 150 words ≈ 600 words ≈ 800 tokens of context, far below both
3,500 and nomic's 2,048-token input limit per chunk.

## Files to create
- `configs/v0.yaml`
- `rag/chunker.py`
- `rag/embedder.py`
- `rag/index.py`
- `rag/retriever.py`
- `rag/pipeline.py`
- `tests/test_03_rag_pipeline_v0.py`

## Files to change
- `llm/client.py`: `seed` and `reasoning_effort` in `generate()`,
  `make_cache_key()`, `_send_request()`, `_call_api()`, both tables and the
  logging; new `embed()` (if Decision 5 is A). Module docstring updated.
- `tests/conftest.py`: `FakeApi.__call__` accepts the new `seed` and
  `reasoning_effort` arguments (done during implementation, records them
  in `calls`); add a `fake_embed` fixture (deterministic
  fake vectors, records inputs) and a `pipeline_config` fixture (v0 values,
  `index.dir` in `tmp_path`, and `rag.index._LOADED` reset).
- `tests/test_02_llm_wrapper.py`: (done during implementation) `FakeSend`
  accepts the two new arguments, `TestCacheKey.BASE` and the spec-formula
  test use 8 fields, the column-set test lists the new columns. Still to
  add: cache-key tests cover the two new inputs;
  column-list test includes the new columns; `_send_request` tests check
  `seed` / `reasoning_effort` are sent only when not None.
- `tests/README.md`: new fixtures and the Step 3 test file.
- `.claude/specs/02-llm-wrapper.md`: mark the three Step 3 carry-forward
  items as handled in Step 3.
- `CLAUDE.md`: Step 3 entry in the Progress section (with the real Done-when
  output, copied from the run and from `llm_calls`).
- `data/llm_cache.sqlite`: deleted and rebuilt (gitignored; not committed).

## New dependencies
No new dependencies. `faiss-cpu`, `numpy`, `openai` and `pyyaml` are already
pinned in `requirements.txt`.

## Tests
`tests/test_03_rag_pipeline_v0.py`. No test reaches Ollama or the network:
embeddings come from a fake (`fake_embed` replaces the one function in
`rag/embedder.py` that calls `llm.client.embed`, and returns deterministic
bag-of-words hash vectors), and generation from a fake `generate`
monkeypatched into `rag.pipeline` that records its arguments and returns an
`LLMResult`. Index files go to `tmp_path`.

Chunker
- `test_chunk_worked_example`: 10 words, size 4, overlap 1 → 3 chunks with
  exactly the word spans from Concepts and IDs `d#000`..`d#002`.
- `test_short_doc_is_one_chunk`: fewer words than `chunk_size` → 1 chunk with
  the whole text.
- `test_overlap_words_are_shared`: the last `chunk_overlap` words of chunk i
  are the first words of chunk i+1.
- `test_every_word_is_covered`: the union of chunk words, in order, rebuilds
  the document's word list.
- `test_chunk_keeps_line_breaks`: a chunk spanning two lines keeps the `\n`.
- `test_bad_chunk_settings_raise`: overlap >= size, size 0, negative overlap.
- `test_real_corpus_chunks`: on `load_docs()`, every chunk has a known
  `doc_id`, non-empty text and a unique `chunk_id`; running twice gives
  identical output.

Embedder (with a fake `llm.client.embed`)
- `test_prefixes_are_applied`: documents get `doc_prefix`, the query gets
  `query_prefix`.
- `test_vectors_are_normalised`: every row has L2 norm 1 (within 1e-5),
  dtype float32.
- `test_batches`: 5 texts with `batch_size=2` → 3 calls, output in input order.
- `test_zero_vector_raises`.

`llm.client.embed` (fake client object, as in Step 2)
- `test_embed_builds_request_and_orders_by_index`: the response lists data
  out of order; the result follows `index`.
- `test_embed_retries_transient_errors`: connection error twice then success.

Index and retriever
- `test_search_returns_expected_order_and_cosine`: hand-made 2-D vectors
  `(1,0)`, `(0.6,0.8)`, `(0,1)` and query `(1,0)` → order 0, 1, 2 with scores
  1.0, 0.6, 0.0.
- `test_save_load_roundtrip`: same search results and chunks after reload.
- `test_load_missing_returns_none`.
- `test_index_key_changes`: changes with doc text, chunk_size,
  chunk_overlap, embedding model, doc_prefix; does NOT change with top_k,
  prompt_template, temperature or query_prefix.
- `test_get_index_builds_once`: two `get_index` calls with the same config →
  documents embedded once (second from memory); a fresh process state
  (memory cleared) loads from disk without embedding.
- `test_retrieve_shape`: `top_k` results, scores descending, keys exactly
  `chunk_id, doc_id, text, score`, score is a Python `float`.

Prompt and contract
- `test_build_prompt`: question and chunk texts appear, chunks in rank order
  joined by the separator; chunk text containing `{` and `}` does not crash.
- `test_answer_contract_shape`: keys exactly `{"answer", "chunks",
  "latency_ms"}`; each chunk's keys exactly the four; types `str`, `list`,
  `float`; `len(chunks) == top_k`.
- `test_answer_passes_generation_settings`: the fake `generate` receives the
  built prompt, and `model`, `temperature`, `seed`, `reasoning_effort` from
  the config.
- `test_latency_uses_model_latency`: fake returns `model_latency_ms=1234.0`
  and wall `latency_ms=999999.0`; `1234.0 <= out["latency_ms"] < 999999.0`.
- `test_llm_error_propagates`: fake raises `LLMOutputError` → `answer`
  raises it.
- `test_empty_question_raises`: `""` and `"   "`.
- `test_context_limit_raises`: fake reports `prompt_tokens` equal to
  `max_prompt_tokens` → `ContextLimitError`.

Config
- `test_v0_config_loads`: `configs/v0.yaml` loads and passes validation.
- `test_missing_key_raises`: for each section and key in
  `REQUIRED_CONFIG_KEYS`, a YAML without it raises `ValueError` naming it.
- `test_bad_values_raise`: overlap >= size, top_k 0, batch_size 0,
  `method: bm25`, template without `{context}`.

Step 2 additions (in `tests/test_02_llm_wrapper.py`)
- Cache key changes with `seed` and with `reasoning_effort`; both stored in
  `llm_cache` and `llm_calls`; sent in the request only when not None.

Architecture
- `test_rag_does_not_import_evalharness` (existing) still passes with the new
  files.

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
- Implement the contract exactly: three keys, four chunk keys, nothing
  extra. Any change needs Gaurav's approval first.
- `answer()` never catches an LLM error to return a fallback string. A
  failed call must surface, so Step 5 can record it as a failure and never
  score it.
- `latency_ms` = measured retrieval time + `LLMResult.model_latency_ms`.
  Never use `LLMResult.latency_ms` (wall time). Index building is not
  included.
- `score` is the FAISS inner product of L2-normalised vectors (= cosine),
  converted to a plain Python `float`. Never re-scale or round it.
- Chunks are returned in rank order (highest score first), the same order
  they appear in the prompt.
- Chunks never cross document boundaries. `chunk_id` is `<doc_id>#<nnn>`,
  stable for a given doc text and chunk settings.
- Normalise vectors in our code (numpy); do not rely on the server
  returning unit vectors.
- The index file pair is rebuilt only when `index_key` changes. A stale
  index must never be loaded: the key covers doc text, chunk settings,
  embedding model and doc_prefix. (The embedding model's build digest is
  not in the key yet; Step 5 handles model identity.)
- The generator's `seed` and `reasoning_effort` are sent only when not
  `None`, and are always part of the cache key and stored rows (Step 2
  rule: any request parameter that changes output enters the key).
- v0 stays plain: no refusal instruction, no "only from context", no
  citations, no reranker, no BM25, no query rewriting, no doc titles added
  to chunks.
- `rag/` may import `llm.client`, but nothing from `evalharness/`.
- Embedding inputs must stay under nomic's 2,048-token limit; Ollama
  truncates longer inputs silently. With chunk_size in words, keep it at or
  below about 1,000 words.
- Done-when numbers (scores, latency, tokens) in the summary are copied from
  the actual run output and `llm_calls`, never estimated. If the intern
  leave policy is not in the top-k for the Done-when question, report that
  plainly; do not tune v0 to fix it.

## Decisions for Gaurav
All resolved on 2026-10-15; Gaurav accepted every recommendation:
1. ~~v0 knob values.~~ `chunk_size: 150` words, `chunk_overlap: 30`,
   `top_k: 4`, generator `temperature: 0.0`.
2. ~~v0 prompt template.~~ The plain template in Config changes, chunks
   joined by `---`, no doc IDs. Stricter wording is held back for Step 11.
3. ~~`seed` and `reasoning_effort`.~~ `seed: 42`, `reasoning_effort: null`
   (not sent; gemma4 thinking stays off). The judge's values are decided in
   Step 7.
4. ~~Context-limit guard.~~ **Option A:** no contract change;
   `max_prompt_tokens: 3500`; `answer()` raises `ContextLimitError` when the
   reply's `prompt_tokens` reaches it.
5. ~~Where embedding calls live.~~ **Option A:** `embed()` in
   `llm/client.py`, sharing its client and retry loop; not cached in SQLite,
   not logged in `llm_calls`.
6. ~~Real calls for the Done-when check.~~ Approved.

## Code review changes (2026-10-15)
The Step 3 review (security, quality, eval-integrity) had no Critical or
High findings. Gaurav approved all 17 action items. Code changes:
`_build_and_save()` helper and `describe_index()` in `rag/index.py`; empty
corpus raises a clear `ValueError`; warm-up embedding after a disk load;
`generation.model_digest` passed through; `prompt_tokens=None` raises
`ContextLimitError`; stricter config checks and `embedding.max_input_words`;
`CHUNKER_VERSION` and the embedding `base_url` in the index key;
`test_setup.py` checks `sentence_transformers` with `find_spec` (see the
libomp note below). Two `test_03` disk-load assertions now expect exactly
one warm-up query embedding and still no chunk re-embedding.

## Carry-forward to later steps
- **macOS libomp clash (Step 11, and any torch use) [resolved in Step 11: the cross-encoder runs in a child process, `rag/rerank_worker.py`]:** faiss-cpu and torch
  each bundle `libomp.dylib`; they cannot run in one process here (checked
  2026-10-15 in every order: `OMP: Error #15` or a silent crash). The test
  suite avoids it by not importing `sentence_transformers` in-process. The
  Step 11 CrossEncoder reranker must solve it for real (separate process,
  numpy search instead of FAISS, or another FAISS build): *decision for
  Gaurav then*. Never set `KMP_DUPLICATE_LIB_OK=TRUE` (OpenMP: may give
  incorrect results).
- **Step 5:** `ContextLimitError` is raised after the reply is cached and
  logged with `error=NULL`; the runner must record it as the item's failure
  and count it in the denominator (larger Step 11 configs may hit the 3500
  guard; dropping those items would bias results). Fill
  `generation.model_digest`; store `describe_index(config)` with each run;
  snapshot the index_key once per run and check it does not change mid-run
  (`get_index` re-reads the corpus on every call).
- **Step 6:** the last chunk of a doc can be almost all overlap, so two
  chunks of one doc can take top-k slots. Define hit@k and rank at doc
  level (first rank at which a source doc appears).
- **Steps 10-11:** reproducibility rests on the gitignored LLM cache; query
  embeddings are not cached, and each answer is a single sample from a not
  fully deterministic cloud model. Reports say so and compare differences
  against the CIs.
- **Steps 7-8 and 11:** prompts put untrusted text (chunks, answers) next to
  instructions. Judges must delimit it (`<context>...</context>`) and say it
  is data; Step 11 prompt experiments may do the same.
- **Step 14:** confine config paths (`docs_dir`, `index.dir`) to the repo in
  `resolve_path` before CI reads configs from PRs; never restore an
  `indexes/` CI cache from an untrusted source.

## Definition of done
- [ ] Done when (CLAUDE.md): `answer("How many casual leaves do interns
      get?", config)` returns an answer plus its chunks and scores. Run
      `python -m rag.pipeline --config configs/v0.yaml --question "How many casual leaves do interns get?"`.
      It prints a non-empty answer, `top_k` chunks each with `chunk_id`,
      `doc_id` and a float score in descending order, and `latency_ms`.
- [ ] Running the same command again loads the index from disk (no
      embedding of documents) and the generator call is a cache hit:
      `sqlite3 data/llm_cache.sqlite "SELECT model, cached, seed, reasoning_effort, prompt_tokens, completion_tokens, model_latency_ms, error FROM llm_calls WHERE model = 'gemma4:31b-cloud'"`
      shows one row with `cached = 0` then one with `cached = 1`, and
      `prompt_tokens` below `max_prompt_tokens`.
- [ ] Contract check from Python prints `True`:
      `python -c "from rag.pipeline import load_pipeline_config, answer; o = answer('How many casual leaves do interns get?', load_pipeline_config('configs/v0.yaml')); print(set(o) == {'answer','chunks','latency_ms'} and all(set(c) == {'chunk_id','doc_id','text','score'} for c in o['chunks']))"`
- [ ] Whether `intern_leave_policy` appears in the returned chunks (and at
      which rank) is reported as observed.
- [ ] `ls indexes/` shows exactly one `.faiss` / `.chunks.json` pair, and
      `git status --porcelain` shows neither, nor `data/llm_cache.sqlite`.
- [ ] `grep -rn "evalharness" rag/` prints nothing.
- [ ] `pytest tests/test_03_rag_pipeline_v0.py` passes with no network access.
- [ ] `pytest` passes (whole suite, including the updated Step 2 tests).
- [ ] The Progress section of CLAUDE.md has a Step 3 entry with the real
      Done-when output, the chunk count, and the Step 3 test count.
- [ ] All changes are committed on `feature/03-rag-pipeline-v0`.
