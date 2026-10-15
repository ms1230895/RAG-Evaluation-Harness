"""Build, save and load the FAISS index of chunk vectors.

The index is an IndexFlatIP: it stores every vector and compares the query
with all of them (exact search). "IP" = inner product, which equals cosine
similarity because every vector is L2-normalised (rag/embedder.py).

Embedding every chunk is the slow part, so a built index is saved as a pair of
files in index.dir:
    <index_key>.faiss         the vectors (row i = chunk i)
    <index_key>.chunks.json   {"meta": {...}, "chunks": [{"chunk_id", "doc_id", "text"}]}
index_key is a SHA-256 of everything that changes the vectors: each doc's ID
and text, chunk_size, chunk_overlap, CHUNKER_VERSION, the embedding model,
doc_prefix and the embedding endpoint (base_url from configs/llm.yaml).
Changing top_k or the prompt reuses the same files; changing chunk size or a
document builds a new pair, so a stale index is never loaded.
(The embedding model's build digest is not in the key yet; Step 5 handles
model identity.)

After an index is loaded from disk, one warm-up query is embedded so the
embedding model is loaded in Ollama before any timed retrieval; otherwise the
first question of a run would pay the model's load time in its latency_ms.
(After a build the model is already warm.)

describe_index(config) returns the index_key and the saved meta, so a run
(Step 5) can record exactly which corpus and index it used.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from llm.client import load_config as load_llm_config
from rag.chunker import CHUNKER_VERSION, chunk_corpus
from rag.embedder import embed_documents, embed_query
from rag.loader import load_docs

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)

# Text embedded once after loading an index from disk, only to make Ollama
# load the embedding model before the first timed retrieval. Its vector is unused.
WARM_UP_TEXT = "warm-up"

# Indexes already loaded in this process, by index_key, so a run of many
# questions loads the files once.
_LOADED: dict[str, tuple[faiss.Index, list[dict]]] = {}


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path: relative paths are relative to the repo root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def index_key(docs: list[dict], config: dict) -> str:
    """SHA-256 hex of the inputs that change the vectors, and nothing else."""
    key_input = json.dumps(
        {
            "docs": [[doc["doc_id"], doc["text"]] for doc in docs],
            "chunk_size": config["chunking"]["chunk_size"],
            "chunk_overlap": config["chunking"]["chunk_overlap"],
            "chunker_version": CHUNKER_VERSION,
            "embedding_model": config["embedding"]["model"],
            "doc_prefix": config["embedding"]["doc_prefix"],
            # Same model tag on a different server may be a different model.
            "embedding_base_url": load_llm_config()["base_url"],
        },
        sort_keys=True,
    )
    return hashlib.sha256(key_input.encode("utf-8")).hexdigest()


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Exact inner-product index over already-normalised float32 vectors (n, dim)."""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def _paths(index_dir: Path, key: str) -> tuple[Path, Path]:
    """The .faiss and .chunks.json paths for one index_key."""
    return index_dir / f"{key}.faiss", index_dir / f"{key}.chunks.json"


def save_index(index: faiss.Index, chunks: list[dict], index_dir: Path, key: str,
               meta: dict) -> None:
    """Write the index file and the chunk list (same order as the index rows)."""
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss_path, chunks_path = _paths(index_dir, key)
    faiss.write_index(index, str(faiss_path))
    chunks_path.write_text(json.dumps({"meta": meta, "chunks": chunks}, indent=1),
                           encoding="utf-8")


def load_index(index_dir: Path, key: str) -> tuple[faiss.Index, list[dict]] | None:
    """Load a saved (index, chunks) pair, or None if either file is missing."""
    faiss_path, chunks_path = _paths(index_dir, key)
    if not (faiss_path.exists() and chunks_path.exists()):
        return None
    index = faiss.read_index(str(faiss_path))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))["chunks"]
    if index.ntotal != len(chunks):
        raise ValueError(f"{faiss_path.name} has {index.ntotal} vectors but "
                         f"{chunks_path.name} has {len(chunks)} chunks; delete both and rebuild")
    return index, chunks


def _build_and_save(docs: list[dict], key: str, config: dict,
                    index_dir: Path) -> tuple[faiss.Index, list[dict]]:
    """Chunk, embed, index and save the corpus; return (index, chunks)."""
    chunking, embedding = config["chunking"], config["embedding"]
    chunks = chunk_corpus(docs, chunking["chunk_size"], chunking["chunk_overlap"])
    if not chunks:
        raise ValueError(f"no chunks produced from corpus.docs_dir="
                         f"{config['corpus']['docs_dir']!r}; is the folder empty?")
    vectors = embed_documents([chunk["text"] for chunk in chunks], embedding)
    index = build_index(vectors)
    meta = {
        "chunk_size": chunking["chunk_size"],
        "chunk_overlap": chunking["chunk_overlap"],
        "chunker_version": CHUNKER_VERSION,
        "embedding_model": embedding["model"],
        "doc_prefix": embedding["doc_prefix"],
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "dim": int(vectors.shape[1]),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_index(index, chunks, index_dir, key, meta)
    return index, chunks


def get_index(config: dict) -> tuple[faiss.Index, list[dict]]:
    """Return (index, chunks) for this config: from memory, else disk, else build."""
    start = time.perf_counter()
    docs = load_docs(resolve_path(config["corpus"]["docs_dir"]))
    key = index_key(docs, config)
    index_dir = resolve_path(config["index"]["dir"])

    if key in _LOADED:
        how = "reused from memory"
        index, chunks = _LOADED[key]
    elif (loaded := load_index(index_dir, key)) is not None:
        how = "loaded from disk"
        index, chunks = loaded
        embed_query(WARM_UP_TEXT, config["embedding"])   # load the embedding model before timing
    else:
        how = "built"
        index, chunks = _build_and_save(docs, key, config, index_dir)
    _LOADED[key] = (index, chunks)

    logger.info("index %s %s: %d chunks (%.1f ms)", key[:12], how, len(chunks),
                (time.perf_counter() - start) * 1000)
    return index, chunks


def describe_index(config: dict) -> dict:
    """index_key plus the saved meta of this config's index (built first if needed).

    For run metadata (Step 5): config_json alone does not say which corpus
    version was indexed, because the documents are not part of the config.
    """
    get_index(config)
    docs = load_docs(resolve_path(config["corpus"]["docs_dir"]))
    key = index_key(docs, config)
    _, chunks_path = _paths(resolve_path(config["index"]["dir"]), key)
    meta = json.loads(chunks_path.read_text(encoding="utf-8"))["meta"]
    return {"index_key": key, **meta}
