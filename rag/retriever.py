"""Retrieval: top-k vector search (v0), plus BM25 and hybrid search (Step 11).

`retrieve` is the v0 vector search and is unchanged. `retrieve_with_method`
picks vector / bm25 / hybrid from the config.
"""

import faiss

from rag.bm25 import bm25_search, rrf_fuse
from rag.embedder import embed_query


def retrieve(question: str, index: faiss.Index, chunks: list[dict], top_k: int,
             embedding_config: dict) -> list[dict]:
    """Return the top_k chunks in rank order (highest score first).

    Each item is {"chunk_id", "doc_id", "text", "score"}; score is the cosine
    similarity (inner product of normalised vectors) as a plain Python float,
    never rounded or re-scaled.
    """
    if top_k > index.ntotal:
        raise ValueError(f"top_k={top_k} but the index has only {index.ntotal} chunks")
    query = embed_query(question, embedding_config)
    # FAISS searches a batch of queries: shape (1, dim) in, (1, top_k) out.
    scores, ids = index.search(query.reshape(1, -1), top_k)
    results = []
    for score, row in zip(scores[0], ids[0]):
        chunk = chunks[row]
        results.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "text": chunk["text"],
            "score": float(score),
        })
    return results


def _chunk_result(chunk: dict, score: float) -> dict:
    """One result in the contract's chunk shape."""
    return {"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
            "text": chunk["text"], "score": float(score)}


def stage_size(config: dict) -> int:
    """How many chunks retrieval returns: top_k, or rerank.pool_k when reranking is on."""
    retrieval = config["retrieval"]
    return retrieval["rerank"]["pool_k"] if retrieval["rerank"]["enabled"] else retrieval["top_k"]


def retrieve_with_method(question: str, index: faiss.Index, chunks: list[dict],
                         config: dict) -> list[dict]:
    """Search with retrieval.method and return up to stage_size(config) chunks, best first.

    vector: cosine similarity, exactly `retrieve`.
    bm25:   BM25 score; fewer chunks come back if few chunks share a query word.
    hybrid: the top hybrid.pool_k of vector and of BM25, fused with RRF; the
            score is the fused RRF score.
    The pool is capped at the number of chunks. `question` is the search query
    (the original question, or its rewrite when query_rewrite is on).
    """
    retrieval = config["retrieval"]
    method = retrieval["method"]
    if retrieval["top_k"] > len(chunks):
        raise ValueError(f"top_k={retrieval['top_k']} but the index has only {len(chunks)} chunks")
    size = min(stage_size(config), len(chunks))   # a rerank pool larger than the corpus is capped
    embedding_config = config["embedding"]

    if method == "vector":
        return retrieve(question, index, chunks, size, embedding_config)

    if method == "bm25":
        return [_chunk_result(chunks[row], score)
                for row, score in bm25_search(question, chunks, size, retrieval["bm25"])]

    if method == "hybrid":
        hybrid = retrieval["hybrid"]
        pool = min(hybrid["pool_k"], len(chunks))
        row_of = {chunk["chunk_id"]: row for row, chunk in enumerate(chunks)}
        vector_rows = [row_of[hit["chunk_id"]]
                       for hit in retrieve(question, index, chunks, pool, embedding_config)]
        bm25_rows = [row for row, _ in bm25_search(question, chunks, pool, retrieval["bm25"])]
        fused = rrf_fuse([vector_rows, bm25_rows], hybrid["rrf_k"])
        return [_chunk_result(chunks[row], score) for row, score in fused[:size]]

    raise ValueError(f"unknown retrieval.method {method!r}")
