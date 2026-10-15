"""Top-k vector retrieval: embed the question, return the k closest chunks."""

import faiss

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
