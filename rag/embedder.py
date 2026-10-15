"""Turn text into L2-normalised embedding vectors.

The model (nomic-embed-text in v0) runs locally in Ollama and is reached
through llm.client.embed. nomic was trained with task prefixes:
"search_document: " before passages and "search_query: " before questions;
both come from the config.

Every vector is scaled to length 1 here (we do not rely on the server doing
it), so the dot product of two vectors equals their cosine similarity, which
is what the FAISS inner-product index computes.
"""

import numpy as np

from llm.client import embed


def _embed_texts(texts: list[str], model: str) -> list[list[float]]:
    """The only function in rag/ that calls the embedding endpoint (tests replace it)."""
    return embed(texts, model)


def _normalise(vectors) -> np.ndarray:
    """Return a float32 copy with every row scaled to length 1."""
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding model returned a zero vector; it cannot be normalised")
    return array / norms


def embed_documents(texts: list[str], embedding_config: dict) -> np.ndarray:
    """Embed passages (with doc_prefix) in batches of batch_size.

    Returns a float32 array of shape (len(texts), dim), rows L2-normalised,
    in the same order as `texts`.
    """
    prefixed = [embedding_config["doc_prefix"] + text for text in texts]
    batch_size = embedding_config["batch_size"]
    vectors = []
    for i in range(0, len(prefixed), batch_size):
        vectors.extend(_embed_texts(prefixed[i:i + batch_size], embedding_config["model"]))
    return _normalise(vectors)


def embed_query(question: str, embedding_config: dict) -> np.ndarray:
    """Embed one question (with query_prefix). Returns float32 shape (dim,), L2-normalised."""
    vectors = _embed_texts([embedding_config["query_prefix"] + question],
                           embedding_config["model"])
    return _normalise(vectors)[0]
