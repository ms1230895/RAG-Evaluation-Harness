"""Keyword search (BM25) and rank fusion (RRF) for the Step 11 retrieval experiments.

BM25 scores a chunk by how often the query's words appear in it, weighting
rare words more and not letting repeats count forever (k1) or long chunks win
just by being long (b). It finds exact words ("B3", "KestrelHub") that an
embedding can blur; the embedding finds paraphrases that BM25 misses.

RRF (reciprocal rank fusion) merges two rankings without comparing their
incompatible scores: a chunk's fused score is the sum, over the rankings it
appears in, of 1 / (rrf_k + rank). Example with rrf_k 60: a chunk that is
rank 1 in one list and rank 3 in the other scores 1/61 + 1/63 = 0.0323; a
chunk that is rank 2 in only one list scores 1/62 = 0.0161.

A "row" below is a position in the chunk list (the same order as the FAISS
index rows), so results can be mapped back to chunks.
"""

import hashlib
import re

from rank_bm25 import BM25Okapi

# A token is a run of letters or digits (any alphabet); "_" and punctuation split.
_TOKEN = re.compile(r"[^\W_]+")

# BM25 objects built in this process, keyed by a hash of the chunk texts and k1 / b,
# so a run of many questions builds the statistics once.
_BM25: dict[str, BM25Okapi] = {}


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens: "B3: 8 CL/yr" -> ["b3", "8", "cl", "yr"]."""
    return _TOKEN.findall(text.lower())


def _get_bm25(chunks: list[dict], bm25_config: dict) -> BM25Okapi:
    """BM25 statistics over the chunk texts, built once per (chunks, k1, b)."""
    digest = hashlib.sha256()
    digest.update(f"{bm25_config['k1']}|{bm25_config['b']}".encode("utf-8"))
    for chunk in chunks:
        digest.update(b"\x00" + chunk["chunk_id"].encode("utf-8"))
        digest.update(b"\x00" + chunk["text"].encode("utf-8"))
    key = digest.hexdigest()
    if key not in _BM25:
        corpus = [tokenize(chunk["text"]) for chunk in chunks]
        _BM25[key] = BM25Okapi(corpus, k1=bm25_config["k1"], b=bm25_config["b"])
    return _BM25[key]


def bm25_search(question: str, chunks: list[dict], k: int,
                bm25_config: dict) -> list[tuple[int, float]]:
    """Top-k (chunk_row, bm25_score) in rank order, best first.

    Chunks whose score is 0 or below are never returned, so a question that
    shares no word with any chunk gives [] instead of an arbitrary list.
    Ties break on the lower chunk row, so the result is deterministic.
    """
    scores = _get_bm25(chunks, bm25_config).get_scores(tokenize(question))
    order = sorted(range(len(scores)), key=lambda row: (-scores[row], row))
    return [(row, float(scores[row])) for row in order if scores[row] > 0][:k]


def rrf_fuse(rankings: list[list[int]], rrf_k: int) -> list[tuple[int, float]]:
    """Fuse ranked lists of chunk rows; return (row, fused_score), best first.

    Ties break on the best rank in the first ranking (a row missing from it
    comes after rows that are in it), then on the lower row.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            scores[row] = scores.get(row, 0.0) + 1.0 / (rrf_k + rank)
    first_rank = {row: rank for rank, row in enumerate(rankings[0], start=1)} if rankings else {}
    absent = len(scores) + 1   # larger than any real rank
    order = sorted(scores, key=lambda row: (-scores[row], first_rank.get(row, absent), row))
    return [(row, scores[row]) for row in order]
