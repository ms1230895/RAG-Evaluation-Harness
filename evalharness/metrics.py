"""Pure metric formulas (Step 6): retrieval metrics, Wilson interval and summaries.

No I/O and no database: every function takes plain lists and returns numbers,
so each one can be checked by hand. Steps 7-13 reuse the summaries and the
Wilson interval.

Retrieval metrics match on doc_id, not chunk_id, because the test set names
source documents. Rank is the chunk's position in the retrieved list (1-based,
in the stored order), so k means the same thing as the pipeline's top_k.
Example: chunk doc IDs [a, a, b, c] with source [b] -> first source rank 3,
hit@1 = 0, hit@4 = 1, reciprocal rank = 1/3.
"""

import math

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------- retrieval

def _check_sources(source_doc_ids: list[str]) -> None:
    """Retrieval metrics are undefined for unanswerable queries (no source docs)."""
    if not source_doc_ids:
        raise ValueError("no source doc IDs (unanswerable query): retrieval metric is undefined")


def first_source_rank(chunk_doc_ids: list[str], source_doc_ids: list[str]) -> int | None:
    """1-based position of the first chunk whose doc_id is a source doc; None if there is none."""
    _check_sources(source_doc_ids)
    sources = set(source_doc_ids)
    for rank, doc_id in enumerate(chunk_doc_ids, start=1):
        if doc_id in sources:
            return rank
    return None


def hit_at_k(chunk_doc_ids: list[str], source_doc_ids: list[str], k: int) -> int:
    """1 if any source doc appears in the first k chunks, else 0.

    Raises ValueError if k < 1 or k > len(chunk_doc_ids): with fewer than k
    chunks we cannot tell what rank k would have held.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > len(chunk_doc_ids):
        raise ValueError(f"k={k} but only {len(chunk_doc_ids)} chunks were retrieved")
    rank = first_source_rank(chunk_doc_ids, source_doc_ids)
    return int(rank is not None and rank <= k)


def reciprocal_rank(chunk_doc_ids: list[str], source_doc_ids: list[str]) -> float:
    """1 / rank of the first source chunk, or 0.0 if no source doc was retrieved."""
    rank = first_source_rank(chunk_doc_ids, source_doc_ids)
    return 0.0 if rank is None else 1.0 / rank


def doc_recall(chunk_doc_ids: list[str], source_doc_ids: list[str]) -> float:
    """Share of distinct source docs that appear anywhere in the retrieved chunks.

    Example: chunks [a, a, c, d], sources [a, b] -> 1 of 2 found -> 0.5.
    """
    _check_sources(source_doc_ids)
    sources = set(source_doc_ids)
    return len(sources & set(chunk_doc_ids)) / len(sources)


# ---------------------------------------------------------------- uncertainty

def wilson_interval(successes: int, n: int, confidence: float) -> tuple[float, float]:
    """Wilson score interval for a rate of successes / n.

        center = (p + z^2/(2n)) / (1 + z^2/n)
        half   = z * sqrt(p(1-p)/n + z^2/(4n^2)) / (1 + z^2/n)

    z = 1.96 for 95%. Unlike p +/- z*sqrt(p(1-p)/n) it stays sensible at
    p = 0 or 1 and small n: 12/12 gives [0.758, 1.0], not [1, 1].
    Both bounds are clamped to [0, 1] (float error can give 1.0000000002).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in 0..{n}, got {successes}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    z = float(norm.ppf(1 - (1 - confidence) / 2))   # plain float, not numpy.float64
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


# ---------------------------------------------------------------- summaries
# In every summary None means "not applicable" (unanswerable, failed output,
# fewer than k chunks). It is counted in n_na and left out of n, never read as 0.

def _split_na(values: list[float | None]) -> tuple[list[float], int]:
    """(the real values, how many were None)."""
    real = [v for v in values if v is not None]
    return real, len(values) - len(real)


def rate_summary(values: list[float | None], confidence: float) -> dict:
    """Summary of 0/1 values: {"n", "n_na", "successes", "rate", "ci_low", "ci_high"}.

    rate and the CI are None when n == 0.
    """
    real, n_na = _split_na(values)
    if any(v not in (0, 1) for v in real):
        raise ValueError("rate_summary expects only 0/1 values (or None)")
    n, successes = len(real), int(sum(real))
    if n == 0:
        return {"n": 0, "n_na": n_na, "successes": 0, "rate": None,
                "ci_low": None, "ci_high": None}
    ci_low, ci_high = wilson_interval(successes, n, confidence)
    return {"n": n, "n_na": n_na, "successes": successes, "rate": successes / n,
            "ci_low": ci_low, "ci_high": ci_high}


def mean_summary(values: list[float | None]) -> dict:
    """{"n", "n_na", "mean"}; mean is None when n == 0. Used for MRR and doc recall."""
    real, n_na = _split_na(values)
    return {"n": len(real), "n_na": n_na, "mean": sum(real) / len(real) if real else None}


def percentile_summary(values: list[float | None]) -> dict:
    """{"n", "n_na", "p50", "p95", "max"} (numpy.percentile, linear); None fields when n == 0."""
    real, n_na = _split_na(values)
    if not real:
        return {"n": 0, "n_na": n_na, "p50": None, "p95": None, "max": None}
    return {"n": len(real), "n_na": n_na,
            "p50": float(np.percentile(real, 50)), "p95": float(np.percentile(real, 95)),
            "max": float(max(real))}
