"""Cross-encoder reranking through a child process (Step 11).

A bi-encoder (our embedder) turns the question and each chunk into vectors
separately, which is fast. A cross-encoder reads the pair (question, chunk)
together and outputs one relevance score: more accurate, much slower, so it
only reorders a short first-stage pool (retrieval.rerank.pool_k) and the top
top_k are kept.

The model runs in rag/rerank_worker.py, a separate Python process, because
faiss-cpu and torch cannot share a process on this macOS setup. THIS file must
never import torch or sentence_transformers. The worker is started on first
use, kept for the life of the process, and closed at exit. Its load and
warm-up are not part of any query's latency.

There is no silent fallback: if the worker dies, times out or returns the
wrong number of scores, RerankError is raised and the item counts as failed.
"""

import atexit
import json
import math
import queue
import subprocess
import sys
import tempfile
import threading

STDERR_TAIL_CHARS = 500   # how much of the worker's stderr goes into an error message


class RerankError(Exception):
    """The reranking worker failed or returned something unusable."""


class WorkerProcess:
    """One running rerank_worker process; ask() sends a request and waits for one reply."""

    def __init__(self, model: str, revision: str | None, timeout_s: int):
        self._stderr = tempfile.TemporaryFile(mode="w+")   # kept to explain a crash
        self._process = subprocess.Popen(
            [sys.executable, "-m", "rag.rerank_worker", model, revision or ""],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            text=True, bufsize=1)
        self._lines: queue.Queue = queue.Queue()
        # A reader thread lets ask() wait with a timeout; readline() itself cannot.
        threading.Thread(target=self._read_lines, daemon=True).start()
        try:
            ready = self._next_message(timeout_s, "start")
            if ready.get("ready") is not True:
                raise RerankError(f"worker did not report ready: {ready!r}")
        except RerankError:
            self.close()   # never leave a half-started process behind
            raise

    def _read_lines(self) -> None:
        for line in self._process.stdout:
            self._lines.put(line)
        self._lines.put(None)   # end of output: the process exited

    def _stderr_tail(self) -> str:
        self._stderr.seek(0)
        return self._stderr.read()[-STDERR_TAIL_CHARS:].strip()

    def _next_message(self, timeout_s: int, what: str) -> dict:
        try:
            line = self._lines.get(timeout=timeout_s)
        except queue.Empty:
            self.close()
            raise RerankError(f"rerank worker gave no reply to {what} within {timeout_s}s") from None
        if line is None:
            raise RerankError(f"rerank worker exited during {what}: {self._stderr_tail()}")
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            raise RerankError(f"rerank worker sent a non-JSON line: {line[:100]!r}") from None
        if "error" in message:
            raise RerankError(f"rerank worker error: {message['error']}")
        return message

    def ask(self, request: dict, timeout_s: int) -> dict:
        """Send one JSON request and return the worker's one JSON reply."""
        try:
            self._process.stdin.write(json.dumps(request) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError):
            raise RerankError(f"rerank worker is not running: {self._stderr_tail()}") from None
        return self._next_message(timeout_s, "a request")

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()
        self._stderr.close()


# Workers started in this process, by (model, revision).
_WORKERS: dict[tuple, WorkerProcess] = {}


def _start_worker(rerank_config: dict) -> WorkerProcess:
    """Start a worker (tests replace this with a fake)."""
    return WorkerProcess(rerank_config["model"], rerank_config["revision"],
                         rerank_config["worker_timeout_s"])


def _get_worker(rerank_config: dict):
    key = (rerank_config["model"], rerank_config["revision"])
    if key not in _WORKERS:
        _WORKERS[key] = _start_worker(rerank_config)
    return _WORKERS[key]


def close_workers() -> None:
    """Stop every worker started by this process."""
    while _WORKERS:
        _, worker = _WORKERS.popitem()
        worker.close()


atexit.register(close_workers)


def rerank(question: str, candidates: list[dict], top_k: int, rerank_config: dict) -> list[dict]:
    """Reorder `candidates` by cross-encoder score and keep the best top_k.

    Each returned chunk is a copy with `score` replaced by the cross-encoder
    score. Ties keep the first-stage order. Raises RerankError on any worker
    failure; the broken worker is dropped so the next call starts a fresh one.
    """
    worker = _get_worker(rerank_config)
    request = {"question": question, "texts": [chunk["text"] for chunk in candidates]}
    try:
        reply = worker.ask(request, rerank_config["worker_timeout_s"])
    except RerankError:
        _WORKERS.pop((rerank_config["model"], rerank_config["revision"]), None)
        worker.close()
        raise
    scores = reply.get("scores")
    if (not isinstance(scores, list) or len(scores) != len(candidates)
            or not all(isinstance(s, (int, float)) and math.isfinite(s) for s in scores)):
        raise RerankError(f"expected {len(candidates)} finite scores, got {scores!r}")
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    return [{**candidates[i], "score": float(scores[i])} for i in order[:top_k]]
