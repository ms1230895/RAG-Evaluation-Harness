"""Cross-encoder reranking worker: runs in its OWN process (Step 11).

faiss-cpu and torch each bundle their own libomp on macOS and cannot share a
process, so rag/reranker.py never imports torch. It starts this module as a
child process instead and talks to it in JSON lines:

    parent -> worker (stdin):   {"question": "...", "texts": ["...", "..."]}
    worker -> parent (stdout):  {"ready": true}                  once, after the model is loaded
                                {"scores": [0.12, -3.4]}          one per request, one score per text
                                {"error": "..."}                  if a request failed

Start it by hand to check the model loads:
    python -m rag.rerank_worker cross-encoder/ms-marco-MiniLM-L-6-v2 ""
(second argument: the pinned Hugging Face revision, or "" for the default branch)
"""

import json
import sys

WARM_UP_PAIR = ("warm-up question", "warm-up passage")


def main() -> None:
    model_name, revision = sys.argv[1], sys.argv[2]

    # Libraries print progress and warnings; keep stdout for the protocol only.
    protocol_out = sys.stdout
    sys.stdout = sys.stderr

    def send(message: dict) -> None:
        protocol_out.write(json.dumps(message) + "\n")
        protocol_out.flush()

    from sentence_transformers import CrossEncoder   # the heavy import lives only here

    model = CrossEncoder(model_name, revision=revision or None)
    model.predict([WARM_UP_PAIR])   # load weights into memory before the first timed call
    send({"ready": True})

    for line in sys.stdin:
        try:
            request = json.loads(line)
            pairs = [(request["question"], text) for text in request["texts"]]
            scores = model.predict(pairs) if pairs else []
            send({"scores": [float(score) for score in scores]})
        except Exception as error:   # report it to the parent instead of dying silently
            send({"error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
