"""Query rewriting (Step 11 experiment): turn the user's question into a search query.

One extra LLM call, through llm/client.py like the generator (cached, retried,
logged). Only retrieval uses the rewrite; the generator and the reranker still
see the user's original question. It uses the generator's model, digest,
temperature and seed (Decision 8), and the prompt in query_rewrite.prompt_template.

There is no fallback: if the call fails or the reply is blank, the error
surfaces and the item counts as failed. Quietly searching with the original
question would make a "rewrite" run measure something else.
"""

from llm.client import generate

QUOTES = ('"', "'")


class RewriteError(Exception):
    """The model's rewrite was blank after cleaning."""


def clean_rewrite(text: str) -> str:
    """Strip whitespace and one pair of surrounding quotes from the model's reply."""
    text = text.strip()
    if len(text) >= 2 and text[0] in QUOTES and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text


def rewrite_query(question: str, config: dict) -> tuple[str, float]:
    """Return (search_query, model_latency_ms) for `question`.

    model_latency_ms is the model's own request time (reported from the
    original call on a cache hit), like the generator's, so the pipeline's
    latency does not look faster on a cached re-run.
    """
    gen = config["generation"]
    template = config["query_rewrite"]["prompt_template"]
    result = generate(template.format(question=question), gen["model"], gen["temperature"],
                      model_digest=gen["model_digest"], seed=gen["seed"],
                      reasoning_effort=gen["reasoning_effort"])
    query = clean_rewrite(result.text)
    if not query:
        raise RewriteError("the model returned a blank query rewrite")
    return query, result.model_latency_ms
