"""Split documents into overlapping word windows (chunks).

chunk_size words per chunk; consecutive chunks share chunk_overlap words, so
a fact that sits on a boundary still appears whole in at least one chunk.
The window start moves by step = chunk_size - chunk_overlap.

Example: 10 words, chunk_size=4, chunk_overlap=1 (step 3) gives windows
[0-3] [3-6] [6-9]: the window starting at 6 reaches the end, so we stop.

Each chunk's text is sliced from the original document between its first and
last word, so line breaks and Markdown lists survive. Chunks never cross
documents, so every chunk has exactly one doc_id.
"""

import re

WORD = re.compile(r"\S+")   # a "word" is any run of non-whitespace characters

# Bump this whenever the chunking rules change (what a "word" is, the window
# or tail rule). It is part of the index key (rag/index.py), so an index built
# by older chunking code is never reused.
CHUNKER_VERSION = "1"


def chunk_document(doc: dict, chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Split one doc ({"doc_id", "title", "text"}) into overlapping word windows.

    Returns [{"chunk_id": "<doc_id>#<nnn>", "doc_id", "text"}] with nnn = 000,
    001, ... A document with no words gives an empty list.
    """
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f"chunk_size must be an integer >= 1, got {chunk_size!r}")
    if not isinstance(chunk_overlap, int) or not 0 <= chunk_overlap < chunk_size:
        raise ValueError(f"chunk_overlap must be an integer with 0 <= overlap < chunk_size "
                         f"({chunk_size}), got {chunk_overlap!r}")

    text = doc["text"]
    words = list(WORD.finditer(text))   # match objects know their start/end positions
    step = chunk_size - chunk_overlap

    chunks = []
    start = 0
    while start < len(words):
        window = words[start:start + chunk_size]
        chunks.append({
            "chunk_id": f"{doc['doc_id']}#{len(chunks):03d}",
            "doc_id": doc["doc_id"],
            # Slice the original text from the first word's start to the last
            # word's end, keeping the whitespace (line breaks) in between.
            "text": text[window[0].start():window[-1].end()],
        })
        if start + chunk_size >= len(words):
            break   # this window reached the last word
        start += step
    return chunks


def chunk_corpus(docs: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    """chunk_document over every doc, in doc order, as one flat list."""
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return chunks
