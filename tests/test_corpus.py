"""Step 1 checks: every doc loads, has a stable ID, and contains no garbage text."""

import re

import pytest

from rag.loader import load_docs

DOCS = load_docs()

# Text that should never appear in a finished document.
GARBAGE_MARKERS = ["lorem", "ipsum", "todo", "tbd", "xxx", "[insert", "placeholder", "<html", "&nbsp;"]

# The only non-ASCII characters we allow: rupee sign and multiplication sign.
ALLOWED_NON_ASCII = {"₹", "×"}


def test_corpus_size():
    assert 15 <= len(DOCS) <= 25


def test_doc_ids_unique_and_safe():
    ids = [doc["doc_id"] for doc in DOCS]
    assert len(ids) == len(set(ids))
    for doc_id in ids:
        # lowercase letters, digits and underscores only, so IDs are safe in JSON, SQL and URLs
        assert re.fullmatch(r"[a-z0-9_]+", doc_id), doc_id


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d["doc_id"])
def test_doc_has_title_and_body(doc):
    assert doc["title"]
    # a real policy, not a stub
    assert len(doc["text"].split()) >= 150


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d["doc_id"])
def test_doc_has_no_garbage(doc):
    text = doc["text"]
    lowered = text.lower()
    for marker in GARBAGE_MARKERS:
        assert marker not in lowered, f"{doc['doc_id']} contains '{marker}'"
    # no broken encoding (replacement character)
    assert "�" not in text
    for ch in text:
        # no control characters except newline
        if ord(ch) < 32:
            assert ch == "\n", f"{doc['doc_id']} has control char {ord(ch)}"
        # no unexpected non-ASCII characters (smart quotes, odd symbols from copy-paste)
        if ord(ch) > 127:
            assert ch in ALLOWED_NON_ASCII, f"{doc['doc_id']} has unexpected char {ch!r}"


def test_loader_rejects_doc_without_title(tmp_path):
    (tmp_path / "bad_doc.md").write_text("No title line here.\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_docs(tmp_path)
