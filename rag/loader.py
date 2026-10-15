"""Load the HR document corpus from data/docs/.

Each document is one Markdown file:
- the filename without ".md" is the stable doc ID (e.g. "leave_policy"),
- the first line is the title, written as "# Title".

The test set (Step 4) refers to documents by these IDs, so a file must never
be renamed once queries point to it.
"""

from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


def load_docs(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """Return one dict per document: {"doc_id", "title", "text"}, sorted by doc_id."""
    docs = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        first_line = lines[0] if lines else ""
        if not first_line.startswith("# "):
            raise ValueError(f"{path.name}: first line must be a title like '# Leave Policy'")
        docs.append({
            "doc_id": path.stem,
            "title": first_line[2:].strip(),
            "text": text,
        })
    return docs


if __name__ == "__main__":
    # Quick look at the corpus: python -m rag.loader
    docs = load_docs()
    print(f"Loaded {len(docs)} documents from {DOCS_DIR}\n")
    for doc in docs:
        words = len(doc["text"].split())
        print(f"{doc['doc_id']:<30} {words:>4} words   {doc['title']}")
    total = sum(len(doc["text"].split()) for doc in docs)
    print(f"\nTotal: {total} words")
