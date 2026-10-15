# RAG Evaluation Harness

A reusable harness that scores RAG answers for faithfulness and relevance on a fixed
100-query test set, tracks results across runs, and flags answers for human review.

Work in progress. See `CLAUDE.md` for the build plan and progress.

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```
