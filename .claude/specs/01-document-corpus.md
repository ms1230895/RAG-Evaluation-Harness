# Spec: Step 01 — Document Corpus

> Written retroactively on 2026-10-15. Step 1 was built (commit `1d19866`)
> before the spec workflow existed. This spec describes what was actually
> built, so `/git-features` can verify and ship it.

## Overview
This step creates the knowledge base the RAG system will answer questions
from: 22 HR policy documents for a fictional company, "Kestrel Labs Pvt. Ltd."
(Bengaluru + Pune offices, bands B1-B6, HR portal "KestrelHub"), stored as one
Markdown file per document in `data/docs/`, plus a small loader. It comes
first in Phase A because everything later depends on it: the pipeline
(Step 3) chunks and indexes these files, and the test set (Step 4) points to
them by doc ID to judge whether retrieval found the right source. A
synthetic corpus lets us plant specific, checkable facts (numbers, dates,
eligibility rules), deliberately similar policies, and deliberate gaps, so
the evaluation has known right answers and known unanswerable questions.

## Concepts to learn
- **Corpus:** the fixed set of documents a RAG system retrieves from. If the
  corpus changes, every earlier evaluation number is no longer comparable, so
  it is treated as frozen once the test set exists.
- **Stable doc ID:** each document's identity is its filename without `.md`
  (`leave_policy.md` -> `leave_policy`). The test set stores the correct
  `source_doc_ids` per query, and retrieval metrics compare these with the
  `doc_id` of retrieved chunks. Renaming a file silently breaks every query
  that points to it.
- **Ground truth without an LLM:** because we wrote the documents, we know
  exactly which doc answers which question. That is what lets hit rate@k and
  MRR (Step 6) be computed with plain Python.
  Example: a query's source is `intern_leave_policy`; the retriever returns
  chunks from [`leave_policy`, `intern_leave_policy`, ...]. Hit@1 = 0,
  hit@3 = 1, reciprocal rank = 1/2 = 0.5.
- **Confusable documents:** pairs of policies on similar topics with
  different facts. They make retrieval non-trivial and create realistic
  hallucination risks (the LLM answers an intern question with the employee
  rule). Pairs in this corpus:
  - `leave_policy` vs `intern_leave_policy` (employees get 8 CL/yr; interns
    get 1 CL per completed month)
  - `expense_reimbursement_policy` vs `travel_policy`
  - `attendance_and_working_hours` vs `remote_hybrid_work_policy`
  - `it_access_policy` vs `device_and_byod_policy` vs
    `information_security_policy`
  - `probation_and_confirmation` vs `resignation_and_exit_policy` (notice
    periods)
- **Deliberate gaps:** topics never covered, used for the unanswerable queries
  in Step 4: ESOPs/stock options, gym or wellness membership, pet policy,
  sabbatical, relocation allowance, office shuttle/transport, and office
  canteen/meals. Note: `travel_policy` does mention *meals* inside the travel
  per diem. Only meals at the office are a true gap, so a vague "meals"
  question is not automatically unanswerable.
- **Garbage text:** leftovers from copy-paste or drafting (`lorem ipsum`,
  `TODO`, HTML tags, smart quotes, replacement characters, control
  characters). Junk gets chunked and embedded like real text, which adds
  noise to retrieval and can leak into answers.

## Depends on
- Step 0: venv, `requirements.txt`, `tests/` package, git repo.

## Interfaces
`rag/loader.py`
```python
DOCS_DIR: Path  # <repo>/data/docs

def load_docs(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """One dict per *.md file, sorted by doc_id:
    {"doc_id": str, "title": str, "text": str}.
    Raises ValueError if a file's first line is not '# Title'."""
```
Example:
```python
from rag.loader import load_docs
docs = load_docs()
docs[0]["doc_id"]  # "attendance_and_working_hours"
```
CLI preview: `python -m rag.loader` prints each doc ID, word count and
title, then the total word count.

This step does not touch the `answer()` contract.

## Data and schema changes
No SQLite schema changes.

New data files: `data/docs/<doc_id>.md`, 22 files, 7,178 words in total.
Format:
- Line 1: `# <Title>`
- Then header lines (effective date, policy owner, who it applies to)
- Then numbered `##` sections with bullet-point rules
- ASCII only, except `₹` and `×`

## Config changes
No config changes.

## Files to create
- `data/docs/*.md` (22 policy documents)
- `rag/loader.py`
- `tests/test_corpus.py`

## Files to change
- `CLAUDE.md`: Step 1 entry in the Progress section.
- `CLAUDE.md`: Step 0 Progress entry now says only `.claude/settings.local.json`
  is ignored; Step 1 gap list says "office canteen/meals".

## New dependencies
No new dependencies.

## Tests
`tests/test_corpus.py`:
- `test_corpus_size`: 15 to 25 documents load.
- `test_doc_ids_unique_and_safe`: IDs are unique and match `[a-z0-9_]+`.
- `test_doc_has_title_and_body` (per doc): non-empty title, at least 150
  words.
- `test_doc_has_no_garbage` (per doc): no garbage markers, no `�`, no
  control characters except newline, no non-ASCII except `₹` and `×`.
- `test_loader_rejects_doc_without_title`: a file without `# Title` raises
  `ValueError`.

No LLM calls are involved.

## Rules for implementation
- Plain Python and only libraries approved in CLAUDE.md
- No LangChain; no RAGAS or DeepEval (except the optional final cross-check)
- `rag/` must never import from `evalharness/`; the harness only calls `answer()`
- All tunable values come from YAML config, with no hardcoded magic numbers
- SQLite via the sqlite3 module with parameterised queries only
- All LLM calls go through `evalharness/llm.py` so they are cached
- API keys only from `.env`, never in code, logs or commits
- The held-out split is never used before Step 12
- Never invent, estimate or adjust metric numbers; report only what a
  stored run produced
- Human labels (calibration and review) come only from Gaurav
- Code is simple, readable and commented

Step-specific:
- Never rename a file in `data/docs/` once Step 4 queries reference it.
- After the test set exists, change document facts only if the affected
  reference answers are updated in the same commit.
- Gap topics must stay absent. Do not add documents that cover them.
- The company is fictional. Do not use real company names or real people.

## Decisions for Gaurav
All resolved on 2026-10-15:
1. ~~CLAUDE.md Progress lists "canteen/meals" as a gap, but `travel_policy`
   mentions meals inside the travel per diem.~~ **Resolved:** the wording is
   now "office canteen/meals". Step 4 unanswerable queries must ask about
   meals at the office, not meals in general.
2. ~~CLAUDE.md's Step 0 Progress entry says `.claude/` is gitignored.~~
   **Resolved:** the entry now says only `.claude/settings.local.json` is
   ignored.

## Definition of done
- [ ] Every doc loads with an ID and title: `python -m rag.loader` lists
      22 documents with no error.
- [ ] No garbage text: `pytest tests/test_corpus.py` passes.
- [ ] Gap topics are absent:
      `grep -rniwE "esops?|stock options?|gym|wellness|pets?|sabbatical|relocation|shuttle|canteen" data/docs`
      returns nothing.
- [ ] `pytest` passes (whole suite).
- [ ] The Progress section of CLAUDE.md has a Step 1 entry.
- [ ] All changes are committed on `feature/01-document-corpus`.
