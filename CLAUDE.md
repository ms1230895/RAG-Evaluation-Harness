# Project: Evaluation Harness for RAG Output

## Who I am and how I want to work
I'm Gaurav, a student comfortable with Python, ML, FastAPI, SQLite and basic RAG (LangChain, FAISS, Hugging Face embeddings). I want to build this project FROM THE BASICS and understand every part well enough to explain it in an interview.

Working rules (follow these strictly):
1. Build ONE step at a time, in the order given under "Build sequence". Never jump ahead or combine steps.
2. Before writing code for a step, explain in plain words: what the step does, why it is needed, the key concept behind it, and the design choices. Then write the code.
3. Keep code simple, readable and commented. Prefer plain Python over clever abstractions. Do not add libraries beyond the list below without asking me.
4. After each step: run it, show me the output, run the tests, summarize what was built, confirm the "Done when" check, update the "Progress" section of CLAUDE.md, make a git commit, then STOP and wait for my go-ahead.
5. When a decision is mine (LLM provider, models, thresholds, corpus content), ask me instead of guessing.
6. Honesty: never invent, estimate or adjust metric numbers. Every number reported must come from an actual run stored in the database. If results are weak, noisy or not statistically meaningful, say so plainly.

## Environment
- macOS, VS Code, Python 3.13 venv. If any package fails to install on 3.13, tell me and we switch to 3.12.
- API keys live in `.env`, loaded with python-dotenv.
- `.gitignore` must cover: `.env`, the venv, caches, the SQLite database, FAISS index files.

## Goal
Build a reusable evaluation harness that scores RAG answers for faithfulness and relevance on a fixed 100-query test set, use it to measurably reduce ungrounded (hallucinated) answers through retrieval and prompt tuning, automate regression runs that flag a subset of answers for human review, and show trends in a Streamlit + Plotly dashboard.

My CV draft says the ungrounded rate dropped from 25% to 12% and that 10-15% of answers get flagged for review. Treat these as hypotheses, NOT targets. The real numbers are whatever we measure, and I will update my CV to match them.

## Core concepts
- A RAG system can fail in two places: retrieval (the right chunk was never fetched) or generation (the right chunk was fetched but the LLM ignored it or added unsupported claims).
- Faithfulness (groundedness): is every claim in the answer supported by the retrieved context? Measured by splitting the answer into atomic claims and checking each one against the context. Score = supported claims / total claims.
- Relevance: does the answer actually address the question? A correct refusal on an unanswerable question counts as relevant.
- Retrieval metrics: hit rate@k (was a correct source doc in the top k?) and MRR (how high was it ranked?). These need no LLM because the test set stores the correct source doc IDs.
- An LLM judge is only trustworthy after it is calibrated against human labels.

## Architecture
Two halves joined by one contract:

```
SYSTEM UNDER TEST (rag/)
  docs -> chunker -> embedder -> FAISS (+ BM25 later) -> retriever
       -> prompt builder -> LLM -> answer

THE CONTRACT
  answer(question: str, config: dict) -> {
      "answer": str,
      "chunks": [{"chunk_id", "doc_id", "text", "score"}],
      "latency_ms": float
  }

EVALUATION HARNESS (evalharness/)
  test set -> runner -> deterministic checks + LLM judges -> SQLite
           -> regression compare + flagging -> Streamlit dashboard
```

The RAG pipeline must know nothing about evaluation. The harness must only call `answer()`, so it can later evaluate any other RAG system (for example my full HR chatbot) unchanged.

We build our own small, config-driven RAG pipeline (v0) so that every knob (chunk size, overlap, top-k, retrieval method, prompt) can be changed through a YAML config.

## Data model (SQLite, using the built-in sqlite3 module)
- `queries`: id, question, type (lookup / multi_chunk / paraphrased / unanswerable), reference_answer, source_doc_ids, split (dev / heldout), reviewed (bool)
- `runs`: run_id, timestamp, git_commit, config_json, judge_version, split
- `outputs`: run_id, query_id, answer, retrieved_chunks_json, latency_ms
- `scores`: run_id, query_id, metric, value, reasoning
- `reviews`: run_id, query_id, flag_reasons, human_label, notes

## Libraries
| Job | Library |
|---|---|
| Embeddings | sentence-transformers with Qwen/Qwen3-Embedding-0.6B (runs locally) |
| Vector search | faiss-cpu |
| Keyword search (later experiment) | rank_bm25 |
| Reranker (later experiment) | sentence-transformers CrossEncoder |
| LLM calls | openai Python SDK pointed at an OpenAI-compatible endpoint (provider chosen in Step 2) |
| Structured judge output and test-set validation | pydantic |
| Storage | sqlite3 (standard library) |
| Analysis and statistics | pandas, numpy, scipy |
| Dashboard | streamlit, plotly |
| Configs | pyyaml |
| Retries and progress | tenacity, tqdm |
| Secrets | python-dotenv |
| Tests | pytest |
| Automation (later) | GitHub Actions |

Do NOT use:
- LangChain for the pipeline: its wrappers hide retrieved chunks and scores, which the harness must see, and I want every step visible.
- RAGAS or DeepEval as the main metrics: we write our own judges. At the very end we may run RAGAS once as a cross-check.

## Conventions
- Every run records run_id, timestamp, git commit hash, full config, and judge prompt version.
- Each experiment is one YAML file in `configs/`. Change ONE variable per experiment and log it in `experiments/log.md`.
- Generator temperature fixed and low; judge temperature 0.
- The judge model must be different from (ideally stronger than) the generator model.
- Cache every LLM call keyed by a hash of (model, prompt, temperature, schema), so re-runs are cheap and repeatable.
- The 20 held-out queries are NEVER used for tuning. They are used once, in Step 12.
- Report rates with 95% confidence intervals (Wilson interval).
- Human labels for calibration and review must come from me, never from an LLM.

## Repo layout
```
rag-eval/
  CLAUDE.md  README.md  requirements.txt  .env  .gitignore
  data/
    docs/               # HR corpus, one file per document, stable doc IDs
    test_set.jsonl
  configs/              # v0.yaml, v1_chunk256.yaml, ...
  rag/
    chunker.py  embedder.py  index.py  retriever.py  pipeline.py
  evalharness/
    llm.py  schemas.py  store.py  runner.py
    checks.py  judges.py  metrics.py  calibration.py  regression.py  cli.py
  dashboard/app.py
  experiments/log.md
  tests/
```

## Build sequence

### Phase A: Foundations
**Step 0. Project setup.** Folder layout, venv, git repo, requirements.txt, .env, .gitignore, CLAUDE.md.
Done when: dependencies install cleanly and the first commit exists.

**Step 1. Document corpus.** 15-25 HR documents (leave, payroll, onboarding, IT access, code of conduct, internship rules, etc.), cleaned to plain text or markdown with stable doc IDs. If synthetic, use a fictional company, include specific checkable facts (numbers, dates, eligibility rules), and include a few closely related policies so retrieval is not trivial.
Done when: every doc loads, has an ID, and contains no garbage text.

**Step 2. LLM wrapper (evalharness/llm.py).** `generate(prompt, model, temperature, schema=None)` with retries, disk/SQLite cache, and token + latency logging. Ask me which provider and models to use for the generator and the judge.
Done when: an identical second call returns instantly from the cache.

### Phase B: System under test
**Step 3. RAG pipeline v0.** Chunker, embedder, FAISS index, top-k retriever, prompt builder, generator, all driven by `configs/v0.yaml`. Implements the contract exactly. v0 should be a plain baseline, not pre-optimized.
Done when: `answer("How many casual leaves do interns get?", config)` returns an answer plus its chunks and scores.

### Phase C: Measurement
**Step 4. Test set (100 queries).** About 50 lookup, 20 multi-chunk, 15 paraphrased/vague, 15 unanswerable (truly not covered by the corpus). Each has a reference answer and source doc IDs (empty for unanswerable). Split 80 dev / 20 held-out, stratified by type. You may draft queries, but I review and edit every one, and each is marked reviewed. Validate with pydantic.
Done when: the file loads, every query validates and is reviewed, and the type counts are correct.

**Step 5. Runner and SQLite store.** Loop over a split, call `answer()`, write outputs under a new run_id with its config and git commit.
Done when: one full run produces the expected rows, queryable with SQL.

**Step 6. Deterministic checks and retrieval metrics.** Empty answer, refusal detection, latency, hit rate@k, MRR, reported overall and by query type.
Done when: these print for a run and are stored in `scores`.

**Step 7. Faithfulness judge.** Call 1: decompose the answer into atomic claims. Call 2: label each claim against the retrieved context as supported / unsupported / contradicted, with reasoning. Store the score and reasoning. An answer is "ungrounded" if its score is below a configurable threshold (start at 0.8) or any claim is contradicted.
Done when: every answer in a run has a faithfulness score and reasoning stored.

**Step 8. Relevance judge.** Rubric-based score with reasoning; a correct refusal on an unanswerable query counts as relevant, and an invented answer to an unanswerable query is both irrelevant and ungrounded.
Done when: every answer has a relevance score and reasoning.

**Step 9. Judge calibration.** Build a simple labeling tool (CLI or Streamlit page) so I can hand-label about 50 answers as grounded / ungrounded. Compute agreement %, precision and recall on "ungrounded", and Cohen's kappa. Iterate on the judge prompt, bumping the judge version each time.
Done when: we have a measured agreement figure we can report honestly.

### Phase D: Improvement
**Step 10. Baseline report.** Run v0 on the dev split: ungrounded rate with a 95% CI, broken down by query type and by failure cause (retrieval miss vs generation error).
Done when: we have a real "before" number.

**Step 11. Experiments, one variable at a time.** Retrieval: chunk size/overlap, top-k, hybrid BM25 + vector, cross-encoder reranker, query rewriting. Prompt: answer-only-from-context, required citations, explicit refusal instruction. One YAML config and one log entry per experiment.
Done when: an experiment table shows which changes helped and by how much.

**Step 12. Held-out confirmation.** Run the best config once on the 20 held-out queries.
Done when: the final "after" number is confirmed on unseen queries, reported with its CI.

### Phase E: Automation and visibility
**Step 13. Regression comparator and flagging.** Compare a new run with the last accepted run. Flag an answer for review if: faithfulness is below threshold, relevance is low, the judges disagree, or the answer changed significantly from the previous run. Tune thresholds against my human labels. Add a pass/fail gate (default: fail if the ungrounded rate rises by more than 3 percentage points; configurable).
Done when: one command outputs the flagged list, the flag rate, and pass/fail.

**Step 14. Automation.** CLI: `python -m evalharness run --config configs/<name>.yaml --split dev`. Optionally a GitHub Actions workflow that runs the regression check on push.
Done when: a full evaluation runs from one command.

**Step 15. Streamlit dashboard.** Pages: trends across runs (Plotly), run-vs-run comparison (queries that improved / regressed), review queue (question, context, answer, judge reasoning, buttons to mark OK / bad, saved to `reviews`), breakdown by query type.
Done when: all pages work on the real database.

**Step 16. README and write-up.** Architecture, metric definitions, judge calibration results, experiment table, before/after results with CIs, and limitations.
Done when: someone new could understand and reproduce the project from the README.

## Start now
Do Step 0 only. As part of it, save this entire prompt as CLAUDE.md at the repo root and add an empty "## Progress" section at the bottom, which you update after every step.

Then ask me: do I have HR documents from my earlier chatbot project, or should you generate a synthetic corpus of fictional company HR policies for Step 1? Then stop and wait.

## Progress

- **Step 0 (done, 2026-10-15):** Folder layout created. Python 3.13.9 venv at `.venv/`; all libraries installed on 3.13 with no issues and pinned in `requirements.txt`. `.gitignore` covers `.env`, venv, caches, SQLite DBs, FAISS index files, and `.claude/settings.local.json` (agents, commands, skills and specs under `.claude/` are committed). `.env` holds placeholders only. Smoke test `tests/test_setup.py` imports every dependency: 15 passed. First git commit made.
- **Step 1 (done, 2026-10-15):** Synthetic corpus of 22 HR policies (7,178 words) for the fictional "Kestrel Labs Pvt. Ltd." (Bengaluru + Pune offices, bands B1-B6, HR portal "KestrelHub"), in `data/docs/<doc_id>.md`; the filename is the stable doc ID and line 1 is `# Title`. Loader: `rag/loader.py` (`load_docs()`, `python -m rag.loader`). Tests: `tests/test_corpus.py` (size, unique safe IDs, title, >=150 words, no garbage markers/control chars/unexpected non-ASCII); 62 tests pass.
  - Deliberately confusable pairs: `leave_policy` vs `intern_leave_policy` (employees 8 CL/yr vs interns 1 CL per completed month); `expense_reimbursement_policy` vs `travel_policy`; `attendance_and_working_hours` vs `remote_hybrid_work_policy`; `it_access_policy` vs `device_and_byod_policy` vs `information_security_policy`; `probation_and_confirmation` vs `resignation_and_exit_policy` (notice periods).
  - Deliberate gaps (never mentioned anywhere, use for unanswerable queries in Step 4): ESOPs/stock options, gym or wellness membership, pet policy, sabbatical, relocation allowance, office shuttle/transport, office canteen/meals (`travel_policy` covers meals only inside the travel per diem).
