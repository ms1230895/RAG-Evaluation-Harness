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

SHARED LLM CLIENT (llm/)
  llm/client.py: generate() with cache, retries, token + latency log.
  Used by the generator in rag/ and the judges in evalharness/; it knows
  nothing about RAG or evaluation, so rag/ still never imports evalharness/.
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
| Embeddings | Ollama `nomic-embed-text` (runs locally, 768-dim), called through Ollama's OpenAI-compatible `/v1/embeddings` endpoint |
| Vector search | faiss-cpu |
| Keyword search (later experiment) | rank_bm25 |
| Reranker (later experiment) | sentence-transformers CrossEncoder |
| LLM calls | openai Python SDK pointed at Ollama's OpenAI-compatible endpoint (`http://localhost:11434/v1`), using Ollama cloud models: generator `gemma4:31b-cloud`, judge `gpt-oss:120b-cloud` (both in the Ollama Free plan) |
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
- Cache every LLM call keyed by a hash of (model, prompt, temperature, schema), so re-runs are cheap and repeatable. Every LLM call goes through `llm/client.py`.
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
  llm/
    client.py           # shared cached LLM client (generator + judges)
  rag/
    chunker.py  embedder.py  index.py  retriever.py  pipeline.py
  evalharness/
    schemas.py  store.py  runner.py
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

**Step 2. LLM wrapper (llm/client.py).** `generate(prompt, model, temperature, schema=None)` with retries, disk/SQLite cache, and token + latency logging. Ask me which provider and models to use for the generator and the judge.
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
- **Step 2 (done, 2026-10-15):** Shared LLM client `llm/client.py` (neutral package so both `rag/` and `evalharness/` can use it; `rag/` still never imports `evalharness/`). `generate(prompt, model, temperature, schema=None, *, config=None) -> LLMResult` calls Ollama's OpenAI-compatible endpoint via the openai SDK (SDK retries off). `configs/llm.yaml` holds `base_url`, timeout, retry and cache settings (all required); `.env` holds only the dummy `LLM_API_KEY`. tenacity retries only transient errors. SQLite cache `data/llm_cache.sqlite` (gitignored, with its journal/WAL files) keyed by SHA-256 of (base_url, model, prompt, temperature, JSON Schema). Every call, including failures (`error` column), is logged in `llm_calls`. Empty, cut-off (`finish_reason == "length"`) or schema-invalid replies raise `LLMOutputError` and are never cached. Unknown token counts are NULL, not 0. Two latencies: `model_latency_ms` (the successful HTTP request only; also returned on cache hits) and `latency_ms` (wall time). Tests: `tests/test_02_llm_wrapper.py`, 94 tests, all offline (fakes, `tmp_path`), plus shared fixtures in `tests/conftest.py`.
  - Models changed: `deepseek-v4.1-flash:cloud` returned `402 Payment Required` (not in Ollama Free plan). Now generator `gemma4:31b-cloud`, judge `gpt-oss:120b-cloud` (judge is the larger model).
  - `gpt-oss:120b-cloud` accepts `response_format` json_schema but still wraps JSON in a ```json fence, so the server does not enforce the schema; `_parse` strips one surrounding fence before pydantic validation.
  - Two code-review rounds (security, quality, eval-integrity) fixed: latency semantics, finish reason (only "stop" is valid), failure logging (hit and miss), NULL tokens, base_url and model_digest in the cache key, `attempts` column, named-column inserts, config key and value checks, connection handling. The cache was rebuilt after each table-layout change.
  - Done-when check (rows from `llm_calls`, after the second rebuild): plain call `gpt-oss:120b-cloud` miss, attempts 1, 73 prompt / 136 completion tokens, finish_reason stop, model_latency_ms 1929.4, latency_ms 1975.2; identical second call cache hit, attempts 0, latency_ms 2.7, same model_latency_ms and text. Structured `DemoClaims` call miss 99 / 247 tokens, model_latency_ms 1939.5, latency_ms 1984.3; hit latency_ms 2.1. Completion tokens include gpt-oss's hidden reasoning tokens. The same prompt at temperature 0 gave three different replies across the three cache rebuilds ("Hello there friend", "Hello there, friend", "Hello there, friend!"): cloud models are not fully deterministic, which is why the cache matters.
  - Carry-forward items for Steps 3, 5, 6, 7 and 14 (seed and reasoning effort, context limit, latency source, model identity, failure counts, strict judge schemas, cost filtering) are listed in `.claude/specs/02-llm-wrapper.md` under "Carry-forward to later steps".
- **Step 3 (done, 2026-10-15):** Config-driven RAG pipeline v0 in `rag/` (`chunker.py`, `embedder.py`, `index.py`, `retriever.py`, `pipeline.py`), driven by `configs/v0.yaml`, implementing the `answer()` contract exactly (3 keys, 4 chunk keys). CLI: `python -m rag.pipeline --config configs/v0.yaml --question "..."`.
  - Decisions (all recommendations accepted): word-window chunks `chunk_size` 150 / `chunk_overlap` 30, nomic prefixes `search_document: ` / `search_query: `, FAISS `IndexFlatIP` on L2-normalised vectors (score = cosine), `top_k` 4, plain prompt (no refusal / only-from-context / citation wording), generator `gemma4:31b-cloud` at temperature 0.0, `seed` 42, `reasoning_effort` null (thinking off). Context guard: `answer()` raises `ContextLimitError` when `prompt_tokens >= 3500` or is not reported (no contract change). `embed()` lives in `llm/client.py` (shared client and retry loop; not cached in SQLite, not logged in `llm_calls`).
  - `llm/client.py`: `seed` and `reasoning_effort` are sent only when set, and are in the cache key (now 8 fields) and both tables; the cache file was deleted and rebuilt (table layout change).
  - Index saved as `indexes/<index_key>.faiss` + `.chunks.json`; `index_key` covers doc IDs and text, chunk settings, `CHUNKER_VERSION`, embedding model, `doc_prefix` and the embedding `base_url`. A warm-up query is embedded after a disk load so the first timed retrieval does not pay Ollama's model load. `describe_index(config)` gives Step 5 the index key and meta. `latency_ms` = retrieval time + `model_latency_ms` (never wall time).
  - Done-when check (from the CLI output and `llm_calls`): 66 chunks; top-4 for "How many casual leaves do interns get?" = `intern_leave_policy#000` 0.780, `intern_leave_policy#001` 0.720, `leave_policy#000` 0.698, `internship_program_rules#000` 0.694 (correct doc at rank 1; the confusable employee `leave_policy` only 0.022 below rank 2). Answer: "Interns earn 1 day of casual leave for every completed month of their internship. For example, an intern on a 6-month internship can earn up to 6 casual leaves in total." Generator: first call `cached=0`, later calls `cached=1`, seed 42, prompt_tokens 887, completion_tokens 41, model_latency_ms 1897.4; latency_ms 1966.4 (first run), 1954.8, 1921.8 and 1953.4 on later runs. On the last run the warm-up embed took 1182.8 ms (model was cold) and the query embed 45.3 ms. This is one question, not an evaluation.
  - Tests: `tests/test_03_rag_pipeline_v0.py` 200 tests, plus 26 new Step 2 cases in `test_02`; full suite 382 passed. All offline (fake embedder, fake generator, `tmp_path`).
  - macOS libomp clash: faiss-cpu and torch each bundle `libomp.dylib` and cannot run in one process (`OMP: Error #15` or a silent crash, every import order). `tests/test_setup.py` now checks `sentence_transformers` with `find_spec` instead of importing it. `KMP_DUPLICATE_LIB_OK` is not used (OpenMP: may give incorrect results). The Step 11 reranker must solve this for real code (decision for Gaurav then).
  - Two code-review rounds (security, quality, eval-integrity), no Critical or High findings. Round 1: all 17 items applied. Round 2 suggestions NOT applied (shipped as is): `describe_index` reads the corpus twice (not atomic); warm-up only after a disk load, so an Ollama model unload after an idle pause still lands in `retrieval_ms`; embed retry waits count as retrieval latency and are not logged; `prompt_tokens=0` passes the guard; `max_input_words` 1000 counts words, not tokens; plus readability nits. Carry-forward items for Steps 5, 6, 10-11, 14 are in `.claude/specs/03-rag-pipeline-v0.md`.
- **Step 4 (done, 2026-10-15):** 100-query test set in `data/test_set.jsonl`, validated with pydantic (`evalharness/schemas.py`) and managed by `evalharness/testset.py` (`python -m evalharness.testset summary | review | assign-splits`). Expected shape lives in `configs/test_set.yaml` (counts, `split_seed` 2026, paths).
  - Counts from `summary`: lookup 40 dev / 10 heldout, multi_chunk 16 / 4, paraphrased 12 / 3, unanswerable 12 / 3; total 80 dev / 20 heldout; `reviewed 100/100`; READY (no problems, exit 0). All 22 docs are the source for at least 2 queries.
  - Review: Gaurav reviewed all 100 queries with the `review` tool and accepted every one as drafted. A field-by-field diff against the draft commit `8ec9ac5` shows 0 content edits; only `reviewed` and `split` changed.
  - Splits assigned once by `assign-splits` (stratified by type, seed 2026); running it again refuses. Held-out (20 queries) is not used before Step 12.
  - Tests: `tests/test_04_test_set.py` 123 tests (offline, `tmp_path`); full suite 505 passed. `evalharness/schemas.py` and `evalharness/testset.py` import nothing from `rag/` or `llm/`.
- **Step 5 (done, 2026-10-15):** Results DB `data/results.sqlite` (gitignored) with all five tables (`evalharness/store.py`: `connect`, `sync_queries`, `start_run`, `write_output`, `finish_run`, `get_run`, `get_outputs`) and the runner `evalharness/runner.py` (`python -m evalharness.runner --config configs/v0.yaml --split dev [--allow-dirty] [--allow-heldout] [--harness-config ...]`). `configs/harness.yaml` holds `db_path`, `test_set_config`, `max_consecutive_failures` 5, `error_max_chars` 500 and the system as `"module:function"` names, loaded with importlib, so `evalharness/` never imports `rag/`.
  - Decisions (all recommendations accepted): extra columns on `runs` (`status` running/completed/aborted/interrupted/invalid, `finished_at`, `note`, `config_path`, `git_dirty`, `test_set_sha256`, `system_json`, `n_queries`/`n_ok`/`n_failed`, `llm_calls_start_id`, `llm_calls_json`) and `outputs.error` (CHECK: exactly one of answer / error); `scores` and `reviews` created empty. Runner refuses (exit 1, nothing written) if the test set is not ready, on `--split heldout` without `--allow-heldout`, on a dirty tree without `--allow-dirty`, or if `describe()` fails. Failures are stored per query and counted, never dropped; abort after 5 in a row. `queries` is frozen: a changed query with an existing id raises.
  - New `rag.pipeline.describe_system(config)` hook (answer() unchanged): index meta + generator and embedding digests from Ollama `/api/tags` (`llm.client.get_model_digest`). `configs/v0.yaml` `generation.model_digest` pinned to `ef09f235533c96cd75e8deed88c628335cb69e2b3ce96275d0d7a67fe9887aba`; the run refuses if the server reports another digest. For `-cloud` models the digest is the local manifest, not the remote weights. Run IDs are `<UTC>-<config name>-<8 hex>`.
  - Done-when check (from `data/results.sqlite`): run `20261015T152638Z-v0-c7c96975` (dev, completed, git_commit `a9c09fe`, git_dirty 0): 80 queries, 80 ok, 0 failed; outputs 80 rows / 80 distinct queries / 80 answers / 0 errors / 0 blank answers; all 80 join to dev queries; `queries` has 100 rows; `test_set_sha256` `0f09379fd1dc4d577991404c0cccd8870234f1e577a2723c512a9c74fd11da5c` (matches `shasum`); index_key `b818ed7d43664cd47707309450937e9efc42b010c8fc575d3c9085cdb7b69b6d`; `gemma4:31b-cloud`: 80 calls, 0 hits, 80 misses, 0 errors; started 15:26:38, finished 15:28:18 UTC. Identical rerun `20261015T152824Z-v0-59c05048`: completed, 80 ok, 0 failed, 80 hits, 0 misses, 0 errors; all 80 answers and chunk lists identical to the first run. `--split heldout` exits 1; no held-out run exists. No metric computed yet (Step 6). Run `20261015T152638Z-v0-c7c96975` is the v0 dev output later steps read.
  - Tests: `tests/test_05_runner_and_store.py` 186 tests (offline: fake system, fake `urlopen`, `tmp_path` DBs); full suite 691 passed. First test run had 1 faulty test (`"text"` listed as a non-string answer), fixed; run-id suffix widened from 4 to 8 hex after the test run showed 100 ids in one second could collide ~7% of the time. One Step 3 assertion updated (`model_digest` is now the pinned 64-hex digest, not null).
  - Not done: `/code-review` was skipped for this step (Gaurav shipped directly).
- **Step 6 (done, 2026-10-15):** Deterministic checks and retrieval metrics, no LLM calls. `evalharness/metrics.py` (pure: `first_source_rank`, `hit_at_k`, `reciprocal_rank`, `doc_recall`, `wilson_interval` clamped to [0, 1], `rate_summary` / `mean_summary` / `percentile_summary`, None = not applicable) and `evalharness/checks.py` (`python -m evalharness.checks --run-id <id> [--allow-heldout]`; only `completed` runs; exit 0 written/unchanged, 1 refused). Settings in the `checks:` block of `configs/harness.yaml` (`hit_ks` [1, 4], `report_confidence` 0.95, 19 refusal phrases).
  - Decisions (all recommendations accepted): hit = any source doc, plus `doc_recall`; rank = chunk position in the stored order; `hit_ks` [1, 4] plus `in_context` (any retrieved chunk, whatever `top_k`; Step 10 uses it for retrieval miss vs generation error); latency report-only. 8 metrics per output: `empty_answer`, `refusal` (phrase match, a heuristic), `latency_ms`, `hit@1`, `hit@4`, `in_context`, `reciprocal_rank`, `doc_recall`. Failed outputs, unanswerable retrieval metrics and k > chunks are stored as `value` NULL with a reason, never 0/1.
  - Schema: `scores` now has `version` (in the PK), `scored_at`, and `reasoning NOT NULL`; the empty Step 5 table was dropped and recreated by `store._migrate_scores` (refuses if it holds rows). New `store.write_scores` (one transaction), `get_scores`, `get_outputs_with_queries`. Version = `checks-<CHECKS_CODE_VERSION>-<sha256(hit_ks + refusal_phrases)[:8]>`; re-scoring with the same version prints `unchanged`, different values under one version are refused.
  - Refusal list review (Decision 4): a trial on a scratch copy of the DB showed `"not covered"` flagging q013's correct answer ("No, outpatient (OPD) consultations and medicines are not covered."); Gaurav removed it before the real scoring.
  - Done-when check (from `data/results.sqlite`, version `checks-1-66c5a50a`, run `20261015T152638Z-v0-c7c96975`, dev): 640 rows = 80 queries x 8 metrics; NULL rows only for the 12 unanswerable queries x 5 retrieval metrics; 0 blank reasonings; rerun prints `unchanged`; unknown run exits 1; no held-out run exists. Report: 80 outputs, 80 ok, 0 failed; empty answers 0/80 [0.000, 0.046]; answerable (n=68): hit@1 62/68 = 0.912 [0.821, 0.959], hit@4 67/68 = 0.985 [0.921, 0.997], in_context 67/68 = 0.985 [0.921, 0.997], MRR 0.944, doc recall 0.985; by type hit@1 lookup 38/40, multi_chunk 15/16, paraphrased 9/12 = 0.750 [0.468, 0.911]; hit@4 lookup 40/40, multi_chunk 16/16, paraphrased 11/12. Retrieval miss: q053 only (paid side projects; `code_of_conduct` not retrieved; the model replied "there is no mention of"), which is also the only false refusal (1/68). Correct refusals (phrase match) 12/12 unanswerable [0.758, 1.000]. Latency (80 real generator calls, 0 cache hits) p50 831 / p95 2117 / max 8824 ms. Retrieval is near ceiling on this corpus; per-type n is 12-16, so per-type differences are inside the CIs and not meaningful. The refusal metric is a phrase match, not a judgement (Step 8 decides relevance).
  - Tests: `tests/test_06_deterministic_checks_retrieval_metrics.py` 181 tests (written by rag-test-writer from the spec, all offline, tmp DBs), plus `scored_run_db` and `checks_config` fixtures; test_05 `scores` column set updated. Full suite 872 passed. `/code-review` not run yet.
- **Step 7 (done, 2026-10-15):** Faithfulness judge, `evalharness/judges.py` (`python -m evalharness.judges faithfulness --run-id <id> [--query-ids ...] [--retry-failed] [--allow-heldout]`; exit 0 complete, 1 refused, 2 partial/aborted). Two cached calls per answer to `gpt-oss:120b-cloud` (judge != generator `gemma4:31b-cloud`): decompose (question + answer -> atomic claims, never sees the context) and verify (numbered claims vs the exact stored chunks, reasoning before label). faithfulness = supported/claims (1.0 for 0 claims); ungrounded = faithfulness < 0.8 or any claim contradicted. Settings and both prompts in `configs/judges.yaml` (`harness.yaml` key `judges_config`); strict pydantic reply schemas `ClaimList` / `ClaimVerdicts` in `evalharness/schemas.py`.
  - Decisions (all recommendations accepted): temperature 0, seed 42, `reasoning_effort: medium`, digest `ac7f7a1e778577c4418f6a25e46e0b45dced6746c75422d4b343aa1495a022ed` pinned and checked; 0 claims = faithfulness 1.0, not ungrounded; statements about the documents are skipped by the decomposer; per-claim verdicts in a new `judge_claims` table (PK run, query, version, claim_index; label CHECK); judge failures stored as NULL "judge failed" (stop after 5 in a row, `--retry-failed`); failed outputs / blank answers stored as NULL "not judged". Version = `faith-v<prompt_version>-<sha256 of all settings + schemas>[:8]`. `runs.judge_version` stays NULL (judging happens after a run; the version lives on each scores / judge_claims row). `checks.get_scorable_run` now holds the shared run guard; `store.get_outputs_with_queries` also returns `question`.
  - Smoke test (`--query-ids q001,q053,q097`): prompt v1 (`faith-v1-73b31388`, 6 calls) ignored the "skip statements about the documents" rule (claims like "the provided text does not mention ..."; q053 became ungrounded through a contradicted "the policy only covers ..." claim). Gaurav chose to make the rule explicit with an example: prompt v2 (`faith-v2-d130a39f`); its smoke test (5 calls, 1 hit, 4 misses) followed Decision 3. The v1 rows (3 items) stay stored under their own version.
  - Done-when check (from `data/results.sqlite`, version `faith-v2-d130a39f`, run `20261015T152638Z-v0-c7c96975`, dev): complete; 80 outputs, 80 judged, 0 judge failed, 0 not judged; full run `gpt-oss:120b-cloud: 144 calls, 0 hits, 144 misses, 0 errors` (plus the 3 smoke items); 80 rows each for faithfulness / n_claims / ungrounded, 0 NULL, 0 blank reasonings; n_claims and faithfulness consistent with `judge_claims` (0 mismatches); rerun `80 skipped (already judged) | no LLM calls`. **Uncalibrated judge, v0 dev (not the Step 10 baseline):** ungrounded all 7/80 = 0.087 [0.043, 0.170]; answerable 7/68 = 0.103 [0.051, 0.198]; lookup 1/40 = 0.025 [0.004, 0.129]; multi_chunk 4/16 = 0.250 [0.102, 0.495]; paraphrased 2/12 = 0.167 [0.047, 0.448]; unanswerable 0/12 = 0.000 [0.000, 0.242]. Mean faithfulness 0.964 (n=80). Claims 155: 145 supported, 5 unsupported, 5 contradicted; 11 answers with 0 claims (10 unanswerable, 1 paraphrased). Ungrounded: q029, q057, q058, q068, q069, q071, q082; all 7 have `in_context = 1` (Step 6), so per this judge they are generation errors. Per-type n is 12-16; the CIs overlap heavily.
  - Possible judge errors noted for Step 9 (not fixed here; prompt tuning belongs to calibration): q057 decomposition split "Public or Internal information" into two "only ..." claims and both were marked contradicted (likely a false ungrounded); q058 "should request admin rights" marked unsupported because the context says rights "can be requested" (borderline).
  - Tests: `tests/test_07_faithfulness_judge.py` 266 tests (rag-test-writer from the spec; fake judge LLM patched at `evalharness.judges.generate`, tmp DBs and tmp LLM cache), plus `judges_config`, `FakeJudgeLLM` / `fake_judge` and `judge_run_db` fixtures. Full suite 1138 passed. `/code-review` not run yet.
- **Step 8 (done, 2026-10-15):** Relevance judge in `evalharness/judges.py` (`python -m evalharness.judges relevance --run-id <id> [--query-ids ...] [--retry-failed] [--allow-heldout]`; same exit codes and refusals as `faithfulness`). One cached `gpt-oss:120b-cloud` call per answer sees the question, the answer and only the rubric that applies (answerable vs not answerable, from the test set: `source_doc_ids == []`); never the chunks or the reference answer. Strict reply schema `RelevanceVerdict` (reasoning, then `score` 0/1/2). Rows per output: `relevance` = score/2 and `irrelevant` = score <= `irrelevant_max_score` (0). Settings and prompt in the `relevance` block of `configs/judges.yaml`; version `rel-v<prompt_version>-<hash>`.
  - Decisions (all recommendations accepted): 0/1/2 rubric; answerability flag only (no reference answer); same judge settings as faithfulness (temperature 0, seed 42, `reasoning_effort: medium`, pinned digest); both judges in `judges.py`. Refactor first (own commit): `_load_block`, `_check_common`, `_check_templates`, `_version` and the shared per-item loop `_judge_items`; Step 7 behaviour and the `faith-v2-d130a39f` version unchanged (266 Step 7 tests pass unchanged).
  - Smoke test (q001, q053, q097): prompt v1 (`rel-v1-3ae4b6e7`, both rubrics plus an ANSWERABLE / NOT ANSWERABLE line, 3 calls) scored q053 (answerable, false refusal) 2 by applying the not-answerable rubric. Gaurav chose prompt v2: show only the applicable rubric, stating that answerability is a fact from the test set and that "not available" scores 0 on an answerable question. v2 smoke (`rel-v2-c825e4f7`, 3 calls): q001 2, q053 0, q097 2. The v1 rows stay stored under their own version.
  - Done-when check (from `data/results.sqlite`, version `rel-v2-c825e4f7`, run `20261015T152638Z-v0-c7c96975`, dev): full run `complete | 80 outputs: 76 judged, 1 judge failed` (`gpt-oss:120b-cloud: 77 calls, 0 hits, 76 misses, 1 errors`); the failure was q098 (`LLMOutputError`: prose before the JSON, schema-invalid, not cached). One `--retry-failed` (1 call, 0 errors) judged q098 (score 2). Final: 80 judged, 0 judge failed, 0 not judged; 80 rows each for `relevance` / `irrelevant`, 0 NULL, 0 blank reasonings; `irrelevant = 1` exactly where `relevance = 0` (0 mismatches); reruns of both judges print `80 skipped (already judged) | no LLM calls`. **Uncalibrated judge, v0 dev:** scores 0 / 1 / 2 = 1 / 0 / 79; mean relevance 0.988 (n=80; answerable 0.985 n=68, paraphrased 0.917 n=12, others 1.000); irrelevant all 1/80 = 0.013 [0.002, 0.067], answerable 1/68 = 0.015 [0.003, 0.079], lookup 0/40 [0.000, 0.088], multi_chunk 0/16 [0.000, 0.194], paraphrased 1/12 = 0.083 [0.015, 0.354], unanswerable 0/12 [0.000, 0.242]. Irrelevant: q053 only (the Step 6 retrieval miss and false refusal); all 12 unanswerable answers scored 2 (clean refusals), matching Step 6's phrase-match 12/12. The judge rates almost every v0 answer fully relevant, so it separates little beyond false refusals; it is not calibrated (Step 9 labels are grounded/ungrounded).
  - Tests: `tests/test_08_relevance_judge.py` 163 tests (rag-test-writer from the spec; fake judge LLM, tmp DBs and tmp LLM cache), plus `relevance_config`, `FakeRelevanceLLM` / `fake_relevance` and `relevance_run_db` fixtures. Full suite 1301 passed. `/code-review` not run yet.
