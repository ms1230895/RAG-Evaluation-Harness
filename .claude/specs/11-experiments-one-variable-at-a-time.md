# Spec: Step 11 — Experiments, One Variable at a Time

## Overview
Step 10 gave the "before" number: the v0 pipeline, scored on the 80 dev
queries, has an ungrounded rate of 7/80 = 8.75% [4.30%, 16.98%] (uncalibrated
judge). This step changes the pipeline one variable at a time and measures
what each change does to that number.

It has two parts.

1. **Make the pipeline able to change.** v0 only supports vector search with
   a fixed prompt. We add the knobs CLAUDE.md lists: hybrid BM25 + vector
   search, a cross-encoder reranker, query rewriting, and a context template
   so that citations are possible. Every knob is a YAML key and is switched
   **off** in `configs/v0.yaml`, so v0 behaves exactly as before.
2. **Make the comparison honest.** A small tool in `evalharness/` checks that
   an experiment config differs from v0 in exactly the declared keys, and
   compares each experiment run with the baseline run query by query (paired),
   with confidence intervals and an exact significance test. The output is the
   experiment table CLAUDE.md asks for, plus one row per experiment in
   `experiments/log.md`.

It comes now because Steps 5-10 gave us the runner, the three scorings and a
baseline. Nothing before this step can say whether a change helped.

Two facts shape the whole step and are stated plainly so nothing is oversold:

- **Headroom is small.** Retrieval is near ceiling (hit@4 67/68, one retrieval
  miss), and all 7 ungrounded answers are generation errors per the judge. So
  retrieval experiments can fix at most about 1 answer, and prompt experiments
  have the real room.
- **Power is small.** With 7 ungrounded answers at baseline, an experiment
  can only reach a two-sided exact p < 0.05 by fixing 6 or 7 of them without
  breaking any (Concepts). Most results will be "direction, not proof". The
  tool prints that explicitly instead of ranking runs by their point estimate.

## Concepts to learn

- **One variable at a time.** If a run changes chunk size and top-k together
  and the ungrounded rate moves, you cannot say which one did it. Each
  experiment copies v0 and changes one thing. Some "things" are really one
  feature spread over a few keys (turning on citations needs a numbered
  context and an instruction); those are declared as one variable with all
  their keys listed, and the tool checks the diff is exactly those keys.
- **Paired comparison.** Baseline and experiment answer the *same* 80
  queries. So we compare per query, not just two percentages. For each query
  the ungrounded flag is either the same in both runs or it flipped.
  - improved: ungrounded in baseline, grounded in experiment
  - regressed: grounded in baseline, ungrounded in experiment
  - Only the flips carry information about the change.
- **Exact McNemar test (sign test on flips).** Under "the change did
  nothing", each flip is equally likely to go either way. With `n` flips and
  `k` improved, the two-sided p-value is `2 * P(X >= max(k, n-k))` for
  `X ~ Binomial(n, 0.5)`.
  Worked example: 4 improved, 1 regressed, so n = 5.
  P(X >= 4) = (C(5,4) + C(5,5)) / 32 = 6/32 = 0.1875, so p = 0.375: not
  significant, even though the rate dropped by 3 answers.
  Detectable effect here: 5 improved / 0 regressed gives p = 2/32 = 0.0625;
  6 / 0 gives 2/64 = 0.031; 7 / 0 gives 0.016; 7 / 1 gives 0.070. Baseline has
  only 7 ungrounded answers, so significance at 0.05 needs 6 or 7 fixed and
  none broken.
- **Wilson interval.** Same as Step 6 and 10: 7/80 gives [4.30%, 16.98%].
  Two intervals that overlap do not prove "no difference", and non-overlap is
  not a test. The paired test above is the test; the intervals show the
  uncertainty of each rate.
- **Paired bootstrap for the mean faithfulness difference.** Faithfulness is
  a score in [0, 1] per answer, so it is more sensitive than the
  ungrounded yes/no. For each query take `d = faithfulness_exp -
  faithfulness_base`. Resample the 80 differences with replacement many
  times (seeded), take the mean each time, and report the 2.5th and 97.5th
  percentiles. If the interval contains 0, the data do not separate the two
  runs. Step 9 already uses the same idea for kappa.
- **Noise control.** The generator is a cloud model at temperature 0 with a
  fixed seed, yet Step 2 saw the same prompt give different text across cache
  rebuilds. So two runs of the *same* config can differ. Experiment E0 changes
  only `generation.seed`; whatever flips there is noise, and every other
  experiment is read against that.
- **BM25.** A keyword score. For query term `t` and chunk `d`:
  `idf(t) * f(t,d)*(k1+1) / (f(t,d) + k1*(1 - b + b*|d|/avgdl))`, summed over
  query terms. `f` is how often `t` is in the chunk, `k1` caps how much
  repeats help, `b` controls length normalisation, `|d|/avgdl` is chunk length
  over average length. Tiny example: f = 2, k1 = 1.5, b = 0.75, chunk of
  average length: 2 * 2.5 / (2 + 1.5) = 1.4286 times idf. BM25 finds exact
  words ("B3", "KestrelHub") that an embedding may blur; the embedding finds
  paraphrases that BM25 misses.
- **Reciprocal rank fusion (RRF).** Combine two rankings without comparing
  their incompatible scores. A chunk's score is the sum over rankings of
  `1 / (rrf_k + rank)`. Example, `rrf_k` 60: chunk A is vector rank 1 and
  BM25 rank 3: 1/61 + 1/63 = 0.03227. Chunk B is vector rank 2 and absent
  from BM25: 1/62 = 0.01613. A wins.
- **Bi-encoder vs cross-encoder.** The embedder (bi-encoder) turns question
  and chunk into vectors separately, so search is fast. A cross-encoder reads
  the pair (question, chunk) together and outputs a relevance score: more
  accurate, much slower, so it only reorders a short pool (for example the
  top 10) from the first-stage search.
- **Query rewriting.** One extra LLM call turns the user's question into a
  better search query (for example expands a vague phrase). Only retrieval
  sees the rewrite; the generator still sees the original question.
- **Winner's curse.** We look at about 13 runs on the same 80 queries. The
  best one looks better than it is, partly by luck. That is exactly why the
  best config is confirmed once on the untouched held-out split in Step 12.
  No held-out query is used in this step.
- **Trade-off, not just a rate.** A prompt that says "refuse if unsure" can
  lower ungrounded by refusing answerable questions. So every row also shows
  the bad-answer rate, false refusals, and latency, and a lower ungrounded
  rate alone is never called a win.

## Depends on
- Step 3: `rag/pipeline.py` (`answer`, `build_prompt`, `load_pipeline_config`,
  `describe_system`), `rag/retriever.py`, `rag/index.py` (`get_index`,
  `index_key`, `resolve_path`), `configs/v0.yaml`.
- Step 2: `llm/client.py` (`generate`, cache, `LLMResult.model_latency_ms`).
  The query rewriter calls it, like the generator does.
- Step 5: `evalharness/runner.py` (one run per config; git commit and clean
  tree rules), `store.py` (`get_run`, `connect`).
- Steps 6-8: `checks`, faithfulness and relevance scores per run.
- Step 10: `evalharness/report.py` (`item_table`, `current_versions`,
  `failure_cause`, `load_report_config`), `store.write_run_report`,
  `experiments/log.md`, `experiments/baseline_v0.md`.
- Step 9 status: both judges are **uncalibrated** (inconclusive check). Every
  judge-based number in this step keeps that label.
- Baseline run: `20261015T152638Z-v0-c7c96975` (dev), scorers `checks-1-66c5a50a`,
  `faith-v2-d130a39f`, `rel-v2-c825e4f7`.

## Interfaces
This step does **not** change the `answer()` contract: same arguments, same
three keys, same four chunk keys, `latency_ms` still retrieval time plus the
model's own time. The harness still only calls `answer()`. `chunks[].score`
keeps its position but its *meaning* depends on the method (cosine for
`vector`, BM25 score for `bm25`, RRF score for `hybrid`, cross-encoder score
after reranking). No harness metric uses the score value, only the order.

### `rag/` (system under test)

`rag/bm25.py` (new)
```python
def tokenize(text: str) -> list[str]
    """Lowercase, split on runs of non-alphanumeric characters (letters+digits).
    tokenize("B3: 8 CL/yr") == ["b3", "8", "cl", "yr"]"""

def bm25_search(question: str, chunks: list[dict], k: int, bm25_config: dict) -> list[tuple[int, float]]
    """Top-k (chunk_row, score) in rank order. Chunks with score <= 0 are never
    returned (a query with no shared term gives an empty list, not an arbitrary
    one). Ties break on the lower chunk row, so the result is deterministic.
    The BM25 object is built once per index_key and kept in memory."""

def rrf_fuse(rankings: list[list[int]], rrf_k: int) -> list[tuple[int, float]]
    """Chunk rows with fused scores, best first. Ties break on the best rank in
    the first ranking, then the lower row.
    rrf_fuse([[7, 2], [2, 9]], 60) -> [(2, 0.03252...), (7, 0.01639...), (9, 0.01613...)]"""
```

`rag/retriever.py` (changed)
```python
def retrieve(question, index, chunks, top_k, embedding_config) -> list[dict]
    # unchanged: vector search, kept as is so v0 is byte-identical

def retrieve_with_method(question: str, index, chunks, config: dict) -> list[dict]
    """Dispatch on config["retrieval"]["method"]: vector | bm25 | hybrid.
    Returns up to the number of chunks the stage needs (top_k, or
    rerank.pool_k when reranking is on), in rank order, as
    {"chunk_id", "doc_id", "text", "score"}.
    hybrid: vector top hybrid.pool_k + BM25 top hybrid.pool_k, fused with RRF,
    then cut to the stage size."""
```

`rag/reranker.py` + `rag/rerank_worker.py` (new): the reranker runs in a
**separate Python process** (Decision 3) so torch is never imported next to
FAISS.
```python
def rerank(question: str, candidates: list[dict], top_k: int, rerank_config: dict) -> list[dict]
    """Send (question, candidate texts) to the worker, get one score per
    candidate, return the top_k candidates re-ordered by that score, with
    score replaced by the cross-encoder score. Raises RerankError if the
    worker dies or returns a wrong count (a failed item, never a silent
    fallback to the unranked list)."""
```
Worker protocol: one JSON object per line on stdin
(`{"question": str, "texts": [str]}`), one JSON object per line on stdout
(`{"scores": [float]}`). The worker loads `sentence_transformers.CrossEncoder`
once, at start, and answers a warm-up pair before the first timed call.

`rag/rewriter.py` (new)
```python
def rewrite_query(question: str, config: dict) -> tuple[str, float]
    """(search_query, model_latency_ms). Uses generation.model / digest /
    temperature / seed through llm.client.generate and
    query_rewrite.prompt_template. The reply is stripped of whitespace and one
    pair of surrounding quotes. A blank reply raises (generate already refuses
    empty text); there is no fallback to the original question."""
```

`rag/pipeline.py` (changed)
- `answer()` flow: optional `rewrite_query` -> `retrieve_with_method` ->
  optional `rerank` (always on the **original** question) -> `build_prompt`
  -> `generate` -> token guard. With every switch off the sequence of
  calls and the prompt string equal v0's.
- `latency_ms` = (BM25/vector/fusion/rerank wall time, round trip to the worker,
  excluding worker start and warm-up) + rewrite `model_latency_ms` + generator
  `model_latency_ms`.
- `build_prompt(question, chunks, template, separator, chunk_template="{text}")`:
  each chunk is rendered by `chunk_template` (fields `{rank}` 1-based and
  `{text}`) before joining.
- `describe_system(config)` also returns, only when reranking is on,
  `reranker_model` and `reranker_revision`.
- `load_pipeline_config`: validates every new key (below) with the same
  fail-with-a-message style; `SUPPORTED_METHODS = ("vector", "bm25", "hybrid")`.

### `evalharness/experiments.py` (new, imports nothing from `rag/` or `llm/`)
```python
def flatten(config: dict) -> dict[str, object]
    """Nested dict -> {"chunking.chunk_size": 150, ...}; lists are leaves."""

def config_diff(base: dict, exp: dict, ignore: tuple[str, ...]) -> list[str]
    """Sorted dotted keys whose value differs (or exists on one side only).
    `ignore` are top-level keys: ("name", "experiment")."""

def check_experiment(base_path, exp_path, ignore) -> list[str]
    """Problems, [] if ok. The experiment YAML must carry
        experiment: {base: configs/v0.yaml, variable: "<label>",
                     changed_keys: [dotted.keys], hypothesis: "<text>"}
    and config_diff(base, exp) must equal changed_keys exactly, and `base`
    must be the file passed in."""

def paired_flips(base_items: list[dict], exp_items: list[dict]) -> dict
    """Per query_id: improved / regressed / both_ungrounded / both_grounded /
    not_paired (a judge failed or output missing on either side). Returns the
    four lists plus counts."""

def mcnemar_exact(improved: int, regressed: int) -> float | None
    """Two-sided exact p; None when there are no flips.
    mcnemar_exact(4, 1) == 0.375; mcnemar_exact(6, 0) == 0.03125."""

def paired_bootstrap_ci(diffs: list[float], n_boot: int, seed: int, confidence: float) -> dict
    """{"mean", "lo", "hi", "n"}; seeded numpy resampling; None when n == 0."""

def compare_runs(conn, base_run_id: str, exp_run_id: str, versions: dict, cfg: dict,
                 *, allow_heldout: bool = False) -> dict
    """One experiment row. Refuses (ExperimentRefused) unless: both runs
    completed; same split; same test_set_sha256; both have rows for all three
    scorer versions; neither is held-out (unless allow_heldout); the runs
    differ. Contains: both ungrounded rates (all, answerable, with Wilson CI),
    delta in points, flip counts and lists (with the judge reasoning of each
    flipped answer), McNemar p, mean-faithfulness difference with bootstrap
    CI, bad-answer rate, false refusals, correct refusals on unanswerable,
    in_context / hit@4 / MRR, latency p50 / p95, failed outputs, config diff,
    index_key. Every number is computed from report.item_table() and stored
    scores; nothing is typed in."""

def format_table(rows: list[dict]) -> str       # terminal text
def format_markdown(rows: list[dict]) -> str    # experiments/experiment_table.md
def log_row(row: dict, created_at: str) -> str  # one experiments/log.md row
def main(argv: list[str] | None = None) -> int
```
CLI:
```
python -m evalharness.experiments check --config configs/v1_chunk75.yaml [--base configs/v0.yaml]
python -m evalharness.experiments compare --baseline 20261015T152638Z-v0-c7c96975 \
    --run <exp_run_id> [--run <exp_run_id> ...] \
    [--markdown experiments/experiment_table.md [--force]] [--log experiments/log.md]
```
- `check`: exit 0 ok; 1 problems printed.
- `compare`: prints the table, stores one `run_reports` row (kind
  `"experiment"`) per experiment run, optionally writes the markdown and
  appends log rows (refuses a duplicate run id in the log, like Step 10).
  Exit 0 ok; 1 refused with nothing written.
- No LLM call, no re-run, no re-scoring. Judges are never edited here.

Each experiment's run, scoring and report use commands that already exist
(the recipe, not a new orchestrator; one-command automation is Step 14):
```
python -m evalharness.experiments check --config configs/v1_x.yaml
git add configs/v1_x.yaml && git commit -m "..."        # clean tree: runner rule
python -m evalharness.runner --config configs/v1_x.yaml --split dev
python -m evalharness.checks --run-id <id>
python -m evalharness.judges faithfulness --run-id <id>
python -m evalharness.judges relevance --run-id <id>
```
All runs are done and judged first; `compare` and the log come at the end, so
the working tree stays clean between runs.

## Data and schema changes
No schema changes. `run_reports.kind` is free text; Step 10 documented it as
`"baseline"` with later kinds to come, and this step adds `"experiment"`. Only
the column comment in `evalharness/store.py` is updated.

New data files: none. `data/` stays untouched; `indexes/` gets one new
`.faiss` + `.chunks.json` pair per distinct (chunk_size, chunk_overlap)
setting, gitignored as before.

## Config changes

### `configs/v0.yaml` — new keys, all OFF or inert (Decision 2)
```yaml
retrieval:
  method: vector            # vector | bm25 | hybrid
  top_k: 4
  bm25:                     # used by bm25 and hybrid; rank_bm25's defaults
    k1: 1.5
    b: 0.75
  hybrid:                   # used only when method: hybrid
    fusion: rrf             # the only supported value
    rrf_k: 60               # the RRF constant (60 is the value from the original paper)
    pool_k: 10              # candidates taken from each of vector and BM25
  rerank:
    enabled: false
    pool_k: 10              # first-stage candidates handed to the cross-encoder
    model: cross-encoder/ms-marco-MiniLM-L-6-v2    # Decision 3
    revision: null          # pinned Hugging Face commit; filled when the model is first downloaded
    worker_timeout_s: 60

query_rewrite:
  enabled: false
  prompt_template: |        # must contain {question}; JSON/other braces doubled
    Rewrite the question below as a short search query for a company HR
    policy knowledge base. Keep every number, name and abbreviation. Reply with
    the query only.

    Question: {question}

generation:
  context_chunk_template: "{text}"   # {rank} (1-based) and {text}; v0 value keeps the v0 prompt
```
Reading rules: `bm25.k1` number > 0; `bm25.b` in [0, 1]; `hybrid.rrf_k`
integer >= 1; `hybrid.pool_k` and `rerank.pool_k` integers >= `top_k`;
`rerank.worker_timeout_s` integer >= 1; `rerank.enabled` and
`query_rewrite.enabled` booleans; both templates must format with only their
named fields. A pool larger than the chunk count is capped at the chunk count
at runtime (top_k itself already raises).

### `configs/harness.yaml` — new `experiments:` block
```yaml
experiments:
  base_config: configs/v0.yaml      # default for `check --base`
  ignore_keys: [name, experiment]   # top-level keys that never count as a change
  n_bootstrap: 2000                 # paired bootstrap resamples
  bootstrap_seed: 2029              # fixed, so a re-run prints the same interval
  table_path: experiments/experiment_table.md
```
The confidence level is the existing `checks.report_confidence` (0.95).

### New experiment configs (Decision 1)
Each is `configs/v0.yaml` plus an `experiment:` block and the single change.
The `hypothesis` is written **before** the run and committed with the config,
so git records it before the result exists.

| Config | `variable` | Change vs v0 |
|---|---|---|
| `v1_seed43.yaml` | generation.seed (noise control) | 42 -> 43 |
| `v1_chunk75.yaml` | chunking.chunk_size | 150 -> 75 |
| `v1_chunk300.yaml` | chunking.chunk_size | 150 -> 300 |
| `v1_overlap0.yaml` | chunking.chunk_overlap | 30 -> 0 |
| `v1_topk2.yaml` | retrieval.top_k | 4 -> 2 |
| `v1_topk6.yaml` | retrieval.top_k | 4 -> 6 |
| `v1_hybrid.yaml` | retrieval.method | vector -> hybrid (RRF) |
| `v1_rerank.yaml` | retrieval.rerank.enabled | false -> true (pool 10 -> top 4) |
| `v1_rewrite.yaml` | query_rewrite.enabled | false -> true |
| `v1_prompt_grounded.yaml` | generation.prompt_template | answer only from the context |
| `v1_prompt_citations.yaml` | generation.prompt_template + context_chunk_template | numbered chunks + cite as [n] |
| `v1_prompt_refusal.yaml` | generation.prompt_template | explicit refusal sentence |

Optional 13th run, created only after the 12 are read: `v2_combined.yaml`
(Decision 6).

Draft prompt wording (Decision 5). All keep the v0 skeleton: Context,
Question, Answer.
```
grounded:
  Answer the question using ONLY the context below. Do not use any other knowledge.
  Context: {context}  Question: {question}  Answer:

citations (context_chunk_template "[{rank}] {text}"):
  Use the following context to answer the question. After each fact in your
  answer, cite the number of the context chunk it comes from, like [2].
  Context: {context}  Question: {question}  Answer:

refusal:
  Use the following context to answer the question. If the context does not
  contain the answer, reply exactly: "The context does not contain this information."
  Context: {context}  Question: {question}  Answer:
```
The refusal sentence contains "does not contain", which is already in the
Step 6 refusal-phrase list, so the refusal heuristic recognises it.

## Files to create
- `rag/bm25.py`, `rag/reranker.py`, `rag/rerank_worker.py`, `rag/rewriter.py`
- `evalharness/experiments.py`
- `configs/v1_seed43.yaml`, `v1_chunk75.yaml`, `v1_chunk300.yaml`,
  `v1_overlap0.yaml`, `v1_topk2.yaml`, `v1_topk6.yaml`, `v1_hybrid.yaml`,
  `v1_rerank.yaml`, `v1_rewrite.yaml`, `v1_prompt_grounded.yaml`,
  `v1_prompt_citations.yaml`, `v1_prompt_refusal.yaml`
- `experiments/experiment_table.md` (generated by `compare --markdown`)
- `tests/test_11_experiments.py`
- `.claude/specs/11-experiments-one-variable-at-a-time.md` (this file)

## Files to change
- `configs/v0.yaml`: the inert keys above (and a comment that the keys are off).
- `configs/harness.yaml`: the `experiments:` block.
- `rag/retriever.py`: add `retrieve_with_method`; `retrieve` unchanged.
- `rag/pipeline.py`: flow, `build_prompt` chunk template, config validation,
  `describe_system` reranker fields, the module docstring (methods list).
- `evalharness/store.py`: the `run_reports.kind` column comment only.
- `experiments/log.md`: one row per experiment (written by `compare --log`).
- `tests/conftest.py`, `tests/README.md`: new fixtures and the Step 11 notes;
  `test_03` and `test_05` assertions that pin the exact v0.yaml key set or
  `SUPPORTED_METHODS` are updated (the v0 *behaviour* assertions stay).
- `CLAUDE.md`: Progress section.
- `.claude/specs/03-rag-pipeline-v0.md`: tick the libomp carry-forward with the
  decision taken.

## New dependencies
No new dependencies. `rank_bm25`, `sentence-transformers`, `scipy` and `numpy`
are already in `requirements.txt`. The reranker model weights are downloaded
once from Hugging Face at first use (data, not a package); see Decision 3.

## Tests
`tests/test_11_experiments.py`, written from this spec by `rag-test-writer`.
No test calls a real LLM, a real embedding endpoint or loads a real
cross-encoder; the worker is replaced by a fake that speaks the same JSON
lines.

- **v0 unchanged:**
  - the new `configs/v0.yaml` loads and has every new key off;
  - `build_prompt` with the v0 template and `"{text}"` gives the exact string
    the pre-Step-11 function gave (a golden string);
  - `answer()` with every switch off returns the same chunks and prompt as
    before on the fixture corpus, and its keys are exactly the contract keys.
- **bm25:** `tokenize` cases (case, punctuation, digits, empty); hand-checked
  ranking on a 3-chunk corpus; a query with no shared term returns `[]`;
  zero-score chunks excluded; tie order is the lower row; k larger than the
  corpus.
- **rrf_fuse:** the worked example (0.03252, 0.01639, 0.01613); a chunk in
  one ranking only; tie-break rule; `rrf_k` changes the order only where the
  formula says so.
- **hybrid retrieval:** fake embedder + tiny corpus; a chunk found by BM25
  only (exact code word) appears; result length = top_k; `method: bm25`
  ignores the embedder (embed never called).
- **rerank:** fake worker reorders by given scores; `top_k` cut; score field is
  the worker's score; wrong-length reply, worker exit and timeout each raise
  `RerankError`; question sent is the original question even when rewriting
  is on; the pool is `rerank.pool_k` and capped at the chunk count.
- **rewrite:** prompt contains the question; surrounding quotes stripped;
  generator receives the **original** question; latency includes the rewrite
  `model_latency_ms`; a blank/failed rewrite raises (no fallback).
- **config validation:** every new key removed in turn names the key; bad
  values (k1 0, b 1.5, pool_k < top_k, non-bool `enabled`, template with a
  stray `{}`); real `v1_*.yaml` files all load and pass `check`.
- **check_experiment / config_diff:** identical configs -> `[]` diff;
  `name` and `experiment` ignored; one changed value; key present on one
  side only; declared keys not equal to the real diff -> problem; wrong
  `base`; missing `experiment` block; two changes under one declared
  variable pass only if both are listed. Read-only check that every real
  `configs/v1_*.yaml` is exactly its declared diff against `configs/v0.yaml`.
- **mcnemar_exact:** 4/1 -> 0.375; 5/0 -> 0.0625; 6/0 -> 0.03125; 7/1 -> 0.0703125;
  3/3 -> 1.0; 0/0 -> None.
- **paired_bootstrap_ci:** a constant difference gives lo = hi = mean; same
  seed gives the same interval; zero pairs gives None.
- **compare_runs** on a hand-made DB (extends the `baseline_db` fixture with
  a second run): flip lists and counts by hand; pairs with a failed judge on
  either side go to `not_paired`, not to a rate; refuses different split,
  different `test_set_sha256`, a missing scorer version, a run not completed,
  held-out without the flag, the same run twice; rates and CI equal Step 10's
  for the same run; numbers equal a direct SQL recomputation; DB-tamper test
  (change a stored score, the table changes).
- **outputs:** `format_markdown` marks "uncalibrated judge", prints n next to
  every rate, never writes the word "significant" without a p-value below
  0.05, and prints "not statistically distinguishable" otherwise; `log_row`
  columns match `experiments/log.md`; duplicate run id refused; `compare`
  makes no LLM call (patched `llm.client.generate` raises).
- **layering:** AST guard: `rag/` imports nothing from `evalharness`;
  `evalharness/experiments.py` imports nothing from `rag` or `llm`;
  `rag/reranker.py` does not import torch or sentence_transformers (only the
  worker does).

## Rules for implementation
- Plain Python and only libraries approved in CLAUDE.md.
- No LangChain; no RAGAS or DeepEval (except the optional final cross-check).
- `rag/` must never import from `evalharness/`; the harness only calls `answer()`.
- All tunable values come from YAML config, with no hardcoded magic numbers
  (RRF constant, BM25 parameters, pool sizes, bootstrap settings, seeds).
- SQLite via the `sqlite3` module with parameterised queries only.
- All LLM calls go through `llm/client.py` so they are cached (the Step 2
  location; the old name `evalharness/llm.py` in the template no longer
  exists).
- API keys only from `.env`, never in code, logs or commits.
- The held-out split is never used before Step 12. `compare` and the runner
  keep refusing `heldout` without the flag, and this step never passes it.
- Never invent, estimate or adjust metric numbers; report only what a stored
  run produced. Hypotheses are written before runs; results are never edited
  to match them.
- Human labels (calibration and review) come only from Gaurav.
- Code is simple, readable and commented.

Specific to this step:
- **One run per experiment, one variable per config.** `check` must pass
  before a run is made. A config that fails it is fixed or declared as a
  combination (Decision 6), not run.
- **Judges and scorer versions are frozen** for the whole step (Decision 7).
  `compare` refuses runs scored by different versions. If a judge prompt must
  change, every run, including the baseline, is re-judged under the new
  version first.
- **Run the baseline's twin first.** After the `v0.yaml` edit, re-run
  `configs/v0.yaml` on dev. It must be 80 generator cache hits and give
  answers and chunk lists identical to run `20261015T152638Z-v0-c7c96975`
  (as Step 5 did). If not, the refactor changed v0 and nothing else proceeds.
- **No silent fallback.** A failed rewrite, rerank or token guard is a failed
  item, stored and counted. A reranker that quietly returns the first-stage
  order would make "rerank" results meaningless.
- **Count failures in the denominator.** Larger chunks or top_k may reach the
  3500-token guard; those items are failures, shown in the table, never
  dropped. Pairs with a failure on either side are `not_paired` and listed.
- **Report the trade-off.** The table always shows ungrounded, bad-answer
  rate, false refusals, correct refusals, retrieval metrics, latency, n, and
  failures. No row is labelled "better" from the ungrounded rate alone.
- **Be honest about power.** Say which flips are inside the noise (use E0),
  print the exact p, and never use "significant" unless p < 0.05. An
  improvement that is not significant is reported as a direction.
- **The reranker runs in a child process** (Decision 3); the parent process
  never imports torch. `KMP_DUPLICATE_LIB_OK` is never set.
- **Look at flips.** For every flipped answer the markdown shows the
  question, the answer and the judge's reasoning, so Gaurav can check by eye
  that a flip is real and not a judge artefact (q057 and q058 were already
  suspected in Step 7). Nothing is auto-corrected.
- **No tuning on held-out, no peeking.** Experiment configs never name a
  held-out query id.

## Decisions for Gaurav
1. **Experiment list and values.** Approve or edit the 12 configs in the
   table above, especially the values (chunk 75 and 300, overlap 0, top_k 2
   and 6, rerank pool 10, hybrid pool 10 and `rrf_k` 60). Cost: about 320
   calls per run (80 generator, up to 240 judge), 400 for the rewrite run, so
   about 4,000 calls for 12 runs plus the v0 twin. Ollama Free plan limits are
   unknown; the runner aborts after 5 failures in a row and every call is
   cached. Pure BM25 (no vector) is left out because CLAUDE.md lists only the
   hybrid; add it if you want the diagnostic.
   *Recommendation:* approve as drafted.
2. **Add inert keys to `configs/v0.yaml`?** With them (all off), every
   experiment differs from v0 by exactly the changed key and `check` is a
   plain diff. Without them, each experiment config would add new sections and
   the diff would be several keys. The old baseline run still holds its old
   `config_json`; the v0 twin run proves the behaviour is identical.
   *Recommendation:* add them.
3. **Reranker and the macOS libomp clash.** faiss-cpu and torch cannot share a
   process here (Step 3). Options:
   - **A: a child process for the cross-encoder** (spec as written). Keeps
     FAISS and v0 untouched. Costs a small worker script and one process.
   - **B: replace FAISS search with numpy** (exact inner product on 66
     vectors gives the same order). Removes the clash but changes the v0 code
     path and drops the FAISS library CLAUDE.md lists.
   - **C: skip the reranker** and record it as a limitation.
   Also decide the model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (about
   90 MB, English, fast) is the draft. It needs a one-time Hugging Face
   download; the revision hash is pinned in the config when downloaded.
   *Recommendation:* A with that model. If the child process also fails on
   Python 3.13, we stop and you choose between B and C.
4. **Noise-control experiment E0 (`v1_seed43`).** One extra run that only
   changes the seed, to show how much the numbers move with no real change.
   *Recommendation:* include it. Without it, a 1-2 answer difference cannot
   be read.
5. **Prompt wording.** Approve or edit the three drafts, and the citation
   format `[n]` with numbered context `[rank] text` (it matches the judge's
   own numbering). The citation run is judged by the unchanged judges, which
   may treat `[2]` as part of a claim; citation *correctness* is not scored in
   this project. *Recommendation:* approve; mention this limit in the README.
6. **Combination run and the rule for Step 12.** Single-variable results do
   not say what happens when winners are combined. Option: after the 12 runs,
   create `v2_combined.yaml` from the individually helpful changes (declared
   as a combination, not a one-variable experiment) and run it on dev once.
   Fix the selection rule **now**, before any result: the config for Step 12
   is the dev run with the lowest ungrounded rate among those whose bad-answer
   rate and false-refusal count are not worse than baseline; ties go to the
   fewer changes, then lower p50 latency.
   *Recommendation:* include the combination run and this rule.
7. **Judges stay frozen.** Both are uncalibrated (Step 9 inconclusive), so
   some flips may be judge artefacts. Options: (a) proceed, show flipped
   answers with reasoning for your eye check (spec as written); (b) first redo
   a careful labelling pass (Step 9 tool) on answers that flipped; judge
   version unchanged. Changing a judge prompt is not offered, because it
   forces re-judging all runs.
   *Recommendation:* (a), and say "uncalibrated judge" on every number.
8. **Query rewriting details.** The rewriter reuses `generation.model`
   (`gemma4:31b-cloud`, temperature 0, seed 42) and rewrites with the prompt
   in `v0.yaml`; the generator and the reranker still see the original
   question; the rewritten text is not stored in `outputs` (contract
   unchanged) and can be read from the `llm_calls` log. Alternative: a
   separate, smaller model for rewriting (adds a model digest to pin).
   *Recommendation:* reuse the generator model as drafted.

## Definition of done
- [ ] Done when (CLAUDE.md): an experiment table shows which changes helped
      and by how much. Run
      `python -m evalharness.experiments compare --baseline 20261015T152638Z-v0-c7c96975 --run <id> ... --markdown experiments/experiment_table.md --log experiments/log.md`.
      It prints and writes one row per experiment run with ungrounded rate
      and 95% Wilson interval, change in points, flips (improved / regressed),
      exact p, mean-faithfulness difference with bootstrap interval,
      bad-answer rate, false refusals, hit@4, MRR, latency p50 / p95, n and
      failures, each labelled "uncalibrated judge".
- [ ] `python -m evalharness.experiments check --config configs/<each v1 config>.yaml`
      exits 0 for all 12 configs.
- [ ] v0 twin: a new `configs/v0.yaml` dev run is `completed`, 80 ok, 80 cache
      hits, 0 misses, and
      `SELECT COUNT(*) FROM outputs a JOIN outputs b USING (query_id) WHERE a.run_id = '<baseline>' AND b.run_id = '<twin>' AND (a.answer != b.answer OR a.retrieved_chunks_json != b.retrieved_chunks_json)`
      returns 0.
- [ ] Each experiment run: `SELECT status, n_ok, n_failed FROM runs WHERE run_id = ...`
      shows `completed` with `n_ok + n_failed = 80`; the three scorers have
      80 rows each (`SELECT metric, COUNT(*) FROM scores WHERE run_id = ... GROUP BY metric`),
      0 judge failures or they are shown in the table; all runs share
      `test_set_sha256` and the three baseline scorer versions.
- [ ] `SELECT COUNT(*) FROM runs WHERE split = 'heldout'` returns 0.
- [ ] `SELECT MAX(id) FROM llm_calls` is unchanged by `compare` (no LLM call).
- [ ] Every number in `experiments/experiment_table.md` and
      `experiments/log.md` matches a stored `run_reports` row
      (`SELECT report_json FROM run_reports WHERE kind = 'experiment'`).
- [ ] One log row per experiment, each with the declared variable and run id.
- [ ] `experiments/experiment_table.md` states the combination rule from
      Decision 6 and which dev run it selects for Step 12 (no held-out run
      made in this step).
- [ ] `pytest` passes (full suite, offline).
- [ ] `/code-review` run on the change set (Steps 5, 6, 7, 8, 9 and 10 skipped
      it; this step is where the pipeline changes, so do not skip).
- [ ] The Progress section of CLAUDE.md is updated with the real numbers from
      the table, the Decision answers, and what the results do and do not
      show.
- [ ] All changes are committed on the feature branch
      `feature/11-experiments-one-variable-at-a-time`.
