# Spec: Step 04 — Test Set

## Overview
This step builds the fixed yardstick every later number is measured with:
100 questions about the Kestrel Labs HR corpus in `data/test_set.jsonl`,
each with a type, a reference answer, the IDs of the documents that answer
it (empty for unanswerable questions), a split (`dev` or `heldout`) and a
`reviewed` flag. It also adds the pydantic schema that validates every line
(`evalharness/schemas.py`) and a small module (`evalharness/testset.py`)
that loads the file, checks it, assigns the stratified split once, helps
Gaurav review each query, and prints a summary. It comes now because the
pipeline (Step 3) exists to be evaluated, and the runner (Step 5), the
retrieval metrics (Step 6) and both judges (Steps 7-8) all need questions
with known right answers and known source documents. Claude drafts the
queries by reading the corpus; Gaurav reviews and edits every one. No LLM
API is called in this step.

## Concepts to learn
- **Test set (golden set):** a fixed list of questions with known answers.
  It must not change between runs, otherwise a change in the ungrounded
  rate could come from different questions instead of a better pipeline.
  Once Step 5 records runs against it, the file is frozen: a query is never
  edited in place, and a removed query's ID is retired, never reused.
- **Query types, and why each exists:**
  - `lookup`: one fact from one place in one document ("How many casual
    leaves does a full-time employee get per year?"). Tests the basic case.
  - `multi_chunk`: the answer needs facts from two or more chunks, in one
    document or across documents ("What is the notice period for a B3
    employee still on probation, and does it change after
    confirmation?"). Tests whether top-k brings in all the pieces, and
    whether the LLM combines them without inventing a link.
  - `paraphrased`: a lookup asked in different words or vaguely, with few
    or none of the document's keywords ("Can I take a day off as a
    trainee?"). Tests whether retrieval works on meaning, not word overlap.
  - `unanswerable`: the corpus truly does not cover it ("Does Kestrel Labs
    offer stock options?"). The right behaviour is a refusal. These are
    where a RAG system is most tempted to hallucinate, so they matter most
    for the ungrounded rate.
- **Source doc IDs as ground truth:** each answerable query stores the doc
  IDs that contain its answer. Step 6 compares them with the `doc_id` of
  retrieved chunks to compute hit rate@k and MRR with plain Python, no LLM.
  Example: sources `["intern_leave_policy"]`, retrieved docs in rank order
  `[leave_policy, intern_leave_policy, ...]` → hit@1 = 0, hit@2 = 1,
  reciprocal rank = 1/2. An unanswerable query has `[]` and is left out of
  retrieval metrics (there is nothing to find).
- **Reference answer:** a short, correct answer written from the source
  documents, with the exact numbers. Later steps can compare an answer with
  it; it also makes Gaurav's review concrete (check the numbers against the
  doc).
- **Dev vs held-out split:** we tune the pipeline (Step 11) by looking at
  dev results. Every look leaks a little information, so the tuned config
  ends up fitted to the dev questions and its dev score is optimistic. The
  20 held-out queries are run exactly once, in Step 12, to get an honest
  "after" number on unseen questions.
- **Stratified split:** the held-out set has the same type mix as the whole
  set. Without stratification, 20 random picks could, by chance, contain
  only 1 unanswerable query, and the held-out ungrounded rate would mostly
  measure luck. With 50/20/15/15 and 20% held out per type, the held-out
  set is exactly 10 lookup, 4 multi_chunk, 3 paraphrased, 3 unanswerable.
  The picks are random with a fixed seed (not hand-picked, so nobody can
  quietly hold out the easy ones) and done once, after review.
- **Why sample size matters (Wilson interval, a preview of Step 10):** for
  `k` bad answers out of `n`, with `p = k/n` and `z = 1.96`:
  `centre = (p + z²/2n) / (1 + z²/n)`,
  `half_width = z·sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n)`.
  Worked example (arithmetic only, not a measured result): 3 of 12 dev
  paraphrased answers ungrounded → p = 0.25, interval about 8.9% to 53.2%.
  The same 25% on all 80 dev queries (20/80) → about 16.8% to 35.5%; 12.5%
  (10/80) → about 6.9% to 21.5%. Two things follow: per-type rates on 12-16
  queries are very rough, and even a 25% → 12.5% drop on 80 queries gives
  overlapping intervals. Reports must say this plainly.
- **Validation with pydantic:** each JSONL line is parsed into a `Query`
  model that rejects unknown fields, wrong types (`"reviewed": "true"`),
  unknown query types and broken rules (an unanswerable query with source
  docs). Bad data fails loudly at load time, with the line number, instead
  of skewing metrics later.
- **Review provenance:** "reviewed" means Gaurav read the query against
  its source document and accepted or edited it. Claude writes every draft
  with `"reviewed": false` and never sets it to `true`. Committing the
  untouched draft first makes `git diff` show exactly what Gaurav changed.

## Depends on
- Step 1: the corpus in `data/docs/<doc_id>.md` (22 docs). Doc IDs are the
  filenames; the deliberate gaps (ESOPs/stock options, gym or wellness
  membership, pet policy, sabbatical, relocation allowance, office
  shuttle/transport, office canteen/meals) and confusable pairs are listed
  in CLAUDE.md Progress and `.claude/specs/01-document-corpus.md`.
- Step 0: `pydantic`, `pyyaml`, `pytest` installed; `evalharness/` package
  exists (only `__init__.py`).
- Steps 2-3 are not used at runtime. `evalharness/testset.py` must not
  import `rag/` or `llm/`: it lists doc IDs by reading filenames in the
  docs directory named in its own config.

## Interfaces
This step does not touch the `answer()` contract.

`evalharness/schemas.py`
```python
QueryType = Literal["lookup", "multi_chunk", "paraphrased", "unanswerable"]
Split = Literal["dev", "heldout"]

class Query(BaseModel):
    """One test-set line. model_config: extra="forbid", strict=True."""
    id: str                     # pattern ^q\d{3}$, e.g. "q007"
    question: str               # non-blank after strip
    type: QueryType
    reference_answer: str       # non-blank after strip
    source_doc_ids: list[str]   # each matches ^[a-z0-9_]+$, no duplicates
    split: Split | None         # None only while drafting / before assign-splits
    reviewed: bool
    # model_validator: type == "unanswerable"  <=>  source_doc_ids == []
```
Example: `Query.model_validate_json('{"id": "q001", "question": "...", "type": "lookup", "reference_answer": "...", "source_doc_ids": ["leave_policy"], "split": null, "reviewed": false}')`

`evalharness/testset.py`
```python
def load_testset_config(path: str | Path = "configs/test_set.yaml") -> dict:
    """Read and check configs/test_set.yaml (required keys, counts are
    non-negative ints, every type present). ValueError names the problem."""

def load_test_set(path: str | Path) -> list[Query]:
    """Parse JSONL, one Query per line. Raises ValueError naming the line
    number on bad JSON or a validation error, and on a duplicate id or a
    duplicate question (compared lowercased, whitespace collapsed).
    Blank lines are an error, except a trailing newline."""

def save_test_set(queries: list[Query], path: str | Path) -> None:
    """Write JSONL sorted by id, fields in schema order, one object per line,
    ensure_ascii=False. Writes a temp file then os.replace (atomic)."""

def corpus_doc_ids(docs_dir: str | Path) -> set[str]:
    """Stems of *.md files in docs_dir (no file contents read)."""

def check_doc_ids(queries: list[Query], known: set[str]) -> list[str]:
    """One problem string per unknown source doc ID; [] if all known."""

def counts(queries: list[Query]) -> dict[str, dict[str, int]]:
    """{type: {"dev": n, "heldout": n, "unassigned": n}} for all 4 types."""

def check_ready(queries: list[Query], config: dict) -> list[str]:
    """Every reason the set is not final: unreviewed ids, split None,
    type x split counts that differ from config, unknown doc IDs.
    [] means ready."""

def filter_split(queries: list[Query], split: Split) -> list[Query]:
    """Queries of one split, in id order. Step 5's runner uses this."""

def assign_splits(queries: list[Query], config: dict) -> list[Query]:
    """Once only. Per type: ids sorted, random.Random(split_seed).sample()
    picks expected_counts[type]["heldout"] ids as heldout, the rest dev.
    Raises ValueError if any query already has a split, if any is not
    reviewed, or if a type's total differs from config. Returns new Query
    objects; no other field changes."""

def review(queries: list[Query], path, docs_dir, input_fn=input) -> list[Query]:
    """Interactive review (Decision 2, option A). For each unreviewed query
    in id order: print id, type, question, reference answer, source doc IDs
    and the full text of each source doc. Keys: [a]ccept -> reviewed=True;
    [e]dit -> choose field (question, reference_answer, type,
    source_doc_ids), enter new value, re-validate, show again;
    [d]elete -> remove (id retired); [g]rep -> search the corpus for a
    word; [s]kip; [q]uit. Saves after every
    change with save_test_set. Only a key pressed by the person sets
    reviewed=True."""
```
CLI:
```
python -m evalharness.testset summary         # counts table, reviewed n/100, queries per doc, problems; exit 1 if not ready
python -m evalharness.testset review          # interactive review of unreviewed queries
python -m evalharness.testset assign-splits   # once, after review; writes split into the file
```
All three read `configs/test_set.yaml` (override with `--config`).
`summary` also lists, for `lookup` and `paraphrased` queries, any number in
the reference answer that is not found in the text of its source docs
(commas removed on both sides), as a review aid.

## Data and schema changes
No SQLite changes. The `queries` table from CLAUDE.md's data model is
created and filled in Step 5 (`evalharness/store.py`), from this file.

New file `data/test_set.jsonl` (committed; not matched by `.gitignore`),
UTF-8, one JSON object per line, sorted by id:
```json
{"id": "q001", "question": "How many casual leaves does a full-time employee get in a year?", "type": "lookup", "reference_answer": "8 casual leaves per calendar year.", "source_doc_ids": ["leave_policy"], "split": "dev", "reviewed": true}
{"id": "q091", "question": "Does Kestrel Labs offer stock options to employees?", "type": "unanswerable", "reference_answer": "The Kestrel Labs HR policies do not cover this.", "source_doc_ids": [], "split": "heldout", "reviewed": true}
```
(Illustrative lines only; the real questions and answers come from the
draft and Gaurav's review.)

Target content (Decisions 1, 3, 4):
| type | total | dev | heldout |
|---|---|---|---|
| lookup | 50 | 40 | 10 |
| multi_chunk | 20 | 16 | 4 |
| paraphrased | 15 | 12 | 3 |
| unanswerable | 15 | 12 | 3 |
| **all** | **100** | **80** | **20** |

Drafting guidelines:
- `lookup` / `paraphrased`: exactly 1 source doc, a reference answer that
  quotes the doc's numbers and terms (so the number check can verify it).
  Lookup questions do not copy a doc sentence word for word; paraphrased
  ones avoid the doc's key terms on purpose.
- `multi_chunk`: 2 or more chunks; `source_doc_ids` lists every doc needed.
- Every one of the 22 docs is a source for at least one query.
- About 12 queries (across types) target the confusable pairs, e.g. intern
  vs employee leave, travel vs expense claims, probation vs resignation
  notice periods.
- `unanswerable`: see Decision 3. Every one is checked as absent with
  `grep -rniw` on the corpus before it is written, and the grep terms are
  shown to Gaurav in the review.

## Config changes
New file `configs/test_set.yaml`:
```yaml
# Test set location and the exact shape it must have (Step 4).
path: data/test_set.jsonl          # relative to the repo root
docs_dir: data/docs                # only filenames are read, to check source_doc_ids
split_seed: 2026                   # random.Random seed for assign-splits; used once
expected_counts:                   # type -> split -> number of queries
  lookup:       {dev: 40, heldout: 10}
  multi_chunk:  {dev: 16, heldout: 4}
  paraphrased:  {dev: 12, heldout: 3}
  unanswerable: {dev: 12, heldout: 3}
```
`configs/v0.yaml` and `configs/llm.yaml` are unchanged.

## Files to create
- `configs/test_set.yaml`
- `evalharness/schemas.py`
- `evalharness/testset.py`
- `data/test_set.jsonl`
- `tests/test_04_test_set.py`

`evalharness/testset.py` is not in CLAUDE.md's repo layout; it is added
because loading, checking and splitting the test set is its own job and
`schemas.py` should hold only models (judge schemas join it in Step 7).

## Files to change
- `tests/README.md`: the Step 4 test file and any fixtures it adds.
- `CLAUDE.md`: Step 4 entry in the Progress section (real counts from
  `summary`, test count, review notes).

## New dependencies
No new dependencies. `pydantic` and `pyyaml` are already pinned.

## Tests
`tests/test_04_test_set.py`. No LLM, no network, no `rag/` import. Unit
tests build small JSONL files in `tmp_path`.

Schema (`Query`)
- `test_valid_query_passes`: a correct lookup and a correct unanswerable.
- `test_extra_field_rejected`, `test_bad_type_rejected`,
  `test_bad_split_rejected`, `test_bad_id_rejected` (`"7"`, `"q7"`,
  `"Q007"`, `"q0007"`).
- `test_blank_question_or_answer_rejected`: `""` and `"   "`.
- `test_strict_types`: `"reviewed": "true"` and `"source_doc_ids": "leave_policy"` rejected.
- `test_unanswerable_needs_no_sources` and `test_answerable_needs_sources`
  (for each of the 3 answerable types).
- `test_duplicate_or_bad_doc_id_rejected`.

Loader and writer
- `test_load_reports_line_number`: a bad line 3 → `ValueError` mentioning
  line 3.
- `test_duplicate_id_rejected`, `test_duplicate_question_rejected` (case and
  spacing differ).
- `test_blank_line_rejected_but_trailing_newline_ok`.
- `test_save_load_roundtrip`: equal after save then load; output sorted by
  id; no temp file left behind.

Checks
- `test_check_doc_ids_names_unknown`.
- `test_counts_shape`: all 4 types present, also when a type has 0 queries.
- `test_check_ready_lists_each_problem`: unreviewed id, `split` None, a
  count mismatch and an unknown doc ID each appear; a correct set gives
  `[]`.
- `test_load_testset_config`: real `configs/test_set.yaml` loads; a missing
  key or a type missing from `expected_counts` raises `ValueError`.

Split assignment
- `test_assign_splits_exact_counts`: per type, heldout and dev match
  config.
- `test_assign_splits_deterministic`: same seed → same ids; another seed →
  different ids (on a set large enough that this is certain to differ).
- `test_assign_splits_refuses`: any split already set; any unreviewed
  query; a type total that differs from config.
- `test_assign_splits_changes_only_split`.
- `test_filter_split`.

Review tool (`input_fn` fake that returns scripted keys)
- `test_review_accept_sets_reviewed_and_saves`.
- `test_review_skip_leaves_unreviewed`.
- `test_review_edit_revalidates`: an edit to an invalid value (unanswerable
  with sources) is rejected and the query is unchanged.
- `test_review_delete_and_quit`: deleted id is gone from the saved file;
  quit saves what was done so far.
- `test_review_grep_prints_matches`: `g` then a word prints the matching
  corpus lines (whole word, any case) and changes nothing in the file.

Real file (these fail until Gaurav's review and `assign-splits` are done;
that is intended, the step is not finished before then)
- `test_real_test_set_loads`: `data/test_set.jsonl` loads with no error.
- `test_real_test_set_ready`: `check_ready(...) == []` (all reviewed, all
  split, counts exact, doc IDs known).
- `test_real_numbers_in_sources`: for `lookup` and `paraphrased`, every
  number in the reference answer appears in its source docs.
- `test_real_every_doc_used`: each corpus doc ID is a source at least once.

Architecture
- `test_testset_does_not_import_rag_or_llm`: `evalharness/schemas.py` and
  `evalharness/testset.py` contain no `import rag`/`from rag`/`llm` imports.
- Existing `test_rag_does_not_import_evalharness` still passes.

## Rules for implementation
- Plain Python and only libraries approved in CLAUDE.md
- No LangChain; no RAGAS or DeepEval (except the optional final cross-check)
- `rag/` must never import from `evalharness/`; the harness only calls `answer()`
- All tunable values come from YAML config — no hardcoded magic numbers
- SQLite via the sqlite3 module with parameterised queries only
- All LLM calls go through `llm/client.py` so they are cached (none in this step)
- API keys only from `.env`, never in code, logs or commits
- The held-out split is never used before Step 12
- Never invent, estimate or adjust metric numbers; report only what a
  stored run produced
- Human labels (calibration and review) come only from Gaurav
- Code is simple, readable and commented

Step-specific:
- Claude drafts queries by reading `data/docs/` directly. No LLM API call
  is made to draft, check or paraphrase queries.
- Every drafted query has `"reviewed": false` and `"split": null`. Claude
  never sets `reviewed` to `true` and never edits a reviewed query without
  Gaurav asking for that specific change.
- Commit the untouched draft before review starts, so `git diff` shows
  Gaurav's edits.
- Every reference answer is checked against its source docs while
  drafting; numbers are copied, never rounded or computed unless the
  question asks for a computation (only `multi_chunk` may do that).
- Every unanswerable query is confirmed absent from the corpus with
  `grep -rniw` before drafting. Questions about meals must ask about the
  office canteen, not travel meals (`travel_policy` covers per-diem meals).
- Splits are assigned once, after all queries are reviewed, by
  `assign-splits` with the config seed. Never hand-pick or re-roll held-out
  queries.
- No pipeline run, retrieval or answer is produced in this step, on any
  split. Do not run `answer()` on test-set questions to "check" them: that
  is tuning on the test set before the baseline exists.
- After this step the corpus and test set change together or not at all
  (Step 1 rule): a doc fact edit updates the affected reference answers in
  the same commit.
- `evalharness/schemas.py` and `evalharness/testset.py` import neither
  `rag` nor `llm`.

## Decisions for Gaurav
All resolved on 2026-10-15; Gaurav accepted every recommendation:
1. ~~Exact counts and split.~~ Exactly 50 lookup / 20 multi_chunk / 15
   paraphrased / 15 unanswerable; held out 10 / 4 / 3 / 3 (20%, stratified).
2. ~~How Gaurav reviews.~~ **Option A:** the
   `python -m evalharness.testset review` tool, plus a `[g]rep` key (added
   in planning): Gaurav types a word and the tool prints every corpus line
   containing it (case-insensitive, whole word), so he can confirm an
   unanswerable query's topic is absent himself.
3. ~~Unanswerable mix.~~ **Option A:** 10 from the 7 gap topics + 5 near
   misses, each confirmed absent with `grep -rniw`.
4. ~~Multi-chunk definition.~~ 2 or more chunks; at least 10 of the 20 span
   2 or more documents; `source_doc_ids` lists all docs needed.
5. ~~Standard unanswerable reference answer.~~ "The Kestrel Labs HR
   policies do not cover this." for all 15.
6. ~~Split timing.~~ Random seeded split (`split_seed: 2026`), once, after
   review.

## Carry-forward to later steps
- **Step 5:** create the `queries` table from `data/test_set.jsonl`; store
  the file's SHA-256 with each run (the `runs` table has no column for it:
  decision for Gaurav then); the runner refuses `--split heldout` until
  Step 12 unless an explicit flag is passed; refuse to run if
  `check_ready()` is not `[]`.
- **Step 6:** retrieval metrics skip unanswerable queries (no sources) and
  say so in the report; decide "any" vs "all" for multi-doc hit@k.
- **Steps 7-8:** the relevance judge treats a refusal on an unanswerable
  query as relevant; the reference sentence from Decision 5 is a guide, not
  an exact match target.
- **Steps 10-12:** per-type n is 12-16 on dev and 3-10 on held-out; report
  per-type rates with their Wilson intervals and do not draw per-type
  conclusions from the held-out set.
- **Limitation for the README (Step 16):** the queries were drafted by
  Claude from the same corpus and reviewed by one person; they may be
  easier or more uniform than real employee questions.

## Definition of done
- [ ] Done when (CLAUDE.md): the file loads, every query validates and is
      reviewed, and the type counts are correct. Run
      `python -m evalharness.testset summary`: it prints 100 queries, the
      type × split table exactly as in Data and schema changes,
      `reviewed 100/100`, no problems, and exits 0 (`echo $?` prints `0`).
- [ ] `git log --oneline -- data/test_set.jsonl` shows the draft commit
      (all `reviewed: false`) before the reviewed version.
- [ ] Held-out was assigned by the tool: re-running
      `python -m evalharness.testset assign-splits` refuses because splits
      are already set.
- [ ] `pytest tests/test_04_test_set.py` passes with no network access.
- [ ] `pytest` passes (whole suite).
- [ ] `grep -rnE "^(import|from) (rag|llm)" evalharness/schemas.py evalharness/testset.py`
      prints nothing, and `grep -rn "evalharness" rag/` prints nothing.
- [ ] `git ls-files data/test_set.jsonl configs/test_set.yaml` lists both.
- [ ] The Progress section of CLAUDE.md has a Step 4 entry with the counts
      from `summary`, the test count, and what changed in review.
- [ ] All changes are committed on `feature/04-test-set`.
