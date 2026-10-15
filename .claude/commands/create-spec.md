---
description: Create a spec file and feature branch for the next RAG eval harness step
argument-hint: "Step number and feature name e.g. 3 rag pipeline v0"
allowed-tools: Read, Write, Glob, Grep, Bash(git:*)
---

You are a senior ML engineer and a patient teacher helping Gaurav build the
"Evaluation Harness for RAG Output" project one step at a time. Always follow
the rules in CLAUDE.md. CLAUDE.md is the source of truth for the build
sequence, architecture, data model, libraries and conventions.

User input: $ARGUMENTS

## Step 1 — Check the working directory is clean
Run `git status`. If there are any uncommitted, unstaged or untracked files,
stop immediately and tell the user to commit or stash them first.
DO NOT CONTINUE until the working directory is clean.

## Step 2 — Parse the arguments
From $ARGUMENTS extract:

1. `step_number` — zero-padded to 2 digits: 3 → 03, 11 → 11
2. `feature_title` — human-readable, Title Case
   - Example: "RAG Pipeline v0" or "Faithfulness Judge"
3. `feature_slug` — git- and file-safe slug
   - Lowercase kebab-case, only a-z, 0-9 and -
   - Maximum 40 characters
   - Example: rag-pipeline-v0, faithfulness-judge
4. `branch_name` — format: `feature/<step_number>-<feature_slug>`
   - Example: `feature/03-rag-pipeline-v0`

If you cannot infer these from $ARGUMENTS, ask the user to clarify before
proceeding.

## Step 3 — Switch to the default branch and pull
- Run `git branch` and use `main` if it exists, otherwise `master`.
- Check it out.
- Run `git remote`. If `origin` exists, run `git pull origin <default_branch>`.
  If there is no remote, skip the pull and mention this in the final report.

## Step 4 — Check this step is next in the sequence
Read the "Build sequence" and "Progress" sections of CLAUDE.md on the
default branch.
- If this step is already marked complete, warn the user and stop.
- If any earlier step is not marked complete, tell the user that steps must
  be done in order, and that the previous step's branch may not have been
  merged into the default branch yet. Then stop.
- If the feature name does not match this step's title in CLAUDE.md, show
  both and ask the user to confirm before continuing.

## Step 5 — Check the branch name is not taken
Run `git branch`. If `branch_name` already exists, append a number:
`feature/03-rag-pipeline-v0-01`, `-02`, etc.

## Step 6 — Create and switch to the feature branch
Run `git checkout -b <branch_name>`

## Step 7 — Research the codebase
Read these before writing the spec (use Glob; some will not exist yet in
early steps, which is fine):
- `CLAUDE.md` — build sequence, architecture, data model, libraries, conventions
- `requirements.txt`
- All files in `configs/`
- All Python files in `rag/`, `evalharness/`, `dashboard/` and `tests/`
- `experiments/log.md`, if it exists
- All files in `.claude/specs/` — to stay consistent with earlier specs and
  avoid duplicating work

## Step 8 — Write the spec
Generate a spec document with this exact structure:

---
# Spec: Step <step_number> — <feature_title>

## Overview
One paragraph: what this step builds and why it is needed at this point in
the build sequence.

## Concepts to learn
The key ideas behind this step, explained in plain words from the basics
(for example: what an embedding is, why LLM calls are cached, what MRR
measures). Include any formula used, with a tiny worked example.

## Depends on
Which previous steps this requires, and the specific files and functions it
builds on.

## Interfaces
Every new or changed public function, class or CLI command: signature,
inputs, outputs, and one example call.
If this step touches the `answer()` contract, say so explicitly. The
contract must not change without the user's approval.

## Data and schema changes
New or changed SQLite tables or columns (verify against
`evalharness/store.py` if it exists), and any new data files with their
format (for example `data/test_set.jsonl`).
If none: state "No data or schema changes".

## Config changes
New YAML keys, their default values, and what each one controls.
If none: state "No config changes".

## Files to create
Every new file.

## Files to change
Every existing file that will be modified, and what changes.

## New dependencies
Only packages from the approved list in CLAUDE.md. If anything else seems
necessary, put it under "Decisions for Gaurav" instead.
If none: state "No new dependencies".

## Tests
The pytest tests to write and what each one checks.
Tests must never call a real LLM API; use a fake or mocked LLM.

## Rules for implementation
Always include:
- Plain Python and only libraries approved in CLAUDE.md
- No LangChain; no RAGAS or DeepEval (except the optional final cross-check)
- `rag/` must never import from `evalharness/`; the harness only calls `answer()`
- All tunable values come from YAML config — no hardcoded magic numbers
- SQLite via the sqlite3 module with parameterised queries only
- All LLM calls go through `evalharness/llm.py` so they are cached
- API keys only from `.env`, never in code, logs or commits
- The held-out split is never used before Step 12
- Never invent, estimate or adjust metric numbers; report only what a
  stored run produced
- Human labels (calibration and review) come only from Gaurav
- Code is simple, readable and commented
Then add any rules specific to this step.

## Decisions for Gaurav
Questions the user must answer before or during implementation (for
example provider choice, thresholds, corpus content).
If none: state "No open decisions".

## Definition of done
A specific, testable checklist. Each item must be verifiable by running a
command (a script, the CLI, an SQL query, or pytest). Always include:
- The "Done when" check for this step from CLAUDE.md
- `pytest` passes
- The Progress section of CLAUDE.md is updated
- All changes are committed on the feature branch
---

## Step 9 — Save the spec
Save to: `.claude/specs/<step_number>-<feature_slug>.md`

## Step 10 — Report to the user
Print a short summary in this exact format:
```
Branch:          <branch_name>
Spec file:       .claude/specs/<step_number>-<feature_slug>.md
Title:           Step <step_number> — <feature_title>
Open decisions:  <number of items under "Decisions for Gaurav">
Remote pull:     <done / skipped (no remote)>
```

Then tell the user:
"Review the spec at `.claude/specs/<step_number>-<feature_slug>.md`,
answer any open decisions, then enter Plan Mode with Shift+Tab twice to
begin implementation. When the step is finished, merge this branch into
the default branch before creating the next spec."

Do not print the full spec in chat unless explicitly asked.