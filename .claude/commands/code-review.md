---
description: Runs parallel security, quality and evaluation-integrity review for one step of the RAG eval harness. Pass the spec name, e.g. /code-review-feature 07-faithfulness-judge
argument-hint: "Spec name e.g. 07-faithfulness-judge"
allowed-tools: Read, Glob, Edit, Write, Agent, Task, Bash(git diff:*), Bash(git status:*), Bash(git ls-files:*), Bash(git branch:*)
---

Run the code review pipeline for the step whose spec is named in
$ARGUMENTS. Always follow the rules in CLAUDE.md.

If no argument is provided, stop immediately and say:
"Please provide a spec name. Usage: /code-review-feature <spec-name>
e.g. /code-review-feature 07-faithfulness-judge"

If `.claude/specs/$ARGUMENTS.md` does not exist, list the files in
`.claude/specs/` and stop with:
"Spec file not found at .claude/specs/$ARGUMENTS.md. Available specs: <list>"

## Pre-flight: collect the full change set

1. Run `git branch`. Set DEFAULT_BRANCH = `main` if it exists, otherwise
   `master`.
2. Run `git diff DEFAULT_BRANCH` — every change to tracked files on this
   branch, whether committed or not.
3. Run `git ls-files --others --exclude-standard` — new untracked files.
   Read each one in full and add it to the change set as a new file.
4. Combine these into CHANGE_SET.

If CHANGE_SET is empty, stop and say:
"No changes found compared with DEFAULT_BRANCH. Implement the step
before running code review."

If CHANGE_SET contains `.env`, a database, a FAISS index or cache files,
pass that fact to the security reviewer.

---

## Step 1: Parallel review

Invoke all three subagents at the same time, in a single message. Do not
wait for one to finish before starting the others. Each receives:
- CHANGE_SET
- Spec: `.claude/specs/$ARGUMENTS.md`
- Project rules: `CLAUDE.md`
- Code to reference: `rag/`, `evalharness/`, `dashboard/`, `configs/`,
  `tests/`

**rag-security-reviewer**
Instruction: Review only the changed code for security issues. Do not
comment on quality, style or metric correctness.

**rag-quality-reviewer**
Instruction: Review only the changed code for quality, Python best
practices, maintainability, and compliance with CLAUDE.md and the spec's
"Rules for implementation". Do not comment on security or on metric
correctness.

**rag-eval-integrity-reviewer**
Instruction: Review the changed code for anything that could make a
reported number wrong, misleading or unreproducible. Do not comment on
style or security.

---

## Step 2: Unified report

When all three have finished, combine their findings. De-duplicate: if
more than one reviewer flagged the same file and line, merge them into one
finding and note each perspective.

### Code Review Report — $ARGUMENTS

**Security findings**
[rag-security-reviewer output]

**Quality findings**
[rag-quality-reviewer output]

**Evaluation-integrity findings**
[rag-eval-integrity-reviewer output]

**Combined action plan**
One numbered checklist, ordered by severity across all reviewers:
Critical first, then High, then Medium, then Low.
Each item: [Reviewer] Severity — file:line — problem — proposed fix

**Overall verdict** — one of:
- APPROVED — no findings above Low. Ready for /ship-feature
- APPROVED WITH SUGGESTIONS — only Medium or Low findings. Can ship;
  note the suggestions for later steps
- CHANGES REQUESTED — at least one Critical or High finding. Must fix
  before /ship-feature

---

## Step 3: Ask for approval

Ask:
"Do you want me to implement the action plan? Reply 'all',
'critical and high only', or list the item numbers."

Wait for an explicit answer. Do not touch any file before it. Then:
- Fix only the approved items, in action-plan order.
- Before each fix, explain in one or two sentences what was wrong and why
  the fix solves it.
- Never resolve a finding by changing test expectations, loosening
  thresholds, skipping tests, or editing stored results.
- When done, list what was changed and tell the user to re-run
  `/test-feature $ARGUMENTS`.

---

## Rules
- Do NOT edit any file before the user approves
- All three reviewers must run in parallel
- Do NOT skip the pre-flight change-set collection
- Do NOT proceed if the spec file does not exist
- If any reviewer fails or returns no output, report it and do not
  present a partial review as complete