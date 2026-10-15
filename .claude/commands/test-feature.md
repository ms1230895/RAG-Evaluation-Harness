---
description: Writes and runs tests for one step of the RAG eval harness. Pass the spec name, e.g. /test-feature 07-faithfulness-judge
argument-hint: "Spec name e.g. 07-faithfulness-judge"
allowed-tools: Read, Glob, Agent, Task
---

Run the testing pipeline for the step whose spec is named in $ARGUMENTS.
Always follow the rules in CLAUDE.md.

If no argument is provided, stop immediately and say:
"Please provide a spec name. Usage: /test-feature <spec-name>
e.g. /test-feature 07-faithfulness-judge"

If `.claude/specs/$ARGUMENTS.md` does not exist, list the files in
`.claude/specs/` and stop with:
"Spec file not found at .claude/specs/$ARGUMENTS.md. Available specs: <list>"

Set TEST_FILE = `tests/test_<$ARGUMENTS with every "-" replaced by "_">.py`
Example: 07-faithfulness-judge → tests/test_07_faithfulness_judge.py

---

## Step 1: Write tests

Invoke the **rag-test-writer** subagent with:

- Spec to base tests on: `.claude/specs/$ARGUMENTS.md`
- Project rules: `CLAUDE.md`
- Output test file: TEST_FILE
- Shared fixtures file: `tests/conftest.py` (may add fixtures, must not
  break existing ones)
- Instruction: Write tests from what the spec says this step SHOULD do.
  Take names, signatures and import paths from the spec's Interfaces
  section. Do NOT derive expected values from the implementation.
  Never call a real LLM, never load the real embedding model, never touch
  the real database — use fakes and tmp_path.

Wait for rag-test-writer to finish and confirm the test file was written
and collected by pytest before proceeding to Step 2.

---

## Step 2: Run tests

Invoke the **rag-test-runner** subagent with:

- Test file to execute: TEST_FILE
- Spec for context: `.claude/specs/$ARGUMENTS.md`
- Source to analyze when diagnosing failures: `rag/`, `evalharness/`,
  `dashboard/`, `configs/`
- Run command: `python -m pytest TEST_FILE -v`
- Instruction: Run ONLY this test file. Classify every failure as BUG,
  MISSING, FAULTY TEST or SPEC UNCLEAR, with its root cause.

---

## Handoff rules

- Do NOT start Step 2 until Step 1 is fully complete
- Do NOT fix any code, whatever the results show
- Do NOT edit, weaken, skip or delete any test after it is written
- Do NOT run any test file other than TEST_FILE
  (the full suite runs later in /ship-feature)
- If rag-test-writer could not write the file, stop and report why —
  do NOT proceed to Step 2

---

## Final output

### Testing Pipeline Report — $ARGUMENTS

**Step 1 — Tests written** (TEST_FILE)
- Each test with a one-line description of the spec requirement it checks
- Any requirement that could not be tested, and why

**Step 2 — Test results**
- Mirror rag-test-runner's report exactly, including the pytest counts

**Verdict** — one of:
- ✅ Ready to ship — all tests pass. Next: /ship-feature
- ❌ Needs fixes — list failing tests with classification and root cause
- ⚠️ Spec needs clarifying — list SPEC UNCLEAR items; resolve these
  before trusting the other results