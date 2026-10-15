---
description: Verify, commit, push, open PR, merge and clean up after a step is complete
allowed-tools: Read, Edit, Glob, Grep, Bash, mcp__github__create_pull_request, mcp__github__merge_pull_request
---

You are shipping a completed step of the "Evaluation Harness for RAG Output"
project. Always follow the rules in CLAUDE.md. Never claim a check passed
unless you actually ran it and saw it pass.

## Step 1 — Identify branches and remote
```bash
git branch --show-current
git branch
git remote
```
- Store the current branch as CURRENT_BRANCH.
- Store `main` as DEFAULT_BRANCH if it exists, otherwise `master`.
- If CURRENT_BRANCH equals DEFAULT_BRANCH, stop and say:
  "You are on the default branch. Run /create-spec to start a feature branch first."
- Store HAS_REMOTE = yes if `origin` exists, otherwise no.

## Step 2 — Find the spec
Branch names look like `feature/<step_number>-<feature_slug>`.
Find the matching file in `.claude/specs/` and read it.
If no matching spec exists, stop and tell the user.

## Step 3 — Verify the step is really done
1. Run `pytest`. If any test fails, stop and report the failures.
2. Go through every item in the spec's "Definition of done" and the step's
   "Done when" line in CLAUDE.md. Run the command that verifies each one.
   - If a check would make uncached real LLM API calls, ask the user
     before running it.
3. Record each item as PASSED or FAILED with a one-line note.
4. If any item FAILED, stop. Report what failed and what is needed.
   Do not commit, do not mark anything as done.

## Step 4 — Update CLAUDE.md Progress
If this step is not already recorded in the "Progress" section of CLAUDE.md,
add one entry:
- Step number and title, today's date, one-line summary of what was built
- Any metrics this step produced, copied exactly from the stored run, with
  the run_id. Never round differently, estimate or adjust numbers.

## Step 5 — Safety check before staging
```bash
git status --porcelain
```
Stop and tell the user (suggest the `.gitignore` fix) if anything about to be
committed matches:
- `.env` or any file containing secrets
- SQLite databases (`*.db`, `*.sqlite`, `*.sqlite3`)
- FAISS index files, LLM response caches, the virtual environment
- Any single file larger than 5 MB
Also scan the diff for strings that look like API keys (for example `sk-`,
`api_key=`, `Authorization: Bearer`). If found, stop.

## Step 6 — Generate the commit message
```bash
git diff --staged
git diff
git log DEFAULT_BRANCH..HEAD --oneline
```
Write a Conventional Commit message:
- feat: new capability
- fix: bug fix
- test: tests only
- refactor: code change with no behaviour change
- chore: config, tooling, experiment configs
- docs: documentation only

Rules:
- Lowercase, no period at the end, under 72 characters
- Describe what someone using the harness can now do, not which files changed

Good: "feat: score answers for faithfulness with claim-level judge"
Good: "feat: flag regressed answers for human review"
Bad:  "feat: added judges.py and updated store.py"

## Step 7 — Commit
```bash
git add -A
git commit -m "<generated-message>"
```
If there is nothing to commit (the work was already committed), skip this
and continue.
Report: "✓ Committed — <message>" or "✓ Nothing new to commit"

## Step 8 — Ship
### If HAS_REMOTE = yes
1. Push:
```bash
   git push -u origin CURRENT_BRANCH
```
   Report: "✓ Pushed — CURRENT_BRANCH"

2. Create a PR from CURRENT_BRANCH into DEFAULT_BRANCH using the GitHub MCP.
   If the GitHub MCP is not connected, check `gh auth status`; if the gh CLI
   is authenticated, use `gh pr create` instead. If neither works, stop and
   say: "GitHub is not connected. Run /mcp to check the GitHub MCP, or
   install and log in to the gh CLI."

   Title: plain English step name, no conventional-commit prefix.
   Example: "Step 07: Faithfulness judge"

   Description:
```markdown
   ## What this step does
   <one paragraph from the spec Overview>

   ## Concepts covered
   <one line per concept from the spec "Concepts to learn">

   ## Changes
   <bullet list of every file changed, one line each>

   ## Results
   <every metric this step produced, copied exactly from the stored run,
   with its run_id — or "No metrics in this step">

   ## Definition of done
   <the checklist from the spec; mark [x] only items that PASSED in Step 3>

   ## How to test
   <exact commands from the spec, for example:
   pytest
   python -m evalharness run --config configs/v0.yaml --split dev
   sqlite3 data/results.db "SELECT ...">
```
   Report: "✓ PR created — <PR URL>"

3. Ask the user: "PR is ready at <PR URL>. Merge it now?"
   Wait for a clear yes before continuing.

4. Squash-merge the PR (GitHub MCP, or `gh pr merge --squash`), using the
   generated commit message as the squash commit message.
   Report: "✓ PR merged to DEFAULT_BRANCH"

5. Delete the remote branch:
```bash
   git push origin --delete CURRENT_BRANCH
```
   Report: "✓ Remote branch deleted"

6. Update local default branch:
```bash
   git checkout DEFAULT_BRANCH
   git pull origin DEFAULT_BRANCH
```

### If HAS_REMOTE = no
1. Ask the user: "No GitHub remote found. Squash-merge CURRENT_BRANCH into
   DEFAULT_BRANCH locally?" Wait for a clear yes.
2. Run:
```bash
   git checkout DEFAULT_BRANCH
   git merge --squash CURRENT_BRANCH
   git commit -m "<generated-message>"
```
   Report: "✓ Squash-merged locally into DEFAULT_BRANCH"

## Step 9 — Delete the local feature branch
```bash
git branch -D CURRENT_BRANCH
```
Report: "✓ Local branch deleted"

## Final summary
Read the Progress and Build sequence sections of CLAUDE.md to find the next
step, then print:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
/ship-feature complete
✓ Verified — <n> of <n> done-checks passed, pytest green
✓ Committed — <message>
✓ Pushed and PR merged   (or: ✓ Squash-merged locally)
✓ Branches cleaned up
✓ On DEFAULT_BRANCH — up to date
Next: /create-spec <next step number> <next step name>
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

## Rules
- Never commit directly to the default branch
- Never ship if pytest or any done-check fails
- Never mark a checklist item [x] unless it was verified in Step 3
- Never commit secrets, databases, indexes, caches or files over 5 MB
- Never invent, round differently or adjust metric numbers
- Always squash merge
- Never merge without the user's explicit yes
- Never proceed to merge if PR creation fails
- Always delete both the remote and local branch after a merge