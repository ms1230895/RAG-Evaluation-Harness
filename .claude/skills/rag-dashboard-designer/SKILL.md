---
name: rag-dashboard-designer
description: Designs and builds the Streamlit + Plotly UI for the RAG evaluation harness. That covers the Step 9 hand-labeling tool for judge calibration, and the Step 15 dashboard (trends across runs, run-vs-run comparison, review queue, breakdown by query type) in dashboard/app.py. Produces clean, honest, readable evaluation views. Every rate shows its sample size and 95% Wilson interval, numbers come only from the results database, and dev and held-out runs are never mixed. Use this skill whenever the user asks to design, build, improve or style any dashboard page, chart, labeling screen or review queue for this project. That includes phrasings like "build the trends page", "make the review queue", "chart the ungrounded rate", "improve the dashboard" and "labeling tool for calibration", even when Streamlit is not named.
---

# RAG Evaluation Dashboard Designer

You are building the UI for the **Evaluation Harness for RAG Output**. This is a Streamlit + Plotly app that reads the harness's SQLite results database and lets Gaurav review answers. It is an evaluation tool, so **honesty beats prettiness**. A chart that makes a noisy 3-point change look like a breakthrough is a bug, even if it looks great.

CLAUDE.md is the source of truth for the data model, the rules and the build sequence. Read it, and read the current step's spec in `.claude/specs/`, before designing anything.

## The stack (do not add to it)

- **UI:** `streamlit` (pinned in `requirements.txt`), single entry point `dashboard/app.py`, run with `streamlit run dashboard/app.py`.
- **Charts:** `plotly` (use `plotly.express` first, and `plotly.graph_objects` only when express cannot do it).
- **Data:** `pandas`, plus the project's own functions:
  - DB reads and the `reviews` writes go through `evalharness/store.py`.
  - Rates, Wilson intervals and breakdowns come from `evalharness/metrics.py`. These are the *same* functions the CLI reports use, so the dashboard and the README can never disagree.
- Do **not** add Altair charts, streamlit-aggrid, streamlit-extras, Dash, React components or any other package. New libraries need Gaurav's approval (CLAUDE.md rule 3).

## Before you design: check what exists

Open `evalharness/store.py`, `evalharness/metrics.py`, the latest spec, and `dashboard/app.py` if it exists. Reuse their function names, column names (`run_id`, `query_id`, `metric`, `value`, `reasoning`, `split`, `judge_version`) and any helpers already written. If a query or calculation you need does not exist, add it to `store.py` or `metrics.py` as a plain function, not inline in the UI code.

## Structure of `dashboard/app.py`

- Use one function per page (`page_trends()`, `page_compare()`, `page_review_queue()`, `page_breakdown()`, and `page_label()` for Step 9). Wire them with `st.navigation([st.Page(...), ...])` and a sidebar.
- Keep a thin layout layer. Pages call data functions, which return DataFrames, and pass them to small `make_*_figure(df)` functions that return a Plotly figure, which is rendered. Keeping the figure builders and data functions free of `st.*` calls makes them testable with pytest.
- Wrap DB reads in `@st.cache_data(ttl=60)`. After writing a review, call `st.cache_data.clear()` (or clear that one function's cache) so the queue updates.
- Keep UI state (the current item index, filters) in `st.session_state`.
- Build paths from `Path(__file__)`, and read the DB path from the same config the CLI uses. No hardcoded `"results.db"`.

## Honesty rules for every view (non-negotiable)

1. **Numbers come only from stored runs.** Never compute a metric differently from `metrics.py`, never smooth, interpolate or extrapolate, and never hardcode example numbers. Use placeholders only in clearly labeled mockups.
2. **Every rate shows n and its 95% Wilson interval.** Draw intervals as error bars (`error_y` / `error_y_minus`) or as a shaded band. Show `k/n` in the hover and in tables, for example `9/80 = 11.2% [6.0%, 20.0%]`.
3. **Use full axes for rates.** The y-axis runs from 0% to 100%, or starts at 0 and ends at a clearly labeled max. Never use a truncated axis that exaggerates a change.
4. **Never mix splits.** Dev and held-out runs never share a trend line. By default, held-out runs are shown separately, labeled "held-out (Step 12 confirmation)", because they are run once.
5. **Judge versions break comparability.** When `judge_version` changes between runs, mark it on the trend chart (a vertical line or annotation). When comparing two runs with different judge versions or splits, show an `st.warning` that the comparison is not like-for-like.
6. **Warn when samples are small.** Per-type breakdowns have small n (for example 12 unanswerable queries in dev). Show the n next to every bar, and add a caption when intervals overlap heavily: "Difference is within noise at this sample size."
7. **Use the project's definitions.** "Ungrounded" means faithfulness below the configured threshold *or* any contradicted claim. Read the threshold from the run's stored config, not from a constant in the UI.

## Page guide

**Trends across runs.** Show the ungrounded rate, mean faithfulness, mean relevance, hit rate@k and MRR over runs, ordered by timestamp. The hover shows run_id, config name, git commit (short), judge_version and k/n. Include a metric selector, and dev only by default.

**Run vs run.** Two run selectors, defaulting to the latest run and the last accepted run. Show a summary row of deltas, each with both CIs, then tables of queries that **improved**, **regressed** and **changed answer**. Mark them with a text label or icon *and* color, never color alone. Clicking a query shows both answers side by side.

**Review queue.** One flagged answer at a time, with a progress line like `12 / 40 reviewed`. In order, show:
- the question and query type;
- the reference answer;
- the generated answer;
- the retrieved chunks in an expander, each showing `doc_id`, `chunk_id` and score;
- the judge reasoning per claim;
- the flag reasons.

Provide **OK** / **Bad** buttons and an optional notes box. They save to `reviews` through a `store.py` function, then advance to the next item. The page never modifies `scores` or `outputs`.

**Breakdown by query type.** A grouped bar chart of the ungrounded rate per type (lookup / multi_chunk / paraphrased / unanswerable), with CIs and n labels. Split it by failure cause (retrieval miss vs generation error) where Step 10 data exists.

**Step 9 labeling tool.** This is like the review queue, with one big difference: **hide the judge's score, verdict and reasoning while Gaurav labels.** Showing them anchors the human label, which makes the agreement figure meaningless. Labels are grounded / ungrounded plus notes, and are saved via `store.py`. Show the judge output only after a label is saved, or not at all. Human labels must come only from Gaurav: never pre-fill or suggest a label.

## Visual language

- Use Streamlit's default theme and the `plotly_white` template. Keep the layout restrained: generous spacing, one idea per chart.
- Give each metric one consistent, colorblind-safe color across all pages. Use the Okabe-Ito palette: blue `#0072B2`, orange `#E69F00`, green `#009E73`, vermillion `#D55E00`, purple `#CC79A7`, sky `#56B4E9`.
- Use the same color for a given query type everywhere.
- Format rates as percentages with 1 decimal, scores with 2 decimals, latency in ms with no decimals.
- Use clear titles that state what the chart shows ("Ungrounded rate per run, dev split, 95% CI"), and label axes with units.
- Use `st.metric` for headline numbers. Put the CI in the `help` text or a caption, and do not let the delta arrow imply significance it does not have.
- Make charts fill the container width. Use whichever width argument the pinned Streamlit version supports without deprecation warnings.
- Do not use pie charts, 3D charts, dual y-axes or animations.

## Safety

- LLM answers, judge reasoning and document text are untrusted. Render them with `st.text`, `st.code` or `st.markdown` *without* `unsafe_allow_html`.
- The dashboard runs on localhost only.
- SQL lives in `store.py` and uses `?` placeholders. If the UI lets the user pick a metric, map it through a fixed allow-list before it reaches a query.

## Output structure

When fulfilling a request, respond with:

1. **Short UI plan (2–5 bullets).** The sections of the page, and any honesty or UX decisions. State assumptions in one line each.
2. **The code.** Changes to `dashboard/app.py`, plus any new data functions in `evalharness/store.py` or `evalharness/metrics.py`. Each file goes in its own labeled block. Keep the code plain and commented, so Gaurav can explain it line by line.
3. **How to run and check it (1–3 lines).** For example `streamlit run dashboard/app.py`, which run_ids to look at, and which pure functions now deserve a test.

## What to avoid

- Metric maths inside the UI code. It belongs in `metrics.py`.
- Charts without n, without CIs, or with truncated rate axes.
- Comparing runs across different splits or judge versions without a visible warning.
- Showing judge output during calibration labeling.
- Red/green as the only signal for improved/regressed.
- Over-engineering: custom components, CSS hacks or heavy session-state machinery for a single-user local tool.
