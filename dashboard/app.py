"""Streamlit dashboard for the RAG evaluation harness (Step 15).

Run:  streamlit run dashboard/app.py     (localhost only)

A thin layer: the pages (dashboard/views.py) call the data functions in
evalharness/dashboard_data.py, hand the stored summaries to the figure builders in
dashboard/figures.py and render the result. No metric is computed in the UI. The ONLY
write is a saved review (store.write_review, human fields of `reviews`). No LLM call
and no run is started.

Honesty rules (CLAUDE.md): every rate shows k/n and its 95% Wilson interval, rate axes
start at 0, dev and held-out never share a figure, and every judge number is labelled
"uncalibrated judge" until a calibration is `measured`.
"""

import sys
from pathlib import Path

# `streamlit run dashboard/app.py` puts dashboard/ on sys.path, not the repo root,
# so `import evalharness` would fail without this line.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

st.set_page_config(page_title="RAG evaluation harness", layout="wide")

from dashboard import views

views.sidebar()
PAGES = [
    st.Page(views.page_trends, title="Trends", url_path="trends", default=True),
    st.Page(views.page_compare, title="Run vs run", url_path="compare"),
    st.Page(views.page_review_queue, title="Review queue", url_path="review"),
    st.Page(views.page_breakdown, title="Breakdown by type", url_path="breakdown"),
]
st.navigation(PAGES).run()
