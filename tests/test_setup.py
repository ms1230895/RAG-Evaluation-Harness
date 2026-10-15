"""Step 0 smoke test: every dependency in requirements.txt can be imported."""

import importlib
import importlib.util

import pytest

# Import names (these can differ from pip package names).
MODULES = [
    "sentence_transformers",
    "faiss",
    "rank_bm25",
    "openai",
    "pydantic",
    "pandas",
    "numpy",
    "scipy",
    "streamlit",
    "plotly",
    "yaml",
    "tenacity",
    "tqdm",
    "dotenv",
]


# Modules only checked for presence, NOT imported in the pytest process.
# sentence_transformers loads torch. On macOS, torch and faiss-cpu each bundle
# their own libomp.dylib, and the two cannot run in one process (checked
# 2026-10-15): once one has used OpenMP, the other aborts the whole run with
# "OMP: Error #15 ... libomp.dylib already initialized". Other test files run
# FAISS searches, so importing torch here would crash the full suite.
# (Step 11's reranker must solve this for real code; see the Step 3 spec.)
CHECK_PRESENCE_ONLY = {"sentence_transformers"}


@pytest.mark.parametrize("name", MODULES)
def test_dependency_imports(name):
    if name in CHECK_PRESENCE_ONLY:
        assert importlib.util.find_spec(name) is not None, f"{name} is not installed"
    else:
        importlib.import_module(name)


def test_project_packages_import():
    # rag/ and evalharness/ must be importable as packages.
    importlib.import_module("rag")
    importlib.import_module("evalharness")
