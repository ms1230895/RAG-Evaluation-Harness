"""Step 0 smoke test: every dependency in requirements.txt can be imported."""

import importlib

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


@pytest.mark.parametrize("name", MODULES)
def test_dependency_imports(name):
    importlib.import_module(name)


def test_project_packages_import():
    # rag/ and evalharness/ must be importable as packages.
    importlib.import_module("rag")
    importlib.import_module("evalharness")
