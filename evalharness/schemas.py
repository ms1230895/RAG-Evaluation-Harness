"""Pydantic models for the evaluation harness.

Step 4: `Query`, one line of data/test_set.jsonl.
(The judges' output schemas join this file in Step 7.)

The models are strict on purpose: bad data should fail loudly when the
test set is loaded, not quietly skew a metric later.
"""

import re
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

QueryType = Literal["lookup", "multi_chunk", "paraphrased", "unanswerable"]
Split = Literal["dev", "heldout"]

# Plain tuples of the allowed values, for loops and messages elsewhere.
QUERY_TYPES: tuple[str, ...] = get_args(QueryType)
SPLITS: tuple[str, ...] = get_args(Split)

# Same rule as the corpus doc IDs (filenames in data/docs/, see tests/test_corpus.py).
DOC_ID_PATTERN = r"^[a-z0-9_]+$"


class Query(BaseModel):
    """One test-set question with its ground truth.

    - id: "q" + 3 digits, e.g. "q007". Never reused once retired.
    - source_doc_ids: docs that contain the answer; [] for unanswerable.
    - split: None only while drafting, before `assign-splits` runs.
    - reviewed: True only after Gaurav accepted the query.
    """

    # extra="forbid": unknown fields are an error (catches typos like "revieved").
    # strict=True: no type coercion, so "reviewed": "true" is rejected.
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(pattern=r"^q\d{3}$")
    question: str
    type: QueryType
    reference_answer: str
    source_doc_ids: list[str]
    split: Split | None
    reviewed: bool

    @field_validator("question", "reference_answer")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("source_doc_ids")
    @classmethod
    def doc_ids_valid(cls, value: list[str]) -> list[str]:
        for doc_id in value:
            if re.fullmatch(DOC_ID_PATTERN, doc_id) is None:
                raise ValueError(f"bad doc ID {doc_id!r} (lowercase letters, digits, _ only)")
        if len(value) != len(set(value)):
            raise ValueError("duplicate doc IDs")
        return value

    @model_validator(mode="after")
    def sources_match_type(self) -> "Query":
        # An unanswerable query has nothing to retrieve; every other type must
        # name at least one source doc, or retrieval metrics cannot score it.
        if self.type == "unanswerable" and self.source_doc_ids:
            raise ValueError("an unanswerable query must have source_doc_ids == []")
        if self.type != "unanswerable" and not self.source_doc_ids:
            raise ValueError(f"a {self.type} query needs at least one source doc ID")
        return self

