"""Pydantic models for the evaluation harness.

Step 4: `Query`, one line of data/test_set.jsonl.
Step 7: the faithfulness judge's reply schemas, `ClaimList` (call 1) and
`ClaimVerdicts` (call 2). llm/client.py validates every judge reply against
them, so a malformed verdict raises and is never cached or scored.
Step 8: the relevance judge's reply schema, `RelevanceVerdict`.

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



# ---------------------------------------------------------------- judge replies (Step 7)

Label = Literal["supported", "unsupported", "contradicted"]
LABELS: tuple[str, ...] = get_args(Label)


class ClaimList(BaseModel):
    """Call 1 reply: the answer split into atomic claims ([] for a pure refusal)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    claims: list[str]

    @field_validator("claims")
    @classmethod
    def claims_not_blank(cls, value: list[str]) -> list[str]:
        stripped = [claim.strip() for claim in value]
        if any(not claim for claim in stripped):
            raise ValueError("a claim must not be blank")
        return stripped


class ClaimVerdict(BaseModel):
    """One claim's verdict.

    Field order matters: models write JSON fields in schema order, so the
    reasoning comes before the label (explain first, then decide).
    No defaults: a reply without a label is invalid, never "supported".
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    claim_index: int = Field(ge=1)       # 1-based, matches the numbered claim list
    reasoning: str
    label: Label

    @field_validator("reasoning")
    @classmethod
    def reasoning_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reasoning must not be blank")
        return value.strip()


class ClaimVerdicts(BaseModel):
    """Call 2 reply: one verdict per claim."""

    model_config = ConfigDict(extra="forbid", strict=True)

    verdicts: list[ClaimVerdict]


# ---------------------------------------------------------------- relevance reply (Step 8)

class RelevanceVerdict(BaseModel):
    """Relevance judge reply: a rubric score of 0 / 1 / 2.

    Reasoning comes before the score (explain first, then decide). No default
    score: a reply without one is invalid. strict=True rejects "2" for 2.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    reasoning: str
    score: Literal[0, 1, 2]

    @field_validator("reasoning")
    @classmethod
    def reasoning_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reasoning must not be blank")
        return value.strip()
