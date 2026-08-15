from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("id must not be blank")
        return value


class Document(Record):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    text: str | None = None


class EvalExample(Record):
    question: str = Field(min_length=1)
    reference_answer: str | None = None
    relevant_doc_ids: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def merge_evidence_ids(self) -> EvalExample:
        self.relevant_doc_ids = list(
            dict.fromkeys([*self.relevant_doc_ids, *(item.doc_id for item in self.evidence)])
        )
        return self


class SFTExample(Record):
    instruction: str = "Answer accurately."
    input: str = ""
    output: str


class RAFTContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    relevant: bool = False

    @field_validator("doc_id")
    @classmethod
    def validate_doc_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("doc_id must not be blank")
        return value


class RAFTExample(Record):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    contexts: list[RAFTContext]
    evidence_doc_ids: list[str]

    @field_validator("evidence_doc_ids")
    @classmethod
    def normalize_evidence_doc_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("evidence_doc_ids must not contain blank IDs")
        return normalized

    @model_validator(mode="after")
    def validate_evidence_metadata(self) -> RAFTExample:
        context_ids = [context.doc_id for context in self.contexts]
        relevant_ids = [context.doc_id for context in self.contexts if context.relevant]
        duplicate_context_ids = sorted(
            doc_id for doc_id, count in Counter(context_ids).items() if count > 1
        )
        duplicate_evidence_ids = sorted(
            doc_id
            for doc_id, count in Counter(self.evidence_doc_ids).items()
            if count > 1
        )
        context_id_set = set(context_ids)
        evidence_id_set = set(self.evidence_doc_ids)
        relevant_id_set = set(relevant_ids)
        missing_context_ids = sorted(evidence_id_set - context_id_set)
        missing_evidence_ids = sorted(relevant_id_set - evidence_id_set)
        evidence_marked_irrelevant = sorted(
            evidence_id_set & {context.doc_id for context in self.contexts if not context.relevant}
        )

        problems: list[str] = []
        if not self.contexts:
            problems.append("contexts must contain at least one item")
        if not self.evidence_doc_ids:
            problems.append("evidence_doc_ids must not be empty")
        if duplicate_context_ids:
            problems.append(f"duplicated context IDs={duplicate_context_ids}")
        if duplicate_evidence_ids:
            problems.append(f"duplicated evidence IDs={duplicate_evidence_ids}")
        if missing_context_ids:
            problems.append(
                f"evidence document IDs missing from contexts={missing_context_ids}"
            )
        if missing_evidence_ids:
            problems.append(f"missing evidence document IDs={missing_evidence_ids}")
            problems.append(f"unexpected relevant document IDs={missing_evidence_ids}")
        if evidence_marked_irrelevant:
            problems.append(f"evidence contexts marked relevant=false={evidence_marked_irrelevant}")
        if not relevant_ids:
            problems.append("no contexts are marked relevant=true")
        if evidence_id_set != relevant_id_set:
            problems.append(
                f"evidence_doc_ids={self.evidence_doc_ids}, "
                f"relevant_context_ids={relevant_ids}"
            )
        if problems:
            raise ValueError(
                f"RAFT example {self.id!r} has inconsistent evidence metadata: "
                + "; ".join(problems)
            )
        return self
