from __future__ import annotations

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


class RAFTExample(Record):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    contexts: list[RAFTContext] = Field(min_length=1)
    evidence_doc_ids: list[str] = Field(min_length=1)
