from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rag_adapt_lab.data.schema import Document


@dataclass(slots=True)
class RetrievalResult:
    document: Document
    score: float
    rank: int


class Retriever(ABC):
    @abstractmethod
    def index(self, documents: list[Document]) -> None: ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]: ...
