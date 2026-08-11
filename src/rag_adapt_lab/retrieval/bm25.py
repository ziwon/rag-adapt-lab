from __future__ import annotations

import re

from rag_adapt_lab.data.schema import Document

from .base import RetrievalResult, Retriever


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class BM25Retriever(Retriever):
    def __init__(self) -> None:
        self.documents: list[Document] = []
        self._bm25 = None

    def index(self, documents: list[Document]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("Install the RAG extras: pip install -e '.[rag]'") from exc
        self.documents = documents
        self._bm25 = BM25Okapi([_tokenize(doc.text) for doc in documents])

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if self._bm25 is None:
            raise RuntimeError("Call index() before search().")
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)[:top_k]
        return [
            RetrievalResult(document=self.documents[idx], score=float(score), rank=rank + 1)
            for rank, (idx, score) in enumerate(ranked)
        ]
