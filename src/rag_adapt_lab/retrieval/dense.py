from __future__ import annotations

import numpy as np

from rag_adapt_lab.data.schema import Document

from .base import RetrievalResult, Retriever


class DenseRetriever(Retriever):
    def __init__(self, model_id: str = "Qwen/Qwen3-Embedding-0.6B") -> None:
        self.model_id = model_id
        self.documents: list[Document] = []
        self._model = None
        self._embeddings: np.ndarray | None = None

    def index(self, documents: list[Document]) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the RAG extras: pip install -e '.[rag]'") from exc
        self.documents = documents
        self._model = SentenceTransformer(self.model_id)
        self._embeddings = np.asarray(
            self._model.encode(
                [doc.text for doc in documents],
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if self._model is None or self._embeddings is None:
            raise RuntimeError("Call index() before search().")
        q = np.asarray(self._model.encode([query], normalize_embeddings=True))[0]
        scores = self._embeddings @ q
        indices = np.argsort(-scores)[:top_k]
        return [
            RetrievalResult(
                document=self.documents[int(idx)],
                score=float(scores[int(idx)]),
                rank=rank + 1,
            )
            for rank, idx in enumerate(indices)
        ]
