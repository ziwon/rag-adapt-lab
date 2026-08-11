from __future__ import annotations

import numpy as np

from rag_adapt_lab.config import require_pinned_hf_revision
from rag_adapt_lab.data.schema import Document

from .base import RetrievalResult, Retriever


class DenseRetriever(Retriever):
    DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
    DEFAULT_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        revision: str | None = None,
        normalize_embeddings: bool = True,
        batch_size: int = 16,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_id = model_id
        if revision is None:
            if model_id != self.DEFAULT_MODEL_ID:
                raise ValueError("A pinned revision is required when using a custom dense model")
            revision = self.DEFAULT_REVISION
        self.revision = require_pinned_hf_revision(revision, model_id=model_id)
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.documents: list[Document] = []
        self._model = None
        self._embeddings: np.ndarray | None = None

    def index(self, documents: list[Document]) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the RAG extras: pip install -e '.[rag]'") from exc
        self.documents = documents
        model = SentenceTransformer(
            self.model_id,
            revision=self.revision,
            trust_remote_code=False,
        )
        self._model = model
        self._embeddings = np.asarray(
            model.encode(
                [doc.text for doc in documents],
                normalize_embeddings=self.normalize_embeddings,
                batch_size=self.batch_size,
                show_progress_bar=True,
            )
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if self._model is None or self._embeddings is None:
            raise RuntimeError("Call index() before search().")
        q = np.asarray(
            self._model.encode(
                [query],
                normalize_embeddings=self.normalize_embeddings,
                batch_size=self.batch_size,
            )
        )[0]
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
