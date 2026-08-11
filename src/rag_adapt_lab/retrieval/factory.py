from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import Retriever
from .bm25 import BM25Retriever
from .dense import DenseRetriever


def create_retriever(config: Mapping[str, Any] | None = None) -> Retriever:
    """Create a retriever from a domain-neutral configuration mapping."""
    values = dict(config or {})
    kind = str(values.get("kind", "bm25"))
    if kind == "bm25":
        return BM25Retriever(lowercase=bool(values.get("lowercase", True)))
    if kind == "dense":
        if values.get("trust_remote_code", False) is not False:
            raise ValueError("Dense retriever remote model code is disabled")
        model_id = str(values.get("model_id", DenseRetriever.DEFAULT_MODEL_ID))
        revision = values.get("revision")
        return DenseRetriever(
            model_id=model_id,
            revision=str(revision) if revision is not None else None,
            normalize_embeddings=bool(values.get("normalize_embeddings", True)),
            batch_size=int(values.get("batch_size", 16)),
        )
    if kind == "hybrid":
        raise ValueError(
            "Hybrid retrieval is configured but no rank-fusion backend is implemented. "
            "Inject a Retriever implementation or use bm25/dense."
        )
    raise ValueError(f"Unsupported retriever kind: {kind!r}")
