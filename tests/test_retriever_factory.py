import pytest

from rag_adapt_lab.retrieval.bm25 import BM25Retriever
from rag_adapt_lab.retrieval.dense import DenseRetriever
from rag_adapt_lab.retrieval.factory import create_retriever


def test_factory_applies_bm25_configuration() -> None:
    retriever = create_retriever({"kind": "bm25", "lowercase": False})
    assert isinstance(retriever, BM25Retriever)
    assert retriever.lowercase is False


def test_factory_applies_pinned_dense_configuration() -> None:
    retriever = create_retriever(
        {
            "kind": "dense",
            "model_id": "example/embedding",
            "revision": "a" * 40,
            "normalize_embeddings": False,
            "batch_size": 7,
            "trust_remote_code": False,
        }
    )
    assert isinstance(retriever, DenseRetriever)
    assert retriever.revision == "a" * 40
    assert retriever.normalize_embeddings is False
    assert retriever.batch_size == 7


def test_factory_rejects_remote_dense_code() -> None:
    with pytest.raises(ValueError, match="remote model code is disabled"):
        create_retriever(
            {
                "kind": "dense",
                "revision": "a" * 40,
                "trust_remote_code": True,
            }
        )
