from rag_adapt_lab.data.raft import build_raft_partitions
from rag_adapt_lab.data.schema import Document, EvalExample
from rag_adapt_lab.data.splitting import source_partition_fingerprint, split_rows


def grouped_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "a-1",
            "question": "Question A",
            "metadata": {"thread_id": "a"},
            "relevant_doc_ids": ["doc-a"],
        },
        {
            "id": "a-2",
            "question": "Related A",
            "metadata": {"thread_id": "a"},
            "relevant_doc_ids": ["doc-a"],
        },
        {
            "id": "b-1",
            "question": "Question B",
            "metadata": {"thread_id": "b"},
            "relevant_doc_ids": ["doc-b"],
        },
        {
            "id": "c-1",
            "question": "Question C",
            "metadata": {"thread_id": "c"},
            "relevant_doc_ids": ["doc-c"],
        },
    ]


def test_group_members_never_cross_partitions_and_split_is_deterministic() -> None:
    first = split_rows(
        grouped_rows(),
        validation_ratio=0.25,
        seed=7,
        strategy="grouped",
        group_by=["metadata.thread_id"],
    )
    second = split_rows(
        grouped_rows(),
        validation_ratio=0.25,
        seed=7,
        strategy="grouped",
        group_by=["metadata.thread_id"],
    )
    assert first == second
    train_threads = {row["metadata"]["thread_id"] for row in first.train_rows}
    validation_threads = {row["metadata"]["thread_id"] for row in first.validation_rows}
    assert train_threads.isdisjoint(validation_threads)
    assert first.audit is not None
    assert first.audit.question_overlap_count == 0


def test_normalized_duplicate_questions_are_automatically_grouped() -> None:
    rows = grouped_rows()
    rows[2]["question"] = "  QUESTION a "
    split = split_rows(
        rows,
        validation_ratio=0.25,
        seed=1,
        strategy="grouped",
        group_by=["metadata.thread_id"],
    )
    train_questions = {str(row["question"]).strip().casefold() for row in split.train_rows}
    validation_questions = {
        str(row["question"]).strip().casefold() for row in split.validation_rows
    }
    assert train_questions.isdisjoint(validation_questions)


def qa(identifier: str, document: str) -> EvalExample:
    return EvalExample(
        id=identifier,
        question=f"question {identifier}",
        reference_answer=f"answer {identifier}",
        relevant_doc_ids=[document],
        metadata={"source_document_id": document},
    )


def documents() -> list[Document]:
    return [Document(id=f"doc-{letter}", text=f"text {letter}") for letter in "abcdef"]


def test_document_disjoint_split_before_mining_prevents_all_document_leakage() -> None:
    partitions = build_raft_partitions(
        documents(),
        [qa("a", "doc-a"), qa("b", "doc-b"), qa("c", "doc-c"), qa("d", "doc-d")],
        validation_ratio=0.5,
        seed=3,
        split_strategy="grouped",
        group_by=("metadata.source_document_id",),
        corpus_policy="document-disjoint",
        distractors=1,
        negative_strategy="random",
    )
    train_documents = {
        context.doc_id for row in partitions.train_rows for context in row.contexts
    }
    validation_documents = {
        context.doc_id for row in partitions.validation_rows for context in row.contexts
    }
    assert train_documents.isdisjoint(validation_documents)
    assert partitions.manifest["document_overlap_count"] == 0
    assert partitions.manifest["negative_mining_scope"] == "split-before-mining"
    assert {
        row.metadata["negative_mining"]["scope"] for row in partitions.train_rows
    } == {"train-partition-only"}
    assert {
        row.metadata["negative_mining"]["scope"] for row in partitions.validation_rows
    } == {"validation-partition-only"}


def test_shared_corpus_allows_document_reuse_but_not_question_overlap() -> None:
    partitions = build_raft_partitions(
        documents(),
        [qa("a", "doc-a"), qa("b", "doc-b"), qa("c", "doc-c"), qa("d", "doc-d")],
        validation_ratio=0.5,
        seed=4,
        corpus_policy="shared-corpus",
        distractors=2,
    )
    assert partitions.manifest["document_pools"]["overlap_count"] == len(documents())
    assert partitions.manifest["question_overlap_count"] == 0


def test_impossible_grouping_constraints_raise_clear_error() -> None:
    rows = [
        {"id": "one", "question": "one", "metadata": {"customer": "same"}},
        {"id": "two", "question": "two", "metadata": {"customer": "same"}},
    ]
    try:
        split_rows(
            rows,
            validation_ratio=0.5,
            seed=1,
            strategy="grouped",
            group_by=["metadata.customer"],
        )
    except ValueError as exc:
        assert "one connected group" in str(exc)
    else:
        raise AssertionError("Expected impossible grouped split to fail")


def test_source_partition_fingerprint_is_representation_independent_but_label_sensitive() -> None:
    sft = [{"id": "q1", "input": "What?", "output": "Answer"}]
    raft = [{"id": "q1", "question": " what? ", "answer": "Answer", "contexts": []}]
    changed_label = [{"id": "q1", "question": "What?", "answer": "Different"}]
    assert source_partition_fingerprint(sft) == source_partition_fingerprint(raft)
    assert source_partition_fingerprint(sft) != source_partition_fingerprint(changed_label)
