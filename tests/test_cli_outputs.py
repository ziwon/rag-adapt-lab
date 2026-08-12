from pathlib import Path

import pytest

from rag_adapt_lab.data.raft import validate_distinct_output_paths


def test_raft_output_paths_must_be_distinct(tmp_path: Path) -> None:
    output = tmp_path / "raft.jsonl"
    with pytest.raises(ValueError, match="must be distinct"):
        validate_distinct_output_paths(
            output,
            output,
            tmp_path / "manifest.json",
        )
