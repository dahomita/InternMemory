"""Tests for the semantic evaluation pipeline."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from internmemory.embedding import EmbeddingVector
from internmemory.semantic_evaluation import (
    run_semantic_calibration,
    run_semantic_dataset,
)


class KeywordVectorProvider:
    """Provide tiny deterministic vectors for an isolated pipeline test."""

    dimensions = 2

    def embed_query(self, query: str) -> EmbeddingVector:
        return self._embed(query)

    def embed_passages(
        self,
        passages: Sequence[str],
    ) -> list[EmbeddingVector]:
        return [self._embed(passage) for passage in passages]

    @staticmethod
    def _embed(text: str) -> EmbeddingVector:
        if "latency" in text.casefold():
            return (1.0, 0.0)
        return (0.0, 1.0)


def test_semantic_dataset_uses_same_retrieval_metrics(
    tmp_path: Path,
) -> None:
    """Semantic and keyword experiments share comparable report fields."""

    fixtures_root = tmp_path / "fixtures"
    fixtures_root.mkdir()
    (fixtures_root / "latency.md").write_text(
        "Tail latency describes the slowest requests.",
        encoding="utf-8",
    )
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "latency",
                        "category": "exact_keyword",
                        "split": "calibration",
                        "query": "latency",
                        "relevant_source": "latency.md",
                        "required_text": "Tail latency",
                    },
                    {
                        "id": "unsupported",
                        "category": "no_answer",
                        "split": "calibration",
                        "query": "encryption",
                        "expected_no_answer": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_semantic_dataset(
        dataset_path,
        fixtures_root,
        KeywordVectorProvider(),
    )

    assert report.case_count == 2
    assert report.recall_at_k == pytest.approx(1.0)
    assert report.mrr == pytest.approx(1.0)
    assert report.no_answer_accuracy == pytest.approx(0.0)

    calibration = run_semantic_calibration(
        dataset_path,
        fixtures_root,
        KeywordVectorProvider(),
    )

    assert calibration.minimum_similarity > 0.0
    assert calibration.limiting_case_id == "unsupported"
    assert calibration.report.recall_at_k == pytest.approx(1.0)
    assert calibration.report.no_answer_accuracy == pytest.approx(1.0)
