"""Tests for the end-to-end hybrid evaluation pipeline."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from internmemory.embedding import EmbeddingVector
from internmemory.hybrid_evaluation import run_hybrid_dataset


class TinyEmbeddingProvider:
    """Return deterministic vectors for pipeline verification."""

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
        return (
            (1.0, 0.0)
            if "latency" in text.casefold()
            else (0.0, 1.0)
        )


def test_hybrid_dataset_reports_fused_quality(
    tmp_path: Path,
) -> None:
    """Keyword and semantic rankings share the standard report format."""

    fixtures_root = tmp_path / "fixtures"
    fixtures_root.mkdir()
    (fixtures_root / "latency.md").write_text(
        "Tail latency describes slow requests.",
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
                        "split": "evaluation",
                        "query": "tail latency",
                        "relevant_source": "latency.md",
                        "required_text": "Tail latency",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_hybrid_dataset(
        dataset_path,
        fixtures_root,
        TinyEmbeddingProvider(),
        split="evaluation",
    )

    assert report.recall_at_k == pytest.approx(1.0)
    assert report.mrr == pytest.approx(1.0)
