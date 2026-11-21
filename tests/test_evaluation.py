"""Tests for the reproducible retrieval evaluation pipeline."""

from pathlib import Path

import pytest

from internmemory.evaluation import run_dataset


def test_fictional_keyword_baseline_retrieves_labeled_evidence() -> None:
    """The checked-in corpus provides a deterministic quality baseline."""

    project_root = Path(__file__).parents[1]

    report = run_dataset(
        project_root / "evals" / "retrieval_cases.json",
        project_root / "evals" / "fixtures",
    )

    assert report.case_count == 6
    assert report.recall_at_k == pytest.approx(1.0)
    assert report.mrr == pytest.approx(1.0)
    assert all(result.rank == 1 for result in report.cases)
    assert all(result.error is None for result in report.cases)
    assert report.median_latency_ms >= 0
    assert report.p95_latency_ms >= report.median_latency_ms
