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

    assert report.case_count == 13
    assert report.answerable_case_count == 11
    assert report.no_answer_case_count == 2
    assert report.recall_at_k == pytest.approx(8 / 11)
    assert report.mrr == pytest.approx(15 / 22)
    assert report.mean_precision_at_k == pytest.approx(15 / 22)
    assert report.no_answer_accuracy == pytest.approx(1.0)
    failed_cases = {
        result.case_id: result.category
        for result in report.cases
        if result.rank is None and not result.expected_no_answer
    }
    assert failed_cases == {
        "batch-semantic-paraphrase": "synonym",
        "sqlite-expanded-abbreviation": "abbreviation",
        "resilience-cross-chunk": "chunk_boundary",
    }
    batch_case = next(
        result
        for result in report.cases
        if result.case_id == "batch-tail-latency"
    )
    assert batch_case.rank == 2
    assert all(
        result.no_answer_correct is True
        for result in report.cases
        if result.expected_no_answer
    )
    assert all(result.error is None for result in report.cases)
    assert report.median_latency_ms >= 0
    assert report.p95_latency_ms >= report.median_latency_ms


def test_any_term_candidates_improve_recall_but_remain_broad() -> None:
    """OR candidates recover split evidence without solving reranking."""

    project_root = Path(__file__).parents[1]

    report = run_dataset(
        project_root / "evals" / "retrieval_cases.json",
        project_root / "evals" / "fixtures",
        strategy="any",
    )

    assert report.recall_at_k == pytest.approx(10 / 11)
    assert report.mrr == pytest.approx(17 / 22)
    assert report.mean_precision_at_k == pytest.approx(1 / 2)
    boundary_case = next(
        result
        for result in report.cases
        if result.case_id == "resilience-cross-chunk"
    )
    assert boundary_case.rank == 2


def test_prioritized_candidates_keep_strict_results_first() -> None:
    """Strict matches lead before broad fallback fills remaining capacity."""

    project_root = Path(__file__).parents[1]

    report = run_dataset(
        project_root / "evals" / "retrieval_cases.json",
        project_root / "evals" / "fixtures",
        strategy="prioritized",
    )

    assert report.recall_at_k == pytest.approx(10 / 11)
    assert report.mrr == pytest.approx(17 / 22)
    assert report.mean_precision_at_k == pytest.approx(1 / 2)
    boundary_case = next(
        result
        for result in report.cases
        if result.case_id == "resilience-cross-chunk"
    )
    assert boundary_case.rank == 2


def test_grouped_evidence_counts_any_member_as_relevant() -> None:
    """A cited group is relevant when any member contains required evidence."""

    project_root = Path(__file__).parents[1]

    report = run_dataset(
        project_root / "evals" / "retrieval_cases.json",
        project_root / "evals" / "fixtures",
        strategy="grouped",
    )

    assert report.recall_at_k == pytest.approx(10 / 11)
    assert report.mrr == pytest.approx(17 / 22)
    assert report.mean_precision_at_k == pytest.approx(17 / 33)
    boundary_case = next(
        result
        for result in report.cases
        if result.case_id == "resilience-cross-chunk"
    )
    assert boundary_case.rank == 2
    assert boundary_case.returned_count == 2
    assert boundary_case.precision_at_k == pytest.approx(1 / 2)


def test_coverage_reranking_promotes_complete_grouped_evidence() -> None:
    """Full combined coverage outranks a dense partial-match distractor."""

    project_root = Path(__file__).parents[1]

    report = run_dataset(
        project_root / "evals" / "retrieval_cases.json",
        project_root / "evals" / "fixtures",
        strategy="grouped_reranked",
    )

    assert report.recall_at_k == pytest.approx(10 / 11)
    assert report.mrr == pytest.approx(9 / 11)
    assert report.mean_precision_at_k == pytest.approx(17 / 33)
    boundary_case = next(
        result
        for result in report.cases
        if result.case_id == "resilience-cross-chunk"
    )
    assert boundary_case.rank == 1
