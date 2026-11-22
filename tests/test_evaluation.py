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

    assert report.case_count == 20
    assert report.answerable_case_count == 17
    assert report.no_answer_case_count == 3
    assert report.recall_at_k == pytest.approx(11 / 17)
    assert report.mrr == pytest.approx(21 / 34)
    assert report.mean_precision_at_k == pytest.approx(21 / 34)
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
        "cache-stampede-paraphrase": "semantic_paraphrase",
        "queue-backpressure-paraphrase": "semantic_paraphrase",
        "trace-context-paraphrase": "semantic_paraphrase",
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

    assert report.recall_at_k == pytest.approx(16 / 17)
    assert report.mrr == pytest.approx(203 / 255)
    assert report.mean_precision_at_k == pytest.approx(23 / 60)
    assert report.no_answer_accuracy == pytest.approx(2 / 3)
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

    assert report.recall_at_k == pytest.approx(16 / 17)
    assert report.mrr == pytest.approx(203 / 255)
    assert report.mean_precision_at_k == pytest.approx(23 / 60)
    assert report.no_answer_accuracy == pytest.approx(2 / 3)
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

    assert report.recall_at_k == pytest.approx(16 / 17)
    assert report.mrr == pytest.approx(163 / 204)
    assert report.mean_precision_at_k == pytest.approx(67 / 170)
    assert report.no_answer_accuracy == pytest.approx(2 / 3)
    boundary_case = next(
        result
        for result in report.cases
        if result.case_id == "resilience-cross-chunk"
    )
    assert boundary_case.rank == 2
    assert boundary_case.returned_count == 3
    assert boundary_case.precision_at_k == pytest.approx(1 / 3)


def test_coverage_reranking_promotes_complete_grouped_evidence() -> None:
    """Full combined coverage outranks a dense partial-match distractor."""

    project_root = Path(__file__).parents[1]

    report = run_dataset(
        project_root / "evals" / "retrieval_cases.json",
        project_root / "evals" / "fixtures",
        strategy="grouped_reranked",
    )

    assert report.recall_at_k == pytest.approx(16 / 17)
    assert report.mrr == pytest.approx(169 / 204)
    assert report.mean_precision_at_k == pytest.approx(67 / 170)
    assert report.no_answer_accuracy == pytest.approx(2 / 3)
    boundary_case = next(
        result
        for result in report.cases
        if result.case_id == "resilience-cross-chunk"
    )
    assert boundary_case.rank == 1
    partial_overlap_case = next(
        result
        for result in report.cases
        if result.case_id == "no-answer-cache-encryption"
    )
    assert partial_overlap_case.no_answer_correct is False
    assert partial_overlap_case.returned_count > 0


def test_coverage_threshold_abstains_from_weak_keyword_evidence() -> None:
    """Weak overlap is rejected instead of becoming grounded context."""

    project_root = Path(__file__).parents[1]

    report = run_dataset(
        project_root / "evals" / "retrieval_cases.json",
        project_root / "evals" / "fixtures",
        strategy="grouped_reranked_thresholded",
    )

    assert report.recall_at_k == pytest.approx(12 / 17)
    assert report.mrr == pytest.approx(23 / 34)
    assert report.mean_precision_at_k == pytest.approx(10 / 17)
    assert report.no_answer_accuracy == pytest.approx(1.0)
    partial_overlap_case = next(
        result
        for result in report.cases
        if result.case_id == "no-answer-cache-encryption"
    )
    assert partial_overlap_case.no_answer_correct is True
    assert partial_overlap_case.returned_count == 0
