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

    assert report.case_count == 36
    assert report.answerable_case_count == 27
    assert report.no_answer_case_count == 9
    assert report.recall_at_k == pytest.approx(16 / 27)
    assert report.mrr == pytest.approx(31 / 54)
    assert report.mean_precision_at_k == pytest.approx(31 / 54)
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
        "rate-token-bucket-paraphrase": "semantic_paraphrase",
        "index-covering-paraphrase": "semantic_paraphrase",
        "canary-rollout-paraphrase": "semantic_paraphrase",
        "queue-load-shedding-control": "morphology",
        "trace-percentile-paraphrase": "semantic_paraphrase",
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

    assert report.recall_at_k == pytest.approx(25 / 27)
    assert report.mrr == pytest.approx(259 / 324)
    assert report.mean_precision_at_k == pytest.approx(583 / 1620)
    assert report.no_answer_accuracy == pytest.approx(2 / 9)
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

    assert report.recall_at_k == pytest.approx(25 / 27)
    assert report.mrr == pytest.approx(259 / 324)
    assert report.mean_precision_at_k == pytest.approx(583 / 1620)
    assert report.no_answer_accuracy == pytest.approx(2 / 9)
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

    assert report.recall_at_k == pytest.approx(25 / 27)
    assert report.mrr == pytest.approx(259 / 324)
    assert report.mean_precision_at_k == pytest.approx(293 / 810)
    assert report.no_answer_accuracy == pytest.approx(2 / 9)
    boundary_case = next(
        result
        for result in report.cases
        if result.case_id == "resilience-cross-chunk"
    )
    assert boundary_case.rank == 2
    assert boundary_case.returned_count == 4
    assert boundary_case.precision_at_k == pytest.approx(1 / 4)


def test_coverage_reranking_promotes_complete_grouped_evidence() -> None:
    """Full combined coverage outranks a dense partial-match distractor."""

    project_root = Path(__file__).parents[1]

    report = run_dataset(
        project_root / "evals" / "retrieval_cases.json",
        project_root / "evals" / "fixtures",
        strategy="grouped_reranked",
    )

    assert report.recall_at_k == pytest.approx(25 / 27)
    assert report.mrr == pytest.approx(277 / 324)
    assert report.mean_precision_at_k == pytest.approx(293 / 810)
    assert report.no_answer_accuracy == pytest.approx(2 / 9)
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

    assert report.recall_at_k == pytest.approx(19 / 27)
    assert report.mrr == pytest.approx(37 / 54)
    assert report.mean_precision_at_k == pytest.approx(11 / 18)
    assert report.no_answer_accuracy == pytest.approx(1.0)
    partial_overlap_case = next(
        result
        for result in report.cases
        if result.case_id == "no-answer-cache-encryption"
    )
    assert partial_overlap_case.no_answer_correct is True
    assert partial_overlap_case.returned_count == 0


def test_calibration_and_evaluation_splits_are_balanced() -> None:
    """Each split has equal answerable and no-answer representation."""

    project_root = Path(__file__).parents[1]

    reports = [
        run_dataset(
            project_root / "evals" / "retrieval_cases.json",
            project_root / "evals" / "fixtures",
            strategy="grouped_reranked_thresholded",
            split=split,
        )
        for split in ("calibration", "regression", "evaluation")
    ]

    assert all(report.case_count == 12 for report in reports)
    assert all(report.answerable_case_count == 9 for report in reports)
    assert all(report.no_answer_case_count == 3 for report in reports)
