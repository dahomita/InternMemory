"""Reproducible keyword-retrieval evaluation runner."""

import argparse
import json
import math
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Literal

from internmemory.chunk_repository import ChunkRepository
from internmemory.database import initialize_database
from internmemory.ingestion_service import IngestionService
from internmemory.retrieval_service import RetrievalService
from internmemory.source_repository import SourceRepository


RetrievalStrategy = Literal[
    "all",
    "any",
    "prioritized",
    "grouped",
    "grouped_reranked",
    "grouped_reranked_thresholded",
]


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    """One human-labeled query and its required evidence."""

    case_id: str
    category: str
    query: str
    relevant_source: str | None
    required_text: str | None
    expected_no_answer: bool


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Observed ranking and latency for one retrieval case."""

    case_id: str
    category: str
    expected_no_answer: bool
    rank: int | None
    reciprocal_rank: float
    precision_at_k: float | None
    returned_count: int
    no_answer_correct: bool | None
    latency_ms: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Aggregate and per-case retrieval measurements."""

    case_count: int
    answerable_case_count: int
    no_answer_case_count: int
    k: int
    recall_at_k: float
    mrr: float
    mean_precision_at_k: float
    no_answer_accuracy: float | None
    median_latency_ms: float
    p95_latency_ms: float
    cases: list[CaseResult]


def load_cases(dataset_path: str | Path) -> list[RetrievalCase]:
    """Load and validate version 1 of the JSON evaluation dataset."""

    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError("unsupported evaluation dataset version")

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("evaluation dataset must contain cases")

    cases: list[RetrievalCase] = []
    for raw_case in raw_cases:
        try:
            case = RetrievalCase(
                case_id=raw_case["id"],
                category=raw_case["category"],
                query=raw_case["query"],
                relevant_source=raw_case.get("relevant_source"),
                required_text=raw_case.get("required_text"),
                expected_no_answer=raw_case.get(
                    "expected_no_answer",
                    False,
                ),
            )
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError("invalid evaluation case") from error

        if not all(
            value.strip()
            for value in (
                case.case_id,
                case.category,
                case.query,
            )
        ):
            raise ValueError("evaluation case values must not be blank")
        if case.expected_no_answer:
            if (
                case.relevant_source is not None
                or case.required_text is not None
            ):
                raise ValueError(
                    "no-answer cases must not label relevant evidence"
                )
        elif (
            not case.relevant_source
            or not case.relevant_source.strip()
            or not case.required_text
            or not case.required_text.strip()
        ):
            raise ValueError(
                "answerable cases require relevant evidence"
            )
        cases.append(case)

    return cases


def evaluate_cases(
    repository: ChunkRepository,
    cases: list[RetrievalCase],
    *,
    k: int = 5,
    strategy: RetrievalStrategy = "all",
) -> EvaluationReport:
    """Measure ranking quality and query latency for labeled cases."""

    if not cases:
        raise ValueError("at least one evaluation case is required")
    if k <= 0:
        raise ValueError("k must be positive")

    retrieval_service = RetrievalService(repository)
    case_results: list[CaseResult] = []
    for case in cases:
        relevant_ranks: list[int] = []
        started_at = perf_counter_ns()
        try:
            if strategy in (
                "grouped",
                "grouped_reranked",
                "grouped_reranked_thresholded",
            ):
                groups = retrieval_service.search(
                    case.query,
                    limit=k,
                    rerank_by_coverage=strategy
                    in (
                        "grouped_reranked",
                        "grouped_reranked_thresholded",
                    ),
                    minimum_coverage=(
                        0.5
                        if strategy == "grouped_reranked_thresholded"
                        else 0.0
                    ),
                )
                returned_count = len(groups)
                if not case.expected_no_answer:
                    required_text = case.required_text
                    if required_text is None:
                        raise RuntimeError(
                            "answerable case lost required text"
                        )
                    relevant_ranks = [
                        index
                        for index, group in enumerate(groups, start=1)
                        if group.source.locator == case.relevant_source
                        and any(
                            _normalize(required_text)
                            in _normalize(chunk.content)
                            for chunk in group.chunks
                        )
                    ]
            else:
                if strategy == "prioritized":
                    results = repository.search_with_fallback(
                        case.query,
                        limit=k,
                    )
                else:
                    results = repository.search(
                        case.query,
                        limit=k,
                        term_match=strategy,
                    )
                returned_count = len(results)
                if not case.expected_no_answer:
                    required_text = case.required_text
                    if required_text is None:
                        raise RuntimeError(
                            "answerable case lost required text"
                        )
                    relevant_ranks = [
                        index
                        for index, result in enumerate(results, start=1)
                        if result.source.locator
                        == case.relevant_source
                        and _normalize(required_text)
                        in _normalize(result.chunk.content)
                    ]
            error = None
        except sqlite3.OperationalError as caught_error:
            returned_count = 0
            error = str(caught_error)
        latency_ms = (perf_counter_ns() - started_at) / 1_000_000

        rank = None
        precision_at_k = None
        no_answer_correct = None
        if case.expected_no_answer:
            no_answer_correct = error is None and returned_count == 0
        else:
            rank = relevant_ranks[0] if relevant_ranks else None
            precision_at_k = (
                len(relevant_ranks) / returned_count
                if returned_count
                else 0.0
            )
        case_results.append(
            CaseResult(
                case_id=case.case_id,
                category=case.category,
                expected_no_answer=case.expected_no_answer,
                rank=rank,
                reciprocal_rank=1 / rank if rank is not None else 0.0,
                precision_at_k=precision_at_k,
                returned_count=returned_count,
                no_answer_correct=no_answer_correct,
                latency_ms=latency_ms,
                error=error,
            )
        )

    return summarize_case_results(case_results, k=k)


def summarize_case_results(
    case_results: list[CaseResult],
    *,
    k: int,
) -> EvaluationReport:
    """Aggregate comparable retrieval metrics from per-case results."""

    answerable_results = [
        result
        for result in case_results
        if not result.expected_no_answer
    ]
    no_answer_results = [
        result
        for result in case_results
        if result.expected_no_answer
    ]
    if not answerable_results:
        raise ValueError("at least one answerable case is required")
    ranks = [result.rank for result in answerable_results]
    latencies = [result.latency_ms for result in case_results]
    return EvaluationReport(
        case_count=len(case_results),
        answerable_case_count=len(answerable_results),
        no_answer_case_count=len(no_answer_results),
        k=k,
        recall_at_k=sum(rank is not None for rank in ranks) / len(ranks),
        mrr=sum(result.reciprocal_rank for result in answerable_results)
        / len(answerable_results),
        mean_precision_at_k=sum(
            result.precision_at_k or 0.0
            for result in answerable_results
        )
        / len(answerable_results),
        no_answer_accuracy=(
            sum(
                result.no_answer_correct is True
                for result in no_answer_results
            )
            / len(no_answer_results)
            if no_answer_results
            else None
        ),
        median_latency_ms=median(latencies),
        p95_latency_ms=_nearest_rank_percentile(latencies, 0.95),
        cases=case_results,
    )


def run_dataset(
    dataset_path: str | Path,
    fixtures_root: str | Path,
    *,
    k: int = 5,
    max_chunk_chars: int = 1_200,
    strategy: RetrievalStrategy = "all",
) -> EvaluationReport:
    """Ingest fixture sources into a temporary database and evaluate them."""

    cases = load_cases(dataset_path)
    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = Path(temporary_directory) / "evaluation.db"
        initialize_database(database_path)
        chunk_repository = ChunkRepository(database_path)
        ingestion_service = IngestionService(
            SourceRepository(database_path),
            chunk_repository,
            allowed_root=fixtures_root,
            max_chunk_chars=max_chunk_chars,
        )
        for locator in sorted(
            {
                case.relevant_source
                for case in cases
                if case.relevant_source is not None
            }
        ):
            ingestion_service.ingest_file(locator)

        return evaluate_cases(
            chunk_repository,
            cases,
            k=k,
            strategy=strategy,
        )


def _normalize(value: str) -> str:
    """Normalize whitespace and case for evidence-label matching."""

    return " ".join(value.casefold().split())


def _nearest_rank_percentile(
    values: list[float],
    percentile: float,
) -> float:
    """Calculate a nearest-rank percentile for a finite sample."""

    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def main() -> None:
    """Run retrieval evaluation from the command line."""

    parser = argparse.ArgumentParser(
        description="Evaluate InternMemory keyword retrieval."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/retrieval_cases.json"),
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("evals/fixtures"),
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-chunk-chars", type=int, default=1_200)
    parser.add_argument(
        "--strategy",
        choices=(
            "all",
            "any",
            "prioritized",
            "grouped",
            "grouped_reranked",
            "grouped_reranked_thresholded",
        ),
        default="all",
    )
    arguments = parser.parse_args()

    report = run_dataset(
        arguments.dataset,
        arguments.fixtures,
        k=arguments.k,
        max_chunk_chars=arguments.max_chunk_chars,
        strategy=arguments.strategy,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
