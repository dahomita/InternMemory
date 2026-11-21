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

from internmemory.chunk_repository import ChunkRepository
from internmemory.database import initialize_database
from internmemory.ingestion_service import IngestionService
from internmemory.source_repository import SourceRepository


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    """One human-labeled query and its required evidence."""

    case_id: str
    query: str
    relevant_source: str
    required_text: str


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Observed ranking and latency for one retrieval case."""

    case_id: str
    rank: int | None
    reciprocal_rank: float
    latency_ms: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Aggregate and per-case retrieval measurements."""

    case_count: int
    k: int
    recall_at_k: float
    mrr: float
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
                query=raw_case["query"],
                relevant_source=raw_case["relevant_source"],
                required_text=raw_case["required_text"],
            )
        except (KeyError, TypeError) as error:
            raise ValueError("invalid evaluation case") from error

        if not all(
            value.strip()
            for value in (
                case.case_id,
                case.query,
                case.relevant_source,
                case.required_text,
            )
        ):
            raise ValueError("evaluation case values must not be blank")
        cases.append(case)

    return cases


def evaluate_cases(
    repository: ChunkRepository,
    cases: list[RetrievalCase],
    *,
    k: int = 5,
) -> EvaluationReport:
    """Measure ranking quality and query latency for labeled cases."""

    if not cases:
        raise ValueError("at least one evaluation case is required")
    if k <= 0:
        raise ValueError("k must be positive")

    case_results: list[CaseResult] = []
    for case in cases:
        started_at = perf_counter_ns()
        try:
            results = repository.search(case.query, limit=k)
            error = None
        except sqlite3.OperationalError as caught_error:
            results = []
            error = str(caught_error)
        latency_ms = (perf_counter_ns() - started_at) / 1_000_000

        rank = next(
            (
                index
                for index, result in enumerate(results, start=1)
                if result.source.locator == case.relevant_source
                and _normalize(case.required_text)
                in _normalize(result.chunk.content)
            ),
            None,
        )
        case_results.append(
            CaseResult(
                case_id=case.case_id,
                rank=rank,
                reciprocal_rank=1 / rank if rank is not None else 0.0,
                latency_ms=latency_ms,
                error=error,
            )
        )

    ranks = [result.rank for result in case_results]
    latencies = [result.latency_ms for result in case_results]
    return EvaluationReport(
        case_count=len(case_results),
        k=k,
        recall_at_k=sum(rank is not None for rank in ranks) / len(ranks),
        mrr=sum(result.reciprocal_rank for result in case_results)
        / len(case_results),
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
            {case.relevant_source for case in cases}
        ):
            ingestion_service.ingest_file(locator)

        return evaluate_cases(chunk_repository, cases, k=k)


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
    arguments = parser.parse_args()

    report = run_dataset(
        arguments.dataset,
        arguments.fixtures,
        k=arguments.k,
        max_chunk_chars=arguments.max_chunk_chars,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
