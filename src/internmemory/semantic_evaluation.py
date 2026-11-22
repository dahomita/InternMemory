"""Reproducible semantic-retrieval evaluation on the keyword corpus."""

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns

from internmemory.chunk_repository import ChunkRepository
from internmemory.database import initialize_database
from internmemory.embedding import EmbeddingProvider, FastEmbedProvider
from internmemory.evaluation import (
    CaseResult,
    EvaluationReport,
    RetrievalCase,
    load_cases,
    summarize_case_results,
)
from internmemory.ingestion_service import IngestionService
from internmemory.models import SourceReference
from internmemory.semantic_retrieval import InMemorySemanticRetriever
from internmemory.source_repository import SourceRepository


DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_MODEL_DIMENSIONS = 384


def evaluate_semantic_cases(
    retriever: InMemorySemanticRetriever,
    cases: list[RetrievalCase],
    *,
    k: int = 5,
    minimum_similarity: float | None = None,
) -> EvaluationReport:
    """Measure semantic ranking against labeled evidence."""

    if not cases:
        raise ValueError("at least one evaluation case is required")
    if k <= 0:
        raise ValueError("k must be positive")

    case_results: list[CaseResult] = []
    for case in cases:
        started_at = perf_counter_ns()
        results = retriever.search(
            case.query,
            limit=k,
            minimum_similarity=minimum_similarity,
        )
        latency_ms = (perf_counter_ns() - started_at) / 1_000_000

        rank = None
        precision_at_k = None
        no_answer_correct = None
        if case.expected_no_answer:
            no_answer_correct = not results
        else:
            required_text = case.required_text
            if required_text is None:
                raise RuntimeError("answerable case lost required text")
            relevant_ranks = [
                index
                for index, result in enumerate(results, start=1)
                if result.source.locator == case.relevant_source
                and _normalize(required_text)
                in _normalize(result.chunk.content)
            ]
            rank = relevant_ranks[0] if relevant_ranks else None
            precision_at_k = (
                len(relevant_ranks) / len(results)
                if results
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
                returned_count=len(results),
                no_answer_correct=no_answer_correct,
                latency_ms=latency_ms,
                error=None,
            )
        )

    return summarize_case_results(case_results, k=k)


def run_semantic_dataset(
    dataset_path: str | Path,
    fixtures_root: str | Path,
    provider: EmbeddingProvider,
    *,
    k: int = 5,
    max_chunk_chars: int = 1_200,
    minimum_similarity: float | None = None,
) -> EvaluationReport:
    """Build a temporary cited corpus and evaluate semantic retrieval."""

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
        passages = []
        for locator in sorted(
            {
                case.relevant_source
                for case in cases
                if case.relevant_source is not None
            }
        ):
            source = ingestion_service.ingest_file(locator)
            reference = SourceReference(
                id=source.id,
                source_type=source.source_type,
                locator=source.locator,
                title=source.title,
                content_hash=source.content_hash,
            )
            passages.extend(
                (reference, chunk)
                for chunk in chunk_repository.list_for_source(source.id)
            )

        retriever = InMemorySemanticRetriever(provider, passages)
        return evaluate_semantic_cases(
            retriever,
            cases,
            k=k,
            minimum_similarity=minimum_similarity,
        )


def _normalize(value: str) -> str:
    """Normalize whitespace and case for evidence-label matching."""

    return " ".join(value.casefold().split())


def main() -> None:
    """Run local BGE semantic evaluation from the command line."""

    parser = argparse.ArgumentParser(
        description="Evaluate InternMemory semantic retrieval."
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
    parser.add_argument("--minimum-similarity", type=float)
    arguments = parser.parse_args()

    provider = FastEmbedProvider(
        model_name=DEFAULT_MODEL_NAME,
        dimensions=DEFAULT_MODEL_DIMENSIONS,
    )
    report = run_semantic_dataset(
        arguments.dataset,
        arguments.fixtures,
        provider,
        k=arguments.k,
        max_chunk_chars=arguments.max_chunk_chars,
        minimum_similarity=arguments.minimum_similarity,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
