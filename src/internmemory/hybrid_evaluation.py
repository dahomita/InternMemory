"""Reproducible reciprocal-rank-fusion evaluation."""

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from internmemory.chunk_repository import ChunkRepository
from internmemory.database import initialize_database
from internmemory.embedding import EmbeddingProvider, FastEmbedProvider
from internmemory.evaluation import (
    CaseSplit,
    EvaluationReport,
    load_cases,
)
from internmemory.hybrid_retrieval import HybridRetriever
from internmemory.ingestion_service import IngestionService
from internmemory.models import SourceReference
from internmemory.semantic_evaluation import (
    DEFAULT_MODEL_DIMENSIONS,
    DEFAULT_MODEL_NAME,
    evaluate_semantic_cases,
)
from internmemory.semantic_retrieval import InMemorySemanticRetriever
from internmemory.source_repository import SourceRepository


def run_hybrid_dataset(
    dataset_path: str | Path,
    fixtures_root: str | Path,
    provider: EmbeddingProvider,
    *,
    split: CaseSplit | None = None,
    k: int = 5,
    candidate_limit: int = 20,
    max_chunk_chars: int = 1_200,
    minimum_similarity: float | None = None,
    keyword_weight: float = 1.0,
    semantic_weight: float = 2.0,
) -> EvaluationReport:
    """Build one temporary corpus and evaluate fused chunk rankings."""

    all_cases = load_cases(dataset_path)
    cases = [
        case
        for case in all_cases
        if split is None or case.split == split
    ]
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
                for case in all_cases
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

        semantic_retriever = InMemorySemanticRetriever(
            provider,
            passages,
        )
        hybrid_retriever = HybridRetriever(
            chunk_repository,
            semantic_retriever,
            candidate_limit=candidate_limit,
            keyword_weight=keyword_weight,
            semantic_weight=semantic_weight,
        )
        return evaluate_semantic_cases(
            hybrid_retriever,
            cases,
            k=k,
            minimum_similarity=minimum_similarity,
        )


def main() -> None:
    """Run local BGE and FTS5 reciprocal-rank fusion."""

    parser = argparse.ArgumentParser(
        description="Evaluate InternMemory hybrid retrieval."
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
    parser.add_argument(
        "--split",
        choices=("calibration", "regression", "evaluation"),
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--max-chunk-chars", type=int, default=1_200)
    parser.add_argument("--minimum-similarity", type=float)
    parser.add_argument("--keyword-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=2.0)
    arguments = parser.parse_args()

    provider = FastEmbedProvider(
        model_name=DEFAULT_MODEL_NAME,
        dimensions=DEFAULT_MODEL_DIMENSIONS,
    )
    report = run_hybrid_dataset(
        arguments.dataset,
        arguments.fixtures,
        provider,
        split=arguments.split,
        k=arguments.k,
        candidate_limit=arguments.candidate_limit,
        max_chunk_chars=arguments.max_chunk_chars,
        minimum_similarity=arguments.minimum_similarity,
        keyword_weight=arguments.keyword_weight,
        semantic_weight=arguments.semantic_weight,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
