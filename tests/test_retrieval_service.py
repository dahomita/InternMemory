"""Tests for context-aware evidence grouping."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from internmemory.chunk_repository import ChunkRepository
from internmemory.database import initialize_database
from internmemory.ingestion_service import IngestionService
from internmemory.retrieval_service import RetrievalService
from internmemory.source_repository import SourceRepository


def test_neighbor_is_attached_when_it_adds_missing_query_terms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adjacent chunks combine split query evidence without duplication."""

    project_root = Path(__file__).parents[1]
    fixtures_root = project_root / "evals" / "fixtures"
    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    chunk_repository = ChunkRepository(database_path)
    ingestion_service = IngestionService(
        SourceRepository(database_path),
        chunk_repository,
        allowed_root=fixtures_root,
        max_chunk_chars=1_200,
    )
    ingestion_service.ingest_file("resilience-boundary.md")
    batch_loader = Mock(wraps=chunk_repository.list_for_sources)
    monkeypatch.setattr(
        chunk_repository,
        "list_for_sources",
        batch_loader,
    )
    service = RetrievalService(chunk_repository)

    groups = service.search(
        "circuit breaker retry budget",
        limit=5,
    )

    assert len(groups) == 1
    assert groups[0].source.locator == "resilience-boundary.md"
    assert [chunk.ordinal for chunk in groups[0].chunks] == [0, 1]
    assert groups[0].query_term_coverage == pytest.approx(1.0)
    batch_loader.assert_called_once()


def test_neighbor_without_new_query_terms_is_not_attached(
    tmp_path: Path,
) -> None:
    """Adjacent context is excluded when it cannot improve coverage."""

    source_root = tmp_path / "documents"
    source_root.mkdir()
    (source_root / "latency.md").write_text(
        "Tail latency affects the slowest requests.\n\n"
        "This paragraph discusses unrelated deployment details.",
        encoding="utf-8",
    )
    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    chunk_repository = ChunkRepository(database_path)
    ingestion_service = IngestionService(
        SourceRepository(database_path),
        chunk_repository,
        allowed_root=source_root,
        max_chunk_chars=50,
    )
    ingestion_service.ingest_file("latency.md")
    service = RetrievalService(chunk_repository)

    groups = service.search("tail latency", limit=5)

    assert len(groups) == 1
    assert len(groups[0].chunks) == 1
    assert groups[0].chunks[0].ordinal == 0
