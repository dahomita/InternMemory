"""Tests for safe local document ingestion."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from internmemory.chunk_repository import ChunkRepository
from internmemory.database import initialize_database
from internmemory.ingestion_service import IngestionService
from internmemory.main import create_app
from internmemory.source_repository import SourceRepository


def test_ingestion_snapshots_hashes_and_updates_document(
    tmp_path: Path,
) -> None:
    """Re-ingesting a changed path updates one stable source record."""

    source_root = tmp_path / "documents"
    source_root.mkdir()
    document = source_root / "retrieval.md"
    document.write_text("# Retrieval\nKeyword baseline.", encoding="utf-8")
    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    chunk_repository = ChunkRepository(database_path)
    service = IngestionService(
        SourceRepository(database_path),
        chunk_repository,
        allowed_root=source_root,
        max_chunk_chars=20,
    )

    first = service.ingest_file("retrieval.md")
    first_chunks = chunk_repository.list_for_source(first.id)
    unchanged = service.ingest_file("retrieval.md")
    unchanged_chunks = chunk_repository.list_for_source(unchanged.id)
    document.write_text(
        "# Retrieval\nUpdated benchmark.",
        encoding="utf-8",
    )
    updated = service.ingest_file("retrieval.md")
    updated_chunks = chunk_repository.list_for_source(updated.id)

    assert first.id == unchanged.id == updated.id
    assert first.content_hash == unchanged.content_hash
    assert updated.content_hash != first.content_hash
    assert updated.content == "# Retrieval\nUpdated benchmark."
    assert updated.locator == "retrieval.md"
    assert [chunk.id for chunk in unchanged_chunks] == [
        chunk.id for chunk in first_chunks
    ]
    assert "".join(
        chunk.content for chunk in updated_chunks
    ) == updated.content


def test_ingestion_rejects_escape_and_oversized_files(
    tmp_path: Path,
) -> None:
    """Files outside the root or beyond the byte limit are rejected."""

    source_root = tmp_path / "documents"
    source_root.mkdir()
    (tmp_path / "outside.txt").write_text("private", encoding="utf-8")
    (source_root / "large.txt").write_text("12345", encoding="utf-8")
    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    service = IngestionService(
        SourceRepository(database_path),
        ChunkRepository(database_path),
        allowed_root=source_root,
        max_bytes=4,
    )

    with pytest.raises(ValueError, match="inside the ingestion root"):
        service.ingest_file("../outside.txt")

    with pytest.raises(ValueError, match="byte limit"):
        service.ingest_file("large.txt")


def test_ingestion_endpoint_returns_metadata_without_full_content(
    tmp_path: Path,
) -> None:
    """The API captures an allowlisted file without echoing its contents."""

    source_root = tmp_path / "documents"
    source_root.mkdir()
    (source_root / "notes.txt").write_text(
        "Public fictional engineering notes.",
        encoding="utf-8",
    )
    app = create_app(
        tmp_path / "internmemory.db",
        ingestion_root=source_root,
    )

    with TestClient(app) as client:
        response = client.post(
            "/sources/ingest",
            json={"relative_path": "notes.txt"},
        )
        missing_response = client.post(
            "/sources/ingest",
            json={"relative_path": "missing.txt"},
        )

    assert response.status_code == 200
    assert response.json()["source_type"] == "text"
    assert response.json()["locator"] == "notes.txt"
    assert response.json()["byte_size"] == 35
    assert "content" not in response.json()
    assert missing_response.status_code == 404
