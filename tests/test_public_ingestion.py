"""Tests for controlled public documentation ingestion."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from internmemory.chunk_repository import ChunkRepository
from internmemory.database import initialize_database
from internmemory.main import create_app
from internmemory.public_ingestion_service import (
    FetchedDocument,
    PublicDocumentIngestionService,
)
from internmemory.source_repository import SourceRepository


class FakeFetcher:
    """Return deterministic page bytes without network access."""

    def __init__(self, document: FetchedDocument) -> None:
        self.document = document
        self.calls = 0

    def fetch(
        self,
        url: str,
        *,
        max_bytes: int,
        timeout_seconds: float,
    ) -> FetchedDocument:
        del url, max_bytes, timeout_seconds
        self.calls += 1
        return self.document


def test_public_html_is_sanitized_snapshotted_and_searchable(
    tmp_path: Path,
) -> None:
    """Visible allowlisted text is indexed without active page content."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    chunk_repository = ChunkRepository(database_path)
    fetcher = FakeFetcher(
        FetchedDocument(
            final_url="https://docs.example.test/guide",
            content_type="text/html",
            body=(
                b"<html><head><title>Retry Guide</title>"
                b"<style>hidden style</style></head><body>"
                b"<h1>Retry budgets</h1>"
                b"<p>Bounded retries protect dependencies.</p>"
                b"<script>malicious instructions</script>"
                b"</body></html>"
            ),
        )
    )
    service = PublicDocumentIngestionService(
        SourceRepository(database_path),
        chunk_repository,
        allowed_domains={"docs.example.test"},
        fetcher=fetcher,
    )

    source = service.ingest_url("https://docs.example.test/guide")
    results = chunk_repository.search("bounded retries")

    assert source.locator == "https://docs.example.test/guide"
    assert source.title == "Retry Guide"
    assert "malicious instructions" not in source.content
    assert results[0].source.id == source.id


def test_public_ingestion_rejects_unallowlisted_request_before_fetch(
    tmp_path: Path,
) -> None:
    """An untrusted hostname cannot trigger outbound network access."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    fetcher = FakeFetcher(
        FetchedDocument(
            final_url="https://evil.example/page",
            content_type="text/plain",
            body=b"untrusted",
        )
    )
    service = PublicDocumentIngestionService(
        SourceRepository(database_path),
        ChunkRepository(database_path),
        allowed_domains={"docs.example.test"},
        fetcher=fetcher,
    )

    with pytest.raises(ValueError, match="not allowlisted"):
        service.ingest_url("https://evil.example/page")

    assert fetcher.calls == 0


def test_public_ingestion_rejects_redirect_outside_allowlist(
    tmp_path: Path,
) -> None:
    """The final redirect target receives the same domain validation."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    service = PublicDocumentIngestionService(
        SourceRepository(database_path),
        ChunkRepository(database_path),
        allowed_domains={"docs.example.test"},
        fetcher=FakeFetcher(
            FetchedDocument(
                final_url="https://evil.example/redirected",
                content_type="text/plain",
                body=b"redirected",
            )
        ),
    )

    with pytest.raises(ValueError, match="not allowlisted"):
        service.ingest_url("https://docs.example.test/start")


@pytest.mark.parametrize(
    ("document", "max_bytes", "message"),
    [
        (
            FetchedDocument(
                final_url="https://docs.example.test/large.txt",
                content_type="text/plain",
                body=b"too large",
            ),
            3,
            "exceeds",
        ),
        (
            FetchedDocument(
                final_url="https://docs.example.test/manual.pdf",
                content_type="application/pdf",
                body=b"pdf",
            ),
            100,
            "content type",
        ),
    ],
)
def test_public_ingestion_enforces_resource_and_type_limits(
    tmp_path: Path,
    document: FetchedDocument,
    max_bytes: int,
    message: str,
) -> None:
    """Oversized and unsupported responses never enter storage."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    service = PublicDocumentIngestionService(
        SourceRepository(database_path),
        ChunkRepository(database_path),
        allowed_domains={"docs.example.test"},
        max_bytes=max_bytes,
        fetcher=FakeFetcher(document),
    )

    with pytest.raises(ValueError, match=message):
        service.ingest_url("https://docs.example.test/document")


def test_public_ingestion_endpoint_returns_cited_source(
    tmp_path: Path,
) -> None:
    """The HTTP endpoint exposes persisted public-source provenance."""

    database_path = tmp_path / "internmemory.db"
    app = create_app(
        database_path,
        ingestion_root=tmp_path,
        public_document_domains={"docs.example.test"},
    )
    app.state.public_ingestion_service = PublicDocumentIngestionService(
        SourceRepository(database_path),
        app.state.chunk_repository,
        allowed_domains={"docs.example.test"},
        fetcher=FakeFetcher(
            FetchedDocument(
                final_url="https://docs.example.test/retries.txt",
                content_type="text/plain",
                body=b"Bounded retry budgets protect dependencies.",
            )
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/sources/ingest-url",
            json={"url": "https://docs.example.test/retries.txt"},
        )

    assert response.status_code == 200
    assert response.json()["locator"] == (
        "https://docs.example.test/retries.txt"
    )
