"""Integration tests for cited document-chunk retrieval."""

from pathlib import Path

from fastapi.testclient import TestClient

from internmemory.main import create_app


def test_document_search_returns_ranked_chunk_with_citation(
    tmp_path: Path,
) -> None:
    """Search results expose exact evidence and source provenance."""

    source_root = tmp_path / "documents"
    source_root.mkdir()
    content = (
        "# Batching\n\n"
        "Larger batches improve throughput.\n\n"
        "Waiting for a batch to fill can increase tail latency."
    )
    (source_root / "batching.md").write_text(content, encoding="utf-8")
    app = create_app(
        tmp_path / "internmemory.db",
        ingestion_root=source_root,
        max_chunk_chars=60,
    )

    with TestClient(app) as client:
        ingest_response = client.post(
            "/sources/ingest",
            json={"relative_path": "batching.md"},
        )
        search_response = client.get(
            "/search/documents",
            params={"q": "tail latency"},
        )

    assert ingest_response.status_code == 200
    assert search_response.status_code == 200
    result = search_response.json()[0]
    assert result["source"]["locator"] == "batching.md"
    assert result["chunk"]["content"] == content[
        result["chunk"]["start_offset"]:result["chunk"]["end_offset"]
    ]
    assert "tail latency" in result["chunk"]["content"]
    assert isinstance(result["score"], float)


def test_document_search_index_tracks_reingestion(
    tmp_path: Path,
) -> None:
    """Changed source snapshots replace stale searchable chunks."""

    source_root = tmp_path / "documents"
    source_root.mkdir()
    document = source_root / "research.txt"
    document.write_text(
        "Investigated obsoletekeyword.",
        encoding="utf-8",
    )
    app = create_app(
        tmp_path / "internmemory.db",
        ingestion_root=source_root,
    )

    with TestClient(app) as client:
        client.post(
            "/sources/ingest",
            json={"relative_path": "research.txt"},
        )
        assert len(
            client.get(
                "/search/documents",
                params={"q": "obsoletekeyword"},
            ).json()
        ) == 1

        document.write_text(
            "Investigated replacementkeyword.",
            encoding="utf-8",
        )
        client.post(
            "/sources/ingest",
            json={"relative_path": "research.txt"},
        )
        stale_response = client.get(
            "/search/documents",
            params={"q": "obsoletekeyword"},
        )
        current_response = client.get(
            "/search/documents",
            params={"q": "replacementkeyword"},
        )

    assert stale_response.json() == []
    assert len(current_response.json()) == 1


def test_document_search_treats_fts_punctuation_as_literal(
    tmp_path: Path,
) -> None:
    """Raw FTS punctuation cannot become executable query syntax."""

    source_root = tmp_path / "documents"
    source_root.mkdir()
    app = create_app(
        tmp_path / "internmemory.db",
        ingestion_root=source_root,
    )

    with TestClient(app) as client:
        response = client.get(
            "/search/documents",
            params={"q": '"unterminated'},
        )

    assert response.status_code == 200
    assert response.json() == []
