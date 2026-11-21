"""Tests for FTS5 keyword retrieval and BM25 ranking."""

from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient

from internmemory.database import initialize_database
from internmemory.main import create_app
from internmemory.note_repository import NoteRepository


def test_bm25_prefers_query_terms_in_title(tmp_path: Path) -> None:
    """The configured title weight influences result order."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    repository = NoteRepository(database_path)
    title_match = repository.create_note(
        title="Tail latency investigation",
        content="Compared two batching strategies.",
        work_date=date(2026, 11, 21),
    )
    repository.create_note(
        title="Batching investigation",
        content="Read documentation about tail latency behavior.",
        work_date=date(2026, 11, 21),
    )

    results = repository.search_notes("tail latency")

    assert [result.note.id for result in results][0] == title_match.id
    assert results[0].score <= results[1].score


def test_search_index_tracks_updates_and_deletes(tmp_path: Path) -> None:
    """Triggers prevent updated or deleted notes from becoming stale."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    repository = NoteRepository(database_path)
    created = repository.create_note(
        content="Investigated obsoletekeyword.",
        work_date=date(2026, 11, 21),
    )

    assert len(repository.search_notes("obsoletekeyword")) == 1

    repository.update_note(
        created.id,
        {"content": "Investigated replacementkeyword."},
    )

    assert repository.search_notes("obsoletekeyword") == []
    assert len(repository.search_notes("replacementkeyword")) == 1

    repository.delete_note(created.id)

    assert repository.search_notes("replacementkeyword") == []


def test_search_endpoint_handles_literal_punctuation_and_empty_terms(
    tmp_path: Path,
) -> None:
    """User punctuation is literal while punctuation-only input is invalid."""

    app = create_app(tmp_path / "internmemory.db")

    with TestClient(app) as client:
        client.post(
            "/notes",
            json={
                "title": "Retrieval latency",
                "content": "Measured p95 latency.",
                "work_date": "2026-11-21",
            },
        )
        search_response = client.get(
            "/search/notes",
            params={"q": "retrieval"},
        )
        literal_response = client.get(
            "/search/notes",
            params={"q": '"unterminated'},
        )
        invalid_response = client.get(
            "/search/notes",
            params={"q": "---"},
        )

    assert search_response.status_code == 200
    assert search_response.json()[0]["note"]["title"] == "Retrieval latency"
    assert isinstance(search_response.json()[0]["score"], float)
    assert literal_response.status_code == 200
    assert literal_response.json() == []
    assert invalid_response.status_code == 422
    assert invalid_response.json() == {
        "detail": "search query must contain a word or number"
    }
