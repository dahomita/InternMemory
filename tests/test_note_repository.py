"""Tests for note persistence operations."""

from datetime import date
from pathlib import Path

from internmemory.database import initialize_database
from internmemory.note_repository import NoteRepository


def test_create_and_get_note_preserves_user_content(tmp_path: Path) -> None:
    """Parameterized SQL stores SQL-looking content as plain data."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    repository = NoteRepository(database_path)
    content = "'); DROP TABLE notes; --"

    created = repository.create_note(
        title="Injection research",
        content=content,
        work_date=date(2026, 11, 21),
    )
    retrieved = repository.get_note(created.id)

    assert retrieved == created
    assert retrieved.content == content
    assert retrieved.work_date == date(2026, 11, 21)
    assert retrieved.created_at.tzinfo is not None


def test_get_note_returns_none_for_unknown_id(tmp_path: Path) -> None:
    """Missing notes are represented explicitly rather than raising."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    repository = NoteRepository(database_path)

    assert repository.get_note(999) is None


def test_list_notes_filters_by_work_date(tmp_path: Path) -> None:
    """Date filtering excludes notes about work from other days."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    repository = NoteRepository(database_path)
    repository.create_note(
        title="Yesterday",
        content="Investigated retrieval latency.",
        work_date=date(2026, 11, 20),
    )
    repository.create_note(
        title="Today, first",
        content="Benchmarked keyword search.",
        work_date=date(2026, 11, 21),
    )
    repository.create_note(
        title="Today, second",
        content="Recorded benchmark results.",
        work_date=date(2026, 11, 21),
    )

    notes = repository.list_notes(work_date=date(2026, 11, 21))

    assert [note.title for note in notes] == [
        "Today, first",
        "Today, second",
    ]


def test_update_note_changes_only_supplied_fields(tmp_path: Path) -> None:
    """Partial updates preserve omitted fields."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    repository = NoteRepository(database_path)
    created = repository.create_note(
        title="Original title",
        content="Original content",
        work_date=date(2026, 11, 21),
    )

    updated = repository.update_note(
        created.id,
        {"content": "Updated content"},
    )

    assert updated is not None
    assert updated.title == "Original title"
    assert updated.content == "Updated content"
    assert updated.work_date == created.work_date


def test_delete_note_reports_whether_note_existed(tmp_path: Path) -> None:
    """Deletion distinguishes an existing note from an unknown ID."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    repository = NoteRepository(database_path)
    created = repository.create_note(
        content="Temporary note",
        work_date=date(2026, 11, 21),
    )

    assert repository.delete_note(created.id) is True
    assert repository.get_note(created.id) is None
    assert repository.delete_note(created.id) is False
