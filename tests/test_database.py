"""Tests for SQLite setup and constraints."""

import sqlite3
from pathlib import Path

import pytest

from internmemory.database import connect_database, initialize_database


def test_initialize_database_creates_tables(tmp_path: Path) -> None:
    """Initialization creates the schema and enables foreign keys."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)

    connection = connect_database(database_path)
    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name IN ('projects', 'notes')"
        ).fetchall()
        foreign_keys_enabled = connection.execute(
            "PRAGMA foreign_keys"
        ).fetchone()[0]
    finally:
        connection.close()

    assert {table["name"] for table in tables} == {"projects", "notes"}
    assert foreign_keys_enabled == 1


def test_project_names_are_unique_ignoring_case(tmp_path: Path) -> None:
    """A project cannot be duplicated using different capitalization."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)

    connection = connect_database(database_path)
    try:
        connection.execute(
            "INSERT INTO projects (name) VALUES (?)",
            ("InternMemory",),
        )

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO projects (name) VALUES (?)",
                ("internmemory",),
            )
    finally:
        connection.close()


def test_note_requires_valid_project_and_content(tmp_path: Path) -> None:
    """Notes reject nonexistent projects and empty content."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)

    connection = connect_database(database_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO notes (project_id, content, work_date) "
                "VALUES (?, ?, ?)",
                (999, "Investigated retrieval latency.", "2026-11-21"),
            )

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO notes (content, work_date) VALUES (?, ?)",
                ("   ", "2026-11-21"),
            )
    finally:
        connection.close()


def test_project_with_notes_cannot_be_deleted(tmp_path: Path) -> None:
    """Deleting a project is restricted while its notes remain."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)

    connection = connect_database(database_path)
    try:
        project = connection.execute(
            "INSERT INTO projects (name) VALUES (?) RETURNING id",
            ("Retrieval research",),
        ).fetchone()
        connection.execute(
            "INSERT INTO notes (project_id, content, work_date) "
            "VALUES (?, ?, ?)",
            (
                project["id"],
                "Compared two batching strategies.",
                "2026-11-21",
            ),
        )

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "DELETE FROM projects WHERE id = ?",
                (project["id"],),
            )
    finally:
        connection.close()
