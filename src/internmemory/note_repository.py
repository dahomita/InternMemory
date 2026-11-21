"""Persistence operations for engineering notes."""

import sqlite3
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path

from internmemory.database import connect_database
from internmemory.models import Note, NoteSearchResult
from internmemory.search_query import build_literal_fts_query


class NoteRepository:
    """Store and retrieve notes without exposing SQL to callers."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = database_path

    def create_note(
        self,
        *,
        content: str,
        work_date: date,
        project_id: int | None = None,
        title: str = "",
    ) -> Note:
        """Persist and return a new note."""

        connection = connect_database(self._database_path)
        try:
            with connection:
                row = connection.execute(
                    """
                    INSERT INTO notes (project_id, title, content, work_date)
                    VALUES (?, ?, ?, ?)
                    RETURNING *
                    """,
                    (project_id, title, content, work_date.isoformat()),
                ).fetchone()
        finally:
            connection.close()

        if row is None:
            raise RuntimeError("SQLite did not return the created note")

        return _row_to_note(row)

    def get_note(self, note_id: int) -> Note | None:
        """Return a note by ID, or None when it does not exist."""

        connection = connect_database(self._database_path)
        try:
            row = connection.execute(
                "SELECT * FROM notes WHERE id = ?",
                (note_id,),
            ).fetchone()
        finally:
            connection.close()

        return _row_to_note(row) if row is not None else None

    def list_notes(
        self,
        *,
        work_date: date | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Note]:
        """Return notes in deterministic order with optional date filtering."""

        where_clause = ""
        parameters: list[object] = []
        if work_date is not None:
            where_clause = "WHERE work_date = ?"
            parameters.append(work_date.isoformat())
        parameters.extend((limit, offset))

        connection = connect_database(self._database_path)
        try:
            rows = connection.execute(
                f"""
                SELECT *
                FROM notes
                {where_clause}
                ORDER BY work_date DESC, created_at ASC, id ASC
                LIMIT ? OFFSET ?
                """,
                parameters,
            ).fetchall()
        finally:
            connection.close()

        return [_row_to_note(row) for row in rows]

    def search_notes(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> list[NoteSearchResult]:
        """Return FTS5 matches ordered by BM25 relevance."""

        fts_query = build_literal_fts_query(query)

        connection = connect_database(self._database_path)
        try:
            rows = connection.execute(
                """
                SELECT n.*, bm25(notes_fts, 5.0, 1.0) AS score
                FROM notes_fts
                JOIN notes AS n ON n.id = notes_fts.rowid
                WHERE notes_fts MATCH ?
                ORDER BY score ASC, n.id ASC
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        finally:
            connection.close()

        return [
            NoteSearchResult(
                note=_row_to_note(row),
                score=row["score"],
            )
            for row in rows
        ]

    def update_note(
        self,
        note_id: int,
        changes: Mapping[str, object],
    ) -> Note | None:
        """Apply supplied fields and return the updated note if it exists."""

        if not changes:
            return self.get_note(note_id)

        allowed_fields = {"project_id", "title", "content", "work_date"}
        unknown_fields = changes.keys() - allowed_fields
        if unknown_fields:
            names = ", ".join(sorted(unknown_fields))
            raise ValueError(f"Unsupported note fields: {names}")

        assignments: list[str] = []
        values: list[object] = []
        for field_name, value in changes.items():
            assignments.append(f"{field_name} = ?")
            if field_name == "work_date":
                if not isinstance(value, date):
                    raise TypeError("work_date must be a date")
                value = value.isoformat()
            values.append(value)

        assignments.append(
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
        )
        values.append(note_id)

        connection = connect_database(self._database_path)
        try:
            with connection:
                row = connection.execute(
                    f"""
                    UPDATE notes
                    SET {", ".join(assignments)}
                    WHERE id = ?
                    RETURNING *
                    """,
                    values,
                ).fetchone()
        finally:
            connection.close()

        return _row_to_note(row) if row is not None else None

    def delete_note(self, note_id: int) -> bool:
        """Delete a note and report whether it existed."""

        connection = connect_database(self._database_path)
        try:
            with connection:
                row = connection.execute(
                    "DELETE FROM notes WHERE id = ? RETURNING id",
                    (note_id,),
                ).fetchone()
        finally:
            connection.close()

        return row is not None


def _row_to_note(row: sqlite3.Row) -> Note:
    """Convert a database row into the domain model."""

    return Note(
        id=row["id"],
        project_id=row["project_id"],
        title=row["title"],
        content=row["content"],
        work_date=date.fromisoformat(row["work_date"]),
        created_at=_parse_timestamp(row["created_at"]),
        updated_at=_parse_timestamp(row["updated_at"]),
    )


def _parse_timestamp(value: str) -> datetime:
    """Parse SQLite's UTC timestamp format on Python 3.10+."""

    return datetime.fromisoformat(value.replace("Z", "+00:00"))
