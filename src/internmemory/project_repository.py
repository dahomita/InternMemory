"""Persistence operations for engineering projects."""

import sqlite3
from datetime import datetime
from pathlib import Path

from internmemory.database import connect_database
from internmemory.models import Project


class ProjectRepository:
    """Store and list projects without exposing SQL to callers."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = database_path

    def create_project(
        self,
        *,
        name: str,
        description: str = "",
    ) -> Project:
        """Persist and return a new project."""

        connection = connect_database(self._database_path)
        try:
            with connection:
                row = connection.execute(
                    """
                    INSERT INTO projects (name, description)
                    VALUES (?, ?)
                    RETURNING *
                    """,
                    (name, description),
                ).fetchone()
        finally:
            connection.close()

        if row is None:
            raise RuntimeError("SQLite did not return the created project")

        return _row_to_project(row)

    def list_projects(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Project]:
        """Return projects in deterministic name order."""

        connection = connect_database(self._database_path)
        try:
            rows = connection.execute(
                """
                SELECT *
                FROM projects
                ORDER BY name COLLATE NOCASE ASC, id ASC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        finally:
            connection.close()

        return [_row_to_project(row) for row in rows]


def _row_to_project(row: sqlite3.Row) -> Project:
    """Convert a database row into the domain model."""

    return Project(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        created_at=_parse_timestamp(row["created_at"]),
        updated_at=_parse_timestamp(row["updated_at"]),
    )


def _parse_timestamp(value: str) -> datetime:
    """Parse SQLite's UTC timestamp format on Python 3.10+."""

    return datetime.fromisoformat(value.replace("Z", "+00:00"))
