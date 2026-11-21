"""Persistence operations for engineering follow-up tasks."""

import sqlite3
from datetime import date, datetime
from pathlib import Path

from internmemory.database import connect_database
from internmemory.models import Task, TaskStatus


class TaskRepository:
    """Store and query tasks without exposing SQL to callers."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = database_path

    def create_task(
        self,
        *,
        description: str,
        project_id: int | None = None,
        status: TaskStatus = TaskStatus.OPEN,
        due_date: date | None = None,
    ) -> Task:
        """Persist and return a new task."""

        connection = connect_database(self._database_path)
        try:
            with connection:
                row = connection.execute(
                    """
                    INSERT INTO tasks (
                        project_id,
                        description,
                        status,
                        due_date,
                        completed_at
                    )
                    VALUES (
                        ?, ?, ?, ?,
                        CASE
                            WHEN ? = 'completed'
                            THEN strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                            ELSE NULL
                        END
                    )
                    RETURNING *
                    """,
                    (
                        project_id,
                        description,
                        status.value,
                        due_date.isoformat() if due_date else None,
                        status.value,
                    ),
                ).fetchone()
        finally:
            connection.close()

        if row is None:
            raise RuntimeError("SQLite did not return the created task")

        return _row_to_task(row)

    def list_tasks(
        self,
        *,
        unfinished_only: bool = True,
        project_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Task]:
        """Return tasks with optional project and completion filtering."""

        conditions: list[str] = []
        parameters: list[object] = []
        if unfinished_only:
            conditions.append("status != 'completed'")
        if project_id is not None:
            conditions.append("project_id = ?")
            parameters.append(project_id)

        where_clause = (
            f"WHERE {' AND '.join(conditions)}"
            if conditions
            else ""
        )
        parameters.extend((limit, offset))

        connection = connect_database(self._database_path)
        try:
            rows = connection.execute(
                f"""
                SELECT *
                FROM tasks
                {where_clause}
                ORDER BY due_date IS NULL, due_date ASC, created_at ASC, id ASC
                LIMIT ? OFFSET ?
                """,
                parameters,
            ).fetchall()
        finally:
            connection.close()

        return [_row_to_task(row) for row in rows]

    def update_status(
        self,
        task_id: int,
        status: TaskStatus,
    ) -> Task | None:
        """Change task status while maintaining completion timestamps."""

        connection = connect_database(self._database_path)
        try:
            with connection:
                row = connection.execute(
                    """
                    UPDATE tasks
                    SET
                        status = ?,
                        completed_at = CASE
                            WHEN ? = 'completed'
                            THEN COALESCE(
                                completed_at,
                                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                            )
                            ELSE NULL
                        END,
                        updated_at = strftime(
                            '%Y-%m-%dT%H:%M:%fZ',
                            'now'
                        )
                    WHERE id = ?
                    RETURNING *
                    """,
                    (status.value, status.value, task_id),
                ).fetchone()
        finally:
            connection.close()

        return _row_to_task(row) if row is not None else None


def _row_to_task(row: sqlite3.Row) -> Task:
    """Convert a database row into the domain model."""

    return Task(
        id=row["id"],
        project_id=row["project_id"],
        description=row["description"],
        status=TaskStatus(row["status"]),
        due_date=(
            date.fromisoformat(row["due_date"])
            if row["due_date"] is not None
            else None
        ),
        completed_at=(
            _parse_timestamp(row["completed_at"])
            if row["completed_at"] is not None
            else None
        ),
        created_at=_parse_timestamp(row["created_at"]),
        updated_at=_parse_timestamp(row["updated_at"]),
    )


def _parse_timestamp(value: str) -> datetime:
    """Parse SQLite's UTC timestamp format on Python 3.10+."""

    return datetime.fromisoformat(value.replace("Z", "+00:00"))
