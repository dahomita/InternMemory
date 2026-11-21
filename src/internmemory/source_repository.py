"""Persistence operations for ingested document sources."""

import sqlite3
from datetime import datetime
from pathlib import Path

from internmemory.database import connect_database
from internmemory.models import Source, SourceType


class SourceRepository:
    """Store source snapshots without exposing SQL to callers."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = database_path

    def upsert_source(
        self,
        *,
        source_type: SourceType,
        locator: str,
        title: str,
        content: str,
        content_hash: str,
        byte_size: int,
    ) -> Source:
        """Create a source or replace its snapshot when content changes."""

        connection = connect_database(self._database_path)
        try:
            with connection:
                existing = connection.execute(
                    "SELECT * FROM sources WHERE locator = ?",
                    (locator,),
                ).fetchone()

                if (
                    existing is not None
                    and existing["source_type"] == source_type.value
                    and existing["title"] == title
                    and existing["content_hash"] == content_hash
                ):
                    return _row_to_source(existing)

                if existing is None:
                    row = connection.execute(
                        """
                        INSERT INTO sources (
                            source_type,
                            locator,
                            title,
                            content,
                            content_hash,
                            byte_size
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        RETURNING *
                        """,
                        (
                            source_type.value,
                            locator,
                            title,
                            content,
                            content_hash,
                            byte_size,
                        ),
                    ).fetchone()
                else:
                    row = connection.execute(
                        """
                        UPDATE sources
                        SET
                            source_type = ?,
                            title = ?,
                            content = ?,
                            content_hash = ?,
                            byte_size = ?,
                            updated_at = strftime(
                                '%Y-%m-%dT%H:%M:%fZ',
                                'now'
                            )
                        WHERE id = ?
                        RETURNING *
                        """,
                        (
                            source_type.value,
                            title,
                            content,
                            content_hash,
                            byte_size,
                            existing["id"],
                        ),
                    ).fetchone()
        finally:
            connection.close()

        if row is None:
            raise RuntimeError("SQLite did not return the ingested source")

        return _row_to_source(row)


def _row_to_source(row: sqlite3.Row) -> Source:
    """Convert a database row into the domain model."""

    return Source(
        id=row["id"],
        source_type=SourceType(row["source_type"]),
        locator=row["locator"],
        title=row["title"],
        content=row["content"],
        content_hash=row["content_hash"],
        byte_size=row["byte_size"],
        created_at=_parse_timestamp(row["created_at"]),
        updated_at=_parse_timestamp(row["updated_at"]),
    )


def _parse_timestamp(value: str) -> datetime:
    """Parse SQLite's UTC timestamp format on Python 3.10+."""

    return datetime.fromisoformat(value.replace("Z", "+00:00"))
