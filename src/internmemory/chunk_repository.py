"""Persistence operations for source-document chunks."""

import sqlite3
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from internmemory.chunking import ChunkDraft
from internmemory.database import connect_database
from internmemory.models import (
    DocumentSearchResult,
    SourceChunk,
    SourceReference,
    SourceType,
)
from internmemory.search_query import TermMatch, build_literal_fts_query


class ChunkRepository:
    """Store deterministic chunks associated with source snapshots."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = database_path

    def replace_for_source(
        self,
        source_id: int,
        drafts: Sequence[ChunkDraft],
    ) -> list[SourceChunk]:
        """Replace changed chunks while preserving unchanged chunk IDs."""

        connection = connect_database(self._database_path)
        try:
            existing_rows = connection.execute(
                """
                SELECT *
                FROM source_chunks
                WHERE source_id = ?
                ORDER BY ordinal ASC
                """,
                (source_id,),
            ).fetchall()

            if _rows_match_drafts(existing_rows, drafts):
                return [_row_to_chunk(row) for row in existing_rows]

            with connection:
                connection.execute(
                    "DELETE FROM source_chunks WHERE source_id = ?",
                    (source_id,),
                )
                connection.executemany(
                    """
                    INSERT INTO source_chunks (
                        source_id,
                        ordinal,
                        start_offset,
                        end_offset,
                        content,
                        content_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            source_id,
                            draft.ordinal,
                            draft.start_offset,
                            draft.end_offset,
                            draft.content,
                            draft.content_hash,
                        )
                        for draft in drafts
                    ],
                )

            rows = connection.execute(
                """
                SELECT *
                FROM source_chunks
                WHERE source_id = ?
                ORDER BY ordinal ASC
                """,
                (source_id,),
            ).fetchall()
        finally:
            connection.close()

        return [_row_to_chunk(row) for row in rows]

    def list_for_source(self, source_id: int) -> list[SourceChunk]:
        """Return a source's chunks in document order."""

        connection = connect_database(self._database_path)
        try:
            rows = connection.execute(
                """
                SELECT *
                FROM source_chunks
                WHERE source_id = ?
                ORDER BY ordinal ASC
                """,
                (source_id,),
            ).fetchall()
        finally:
            connection.close()

        return [_row_to_chunk(row) for row in rows]

    def list_for_sources(
        self,
        source_ids: Sequence[int],
    ) -> list[SourceChunk]:
        """Return chunks for multiple sources using one database query."""

        unique_source_ids = tuple(dict.fromkeys(source_ids))
        if not unique_source_ids:
            return []

        placeholders = ", ".join("?" for _ in unique_source_ids)
        connection = connect_database(self._database_path)
        try:
            rows = connection.execute(
                f"""
                SELECT *
                FROM source_chunks
                WHERE source_id IN ({placeholders})
                ORDER BY source_id ASC, ordinal ASC
                """,
                unique_source_ids,
            ).fetchall()
        finally:
            connection.close()

        return [_row_to_chunk(row) for row in rows]

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        term_match: TermMatch = "all",
    ) -> list[DocumentSearchResult]:
        """Return source chunks ordered by FTS5 BM25 relevance."""

        fts_query = build_literal_fts_query(
            query,
            term_match=term_match,
        )

        connection = connect_database(self._database_path)
        try:
            rows = connection.execute(
                """
                SELECT
                    c.id AS chunk_id,
                    c.source_id,
                    c.ordinal,
                    c.start_offset,
                    c.end_offset,
                    c.content AS chunk_content,
                    c.content_hash AS chunk_content_hash,
                    c.created_at AS chunk_created_at,
                    s.source_type,
                    s.locator,
                    s.title,
                    s.content_hash AS source_content_hash,
                    bm25(source_chunks_fts) AS score
                FROM source_chunks_fts
                JOIN source_chunks AS c
                    ON c.id = source_chunks_fts.rowid
                JOIN sources AS s
                    ON s.id = c.source_id
                WHERE source_chunks_fts MATCH ?
                ORDER BY score ASC, c.id ASC
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        finally:
            connection.close()

        return [_search_row_to_result(row) for row in rows]

    def search_with_fallback(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> list[DocumentSearchResult]:
        """Prioritize all-term matches, then fill with any-term matches."""

        strict_results = self.search(
            query,
            limit=limit,
            term_match="all",
        )
        if len(strict_results) >= limit:
            return strict_results

        seen_chunk_ids = {
            result.chunk.id
            for result in strict_results
        }
        results = list(strict_results)
        broad_results = self.search(
            query,
            limit=limit + len(strict_results),
            term_match="any",
        )
        for result in broad_results:
            if result.chunk.id in seen_chunk_ids:
                continue
            results.append(result)
            seen_chunk_ids.add(result.chunk.id)
            if len(results) == limit:
                break

        return results


def _rows_match_drafts(
    rows: Sequence[sqlite3.Row],
    drafts: Sequence[ChunkDraft],
) -> bool:
    """Return whether persisted chunks exactly match generated drafts."""

    if len(rows) != len(drafts):
        return False

    return all(
        row["ordinal"] == draft.ordinal
        and row["start_offset"] == draft.start_offset
        and row["end_offset"] == draft.end_offset
        and row["content_hash"] == draft.content_hash
        for row, draft in zip(rows, drafts)
    )


def _row_to_chunk(row: sqlite3.Row) -> SourceChunk:
    """Convert a database row into the domain model."""

    return SourceChunk(
        id=row["id"],
        source_id=row["source_id"],
        ordinal=row["ordinal"],
        start_offset=row["start_offset"],
        end_offset=row["end_offset"],
        content=row["content"],
        content_hash=row["content_hash"],
        created_at=_parse_timestamp(row["created_at"]),
    )


def _search_row_to_result(row: sqlite3.Row) -> DocumentSearchResult:
    """Convert a joined FTS row into a cited search result."""

    return DocumentSearchResult(
        source=SourceReference(
            id=row["source_id"],
            source_type=SourceType(row["source_type"]),
            locator=row["locator"],
            title=row["title"],
            content_hash=row["source_content_hash"],
        ),
        chunk=SourceChunk(
            id=row["chunk_id"],
            source_id=row["source_id"],
            ordinal=row["ordinal"],
            start_offset=row["start_offset"],
            end_offset=row["end_offset"],
            content=row["chunk_content"],
            content_hash=row["chunk_content_hash"],
            created_at=_parse_timestamp(row["chunk_created_at"]),
        ),
        score=row["score"],
    )


def _parse_timestamp(value: str) -> datetime:
    """Parse SQLite's UTC timestamp format on Python 3.10+."""

    return datetime.fromisoformat(value.replace("Z", "+00:00"))
