"""Tests for plain-text to FTS5 query conversion."""

import pytest

from internmemory.search_query import build_literal_fts_query


def test_hyphenated_terms_become_safe_literal_tokens() -> None:
    """Punctuation cannot be interpreted as an FTS5 operator."""

    query = build_literal_fts_query(
        "throughput per-request overhead"
    )

    assert query == (
        '"throughput" AND "per" AND "request" AND "overhead"'
    )


def test_duplicate_terms_are_removed_in_original_order() -> None:
    """Repeated user terms do not enlarge the generated expression."""

    assert build_literal_fts_query("latency tail latency") == (
        '"latency" AND "tail"'
    )


def test_query_requires_at_least_one_searchable_token() -> None:
    """Punctuation-only input is rejected before SQLite."""

    with pytest.raises(ValueError, match="word or number"):
        build_literal_fts_query("---")
