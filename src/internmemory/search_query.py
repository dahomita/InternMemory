"""Conversion of plain user text into safe FTS5 query syntax."""

import re


WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def build_literal_fts_query(query: str) -> str:
    """Build an AND query whose terms cannot become FTS5 operators."""

    terms = list(dict.fromkeys(WORD_PATTERN.findall(query)))
    if not terms:
        raise ValueError("search query must contain a word or number")

    return " AND ".join(f'"{term}"' for term in terms)
