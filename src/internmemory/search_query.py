"""Conversion of plain user text into safe FTS5 query syntax."""

import re
from typing import Literal


WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
TermMatch = Literal["all", "any"]


def extract_literal_terms(text: str) -> tuple[str, ...]:
    """Return unique case-normalized word and number tokens."""

    return tuple(
        dict.fromkeys(
            term.casefold()
            for term in WORD_PATTERN.findall(text)
        )
    )


def build_literal_fts_query(
    query: str,
    *,
    term_match: TermMatch = "all",
) -> str:
    """Build a literal all-term or any-term FTS5 query."""

    terms = extract_literal_terms(query)
    if not terms:
        raise ValueError("search query must contain a word or number")
    if term_match not in ("all", "any"):
        raise ValueError("term_match must be 'all' or 'any'")

    operator = " AND " if term_match == "all" else " OR "
    return operator.join(f'"{term}"' for term in terms)
