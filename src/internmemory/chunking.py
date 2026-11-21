"""Deterministic character-bounded document chunking."""

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """A chunk ready to be associated with a persisted source."""

    ordinal: int
    start_offset: int
    end_offset: int
    content: str
    content_hash: str


def chunk_text(content: str, *, max_chars: int = 1_200) -> list[ChunkDraft]:
    """Split text near paragraph or word boundaries without overlap."""

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    chunks: list[ChunkDraft] = []
    cursor = 0
    content_length = len(content)

    while cursor < content_length:
        while cursor < content_length and content[cursor].isspace():
            cursor += 1
        if cursor >= content_length:
            break

        hard_end = min(cursor + max_chars, content_length)
        end = hard_end

        if hard_end < content_length:
            minimum_boundary = cursor + max(1, max_chars // 2)
            paragraph_break = content.rfind(
                "\n\n",
                minimum_boundary,
                hard_end,
            )
            if paragraph_break >= minimum_boundary:
                end = paragraph_break + 2
            else:
                whitespace_positions = (
                    content.rfind(" ", cursor, hard_end),
                    content.rfind("\n", cursor, hard_end),
                    content.rfind("\t", cursor, hard_end),
                )
                word_break = max(whitespace_positions)
                if word_break > cursor:
                    end = word_break + 1

        chunk_content = content[cursor:end]
        chunks.append(
            ChunkDraft(
                ordinal=len(chunks),
                start_offset=cursor,
                end_offset=end,
                content=chunk_content,
                content_hash=hashlib.sha256(
                    chunk_content.encode("utf-8")
                ).hexdigest(),
            )
        )
        cursor=end

    return chunks
