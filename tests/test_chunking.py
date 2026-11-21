"""Tests for deterministic document chunk boundaries."""

from internmemory.chunking import chunk_text


def test_chunks_preserve_exact_source_offsets_and_size_bound() -> None:
    """Every chunk maps exactly back to its source snapshot."""

    content = (
        "Retrieval finds candidate evidence.\n\n"
        "BM25 ranks candidates using rarity and frequency.\n\n"
        "Evaluation measures whether the expected evidence was retrieved."
    )

    chunks = chunk_text(content, max_chars=55)

    assert len(chunks) == 4
    for ordinal, chunk in enumerate(chunks):
        assert chunk.ordinal == ordinal
        assert chunk.content == content[
            chunk.start_offset:chunk.end_offset
        ]
        assert 0 < len(chunk.content) <= 55
    assert all(
        left.end_offset <= right.start_offset
        for left, right in zip(chunks, chunks[1:])
    )
    assert "".join(chunk.content for chunk in chunks) == content


def test_oversized_word_is_split_without_empty_chunks() -> None:
    """A token longer than the bound cannot stall the chunker."""

    chunks = chunk_text("x" * 25, max_chars=10)

    assert [len(chunk.content) for chunk in chunks] == [10, 10, 5]
