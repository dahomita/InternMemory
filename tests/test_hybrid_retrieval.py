"""Tests for reciprocal-rank-fusion hybrid retrieval."""

from datetime import datetime, timezone

import pytest

from internmemory.hybrid_retrieval import reciprocal_rank_fusion
from internmemory.models import (
    DocumentSearchResult,
    SourceChunk,
    SourceReference,
    SourceType,
)


def test_rrf_promotes_results_supported_by_both_rankings() -> None:
    """Agreement across retrievers outranks one retriever's top result."""

    keyword = [_result(1), _result(2)]
    semantic = [_result(2), _result(3)]

    fused = reciprocal_rank_fusion(
        [keyword, semantic],
        limit=3,
        rrf_k=60,
    )

    assert [result.chunk.id for result in fused] == [2, 1, 3]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_deduplicates_repeated_chunks_within_one_ranking() -> None:
    """One retriever cannot inflate a chunk by returning it twice."""

    duplicate = _result(1)

    fused = reciprocal_rank_fusion(
        [[duplicate, duplicate]],
        limit=5,
    )

    assert len(fused) == 1
    assert fused[0].score == pytest.approx(1 / 61)


def test_rrf_applies_explicit_ranking_weights() -> None:
    """A trusted retriever can contribute more to the fused order."""

    fused = reciprocal_rank_fusion(
        [[_result(1)], [_result(2)]],
        ranking_weights=[1.0, 2.0],
        limit=2,
    )

    assert [result.chunk.id for result in fused] == [2, 1]


def _result(chunk_id: int) -> DocumentSearchResult:
    """Build a cited result whose raw score is intentionally irrelevant."""

    source = SourceReference(
        id=chunk_id,
        source_type=SourceType.MARKDOWN,
        locator=f"source-{chunk_id}.md",
        title=f"source-{chunk_id}",
        content_hash=f"source-hash-{chunk_id}",
    )
    chunk = SourceChunk(
        id=chunk_id,
        source_id=chunk_id,
        ordinal=0,
        start_offset=0,
        end_offset=10,
        content=f"chunk {chunk_id}",
        content_hash=f"chunk-hash-{chunk_id}",
        created_at=datetime.now(timezone.utc),
    )
    return DocumentSearchResult(
        source=source,
        chunk=chunk,
        score=float(chunk_id),
    )
