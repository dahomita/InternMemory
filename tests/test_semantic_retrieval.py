"""Tests for deterministic in-memory semantic ranking."""

from collections.abc import Sequence
from datetime import datetime, timezone

import pytest

from internmemory.embedding import EmbeddingVector
from internmemory.models import SourceChunk, SourceReference, SourceType
from internmemory.semantic_retrieval import InMemorySemanticRetriever


class FakeEmbeddingProvider:
    """Map fixture text to small vectors with known similarity."""

    dimensions = 2

    def embed_query(self, query: str) -> EmbeddingVector:
        assert query == "slow requests"
        return (1.0, 0.0)

    def embed_passages(
        self,
        passages: Sequence[str],
    ) -> list[EmbeddingVector]:
        vectors = {
            "tail latency evidence": (0.9, 0.1),
            "database schema evidence": (0.0, 1.0),
        }
        return [vectors[passage] for passage in passages]


def test_semantic_retrieval_ranks_by_cosine_similarity() -> None:
    """The passage nearest to the query vector ranks first."""

    source = SourceReference(
        id=1,
        source_type=SourceType.MARKDOWN,
        locator="fictional.md",
        title="fictional",
        content_hash="source-hash",
    )
    passages = [
        (
            source,
            _chunk(
                chunk_id=1,
                ordinal=0,
                content="database schema evidence",
            ),
        ),
        (
            source,
            _chunk(
                chunk_id=2,
                ordinal=1,
                content="tail latency evidence",
            ),
        ),
    ]
    retriever = InMemorySemanticRetriever(
        FakeEmbeddingProvider(),
        passages,
    )

    results = retriever.search("slow requests", limit=2)

    assert [result.chunk.id for result in results] == [2, 1]
    assert results[0].score == pytest.approx(0.9938837)
    assert results[1].score == pytest.approx(0.0)


def test_semantic_retrieval_can_abstain_below_similarity_threshold() -> None:
    """A minimum score can exclude weak semantic candidates."""

    source = SourceReference(
        id=1,
        source_type=SourceType.TEXT,
        locator="fictional.txt",
        title="fictional",
        content_hash="source-hash",
    )
    retriever = InMemorySemanticRetriever(
        FakeEmbeddingProvider(),
        [
            (
                source,
                _chunk(
                    chunk_id=1,
                    ordinal=0,
                    content="database schema evidence",
                ),
            )
        ],
    )

    assert retriever.search(
        "slow requests",
        minimum_similarity=0.5,
    ) == []


def _chunk(
    *,
    chunk_id: int,
    ordinal: int,
    content: str,
) -> SourceChunk:
    """Build a small cited chunk for retrieval tests."""

    return SourceChunk(
        id=chunk_id,
        source_id=1,
        ordinal=ordinal,
        start_offset=ordinal * 100,
        end_offset=ordinal * 100 + len(content),
        content=content,
        content_hash=f"chunk-{chunk_id}",
        created_at=datetime.now(timezone.utc),
    )
