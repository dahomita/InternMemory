"""Small-corpus semantic retrieval with cached in-memory vectors."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from internmemory.embedding import EmbeddingProvider, EmbeddingVector
from internmemory.models import (
    DocumentSearchResult,
    SourceChunk,
    SourceReference,
)


@dataclass(frozen=True, slots=True)
class SemanticPassage:
    """A cited source chunk paired with its cached embedding."""

    source: SourceReference
    chunk: SourceChunk
    vector: EmbeddingVector


class InMemorySemanticRetriever:
    """Rank a bounded passage corpus using cosine similarity."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        passages: Sequence[tuple[SourceReference, SourceChunk]],
    ) -> None:
        self._provider = provider
        vectors = provider.embed_passages(
            [chunk.content for _, chunk in passages]
        )
        if len(vectors) != len(passages):
            raise RuntimeError(
                "embedding provider returned the wrong passage count"
            )
        self._passages = [
            SemanticPassage(
                source=source,
                chunk=chunk,
                vector=vector,
            )
            for (source, chunk), vector in zip(passages, vectors)
        ]

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        minimum_similarity: float | None = None,
    ) -> list[DocumentSearchResult]:
        """Return passages ordered by descending cosine similarity."""

        if limit <= 0:
            raise ValueError("limit must be positive")
        if (
            minimum_similarity is not None
            and not -1.0 <= minimum_similarity <= 1.0
        ):
            raise ValueError(
                "minimum_similarity must be between -1 and 1"
            )

        query_vector = self._provider.embed_query(query)
        scored = [
            (
                _cosine_similarity(query_vector, passage.vector),
                passage,
            )
            for passage in self._passages
        ]
        scored.sort(
            key=lambda item: (-item[0], item[1].chunk.id)
        )
        return [
            DocumentSearchResult(
                source=passage.source,
                chunk=passage.chunk,
                score=similarity,
            )
            for similarity, passage in scored
            if (
                minimum_similarity is None
                or similarity >= minimum_similarity
            )
        ][:limit]


def _cosine_similarity(
    left: EmbeddingVector,
    right: EmbeddingVector,
) -> float:
    """Return cosine similarity for two equal, nonzero vectors."""

    if len(left) != len(right):
        raise RuntimeError("embedding dimensions do not match")
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise RuntimeError("embedding vectors must be nonzero")
    dot_product = math.fsum(
        left_value * right_value
        for left_value, right_value in zip(left, right)
    )
    similarity = dot_product / (left_norm * right_norm)
    return max(-1.0, min(1.0, similarity))
