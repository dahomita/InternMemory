"""Model-independent text embedding interfaces and local adapters."""

import math
from collections.abc import Sequence
from typing import Protocol


EmbeddingVector = tuple[float, ...]


class EmbeddingProvider(Protocol):
    """Convert queries and passages into vectors in one shared space."""

    @property
    def dimensions(self) -> int:
        """Return the fixed number of values in each vector."""

    def embed_query(self, query: str) -> EmbeddingVector:
        """Embed one retrieval query."""

    def embed_passages(
        self,
        passages: Sequence[str],
    ) -> list[EmbeddingVector]:
        """Embed document passages in input order."""


class FastEmbedProvider:
    """CPU-only FastEmbed adapter with explicit model dimensions."""

    def __init__(
        self,
        *,
        model_name: str,
        dimensions: int,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")

        try:
            from fastembed import TextEmbedding
        except ImportError as error:
            raise RuntimeError(
                "install InternMemory with the 'semantic' extra"
            ) from error

        self._dimensions = dimensions
        self._model = TextEmbedding(
            model_name=model_name,
            providers=["CPUExecutionProvider"],
        )

    @property
    def dimensions(self) -> int:
        """Return the configured model dimensions."""

        return self._dimensions

    def embed_query(self, query: str) -> EmbeddingVector:
        """Embed one query using model-specific query behavior."""

        if not query.strip():
            raise ValueError("query must not be blank")
        vectors = list(self._model.query_embed(query))
        if len(vectors) != 1:
            raise RuntimeError("embedding model returned an invalid query")
        return self._validated_vector(vectors[0])

    def embed_passages(
        self,
        passages: Sequence[str],
    ) -> list[EmbeddingVector]:
        """Embed nonblank passages using model-specific passage behavior."""

        if any(not passage.strip() for passage in passages):
            raise ValueError("passages must not be blank")
        if not passages:
            return []
        return [
            self._validated_vector(vector)
            for vector in self._model.passage_embed(passages)
        ]

    def _validated_vector(
        self,
        vector: Sequence[float],
    ) -> EmbeddingVector:
        """Validate provider output before it enters retrieval state."""

        converted = tuple(float(value) for value in vector)
        if len(converted) != self._dimensions:
            raise RuntimeError(
                "embedding model returned unexpected dimensions"
            )
        if not all(math.isfinite(value) for value in converted):
            raise RuntimeError(
                "embedding model returned a non-finite value"
            )
        return converted
