"""Rank-fusion retrieval combining keyword and semantic candidates."""

from collections.abc import Sequence

from internmemory.chunk_repository import ChunkRepository
from internmemory.models import DocumentSearchResult
from internmemory.semantic_retrieval import InMemorySemanticRetriever


class HybridRetriever:
    """Fuse keyword and semantic chunk ranks using RRF."""

    def __init__(
        self,
        keyword_repository: ChunkRepository,
        semantic_retriever: InMemorySemanticRetriever,
        *,
        candidate_limit: int = 20,
        rrf_k: int = 60,
        keyword_weight: float = 1.0,
        semantic_weight: float = 1.0,
    ) -> None:
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if keyword_weight <= 0 or semantic_weight <= 0:
            raise ValueError("retrieval weights must be positive")
        self._keyword_repository = keyword_repository
        self._semantic_retriever = semantic_retriever
        self._candidate_limit = candidate_limit
        self._rrf_k = rrf_k
        self._keyword_weight = keyword_weight
        self._semantic_weight = semantic_weight

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        minimum_similarity: float | None = None,
    ) -> list[DocumentSearchResult]:
        """Return fused candidates ordered by descending RRF score."""

        if limit <= 0:
            raise ValueError("limit must be positive")
        keyword_results = self._keyword_repository.search_with_fallback(
            query,
            limit=self._candidate_limit,
        )
        semantic_results = self._semantic_retriever.search(
            query,
            limit=self._candidate_limit,
            minimum_similarity=minimum_similarity,
        )
        return reciprocal_rank_fusion(
            [keyword_results, semantic_results],
            ranking_weights=[
                self._keyword_weight,
                self._semantic_weight,
            ],
            limit=limit,
            rrf_k=self._rrf_k,
        )


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[DocumentSearchResult]],
    *,
    limit: int,
    rrf_k: int = 60,
    ranking_weights: Sequence[float] | None = None,
) -> list[DocumentSearchResult]:
    """Fuse rankings without comparing their incompatible raw scores."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    weights = (
        list(ranking_weights)
        if ranking_weights is not None
        else [1.0] * len(rankings)
    )
    if len(weights) != len(rankings):
        raise ValueError("one weight is required per ranking")
    if any(weight <= 0 for weight in weights):
        raise ValueError("ranking weights must be positive")

    scores: dict[int, float] = {}
    best_ranks: dict[int, int] = {}
    results_by_chunk: dict[int, DocumentSearchResult] = {}
    for ranking, weight in zip(rankings, weights):
        seen_in_ranking: set[int] = set()
        for rank, result in enumerate(ranking, start=1):
            chunk_id = result.chunk.id
            if chunk_id in seen_in_ranking:
                continue
            seen_in_ranking.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                weight / (rrf_k + rank)
            )
            best_ranks[chunk_id] = min(
                best_ranks.get(chunk_id, rank),
                rank,
            )
            results_by_chunk.setdefault(chunk_id, result)

    ordered_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            best_ranks[chunk_id],
            chunk_id,
        ),
    )
    return [
        DocumentSearchResult(
            source=results_by_chunk[chunk_id].source,
            chunk=results_by_chunk[chunk_id].chunk,
            score=scores[chunk_id],
        )
        for chunk_id in ordered_ids[:limit]
    ]
