"""Application service for building bounded, cited evidence groups."""

from dataclasses import dataclass

from internmemory.chunk_repository import ChunkRepository
from internmemory.models import SourceChunk, SourceReference
from internmemory.search_query import extract_literal_terms


@dataclass(frozen=True, slots=True)
class EvidenceGroup:
    """One ranked source passage containing one or more cited chunks."""

    source: SourceReference
    chunks: tuple[SourceChunk, ...]
    anchor_score: float
    query_term_coverage: float


class RetrievalService:
    """Retrieve strict-first candidates and attach useful neighbors."""

    def __init__(self, chunk_repository: ChunkRepository) -> None:
        self._chunk_repository = chunk_repository

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        rerank_by_coverage: bool = False,
    ) -> list[EvidenceGroup]:
        """Return deduplicated evidence groups in candidate rank order."""

        query_terms = set(extract_literal_terms(query))
        if not query_terms:
            raise ValueError("search query must contain a word or number")

        candidates = self._chunk_repository.search_with_fallback(
            query,
            limit=limit,
        )
        chunks_by_source: dict[int, dict[int, SourceChunk]] = {}
        for chunk in self._chunk_repository.list_for_sources(
            [candidate.source.id for candidate in candidates]
        ):
            chunks_by_source.setdefault(
                chunk.source_id,
                {},
            )[chunk.ordinal] = chunk

        groups: list[EvidenceGroup] = []
        seen_groups: set[tuple[int, ...]] = set()

        for candidate in candidates:
            chunks_by_ordinal = chunks_by_source[candidate.source.id]
            selected = [candidate.chunk]
            covered_terms = self._covered_terms(
                candidate.chunk,
                query_terms,
            )

            for neighbor_ordinal in (
                candidate.chunk.ordinal - 1,
                candidate.chunk.ordinal + 1,
            ):
                neighbor = chunks_by_ordinal.get(neighbor_ordinal)
                if neighbor is None:
                    continue
                neighbor_terms = self._covered_terms(
                    neighbor,
                    query_terms,
                )
                if neighbor_terms <= covered_terms:
                    continue
                selected.append(neighbor)
                covered_terms.update(neighbor_terms)

            ordered_chunks = tuple(
                sorted(selected, key=lambda chunk: chunk.ordinal)
            )
            group_key = tuple(chunk.id for chunk in ordered_chunks)
            if group_key in seen_groups:
                continue
            seen_groups.add(group_key)
            groups.append(
                EvidenceGroup(
                    source=candidate.source,
                    chunks=ordered_chunks,
                    anchor_score=candidate.score,
                    query_term_coverage=(
                        len(covered_terms) / len(query_terms)
                    ),
                )
            )

        if rerank_by_coverage:
            groups.sort(
                key=lambda group: (
                    -group.query_term_coverage,
                    len(group.chunks),
                )
            )

        return groups

    @staticmethod
    def _covered_terms(
        chunk: SourceChunk,
        query_terms: set[str],
    ) -> set[str]:
        """Return query terms present in one chunk."""

        return set(extract_literal_terms(chunk.content)) & query_terms
