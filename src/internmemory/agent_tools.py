"""Read-only agent tools backed by InternMemory application services."""

from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from internmemory.agent_models import JsonValue
from internmemory.retrieval_service import RetrievalService
from internmemory.tool_registry import StructuredTool


class SearchEvidenceInput(BaseModel):
    """Strict, bounded arguments for grounded evidence retrieval."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "Concise literal keywords likely to appear in the source. "
            "Preserve every essential concept from the user's question, "
            "but avoid conversational filler because retrieval is lexical."
        ),
    )
    limit: int = Field(default=5, ge=1, le=10)


def create_search_evidence_tool(
    service: RetrievalService,
) -> StructuredTool[SearchEvidenceInput]:
    """Adapt grounded keyword retrieval into a read-only agent tool."""

    def search(request: SearchEvidenceInput) -> JsonValue:
        groups = service.search(
            request.query,
            limit=request.limit,
            rerank_by_coverage=True,
            minimum_coverage=0.5,
        )
        result = {
            "abstained": not groups,
            "evidence": [
                {
                    "source": {
                        "id": group.source.id,
                        "type": group.source.source_type.value,
                        "locator": group.source.locator,
                        "title": group.source.title,
                        "content_hash": group.source.content_hash,
                    },
                    "chunks": [
                        {
                            "id": chunk.id,
                            "ordinal": chunk.ordinal,
                            "start_offset": chunk.start_offset,
                            "end_offset": chunk.end_offset,
                            "content": chunk.content,
                            "content_hash": chunk.content_hash,
                        }
                        for chunk in group.chunks
                    ],
                    "anchor_score": group.anchor_score,
                    "query_term_coverage": (
                        group.query_term_coverage
                    ),
                }
                for group in groups
            ],
        }
        return cast(JsonValue, result)

    return StructuredTool(
        name="search_evidence",
        description=(
            "Search stored engineering documents using literal keywords. "
            "Use concise source terms likely to appear in notes, not a full "
            "natural-language paraphrase. If a search abstains, reformulate "
            "once with short systems nouns (queue, backpressure, latency, "
            "retry, timeout, cache) instead of restating the question. "
            "Returns cited passages or abstains when lexical support is weak."
        ),
        input_model=SearchEvidenceInput,
        handler=search,
    )
