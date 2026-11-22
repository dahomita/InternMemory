"""Integration tests for grounded read-only agent tools."""

import json
from pathlib import Path

from internmemory.agent_models import (
    AgentStatus,
    FinalAnswerAction,
    ModelResponse,
    ToolCallAction,
)
from internmemory.agent_runner import AgentRunner
from internmemory.agent_tools import create_search_evidence_tool
from internmemory.chunk_repository import ChunkRepository
from internmemory.database import initialize_database
from internmemory.ingestion_service import IngestionService
from internmemory.model_provider import ScriptedModelProvider
from internmemory.retrieval_service import RetrievalService
from internmemory.source_repository import SourceRepository
from internmemory.tool_registry import ToolRegistry


def create_registry(tmp_path: Path) -> ToolRegistry:
    source_root = tmp_path / "documents"
    source_root.mkdir()
    (source_root / "batching.md").write_text(
        "Larger batches can improve throughput, but waiting for a batch "
        "to fill can increase tail latency.",
        encoding="utf-8",
    )
    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    chunk_repository = ChunkRepository(database_path)
    IngestionService(
        SourceRepository(database_path),
        chunk_repository,
        allowed_root=source_root,
    ).ingest_file("batching.md")
    return ToolRegistry(
        [
            create_search_evidence_tool(
                RetrievalService(chunk_repository)
            )
        ]
    )


def test_search_evidence_tool_returns_cited_grounded_passage(
    tmp_path: Path,
) -> None:
    registry = create_registry(tmp_path)

    observation = registry.execute(
        ToolCallAction(
            call_id="call-1",
            tool_name="search_evidence",
            arguments={"query": "batch throughput tail latency"},
        )
    )

    payload = json.loads(observation.content)
    assert payload["ok"] is True
    assert payload["result"]["abstained"] is False
    evidence = payload["result"]["evidence"][0]
    assert evidence["source"]["locator"] == "batching.md"
    assert evidence["source"]["content_hash"]
    assert "tail latency" in evidence["chunks"][0]["content"]
    assert evidence["query_term_coverage"] == 1.0


def test_search_evidence_tool_explicitly_abstains_without_support(
    tmp_path: Path,
) -> None:
    registry = create_registry(tmp_path)

    observation = registry.execute(
        ToolCallAction(
            call_id="call-1",
            tool_name="search_evidence",
            arguments={"query": "kubernetes pod autoscaling"},
        )
    )

    payload = json.loads(observation.content)
    assert payload["result"] == {
        "abstained": True,
        "evidence": [],
    }


def test_scripted_agent_receives_search_evidence_observation(
    tmp_path: Path,
) -> None:
    registry = create_registry(tmp_path)
    provider = ScriptedModelProvider(
        [
            ModelResponse(
                action=ToolCallAction(
                    call_id="call-1",
                    tool_name="search_evidence",
                    arguments={"query": "tail latency"},
                )
            ),
            ModelResponse(
                action=FinalAnswerAction(
                    content="Batch waiting can increase tail latency."
                )
            ),
        ]
    )

    state = AgentRunner(provider, registry).run(
        "What did I learn about batching?"
    )

    assert state.status is AgentStatus.COMPLETED
    tool_observation = provider.requests[1].messages[-1]
    assert tool_observation.tool_call_id == "call-1"
    assert "batching.md" in tool_observation.content
    assert provider.requests[1].tools == ()


def test_scripted_agent_receives_explicit_abstention(
    tmp_path: Path,
) -> None:
    registry = create_registry(tmp_path)
    provider = ScriptedModelProvider(
        [
            ModelResponse(
                action=ToolCallAction(
                    call_id="call-1",
                    tool_name="search_evidence",
                    arguments={
                        "query": "kubernetes pod autoscaling"
                    },
                )
            ),
            ModelResponse(
                action=FinalAnswerAction(
                    content="I do not have supporting evidence."
                )
            ),
        ]
    )

    state = AgentRunner(provider, registry).run(
        "What did I record about Kubernetes autoscaling?"
    )

    observation = json.loads(
        provider.requests[1].messages[-1].content
    )
    assert observation["result"]["abstained"] is True
    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "I do not have supporting evidence."


def test_scripted_agent_can_correct_invalid_tool_arguments(
    tmp_path: Path,
) -> None:
    registry = create_registry(tmp_path)
    provider = ScriptedModelProvider(
        [
            ModelResponse(
                action=ToolCallAction(
                    call_id="call-1",
                    tool_name="search_evidence",
                    arguments={
                        "query": "tail latency",
                        "limit": "five",
                    },
                )
            ),
            ModelResponse(
                action=ToolCallAction(
                    call_id="call-2",
                    tool_name="search_evidence",
                    arguments={
                        "query": "tail latency",
                        "limit": 5,
                    },
                )
            ),
            ModelResponse(
                action=FinalAnswerAction(
                    content="Batch waiting can increase tail latency."
                )
            ),
        ]
    )

    state = AgentRunner(provider, registry).run(
        "What did I learn about tail latency?"
    )

    invalid_observation = json.loads(
        provider.requests[1].messages[-1].content
    )
    corrected_observation = json.loads(
        provider.requests[2].messages[-1].content
    )
    assert invalid_observation["error"]["type"] == "invalid_arguments"
    assert corrected_observation["ok"] is True
    assert provider.requests[1].tools
    assert provider.requests[2].tools == ()
    assert state.status is AgentStatus.COMPLETED
    assert state.step_count == 3
