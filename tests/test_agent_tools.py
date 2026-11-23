"""Integration tests for grounded read-only agent tools."""

import json
from datetime import date, datetime, timezone
from pathlib import Path

from internmemory.agent_models import (
    AgentStatus,
    FinalAnswerAction,
    ModelResponse,
    ToolCallAction,
)
from internmemory.agent_runner import AgentRunner

from internmemory.agent_tools import (
    create_get_daily_summary_tool,
    create_save_note_tool,
    create_search_evidence_tool,
    create_search_notes_tool,
)
from internmemory.chunk_repository import ChunkRepository
from internmemory.database import initialize_database
from internmemory.ingestion_service import IngestionService
from internmemory.model_provider import ScriptedModelProvider
from internmemory.note_repository import NoteRepository
from internmemory.recap_service import RecapService
from internmemory.retrieval_service import RetrievalService
from internmemory.source_repository import SourceRepository
from internmemory.task_repository import TaskRepository
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


def create_notes_registry(tmp_path: Path) -> ToolRegistry:
    database_path = tmp_path / "notes.db"
    initialize_database(database_path)
    repository = NoteRepository(database_path)
    repository.create_note(
        title="Batching follow-up",
        content=(
            "Compared two batching strategies and planned a tail "
            "latency benchmark for tomorrow."
        ),
        work_date=date(2026, 11, 21),
    )
    return ToolRegistry([create_search_notes_tool(repository)])


def test_search_notes_tool_returns_matching_personal_notes(
    tmp_path: Path,
) -> None:
    registry = create_notes_registry(tmp_path)

    observation = registry.execute(
        ToolCallAction(
            call_id="call-1",
            tool_name="search_notes",
            arguments={"query": "tail latency benchmark"},
        )
    )

    payload = json.loads(observation.content)
    assert payload["ok"] is True
    assert payload["result"]["abstained"] is False
    note = payload["result"]["notes"][0]
    assert note["title"] == "Batching follow-up"
    assert note["work_date"] == "2026-11-21"
    assert "tail latency" in note["content"]


def test_search_notes_tool_abstains_without_matches(
    tmp_path: Path,
) -> None:
    registry = create_notes_registry(tmp_path)

    observation = registry.execute(
        ToolCallAction(
            call_id="call-1",
            tool_name="search_notes",
            arguments={"query": "kubernetes autoscaling"},
        )
    )

    payload = json.loads(observation.content)
    assert payload["result"] == {
        "abstained": True,
        "notes": [],
    }


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


def create_recap_registry(tmp_path: Path) -> ToolRegistry:
    database_path = tmp_path / "recap.db"
    initialize_database(database_path)
    note_repository = NoteRepository(database_path)
    task_repository = TaskRepository(database_path)
    note_repository.create_note(
        title="Yesterday batching",
        content="Planned a throughput versus tail latency benchmark.",
        work_date=date(2026, 11, 21),
    )
    task_repository.create_task(
        description="Run the batching benchmark",
    )
    service = RecapService(
        note_repository,
        task_repository,
        timezone_name="UTC",
        now_provider=lambda: datetime(
            2026,
            11,
            22,
            13,
            0,
            tzinfo=timezone.utc,
        ),
    )
    return ToolRegistry([create_get_daily_summary_tool(service)])


def test_get_daily_summary_tool_returns_notes_and_tasks(
    tmp_path: Path,
) -> None:
    registry = create_recap_registry(tmp_path)

    observation = registry.execute(
        ToolCallAction(
            call_id="call-1",
            tool_name="get_daily_summary",
            arguments={},
        )
    )

    payload = json.loads(observation.content)
    assert payload["ok"] is True
    assert payload["result"]["work_date"] == "2026-11-21"
    assert payload["result"]["note_count"] == 1
    assert payload["result"]["unfinished_task_count"] == 1
    assert (
        payload["result"]["notes"][0]["title"] == "Yesterday batching"
    )
    assert (
        payload["result"]["unfinished_tasks"][0]["description"]
        == "Run the batching benchmark"
    )


def test_save_note_requires_confirmation_before_write(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "save.db"
    initialize_database(database_path)
    repository = NoteRepository(database_path)
    registry = ToolRegistry([create_save_note_tool(repository)])
    draft = {
        "title": "End of day",
        "content": "Finished the batching notes.",
        "work_date": "2026-11-22",
    }

    preview = registry.execute(
        ToolCallAction(
            call_id="call-1",
            tool_name="save_note",
            arguments=draft,
        )
    )
    preview_payload = json.loads(preview.content)
    assert preview_payload["result"]["written"] is False
    assert preview_payload["result"]["requires_confirmation"] is True
    assert repository.search_notes("batching notes") == []

    saved = registry.execute(
        ToolCallAction(
            call_id="call-2",
            tool_name="save_note",
            arguments={**draft, "confirmed": True},
        )
    )
    saved_payload = json.loads(saved.content)
    assert saved_payload["result"]["written"] is True
    assert saved_payload["result"]["note"]["id"] == 1
    assert len(repository.search_notes("batching notes")) == 1


def test_scripted_agent_keeps_tools_after_save_preview(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "save-agent.db"
    initialize_database(database_path)
    repository = NoteRepository(database_path)
    registry = ToolRegistry([create_save_note_tool(repository)])
    provider = ScriptedModelProvider(
        [
            ModelResponse(
                action=ToolCallAction(
                    call_id="call-1",
                    tool_name="save_note",
                    arguments={
                        "title": "Draft",
                        "content": "Needs confirmation.",
                        "work_date": "2026-11-22",
                    },
                )
            ),
            ModelResponse(
                action=ToolCallAction(
                    call_id="call-2",
                    tool_name="save_note",
                    arguments={
                        "title": "Draft",
                        "content": "Needs confirmation.",
                        "work_date": "2026-11-22",
                        "confirmed": True,
                    },
                )
            ),
            ModelResponse(
                action=FinalAnswerAction(
                    content="Saved note id 1 after confirmation."
                )
            ),
        ]
    )

    state = AgentRunner(provider, registry).run(
        "Save a note about confirmation."
    )

    assert state.status is AgentStatus.COMPLETED
    assert provider.requests[1].tools
    assert provider.requests[2].tools == ()
    assert repository.search_notes("confirmation")[0].note.id == 1
