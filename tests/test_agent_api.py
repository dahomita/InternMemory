"""Integration tests for the product agent HTTP entrypoint."""

import json
from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient

from internmemory.agent_models import (
    FinalAnswerAction,
    ModelResponse,
    TokenUsage,
    ToolCallAction,
)
from internmemory.agent_service import (
    AgentService,
    build_default_tool_registry,
)
from internmemory.chunk_repository import ChunkRepository
from internmemory.database import initialize_database
from internmemory.main import create_app
from internmemory.model_provider import ScriptedModelProvider
from internmemory.note_repository import NoteRepository
from internmemory.public_ingestion_service import PublicDocumentIngestionService
from internmemory.recap_service import RecapService
from internmemory.retrieval_service import RetrievalService
from internmemory.source_repository import SourceRepository
from internmemory.task_repository import TaskRepository


def _tool_call(call_id: str, tool_name: str, **arguments: object) -> ModelResponse:
    return ModelResponse(
        action=ToolCallAction(
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
        ),
        usage=TokenUsage(input_tokens=10, output_tokens=4),
    )


def test_agent_run_returns_503_without_model_provider(
    tmp_path: Path,
) -> None:
    app = create_app(
        tmp_path / "internmemory.db",
        enable_groq=False,
    )

    with TestClient(app) as client:
        status_response = client.get("/agent/status")
        response = client.post(
            "/agent/run",
            json={"message": "What did I learn about batching?"},
        )

    assert status_response.status_code == 200
    assert status_response.json()["available"] is False
    assert response.status_code == 503
    assert "model provider" in response.json()["detail"].lower()


def test_agent_run_uses_full_tool_registry_with_scripted_provider(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    NoteRepository(database_path).create_note(
        title="Batching",
        content="Batching reduced p95 latency for the queue.",
        work_date=date(2026, 11, 21),
    )
    provider = ScriptedModelProvider(
        [
            _tool_call(
                "call-1",
                "search_notes",
                query="batching latency",
                limit=5,
            ),
            ModelResponse(
                action=FinalAnswerAction(
                    content=(
                        "Your note says batching reduced p95 latency."
                    ),
                    abstained=False,
                ),
                usage=TokenUsage(input_tokens=20, output_tokens=8),
            ),
        ]
    )
    app = create_app(
        database_path,
        model_provider=provider,
        enable_groq=False,
    )

    with TestClient(app) as client:
        response = client.post(
            "/agent/run",
            json={
                "message": "What did I note about batching?",
                "include_trace": True,
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "completed"
    assert payload["answer"] == (
        "Your note says batching reduced p95 latency."
    )
    assert payload["abstained"] is False
    assert payload["tools"] == [
        "search_notes",
        "search_evidence",
        "get_daily_summary",
        "save_note",
        "fetch_public_page",
        "ingest_public_page",
    ]
    assert payload["step_count"] == 2
    assert payload["usage"]["input_tokens"] == 30
    assert payload["trace"] is not None
    assert payload["trace"][0]["kind"] == "tool_call"
    assert payload["trace"][0]["tool_call"]["tool_name"] == (
        "search_notes"
    )
    observation = json.loads(payload["trace"][0]["observation"])
    assert observation["ok"] is True
    assert observation["result"]["abstained"] is False
    assert payload["trace"][1]["kind"] == "final_answer"
    assert payload["pending_confirmation"] is None


def test_agent_run_surfaces_pending_note_confirmation(
    tmp_path: Path,
) -> None:
    provider = ScriptedModelProvider(
        [
            _tool_call(
                "call-1",
                "save_note",
                title="Draft",
                content="Needs confirmation before write.",
                work_date="2026-11-22",
            ),
            ModelResponse(
                action=FinalAnswerAction(
                    content=(
                        "Preview ready. Confirm to save the note."
                    ),
                    abstained=False,
                ),
                usage=TokenUsage(input_tokens=12, output_tokens=6),
            ),
        ]
    )
    app = create_app(
        tmp_path / "internmemory.db",
        model_provider=provider,
        enable_groq=False,
    )

    with TestClient(app) as client:
        response = client.post(
            "/agent/run",
            json={"message": "Save a note about confirmation."},
        )
        payload = response.json()
        listed = client.get("/notes")

    assert response.status_code == 200
    assert payload["pending_confirmation"] == {
        "kind": "note",
        "title": "Draft",
        "content": "Needs confirmation before write.",
        "work_date": "2026-11-22",
        "project_id": None,
        "url": None,
        "message": (
            "Preview only. Call save_note again with the same "
            "fields and confirmed=true to persist."
        ),
    }
    assert listed.json() == []


def test_confirm_note_persists_after_explicit_approval(
    tmp_path: Path,
) -> None:
    app = create_app(
        tmp_path / "internmemory.db",
        enable_groq=False,
    )
    draft = {
        "title": "End of day",
        "content": "Finished the batching notes.",
        "work_date": "2026-11-22",
        "project_id": None,
        "confirmed": True,
    }

    with TestClient(app) as client:
        rejected = client.post(
            "/agent/confirm-note",
            json={**draft, "confirmed": False},
        )
        created = client.post("/agent/confirm-note", json=draft)
        listed = client.get("/notes")

    assert rejected.status_code == 422
    assert created.status_code == 201
    assert created.json()["written"] is True
    assert created.json()["note"]["content"] == draft["content"]
    assert len(listed.json()) == 1


def test_agent_run_rejects_blank_message(tmp_path: Path) -> None:
    provider = ScriptedModelProvider(
        [
            ModelResponse(
                action=FinalAnswerAction(content="unused"),
                usage=TokenUsage(),
            )
        ]
    )
    app = create_app(
        tmp_path / "internmemory.db",
        model_provider=provider,
        enable_groq=False,
    )

    with TestClient(app) as client:
        response = client.post(
            "/agent/run",
            json={"message": "   "},
        )

    assert response.status_code == 422


def test_agent_service_exposes_all_product_tools(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "tools.db"
    initialize_database(database_path)
    notes = NoteRepository(database_path)
    public = PublicDocumentIngestionService(
        SourceRepository(database_path),
        ChunkRepository(database_path),
        allowed_domains={"docs.example.test"},
    )
    service = AgentService(
        build_default_tool_registry(
            notes,
            RetrievalService(ChunkRepository(database_path)),
            RecapService(
                notes,
                TaskRepository(database_path),
                timezone_name="America/New_York",
            ),
            public,
        ),
        notes,
        public_ingestion_service=public,
    )

    assert service.tool_names == (
        "search_notes",
        "search_evidence",
        "get_daily_summary",
        "save_note",
        "fetch_public_page",
        "ingest_public_page",
    )
    assert service.is_available is False
