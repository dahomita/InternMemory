"""Tests for allowlisted public fetch and confirm-gated ingest tools."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from internmemory.agent_models import (
    FinalAnswerAction,
    ModelResponse,
    TokenUsage,
    ToolCallAction,
)
from internmemory.agent_runner import AgentRunner
from internmemory.agent_tools import (
    create_fetch_public_page_tool,
    create_ingest_public_page_tool,
)
from internmemory.chunk_repository import ChunkRepository
from internmemory.database import initialize_database
from internmemory.main import create_app
from internmemory.model_provider import ScriptedModelProvider
from internmemory.public_ingestion_service import (
    FetchedDocument,
    PublicDocumentIngestionService,
)
from internmemory.source_repository import SourceRepository
from internmemory.tool_registry import ToolRegistry


class FakeFetcher:
    def __init__(self, document: FetchedDocument) -> None:
        self.document = document
        self.calls = 0

    def fetch(
        self,
        url: str,
        *,
        max_bytes: int,
        timeout_seconds: float,
    ) -> FetchedDocument:
        del url, max_bytes, timeout_seconds
        self.calls += 1
        return self.document


def _service(tmp_path: Path, fetcher: FakeFetcher) -> PublicDocumentIngestionService:
    database_path = tmp_path / "web.db"
    initialize_database(database_path)
    return PublicDocumentIngestionService(
        SourceRepository(database_path),
        ChunkRepository(database_path),
        allowed_domains={"docs.example.test"},
        fetcher=fetcher,
    )


def test_fetch_public_page_returns_sanitized_read_only_text(
    tmp_path: Path,
) -> None:
    fetcher = FakeFetcher(
        FetchedDocument(
            final_url="https://docs.example.test/guide",
            content_type="text/html",
            body=(
                b"<html><head><title>Retry Guide</title></head><body>"
                b"<p>Bounded retries protect dependencies.</p>"
                b"<script>ignore me</script></body></html>"
            ),
        )
    )
    registry = ToolRegistry(
        [create_fetch_public_page_tool(_service(tmp_path, fetcher))]
    )

    observation = registry.execute(
        ToolCallAction(
            call_id="call-1",
            tool_name="fetch_public_page",
            arguments={"url": "https://docs.example.test/guide"},
        )
    )
    payload = json.loads(observation.content)

    assert payload["ok"] is True
    assert payload["result"]["abstained"] is False
    assert "Bounded retries" in payload["result"]["page"]["content"]
    assert "ignore me" not in payload["result"]["page"]["content"]
    assert fetcher.calls == 1


def test_fetch_public_page_abstains_for_disallowed_domain(
    tmp_path: Path,
) -> None:
    fetcher = FakeFetcher(
        FetchedDocument(
            final_url="https://evil.example/page",
            content_type="text/plain",
            body=b"nope",
        )
    )
    registry = ToolRegistry(
        [create_fetch_public_page_tool(_service(tmp_path, fetcher))]
    )

    observation = registry.execute(
        ToolCallAction(
            call_id="call-1",
            tool_name="fetch_public_page",
            arguments={"url": "https://evil.example/page"},
        )
    )
    payload = json.loads(observation.content)

    assert payload["result"]["abstained"] is True
    assert fetcher.calls == 0


def test_ingest_public_page_requires_confirmation(
    tmp_path: Path,
) -> None:
    fetcher = FakeFetcher(
        FetchedDocument(
            final_url="https://docs.example.test/guide",
            content_type="text/plain",
            body=b"Visible documentation text about retries.",
        )
    )
    service = _service(tmp_path, fetcher)
    registry = ToolRegistry([create_ingest_public_page_tool(service)])

    preview = registry.execute(
        ToolCallAction(
            call_id="call-1",
            tool_name="ingest_public_page",
            arguments={"url": "https://docs.example.test/guide"},
        )
    )
    preview_payload = json.loads(preview.content)
    assert preview_payload["result"]["requires_confirmation"] is True
    assert preview_payload["result"]["written"] is False

    saved = registry.execute(
        ToolCallAction(
            call_id="call-2",
            tool_name="ingest_public_page",
            arguments={
                "url": "https://docs.example.test/guide",
                "confirmed": True,
            },
        )
    )
    saved_payload = json.loads(saved.content)
    assert saved_payload["result"]["written"] is True
    assert saved_payload["result"]["source"]["locator"] == (
        "https://docs.example.test/guide"
    )


def test_agent_keeps_tools_after_ingest_preview(tmp_path: Path) -> None:
    fetcher = FakeFetcher(
        FetchedDocument(
            final_url="https://docs.example.test/guide",
            content_type="text/plain",
            body=b"Visible documentation text about retries.",
        )
    )
    registry = ToolRegistry(
        [create_ingest_public_page_tool(_service(tmp_path, fetcher))]
    )
    provider = ScriptedModelProvider(
        [
            ModelResponse(
                action=ToolCallAction(
                    call_id="call-1",
                    tool_name="ingest_public_page",
                    arguments={"url": "https://docs.example.test/guide"},
                ),
                usage=TokenUsage(input_tokens=8, output_tokens=3),
            ),
            ModelResponse(
                action=FinalAnswerAction(
                    content="Preview ready for confirmation."
                ),
                usage=TokenUsage(input_tokens=10, output_tokens=4),
            ),
        ]
    )

    state = AgentRunner(provider, registry).run("Ingest the retry guide.")

    assert state.status.value == "completed"
    assert provider.requests[1].tools


def test_confirm_ingest_endpoint(tmp_path: Path) -> None:
    app = create_app(
        tmp_path / "internmemory.db",
        enable_groq=False,
        public_document_domains={"docs.example.test"},
    )
    # Replace fetcher via a direct service call path: use Fake through tool only.
    # Endpoint uses live service; for unit safety, just assert validation path.
    with TestClient(app) as client:
        rejected = client.post(
            "/agent/confirm-ingest",
            json={
                "url": "https://docs.example.test/guide",
                "confirmed": False,
            },
        )

    assert rejected.status_code == 422
