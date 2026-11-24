"""Tests for the minimal product chat UI and agent status probe."""

from pathlib import Path

from fastapi.testclient import TestClient

from internmemory.agent_models import FinalAnswerAction, ModelResponse
from internmemory.main import create_app
from internmemory.model_provider import ScriptedModelProvider


def test_ui_home_serves_chat_page(tmp_path: Path) -> None:
    app = create_app(tmp_path / "internmemory.db", enable_groq=False)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "InternMemory" in response.text
    assert "/static/app.js" in response.text


def test_static_assets_are_served(tmp_path: Path) -> None:
    app = create_app(tmp_path / "internmemory.db", enable_groq=False)

    with TestClient(app) as client:
        css = client.get("/static/app.css")
        js = client.get("/static/app.js")

    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]
    assert js.status_code == 200
    assert "agent/confirm-note" in js.text
    assert "agent/confirm-ingest" in js.text
    assert "/notes/" in js.text


def test_agent_status_reports_unavailable_without_provider(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path / "internmemory.db", enable_groq=False)

    with TestClient(app) as client:
        response = client.get("/agent/status")

    payload = response.json()
    assert response.status_code == 200
    assert payload["available"] is False
    assert "save_note" in payload["tools"]
    assert payload["detail"]


def test_agent_status_reports_available_with_provider(
    tmp_path: Path,
) -> None:
    provider = ScriptedModelProvider(
        [ModelResponse(action=FinalAnswerAction(content="unused"))]
    )
    app = create_app(
        tmp_path / "internmemory.db",
        model_provider=provider,
        enable_groq=False,
    )

    with TestClient(app) as client:
        response = client.get("/agent/status")

    payload = response.json()
    assert response.status_code == 200
    assert payload["available"] is True
    assert payload["detail"] is None
