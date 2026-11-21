"""Tests for service health reporting."""

from pathlib import Path

from fastapi.testclient import TestClient

from internmemory.main import create_app


def test_health_returns_ok(tmp_path: Path) -> None:
    """The service exposes a stable health-check contract."""

    app = create_app(tmp_path / "internmemory.db")
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
