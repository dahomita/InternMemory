"""Integration tests for project HTTP endpoints."""

from pathlib import Path

from fastapi.testclient import TestClient

from internmemory.main import create_app


def test_create_list_and_reference_project(tmp_path: Path) -> None:
    """A created project can organize a new engineering note."""

    app = create_app(tmp_path / "internmemory.db")

    with TestClient(app) as client:
        project_response = client.post(
            "/projects",
            json={
                "name": " InternMemory ",
                "description": "Personal engineering memory",
            },
        )
        project = project_response.json()
        note_response = client.post(
            "/notes",
            json={
                "project_id": project["id"],
                "content": "Implemented project APIs.",
                "work_date": "2026-11-21",
            },
        )
        list_response = client.get("/projects")

    assert project_response.status_code == 201
    assert project["name"] == "InternMemory"
    assert note_response.status_code == 201
    assert note_response.json()["project_id"] == project["id"]
    assert list_response.status_code == 200
    assert list_response.json() == [project]


def test_duplicate_project_name_returns_conflict(tmp_path: Path) -> None:
    """Case-only duplicate names produce a visible conflict."""

    app = create_app(tmp_path / "internmemory.db")

    with TestClient(app) as client:
        client.post("/projects", json={"name": "InternMemory"})
        response = client.post(
            "/projects",
            json={"name": "internmemory"},
        )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "A project with this name already exists"
    }
