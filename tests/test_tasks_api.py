"""Integration tests for task HTTP endpoints."""

from pathlib import Path

from fastapi.testclient import TestClient

from internmemory.main import create_app


def test_task_lifecycle_and_unfinished_filter(tmp_path: Path) -> None:
    """Completed tasks are hidden from the default task list."""

    app = create_app(tmp_path / "internmemory.db")

    with TestClient(app) as client:
        created_response = client.post(
            "/tasks",
            json={
                "description": "Benchmark retrieval approaches",
                "due_date": "2026-11-22",
            },
        )
        task = created_response.json()
        blocked_response = client.patch(
            f"/tasks/{task['id']}/status",
            json={"status": "blocked"},
        )
        completed_response = client.patch(
            f"/tasks/{task['id']}/status",
            json={"status": "completed"},
        )
        unfinished_response = client.get("/tasks")
        all_tasks_response = client.get(
            "/tasks",
            params={"unfinished_only": "false"},
        )

    assert created_response.status_code == 201
    assert task["status"] == "open"
    assert blocked_response.json()["status"] == "blocked"
    assert completed_response.json()["status"] == "completed"
    assert completed_response.json()["completed_at"] is not None
    assert unfinished_response.json() == []
    assert len(all_tasks_response.json()) == 1


def test_task_input_rejects_unknown_status_and_project(
    tmp_path: Path,
) -> None:
    """API and database boundaries reject invalid task relationships."""

    app = create_app(tmp_path / "internmemory.db")

    with TestClient(app) as client:
        invalid_status_response = client.post(
            "/tasks",
            json={
                "description": "Invalid state",
                "status": "waiting_forever",
            },
        )
        invalid_project_response = client.post(
            "/tasks",
            json={
                "description": "Unknown project",
                "project_id": 999,
            },
        )

    assert invalid_status_response.status_code == 422
    assert invalid_project_response.status_code == 409


def test_updating_unknown_task_returns_not_found(tmp_path: Path) -> None:
    """Status updates distinguish a missing task."""

    app = create_app(tmp_path / "internmemory.db")

    with TestClient(app) as client:
        response = client.patch(
            "/tasks/999/status",
            json={"status": "completed"},
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Task not found"}
