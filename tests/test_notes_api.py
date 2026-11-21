"""Integration tests for note HTTP endpoints."""

from pathlib import Path

from fastapi.testclient import TestClient

from internmemory.main import create_app


def test_create_then_get_note(tmp_path: Path) -> None:
    """A valid note can be created and retrieved through HTTP."""

    app = create_app(tmp_path / "internmemory.db")
    payload = {
        "title": "Retrieval batching",
        "content": "Compared throughput against p95 latency.",
        "work_date": "2026-11-21",
    }

    with TestClient(app) as client:
        create_response = client.post("/notes", json=payload)
        note_id = create_response.json()["id"]
        get_response = client.get(f"/notes/{note_id}")

    assert create_response.status_code == 201
    assert create_response.json()["content"] == payload["content"]
    assert get_response.status_code == 200
    assert get_response.json() == create_response.json()


def test_create_note_rejects_blank_content(tmp_path: Path) -> None:
    """Invalid input is rejected before reaching the repository."""

    app = create_app(tmp_path / "internmemory.db")

    with TestClient(app) as client:
        response = client.post(
            "/notes",
            json={"content": "   ", "work_date": "2026-11-21"},
        )

    assert response.status_code == 422


def test_list_notes_filters_by_work_date_and_allows_empty_results(
    tmp_path: Path,
) -> None:
    """Collection queries return matching notes or an empty list."""

    app = create_app(tmp_path / "internmemory.db")

    with TestClient(app) as client:
        client.post(
            "/notes",
            json={
                "title": "Friday work",
                "content": "Measured retrieval latency.",
                "work_date": "2026-11-20",
            },
        )
        client.post(
            "/notes",
            json={
                "title": "Saturday work",
                "content": "Reviewed the measurements.",
                "work_date": "2026-11-21",
            },
        )
        filtered_response = client.get(
            "/notes",
            params={"work_date": "2026-11-20"},
        )
        empty_response = client.get(
            "/notes",
            params={"work_date": "2026-11-19"},
        )

    assert filtered_response.status_code == 200
    assert [
        note["title"] for note in filtered_response.json()
    ] == ["Friday work"]
    assert empty_response.status_code == 200
    assert empty_response.json() == []


def test_get_unknown_note_returns_not_found(tmp_path: Path) -> None:
    """A singular missing resource produces HTTP 404."""

    app = create_app(tmp_path / "internmemory.db")

    with TestClient(app) as client:
        response = client.get("/notes/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Note not found"}


def test_patch_preserves_omitted_fields(tmp_path: Path) -> None:
    """PATCH changes only explicitly supplied fields."""

    app = create_app(tmp_path / "internmemory.db")
    payload = {
        "title": "Keep this title",
        "content": "Original content",
        "work_date": "2026-11-21",
    }

    with TestClient(app) as client:
        created = client.post("/notes", json=payload).json()
        response = client.patch(
            f"/notes/{created['id']}",
            json={"content": "Updated content"},
        )

    assert response.status_code == 200
    assert response.json()["title"] == "Keep this title"
    assert response.json()["content"] == "Updated content"


def test_patch_rejects_empty_body_and_null_content(tmp_path: Path) -> None:
    """PATCH requires at least one valid field."""

    app = create_app(tmp_path / "internmemory.db")

    with TestClient(app) as client:
        empty_response = client.patch("/notes/1", json={})
        null_response = client.patch("/notes/1", json={"content": None})

    assert empty_response.status_code == 422
    assert null_response.status_code == 422


def test_delete_note_returns_no_content_then_not_found(
    tmp_path: Path,
) -> None:
    """A deleted note disappears and cannot be deleted twice."""

    app = create_app(tmp_path / "internmemory.db")

    with TestClient(app) as client:
        created = client.post(
            "/notes",
            json={
                "content": "Delete after verification.",
                "work_date": "2026-11-21",
            },
        ).json()
        delete_response = client.delete(f"/notes/{created['id']}")
        get_response = client.get(f"/notes/{created['id']}")
        second_delete_response = client.delete(f"/notes/{created['id']}")

    assert delete_response.status_code == 204
    assert delete_response.content == b""
    assert get_response.status_code == 404
    assert second_delete_response.status_code == 404
