"""Integration tests for deterministic morning recaps."""

from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from internmemory.main import create_app


def test_morning_recap_uses_configured_timezone_and_unfinished_tasks(
    tmp_path: Path,
) -> None:
    """Automatic yesterday is local, while explicit dates remain available."""

    fixed_now = datetime(2026, 11, 22, 2, 0, tzinfo=timezone.utc)
    app = create_app(
        tmp_path / "internmemory.db",
        timezone_name="America/New_York",
        now_provider=lambda: fixed_now,
    )

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
                "content": "Reviewed benchmark results.",
                "work_date": "2026-11-21",
            },
        )
        client.post(
            "/tasks",
            json={"description": "Compare BM25 configurations"},
        )
        client.post(
            "/tasks",
            json={
                "description": "Create the FTS index",
                "status": "completed",
            },
        )

        automatic_response = client.get("/recap/morning")
        explicit_response = client.get(
            "/recap/morning",
            params={"work_date": "2026-11-21"},
        )

    automatic = automatic_response.json()
    explicit = explicit_response.json()

    assert automatic_response.status_code == 200
    assert automatic["work_date"] == "2026-11-20"
    assert [note["title"] for note in automatic["notes"]] == [
        "Friday work"
    ]
    assert [
        task["description"] for task in automatic["unfinished_tasks"]
    ] == ["Compare BM25 configurations"]
    assert explicit["work_date"] == "2026-11-21"
    assert [note["title"] for note in explicit["notes"]] == [
        "Saturday work"
    ]
