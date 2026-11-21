"""Tests for task persistence and lifecycle behavior."""

from datetime import date
from pathlib import Path

from internmemory.database import initialize_database
from internmemory.models import TaskStatus
from internmemory.task_repository import TaskRepository


def test_completed_tasks_leave_unfinished_list(tmp_path: Path) -> None:
    """Completing a task records completion and removes it from defaults."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    repository = TaskRepository(database_path)
    task = repository.create_task(
        description="Benchmark retrieval approaches",
        due_date=date(2026, 11, 22),
    )

    completed = repository.update_status(task.id, TaskStatus.COMPLETED)

    assert completed is not None
    assert completed.status is TaskStatus.COMPLETED
    assert completed.completed_at is not None
    assert repository.list_tasks() == []
    assert repository.list_tasks(unfinished_only=False) == [completed]


def test_reopening_task_clears_completion_timestamp(tmp_path: Path) -> None:
    """A reopened task is no longer marked as completed."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    repository = TaskRepository(database_path)
    task = repository.create_task(
        description="Review benchmark results",
        status=TaskStatus.COMPLETED,
    )

    reopened = repository.update_status(task.id, TaskStatus.BLOCKED)

    assert reopened is not None
    assert reopened.status is TaskStatus.BLOCKED
    assert reopened.completed_at is None
