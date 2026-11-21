"""Tests for project persistence operations."""

import sqlite3
from pathlib import Path

import pytest

from internmemory.database import initialize_database
from internmemory.project_repository import ProjectRepository


def test_create_and_list_projects_in_name_order(tmp_path: Path) -> None:
    """Projects are persisted and listed case-insensitively by name."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    repository = ProjectRepository(database_path)
    repository.create_project(name="Search")
    repository.create_project(name="agent harness")

    projects = repository.list_projects()

    assert [project.name for project in projects] == [
        "agent harness",
        "Search",
    ]


def test_project_names_are_unique_ignoring_case(tmp_path: Path) -> None:
    """The repository surfaces the database uniqueness constraint."""

    database_path = tmp_path / "internmemory.db"
    initialize_database(database_path)
    repository = ProjectRepository(database_path)
    repository.create_project(name="InternMemory")

    with pytest.raises(sqlite3.IntegrityError):
        repository.create_project(name="internmemory")
