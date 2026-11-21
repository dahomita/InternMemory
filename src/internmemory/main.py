"""FastAPI application entry point."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI

from internmemory.api import router
from internmemory.database import initialize_database
from internmemory.note_repository import NoteRepository
from internmemory.project_repository import ProjectRepository
from internmemory.recap_service import RecapService
from internmemory.task_repository import TaskRepository


DEFAULT_DATABASE_PATH = Path("internmemory.db")
DEFAULT_TIMEZONE = "America/New_York"


def create_app(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    timezone_name: str = DEFAULT_TIMEZONE,
    now_provider: Callable[[], datetime] | None = None,
) -> FastAPI:
    """Build an application configured for one database."""

    note_repository = NoteRepository(database_path)
    project_repository = ProjectRepository(database_path)
    task_repository = TaskRepository(database_path)
    recap_service = RecapService(
        note_repository,
        task_repository,
        timezone_name=timezone_name,
        now_provider=now_provider,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        initialize_database(database_path)
        yield

    application = FastAPI(
        title="InternMemory",
        description="A personal engineering knowledge engine.",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.note_repository = note_repository
    application.state.project_repository = project_repository
    application.state.task_repository = task_repository
    application.state.recap_service = recap_service
    application.include_router(router)
    return application


app = create_app()
