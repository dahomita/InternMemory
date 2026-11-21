"""FastAPI application entry point."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI

from internmemory.api import router
from internmemory.chunk_repository import ChunkRepository
from internmemory.database import initialize_database
from internmemory.ingestion_service import IngestionService
from internmemory.note_repository import NoteRepository
from internmemory.project_repository import ProjectRepository
from internmemory.recap_service import RecapService
from internmemory.retrieval_service import RetrievalService
from internmemory.source_repository import SourceRepository
from internmemory.task_repository import TaskRepository


DEFAULT_DATABASE_PATH = Path("internmemory.db")
DEFAULT_INGESTION_ROOT = Path("documents")
DEFAULT_TIMEZONE = "America/New_York"


def create_app(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    timezone_name: str = DEFAULT_TIMEZONE,
    now_provider: Callable[[], datetime] | None = None,
    ingestion_root: str | Path = DEFAULT_INGESTION_ROOT,
    max_source_bytes: int = 1_000_000,
    max_chunk_chars: int = 1_200,
) -> FastAPI:
    """Build an application configured for one database."""

    note_repository = NoteRepository(database_path)
    project_repository = ProjectRepository(database_path)
    source_repository = SourceRepository(database_path)
    chunk_repository = ChunkRepository(database_path)
    task_repository = TaskRepository(database_path)
    ingestion_service = IngestionService(
        source_repository,
        chunk_repository,
        allowed_root=ingestion_root,
        max_bytes=max_source_bytes,
        max_chunk_chars=max_chunk_chars,
    )
    recap_service = RecapService(
        note_repository,
        task_repository,
        timezone_name=timezone_name,
        now_provider=now_provider,
    )
    retrieval_service = RetrievalService(chunk_repository)

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
    application.state.chunk_repository = chunk_repository
    application.state.note_repository = note_repository
    application.state.project_repository = project_repository
    application.state.ingestion_service = ingestion_service
    application.state.task_repository = task_repository
    application.state.recap_service = recap_service
    application.state.retrieval_service = retrieval_service
    application.include_router(router)
    return application


app = create_app()
