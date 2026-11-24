"""FastAPI application entry point."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from internmemory.agent_service import (
    AgentService,
    build_default_tool_registry,
)
from internmemory.api import router
from internmemory.chunk_repository import ChunkRepository
from internmemory.database import initialize_database
from internmemory.groq_provider import GroqModelProvider
from internmemory.ingestion_service import IngestionService
from internmemory.model_provider import ModelProvider
from internmemory.note_repository import NoteRepository
from internmemory.project_repository import ProjectRepository
from internmemory.public_ingestion_service import (
    PublicDocumentIngestionService,
)
from internmemory.recap_service import RecapService
from internmemory.retrieval_service import RetrievalService
from internmemory.source_repository import SourceRepository
from internmemory.task_repository import TaskRepository


DEFAULT_DATABASE_PATH = Path("internmemory.db")
DEFAULT_INGESTION_ROOT = Path("documents")
DEFAULT_TIMEZONE = "America/New_York"
STATIC_DIRECTORY = Path(__file__).resolve().parent / "static"
DEFAULT_PUBLIC_DOCUMENT_DOMAINS = {
    "docs.python.org",
    "fastapi.tiangolo.com",
    "www.sqlite.org",
}


def _resolve_model_provider(
    model_provider: ModelProvider | None,
    *,
    enable_groq: bool,
) -> ModelProvider | None:
    """Use an injected provider, otherwise optional Groq from the environment."""

    if model_provider is not None:
        return model_provider
    if not enable_groq:
        return None
    try:
        return GroqModelProvider.from_environment()
    except ValueError:
        return None


def create_app(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    timezone_name: str = DEFAULT_TIMEZONE,
    now_provider: Callable[[], datetime] | None = None,
    ingestion_root: str | Path = DEFAULT_INGESTION_ROOT,
    max_source_bytes: int = 1_000_000,
    max_chunk_chars: int = 1_200,
    public_document_domains: set[str] | None = None,
    model_provider: ModelProvider | None = None,
    enable_groq: bool = True,
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
    public_ingestion_service = PublicDocumentIngestionService(
        source_repository,
        chunk_repository,
        allowed_domains=(
            public_document_domains
            if public_document_domains is not None
            else DEFAULT_PUBLIC_DOCUMENT_DOMAINS
        ),
        max_bytes=max_source_bytes,
        max_chunk_chars=max_chunk_chars,
    )
    agent_service = AgentService(
        build_default_tool_registry(
            note_repository,
            retrieval_service,
            recap_service,
        ),
        note_repository,
        _resolve_model_provider(
            model_provider,
            enable_groq=enable_groq,
        ),
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
    application.state.chunk_repository = chunk_repository
    application.state.note_repository = note_repository
    application.state.project_repository = project_repository
    application.state.ingestion_service = ingestion_service
    application.state.task_repository = task_repository
    application.state.recap_service = recap_service
    application.state.retrieval_service = retrieval_service
    application.state.public_ingestion_service = public_ingestion_service
    application.state.agent_service = agent_service
    application.include_router(router)
    application.mount(
        "/static",
        StaticFiles(directory=STATIC_DIRECTORY),
        name="static",
    )

    @application.get("/", include_in_schema=False)
    def ui_home() -> FileResponse:
        """Serve the minimal chat and confirmation UI."""

        return FileResponse(STATIC_DIRECTORY / "index.html")

    return application


app = create_app()
