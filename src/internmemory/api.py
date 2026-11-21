"""HTTP routes for the InternMemory API."""

import sqlite3
from datetime import date
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    status,
)

from internmemory.chunk_repository import ChunkRepository
from internmemory.ingestion_service import IngestionService
from internmemory.note_repository import NoteRepository
from internmemory.project_repository import ProjectRepository
from internmemory.recap_service import RecapService
from internmemory.retrieval_service import RetrievalService
from internmemory.task_repository import TaskRepository
from internmemory.schemas import (
    DocumentSearchResultResponse,
    EvidenceGroupResponse,
    HealthResponse,
    MorningRecapResponse,
    NoteCreate,
    NoteResponse,
    NoteSearchResultResponse,
    NoteUpdate,
    ProjectCreate,
    ProjectResponse,
    SourceIngestRequest,
    SourceResponse,
    TaskCreate,
    TaskResponse,
    TaskStatusUpdate,
)


router = APIRouter()


def get_note_repository(request: Request) -> NoteRepository:
    """Return the repository configured for this application."""

    repository: NoteRepository = request.app.state.note_repository
    return repository


NoteRepositoryDependency = Annotated[
    NoteRepository,
    Depends(get_note_repository),
]


def get_project_repository(request: Request) -> ProjectRepository:
    """Return the project repository configured for this application."""

    repository: ProjectRepository = request.app.state.project_repository
    return repository


ProjectRepositoryDependency = Annotated[
    ProjectRepository,
    Depends(get_project_repository),
]


def get_task_repository(request: Request) -> TaskRepository:
    """Return the task repository configured for this application."""

    repository: TaskRepository = request.app.state.task_repository
    return repository


TaskRepositoryDependency = Annotated[
    TaskRepository,
    Depends(get_task_repository),
]


def get_recap_service(request: Request) -> RecapService:
    """Return the recap service configured for this application."""

    service: RecapService = request.app.state.recap_service
    return service


RecapServiceDependency = Annotated[
    RecapService,
    Depends(get_recap_service),
]


def get_ingestion_service(request: Request) -> IngestionService:
    """Return the ingestion service configured for this application."""

    service: IngestionService = request.app.state.ingestion_service
    return service


IngestionServiceDependency = Annotated[
    IngestionService,
    Depends(get_ingestion_service),
]


def get_chunk_repository(request: Request) -> ChunkRepository:
    """Return the chunk repository configured for this application."""

    repository: ChunkRepository = request.app.state.chunk_repository
    return repository


ChunkRepositoryDependency = Annotated[
    ChunkRepository,
    Depends(get_chunk_repository),
]


def get_retrieval_service(request: Request) -> RetrievalService:
    """Return the retrieval service configured for this application."""

    service: RetrievalService = request.app.state.retrieval_service
    return service


RetrievalServiceDependency = Annotated[
    RetrievalService,
    Depends(get_retrieval_service),
]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether the API process can serve requests."""

    return HealthResponse(status="ok")


@router.get("/recap/morning", response_model=MorningRecapResponse)
def get_morning_recap(
    service: RecapServiceDependency,
    work_date: Annotated[date | None, Query()] = None,
) -> MorningRecapResponse:
    """Return notes for a work date and all unfinished tasks."""

    recap = service.build_morning_recap(work_date)
    return MorningRecapResponse.model_validate(recap)


@router.post("/sources/ingest", response_model=SourceResponse)
def ingest_source(
    request: SourceIngestRequest,
    service: IngestionServiceDependency,
) -> SourceResponse:
    """Capture one allowlisted local Markdown or text document."""

    try:
        source = service.ingest_file(request.relative_path)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source file not found",
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error

    return SourceResponse.model_validate(source)


@router.post(
    "/projects",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_project(
    project: ProjectCreate,
    repository: ProjectRepositoryDependency,
) -> ProjectResponse:
    """Validate and persist an engineering project."""

    try:
        created = repository.create_project(
            name=project.name,
            description=project.description,
        )
    except sqlite3.IntegrityError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A project with this name already exists",
        ) from error

    return ProjectResponse.model_validate(created)


@router.get("/projects", response_model=list[ProjectResponse])
def list_projects(
    repository: ProjectRepositoryDependency,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ProjectResponse]:
    """List projects in case-insensitive name order."""

    projects = repository.list_projects(limit=limit, offset=offset)
    return [
        ProjectResponse.model_validate(project)
        for project in projects
    ]


@router.post(
    "/tasks",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_task(
    task: TaskCreate,
    repository: TaskRepositoryDependency,
) -> TaskResponse:
    """Validate and persist an engineering follow-up task."""

    try:
        created = repository.create_task(
            project_id=task.project_id,
            description=task.description,
            status=task.status,
            due_date=task.due_date,
        )
    except sqlite3.IntegrityError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The task conflicts with existing data",
        ) from error

    return TaskResponse.model_validate(created)


@router.get("/tasks", response_model=list[TaskResponse])
def list_tasks(
    repository: TaskRepositoryDependency,
    unfinished_only: bool = True,
    project_id: Annotated[int | None, Query(gt=0)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[TaskResponse]:
    """List unfinished tasks by default, optionally scoped to a project."""

    tasks = repository.list_tasks(
        unfinished_only=unfinished_only,
        project_id=project_id,
        limit=limit,
        offset=offset,
    )
    return [TaskResponse.model_validate(task) for task in tasks]


@router.patch("/tasks/{task_id}/status", response_model=TaskResponse)
def update_task_status(
    update: TaskStatusUpdate,
    repository: TaskRepositoryDependency,
    task_id: Annotated[int, Path(gt=0)],
) -> TaskResponse:
    """Update task status and completion metadata."""

    task = repository.update_status(task_id, update.status)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    return TaskResponse.model_validate(task)


@router.post(
    "/notes",
    response_model=NoteResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_note(
    note: NoteCreate,
    repository: NoteRepositoryDependency,
) -> NoteResponse:
    """Validate and persist an engineering note."""

    try:
        created = repository.create_note(
            project_id=note.project_id,
            title=note.title,
            content=note.content,
            work_date=note.work_date,
        )
    except sqlite3.IntegrityError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The note conflicts with existing data",
        ) from error

    return NoteResponse.model_validate(created)


@router.get("/notes", response_model=list[NoteResponse])
def list_notes(
    repository: NoteRepositoryDependency,
    work_date: Annotated[date | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[NoteResponse]:
    """List notes, optionally filtering by the date work occurred."""

    notes = repository.list_notes(
        work_date=work_date,
        limit=limit,
        offset=offset,
    )
    return [NoteResponse.model_validate(note) for note in notes]


@router.get(
    "/search/notes",
    response_model=list[NoteSearchResultResponse],
)
def search_notes(
    repository: NoteRepositoryDependency,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[NoteSearchResultResponse]:
    """Search note titles and content using FTS5 and BM25."""

    if not q.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Search query must not be blank",
        )

    try:
        results = repository.search_notes(q, limit=limit)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except sqlite3.OperationalError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid search query",
        ) from error

    return [
        NoteSearchResultResponse.model_validate(result)
        for result in results
    ]


@router.get(
    "/search/documents",
    response_model=list[DocumentSearchResultResponse],
)
def search_documents(
    repository: ChunkRepositoryDependency,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[DocumentSearchResultResponse]:
    """Search ingested source chunks using FTS5 and BM25."""

    if not q.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Search query must not be blank",
        )

    try:
        results = repository.search(q, limit=limit)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except sqlite3.OperationalError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid search query",
        ) from error

    return [
        DocumentSearchResultResponse.model_validate(result)
        for result in results
    ]


@router.get(
    "/search/evidence",
    response_model=list[EvidenceGroupResponse],
)
def search_evidence(
    service: RetrievalServiceDependency,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[EvidenceGroupResponse]:
    """Return grouped evidence with coverage-gated adjacent chunks."""

    if not q.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Search query must not be blank",
        )

    try:
        groups = service.search(
            q,
            limit=limit,
            rerank_by_coverage=True,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except sqlite3.OperationalError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid search query",
        ) from error

    return [
        EvidenceGroupResponse.model_validate(group)
        for group in groups
    ]


@router.get("/notes/{note_id}", response_model=NoteResponse)
def get_note(
    repository: NoteRepositoryDependency,
    note_id: Annotated[int, Path(gt=0)],
) -> NoteResponse:
    """Return one note or report that it does not exist."""

    note = repository.get_note(note_id)
    if note is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found",
        )

    return NoteResponse.model_validate(note)


@router.patch("/notes/{note_id}", response_model=NoteResponse)
def update_note(
    note: NoteUpdate,
    repository: NoteRepositoryDependency,
    note_id: Annotated[int, Path(gt=0)],
) -> NoteResponse:
    """Update only the fields supplied by the client."""

    try:
        updated = repository.update_note(
            note_id,
            note.model_dump(exclude_unset=True),
        )
    except sqlite3.IntegrityError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The note conflicts with existing data",
        ) from error

    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found",
        )

    return NoteResponse.model_validate(updated)


@router.delete(
    "/notes/{note_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_note(
    repository: NoteRepositoryDependency,
    note_id: Annotated[int, Path(gt=0)],
) -> Response:
    """Delete one note if it exists."""

    if not repository.delete_note(note_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found",
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)
