"""Domain models shared across storage and API layers."""

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Allowed lifecycle states for an engineering task."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"


class SourceType(str, Enum):
    """Supported local document formats."""

    MARKDOWN = "markdown"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class Project:
    """A persisted engineering project."""

    id: int
    name: str
    description: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class Note:
    """A persisted engineering note."""

    id: int
    project_id: int | None
    title: str
    content: str
    work_date: date
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class NoteSearchResult:
    """A note paired with its FTS5 BM25 ranking score."""

    note: Note
    score: float


@dataclass(frozen=True, slots=True)
class Task:
    """A persisted engineering follow-up task."""

    id: int
    project_id: int | None
    description: str
    status: TaskStatus
    due_date: date | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class Source:
    """A captured local document and its provenance."""

    id: int
    source_type: SourceType
    locator: str
    title: str
    content: str
    content_hash: str
    byte_size: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class SourceChunk:
    """A searchable evidence unit with exact source character offsets."""

    id: int
    source_id: int
    ordinal: int
    start_offset: int
    end_offset: int
    content: str
    content_hash: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class SourceReference:
    """Citation metadata without the source's full stored content."""

    id: int
    source_type: SourceType
    locator: str
    title: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class DocumentSearchResult:
    """A ranked source chunk paired with citation metadata."""

    source: SourceReference
    chunk: SourceChunk
    score: float
