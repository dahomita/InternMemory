"""HTTP request and response schemas."""

from datetime import date, datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from internmemory.models import TaskStatus


class HealthResponse(BaseModel):
    """Response returned when the service is available."""

    status: str


class ProjectCreate(BaseModel):
    """Validated input for creating an engineering project."""

    name: str = Field(min_length=1, max_length=100)
    description: str = ""

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only names and store a trimmed value."""

        name = value.strip()
        if not name:
            raise ValueError("name must not be blank")
        return name


class ProjectResponse(BaseModel):
    """Public representation of a persisted project."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str
    created_at: datetime
    updated_at: datetime


class TaskCreate(BaseModel):
    """Validated input for creating an engineering task."""

    project_id: int | None = Field(default=None, gt=0)
    description: str
    status: TaskStatus = TaskStatus.OPEN
    due_date: date | None = None

    @field_validator("description")
    @classmethod
    def description_must_not_be_blank(cls, value: str) -> str:
        """Reject descriptions containing only whitespace."""

        if not value.strip():
            raise ValueError("description must not be blank")
        return value


class TaskStatusUpdate(BaseModel):
    """Validated task status transition request."""

    status: TaskStatus


class TaskResponse(BaseModel):
    """Public representation of a persisted task."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int | None
    description: str
    status: TaskStatus
    due_date: date | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class NoteCreate(BaseModel):
    """Validated input for creating an engineering note."""

    project_id: int | None = Field(default=None, gt=0)
    title: str = Field(default="", max_length=200)
    content: str
    work_date: date

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        """Reject content containing only whitespace."""

        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class NoteUpdate(BaseModel):
    """Fields that may be changed on an existing note."""

    project_id: int | None = Field(default=None, gt=0)
    title: str | None = Field(default=None, max_length=200)
    content: str | None = None
    work_date: date | None = None

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str | None) -> str | None:
        """Reject blank content when the field is supplied."""

        if value is not None and not value.strip():
            raise ValueError("content must not be blank")
        return value

    @model_validator(mode="after")
    def validate_patch_fields(self) -> "NoteUpdate":
        """Distinguish omitted fields from invalid explicit nulls."""

        if not self.model_fields_set:
            raise ValueError("at least one field must be supplied")

        for field_name in ("title", "content", "work_date"):
            if (
                field_name in self.model_fields_set
                and getattr(self, field_name) is None
            ):
                raise ValueError(f"{field_name} must not be null")

        return self


class NoteResponse(BaseModel):
    """Public representation of a persisted note."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int | None
    title: str
    content: str
    work_date: date
    created_at: datetime
    updated_at: datetime


class NoteSearchResultResponse(BaseModel):
    """A search match and its FTS5 ranking score."""

    model_config = ConfigDict(from_attributes=True)

    note: NoteResponse
    score: float


class MorningRecapResponse(BaseModel):
    """Deterministic recap of prior work and unfinished tasks."""

    model_config = ConfigDict(from_attributes=True)

    work_date: date
    notes: list[NoteResponse]
    unfinished_tasks: list[TaskResponse]
