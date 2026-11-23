"""Agent tools backed by InternMemory application services."""

from datetime import date
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from internmemory.agent_models import JsonValue
from internmemory.note_repository import NoteRepository
from internmemory.recap_service import RecapService
from internmemory.retrieval_service import RetrievalService
from internmemory.tool_registry import StructuredTool


def _parse_optional_iso_date(value: object) -> object:
    """Accept ISO date strings from JSON tool arguments."""

    if isinstance(value, str):
        return date.fromisoformat(value)
    return value


class SearchNotesInput(BaseModel):
    """Strict, bounded arguments for personal-note retrieval."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "Concise literal keywords likely to appear in personal notes. "
            "Prefer engineering terms from the user's work, not filler."
        ),
    )
    limit: int = Field(default=5, ge=1, le=10)


class SearchEvidenceInput(BaseModel):
    """Strict, bounded arguments for grounded evidence retrieval."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "Concise literal keywords likely to appear in the source. "
            "Preserve every essential concept from the user's question, "
            "but avoid conversational filler because retrieval is lexical."
        ),
    )
    limit: int = Field(default=5, ge=1, le=10)


class GetDailySummaryInput(BaseModel):
    """Optional date-scoped morning/daily recap request."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    work_date: date | None = Field(
        default=None,
        description=(
            "ISO work date to summarize. Omit to use local yesterday."
        ),
    )
    note_limit: int = Field(default=20, ge=1, le=50)
    task_limit: int = Field(default=20, ge=1, le=50)

    @field_validator("work_date", mode="before")
    @classmethod
    def parse_work_date(cls, value: object) -> object:
        return _parse_optional_iso_date(value)


class SaveNoteInput(BaseModel):
    """Bounded note draft that requires confirmation before writing."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    content: str = Field(min_length=1, max_length=8_000)
    title: str = Field(default="", max_length=200)
    work_date: date
    project_id: int | None = Field(default=None, gt=0)
    confirmed: bool = Field(
        default=False,
        description=(
            "Must be true to persist the note. First call with false to "
            "preview the draft, then call again with confirmed=true."
        ),
    )

    @field_validator("work_date", mode="before")
    @classmethod
    def parse_work_date(cls, value: object) -> object:
        return _parse_optional_iso_date(value)


def create_search_notes_tool(
    repository: NoteRepository,
) -> StructuredTool[SearchNotesInput]:
    """Adapt personal-note FTS search into a read-only agent tool."""

    def search(request: SearchNotesInput) -> JsonValue:
        matches = repository.search_notes(
            request.query,
            limit=request.limit,
        )
        result = {
            "abstained": not matches,
            "notes": [
                {
                    "id": match.note.id,
                    "project_id": match.note.project_id,
                    "title": match.note.title,
                    "content": match.note.content,
                    "work_date": match.note.work_date.isoformat(),
                    "score": match.score,
                }
                for match in matches
            ],
        }
        return cast(JsonValue, result)

    return StructuredTool(
        name="search_notes",
        description=(
            "Search personal engineering notes by literal keywords. "
            "Use for questions about the user's own recorded work, "
            "decisions, and follow-ups. Do not use this for ingested "
            "documentation; use search_evidence for documents. Returns "
            "matching notes or abstains when nothing matches."
        ),
        input_model=SearchNotesInput,
        handler=search,
    )


def create_search_evidence_tool(
    service: RetrievalService,
) -> StructuredTool[SearchEvidenceInput]:
    """Adapt grounded keyword retrieval into a read-only agent tool."""

    def search(request: SearchEvidenceInput) -> JsonValue:
        groups = service.search(
            request.query,
            limit=request.limit,
            rerank_by_coverage=True,
            minimum_coverage=0.5,
        )
        result = {
            "abstained": not groups,
            "evidence": [
                {
                    "source": {
                        "id": group.source.id,
                        "type": group.source.source_type.value,
                        "locator": group.source.locator,
                        "title": group.source.title,
                        "content_hash": group.source.content_hash,
                    },
                    "chunks": [
                        {
                            "id": chunk.id,
                            "ordinal": chunk.ordinal,
                            "start_offset": chunk.start_offset,
                            "end_offset": chunk.end_offset,
                            "content": chunk.content,
                            "content_hash": chunk.content_hash,
                        }
                        for chunk in group.chunks
                    ],
                    "anchor_score": group.anchor_score,
                    "query_term_coverage": (
                        group.query_term_coverage
                    ),
                }
                for group in groups
            ],
        }
        return cast(JsonValue, result)

    return StructuredTool(
        name="search_evidence",
        description=(
            "Search stored engineering documents using literal keywords. "
            "Use concise source terms likely to appear in notes, not a full "
            "natural-language paraphrase. If a search abstains, reformulate "
            "once with short systems nouns (queue, backpressure, latency, "
            "retry, timeout, cache) instead of restating the question. "
            "Returns cited passages or abstains when lexical support is weak."
        ),
        input_model=SearchEvidenceInput,
        handler=search,
    )


def create_get_daily_summary_tool(
    service: RecapService,
) -> StructuredTool[GetDailySummaryInput]:
    """Expose deterministic morning/daily recap as a read-only tool."""

    def summarize(request: GetDailySummaryInput) -> JsonValue:
        recap = service.build_morning_recap(request.work_date)
        notes = recap.notes[: request.note_limit]
        tasks = recap.unfinished_tasks[: request.task_limit]
        result = {
            "work_date": recap.work_date.isoformat(),
            "notes": [
                {
                    "id": note.id,
                    "project_id": note.project_id,
                    "title": note.title,
                    "content": note.content,
                    "work_date": note.work_date.isoformat(),
                }
                for note in notes
            ],
            "unfinished_tasks": [
                {
                    "id": task.id,
                    "project_id": task.project_id,
                    "description": task.description,
                    "status": task.status.value,
                    "due_date": (
                        None
                        if task.due_date is None
                        else task.due_date.isoformat()
                    ),
                }
                for task in tasks
            ],
            "note_count": len(notes),
            "unfinished_task_count": len(tasks),
            "truncated": (
                len(recap.notes) > len(notes)
                or len(recap.unfinished_tasks) > len(tasks)
            ),
        }
        return cast(JsonValue, result)

    return StructuredTool(
        name="get_daily_summary",
        description=(
            "Return a deterministic daily/morning recap: notes for one work "
            "date plus currently unfinished tasks. Omit work_date to use "
            "local yesterday. Use for recap questions; do not invent missing "
            "notes or tasks."
        ),
        input_model=GetDailySummaryInput,
        handler=summarize,
    )


def create_save_note_tool(
    repository: NoteRepository,
) -> StructuredTool[SaveNoteInput]:
    """Create notes only after an explicit confirmation flag."""

    def save(request: SaveNoteInput) -> JsonValue:
        preview = {
            "title": request.title,
            "content": request.content,
            "work_date": request.work_date.isoformat(),
            "project_id": request.project_id,
        }
        if not request.confirmed:
            return cast(
                JsonValue,
                {
                    "written": False,
                    "requires_confirmation": True,
                    "preview": preview,
                    "message": (
                        "Preview only. Call save_note again with the same "
                        "fields and confirmed=true to persist."
                    ),
                },
            )

        created = repository.create_note(
            title=request.title,
            content=request.content,
            work_date=request.work_date,
            project_id=request.project_id,
        )
        return cast(
            JsonValue,
            {
                "written": True,
                "requires_confirmation": False,
                "note": {
                    "id": created.id,
                    "project_id": created.project_id,
                    "title": created.title,
                    "content": created.content,
                    "work_date": created.work_date.isoformat(),
                },
            },
        )

    return StructuredTool(
        name="save_note",
        description=(
            "Draft or save a personal engineering note. Always call first "
            "with confirmed=false to preview. Only persist when the user has "
            "approved and confirmed=true. Never invent confidential employer "
            "data."
        ),
        input_model=SaveNoteInput,
        handler=save,
    )
