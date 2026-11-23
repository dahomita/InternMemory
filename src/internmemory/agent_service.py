"""Product-facing agent assembly over application services and tools."""

import json
from dataclasses import dataclass
from datetime import date

from internmemory.agent_models import AgentState
from internmemory.agent_runner import AgentRunner
from internmemory.agent_tools import (
    create_get_daily_summary_tool,
    create_save_note_tool,
    create_search_evidence_tool,
    create_search_notes_tool,
)
from internmemory.model_provider import ModelProvider
from internmemory.models import Note
from internmemory.note_repository import NoteRepository
from internmemory.recap_service import RecapService
from internmemory.retrieval_service import RetrievalService
from internmemory.tool_registry import ToolRegistry


class AgentUnavailableError(RuntimeError):
    """Raised when the product agent has no configured model provider."""


@dataclass(frozen=True, slots=True)
class PendingNoteConfirmation:
    """A save_note preview waiting for explicit user approval."""

    title: str
    content: str
    work_date: date
    project_id: int | None = None
    message: str = (
        "Preview only. Call POST /agent/confirm-note with confirmed=true "
        "to persist."
    )


def build_default_tool_registry(
    note_repository: NoteRepository,
    retrieval_service: RetrievalService,
    recap_service: RecapService,
) -> ToolRegistry:
    """Wire every product tool against live application services."""

    return ToolRegistry(
        [
            create_search_notes_tool(note_repository),
            create_search_evidence_tool(retrieval_service),
            create_get_daily_summary_tool(recap_service),
            create_save_note_tool(note_repository),
        ]
    )


def extract_pending_note_confirmation(
    state: AgentState,
) -> PendingNoteConfirmation | None:
    """Return the latest unresolved save_note preview from a run."""

    pending: PendingNoteConfirmation | None = None
    for step in state.trace:
        observation = step.tool_observation
        if observation is None:
            continue
        try:
            payload = json.loads(observation.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            continue
        result = payload.get("result")
        if not isinstance(result, dict):
            continue
        if result.get("written") is True:
            pending = None
            continue
        if result.get("requires_confirmation") is not True:
            continue
        preview = result.get("preview")
        if not isinstance(preview, dict):
            continue
        content = preview.get("content")
        work_date = preview.get("work_date")
        if not isinstance(content, str) or not isinstance(work_date, str):
            continue
        title = preview.get("title", "")
        project_id = preview.get("project_id")
        if project_id is not None and not isinstance(project_id, int):
            continue
        if not isinstance(title, str):
            continue
        message = result.get("message")
        pending = PendingNoteConfirmation(
            title=title,
            content=content,
            work_date=date.fromisoformat(work_date),
            project_id=project_id,
            message=(
                message
                if isinstance(message, str) and message.strip()
                else PendingNoteConfirmation.message
            ),
        )
    return pending


class AgentService:
    """Run one bounded agent turn with the full InternMemory tool set."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        note_repository: NoteRepository,
        model_provider: ModelProvider | None = None,
        *,
        max_successful_reads: int = 2,
    ) -> None:
        if max_successful_reads <= 0:
            raise ValueError("max_successful_reads must be positive")
        self._tool_registry = tool_registry
        self._note_repository = note_repository
        self._model_provider = model_provider
        self._max_successful_reads = max_successful_reads

    @property
    def is_available(self) -> bool:
        """Return whether a model provider is configured."""

        return self._model_provider is not None

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Return registered tool names in registry order."""

        return tuple(
            definition.name
            for definition in self._tool_registry.definitions
        )

    def run(
        self,
        message: str,
        *,
        max_steps: int = 5,
        max_context_chars: int = 32_000,
        max_total_tokens: int = 20_000,
    ) -> AgentState:
        """Execute one user message through the bounded harness."""

        if self._model_provider is None:
            raise AgentUnavailableError(
                "No model provider is configured. Set GROQ_API_KEY or "
                "inject a ModelProvider when creating the application."
            )
        if not message.strip():
            raise ValueError("message must not be blank")

        runner = AgentRunner(
            self._model_provider,
            self._tool_registry,
            max_successful_reads=self._max_successful_reads,
        )
        return runner.run(
            message,
            max_steps=max_steps,
            max_context_chars=max_context_chars,
            max_total_tokens=max_total_tokens,
        )

    def confirm_note(
        self,
        *,
        content: str,
        work_date: date,
        title: str = "",
        project_id: int | None = None,
        confirmed: bool,
    ) -> Note:
        """Persist a previously previewed note after explicit confirmation."""

        if not confirmed:
            raise ValueError(
                "confirmed must be true to persist a note; "
                "previews are returned by /agent/run"
            )
        if not content.strip():
            raise ValueError("content must not be blank")
        return self._note_repository.create_note(
            title=title,
            content=content,
            work_date=work_date,
            project_id=project_id,
        )
