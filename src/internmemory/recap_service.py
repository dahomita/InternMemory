"""Application service for deterministic morning recaps."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from internmemory.models import Note, Task
from internmemory.note_repository import NoteRepository
from internmemory.task_repository import TaskRepository


@dataclass(frozen=True, slots=True)
class MorningRecap:
    """Notes from one work date and currently unfinished tasks."""

    work_date: date
    notes: list[Note]
    unfinished_tasks: list[Task]


class RecapService:
    """Combine memory and task data without using an LLM."""

    def __init__(
        self,
        note_repository: NoteRepository,
        task_repository: TaskRepository,
        *,
        timezone_name: str,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._note_repository = note_repository
        self._task_repository = task_repository
        self._timezone = ZoneInfo(timezone_name)
        self._now_provider = now_provider or (
            lambda: datetime.now(timezone.utc)
        )

    def build_morning_recap(
        self,
        work_date: date | None = None,
    ) -> MorningRecap:
        """Build a recap for an explicit date or local yesterday."""

        target_date = work_date or self._local_yesterday()
        return MorningRecap(
            work_date=target_date,
            notes=self._note_repository.list_notes(
                work_date=target_date,
                limit=500,
            ),
            unfinished_tasks=self._task_repository.list_tasks(
                unfinished_only=True,
                limit=500,
            ),
        )

    def _local_yesterday(self) -> date:
        """Calculate yesterday using the configured local timezone."""

        now = self._now_provider()
        if now.tzinfo is None:
            raise ValueError("now_provider must return an aware datetime")
        local_today = now.astimezone(self._timezone).date()
        return local_today - timedelta(days=1)
