"""Typed domain models for model-independent agent execution."""

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias


JsonValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)


class MessageRole(str, Enum):
    """Allowed participants in an agent conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class AgentStatus(str, Enum):
    """Lifecycle states for one bounded agent run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STEP_LIMIT_REACHED = "step_limit_reached"
    CONTEXT_LIMIT_REACHED = "context_limit_reached"
    TOKEN_LIMIT_REACHED = "token_limit_reached"


@dataclass(frozen=True, slots=True)
class ToolCallAction:
    """A model request to invoke one registered tool."""

    call_id: str
    tool_name: str
    arguments: dict[str, JsonValue]

    def __post_init__(self) -> None:
        if not self.call_id.strip() or not self.tool_name.strip():
            raise ValueError("tool call identifiers must not be blank")


@dataclass(frozen=True, slots=True)
class AgentMessage:
    """One model-visible message."""

    role: MessageRole
    content: str
    tool_call_id: str | None = None
    tool_call: ToolCallAction | None = None

    def __post_init__(self) -> None:
        if not self.content.strip() and self.tool_call is None:
            raise ValueError(
                "message requires content or an assistant tool call"
            )
        if self.role is MessageRole.TOOL and not self.tool_call_id:
            raise ValueError("tool messages require a tool_call_id")
        if self.role is not MessageRole.TOOL and self.tool_call_id is not None:
            raise ValueError(
                "only tool messages may contain a tool_call_id"
            )
        if (
            self.tool_call is not None
            and self.role is not MessageRole.ASSISTANT
        ):
            raise ValueError(
                "only assistant messages may contain a tool_call"
            )


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Model-visible name, purpose, and JSON input contract."""

    name: str
    description: str
    input_schema: dict[str, JsonValue]

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.description.strip():
            raise ValueError("tool metadata must not be blank")


@dataclass(frozen=True, slots=True)
class EvidenceCitation:
    """A final-answer citation tied to an observed source snapshot."""

    source_locator: str
    source_content_hash: str
    chunk_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not self.source_locator.strip()
            or not self.source_content_hash.strip()
        ):
            raise ValueError("citation source metadata must not be blank")
        if not self.chunk_ids or any(
            chunk_id <= 0 for chunk_id in self.chunk_ids
        ):
            raise ValueError("citation chunk_ids must be positive")


@dataclass(frozen=True, slots=True)
class FinalAnswerAction:
    """A model request to terminate with a user-facing answer."""

    content: str
    citations: tuple[EvidenceCitation, ...] = ()
    abstained: bool = False

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("final answer must not be blank")


ModelAction: TypeAlias = ToolCallAction | FinalAnswerAction


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Immutable context supplied for one model decision."""

    messages: tuple[AgentMessage, ...]
    tools: tuple[ToolDefinition, ...]

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("model request requires at least one message")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Provider-reported token counts for one model decision."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must not be negative")


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """One validated model action and its usage metadata."""

    action: ModelAction
    usage: TokenUsage = TokenUsage()


@dataclass(frozen=True, slots=True)
class AgentStepTrace:
    """Structured diagnostic record for one model decision."""

    step_number: int
    action: ModelAction | None
    usage: TokenUsage
    model_duration_ms: float
    tool_duration_ms: float | None = None
    tool_observation: AgentMessage | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.step_number <= 0:
            raise ValueError("trace step_number must be positive")
        if self.model_duration_ms < 0:
            raise ValueError("model duration must not be negative")
        if (
            self.tool_duration_ms is not None
            and self.tool_duration_ms < 0
        ):
            raise ValueError("tool duration must not be negative")


@dataclass(slots=True)
class AgentState:
    """Mutable state owned by one bounded harness execution."""

    messages: list[AgentMessage]
    max_steps: int
    max_context_chars: int = 32_000
    max_total_tokens: int = 20_000
    step_count: int = 0
    status: AgentStatus = AgentStatus.RUNNING
    final_answer: str | None = None
    final_action: FinalAnswerAction | None = None
    error: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    trace: list[AgentStepTrace] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("agent state requires at least one message")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")
        if self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")
        if self.step_count < 0:
            raise ValueError("step_count must not be negative")
