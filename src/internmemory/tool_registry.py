"""Typed validation and dispatch for model-requested tools."""

import json
from collections.abc import Callable, Sequence
from typing import Generic, Protocol, TypeVar, cast

from pydantic import BaseModel, ValidationError

from internmemory.agent_models import (
    AgentMessage,
    JsonValue,
    MessageRole,
    ToolCallAction,
    ToolDefinition,
)


class AgentTool(Protocol):
    """A validated tool that can be exposed to a model."""

    @property
    def definition(self) -> ToolDefinition:
        """Return the model-visible tool contract."""

    def invoke(self, arguments: dict[str, JsonValue]) -> JsonValue:
        """Validate arguments and invoke the underlying operation."""


InputModel = TypeVar("InputModel", bound=BaseModel)


class StructuredTool(Generic[InputModel]):
    """Adapt a typed Pydantic input model and handler into an agent tool."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_model: type[InputModel],
        handler: Callable[[InputModel], JsonValue],
    ) -> None:
        self._input_model = input_model
        self._handler = handler
        self._definition = ToolDefinition(
            name=name,
            description=description,
            input_schema=cast(
                dict[str, JsonValue],
                input_model.model_json_schema(),
            ),
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def invoke(self, arguments: dict[str, JsonValue]) -> JsonValue:
        validated = self._input_model.model_validate(
            arguments,
            strict=True,
        )
        return self._handler(validated)


class ToolRegistry:
    """Expose unique tool definitions and safely dispatch tool calls."""

    def __init__(
        self,
        tools: Sequence[AgentTool],
        *,
        max_observation_chars: int = 16_000,
    ) -> None:
        if max_observation_chars <= 0:
            raise ValueError("max_observation_chars must be positive")
        self._max_observation_chars = max_observation_chars
        self._tools: dict[str, AgentTool] = {}
        for tool in tools:
            name = tool.definition.name
            if name in self._tools:
                raise ValueError(f"duplicate tool name: {name}")
            self._tools[name] = tool

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool.definition for tool in self._tools.values())

    def execute(self, action: ToolCallAction) -> AgentMessage:
        tool = self._tools.get(action.tool_name)
        if tool is None:
            return self._error_observation(
                action.call_id,
                error_type="unknown_tool",
                message=f"unknown tool: {action.tool_name}",
            )

        try:
            result = tool.invoke(action.arguments)
            content = json.dumps(
                {"ok": True, "result": result},
                separators=(",", ":"),
                sort_keys=True,
            )
            if len(content) > self._max_observation_chars:
                return self._error_observation(
                    action.call_id,
                    error_type="output_too_large",
                    message=(
                        "tool output exceeds the configured size limit; "
                        "request fewer or narrower results"
                    ),
                )
        except ValidationError as exc:
            fields = sorted(
                {
                    ".".join(str(part) for part in error["loc"])
                    for error in exc.errors()
                }
            )
            return self._error_observation(
                action.call_id,
                error_type="invalid_arguments",
                message=(
                    "invalid arguments"
                    + (f" for: {', '.join(fields)}" if fields else "")
                ),
            )
        except (TypeError, ValueError):
            return self._error_observation(
                action.call_id,
                error_type="execution_error",
                message="tool execution failed",
            )

        return AgentMessage(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=action.call_id,
        )

    @staticmethod
    def _error_observation(
        call_id: str,
        *,
        error_type: str,
        message: str,
    ) -> AgentMessage:
        return AgentMessage(
            role=MessageRole.TOOL,
            content=json.dumps(
                {
                    "ok": False,
                    "error": {
                        "type": error_type,
                        "message": message,
                    },
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            tool_call_id=call_id,
        )
