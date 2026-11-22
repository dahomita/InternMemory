"""Tests for typed agent-tool validation and dispatch."""

import json

import pytest
from pydantic import BaseModel, Field

from internmemory.agent_models import JsonValue, ToolCallAction
from internmemory.tool_registry import StructuredTool, ToolRegistry


class SearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=5, ge=1, le=10)


def action(
    *,
    tool_name: str = "search_evidence",
    arguments: dict[str, JsonValue] | None = None,
) -> ToolCallAction:
    return ToolCallAction(
        call_id="call-1",
        tool_name=tool_name,
        arguments=arguments or {"query": "tail latency"},
    )


def test_registry_exposes_json_schema_and_dispatches_valid_input() -> None:
    received: list[SearchInput] = []
    registry = ToolRegistry(
        [
            StructuredTool(
                name="search_evidence",
                description="Search grounded engineering evidence.",
                input_model=SearchInput,
                handler=lambda request: (
                    received.append(request)
                    or {"count": request.limit}
                ),
            )
        ]
    )

    observation = registry.execute(
        action(arguments={"query": "tail latency", "limit": 3})
    )

    schema = registry.definitions[0].input_schema
    assert schema["properties"]["limit"]["maximum"] == 10
    assert received == [SearchInput(query="tail latency", limit=3)]
    assert json.loads(observation.content) == {
        "ok": True,
        "result": {"count": 3},
    }
    assert observation.tool_call_id == "call-1"


def test_registry_rejects_wrong_type_before_handler_runs() -> None:
    call_count = 0

    def handler(_: SearchInput) -> JsonValue:
        nonlocal call_count
        call_count += 1
        return None

    registry = ToolRegistry(
        [
            StructuredTool(
                name="search_evidence",
                description="Search grounded engineering evidence.",
                input_model=SearchInput,
                handler=handler,
            )
        ]
    )

    observation = registry.execute(
        action(arguments={"query": "tail latency", "limit": "five"})
    )

    assert call_count == 0
    assert json.loads(observation.content) == {
        "ok": False,
        "error": {
            "type": "invalid_arguments",
            "message": "invalid arguments for: limit",
        },
    }


def test_registry_returns_structured_error_for_unknown_tool() -> None:
    registry = ToolRegistry([])

    observation = registry.execute(action(tool_name="delete_database"))

    assert json.loads(observation.content) == {
        "ok": False,
        "error": {
            "type": "unknown_tool",
            "message": "unknown tool: delete_database",
        },
    }


def test_registry_rejects_duplicate_names() -> None:
    tool = StructuredTool(
        name="search_evidence",
        description="Search grounded engineering evidence.",
        input_model=SearchInput,
        handler=lambda _: None,
    )

    with pytest.raises(ValueError, match="duplicate tool name"):
        ToolRegistry([tool, tool])


def test_registry_replaces_oversized_output_with_bounded_error() -> None:
    registry = ToolRegistry(
        [
            StructuredTool(
                name="search_evidence",
                description="Search grounded engineering evidence.",
                input_model=SearchInput,
                handler=lambda _: {"content": "x" * 500},
            )
        ],
        max_observation_chars=100,
    )

    observation = registry.execute(action())

    assert json.loads(observation.content) == {
        "ok": False,
        "error": {
            "type": "output_too_large",
            "message": (
                "tool output exceeds the configured size limit; "
                "request fewer or narrower results"
            ),
        },
    }
