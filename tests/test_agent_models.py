"""Tests for agent domain models and the deterministic provider."""

from collections.abc import Callable

import pytest

from internmemory.agent_models import (
    AgentMessage,
    AgentState,
    FinalAnswerAction,
    MessageRole,
    ModelResponse,
    TokenUsage,
    ToolCallAction,
    ToolDefinition,
)
from internmemory.model_provider import ScriptedModelProvider


def test_scripted_provider_returns_actions_and_records_requests() -> None:
    """Harness tests can inspect exactly what context reached the model."""

    responses = [
        ModelResponse(
            action=ToolCallAction(
                call_id="call-1",
                tool_name="search_evidence",
                arguments={"query": "tail latency"},
            ),
            usage=TokenUsage(input_tokens=10, output_tokens=4),
        ),
        ModelResponse(
            action=FinalAnswerAction(content="Supported answer."),
        ),
    ]
    provider = ScriptedModelProvider(responses)
    messages = [
        AgentMessage(
            role=MessageRole.USER,
            content="What did I learn?",
        )
    ]
    tools = [
        ToolDefinition(
            name="search_evidence",
            description="Search cited engineering evidence.",
            input_schema={"type": "object"},
        )
    ]

    first = provider.generate(messages, tools)
    messages.append(
        AgentMessage(
            role=MessageRole.ASSISTANT,
            content="Searching evidence.",
        )
    )
    second = provider.generate(messages, tools)

    assert isinstance(first.action, ToolCallAction)
    assert isinstance(second.action, FinalAnswerAction)
    assert len(provider.requests) == 2
    assert len(provider.requests[0].messages) == 1
    assert len(provider.requests[1].messages) == 2


def test_scripted_provider_fails_when_responses_are_exhausted() -> None:
    """Unexpected extra model steps fail deterministically."""

    provider = ScriptedModelProvider(
        [
            ModelResponse(
                action=FinalAnswerAction(content="Done."),
            )
        ]
    )
    messages = [
        AgentMessage(role=MessageRole.USER, content="Question")
    ]

    provider.generate(messages, [])

    with pytest.raises(RuntimeError, match="exhausted"):
        provider.generate(messages, [])


def test_tool_observation_requires_call_identifier() -> None:
    """Tool results remain correlated with their originating request."""

    with pytest.raises(ValueError, match="tool_call_id"):
        AgentMessage(
            role=MessageRole.TOOL,
            content="result",
        )


def test_non_tool_message_rejects_call_identifier() -> None:
    """Only tool observations may reference a tool invocation."""

    with pytest.raises(ValueError, match="only tool messages"):
        AgentMessage(
            role=MessageRole.USER,
            content="Question",
            tool_call_id="call-1",
        )


@pytest.mark.parametrize(
    ("factory", "error"),
    [
        (lambda: FinalAnswerAction(content="  "), "final answer"),
        (
            lambda: ToolCallAction(
                call_id=" ",
                tool_name="search_evidence",
                arguments={},
            ),
            "identifiers",
        ),
        (
            lambda: ToolDefinition(
                name=" ",
                description="Search evidence.",
                input_schema={},
            ),
            "metadata",
        ),
    ],
)
def test_model_actions_and_tools_reject_blank_identifiers(
    factory: Callable[[], object],
    error: str,
) -> None:
    """Malformed model output must fail before the harness executes it."""

    with pytest.raises(ValueError, match=error):
        factory()


def test_token_usage_rejects_negative_counts() -> None:
    """Provider accounting cannot silently reduce accumulated usage."""

    with pytest.raises(ValueError, match="negative"):
        TokenUsage(input_tokens=-1)


def test_agent_state_requires_a_positive_step_bound() -> None:
    """No agent run may begin with unlimited or zero execution steps."""

    with pytest.raises(ValueError, match="max_steps"):
        AgentState(
            messages=[
                AgentMessage(
                    role=MessageRole.USER,
                    content="Question",
                )
            ],
            max_steps=0,
        )
