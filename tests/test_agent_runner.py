"""Tests for bounded model-independent agent execution."""

import json
from collections.abc import Sequence

import pytest

from internmemory.agent_models import (
    AgentMessage,
    AgentStatus,
    FinalAnswerAction,
    MessageRole,
    ModelResponse,
    TokenUsage,
    ToolCallAction,
    ToolDefinition,
)
from internmemory.agent_runner import AgentRunner
from internmemory.model_provider import ScriptedModelProvider


class RecordingToolExecutor:
    """Small deterministic executor used to isolate runner behavior."""

    def __init__(
        self,
        *,
        observations: Sequence[str] | None = None,
    ) -> None:
        self.actions: list[ToolCallAction] = []
        self._observations = list(observations or [])
        self._definitions = (
            ToolDefinition(
                name="search_evidence",
                description="Search grounded evidence.",
                input_schema={"type": "object"},
            ),
        )

    @property
    def definitions(self) -> Sequence[ToolDefinition]:
        return self._definitions

    def execute(self, action: ToolCallAction) -> AgentMessage:
        self.actions.append(action)
        if self._observations:
            content = self._observations.pop(0)
        else:
            content = '{"results": []}'
        return AgentMessage(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=action.call_id,
        )


def tool_call(call_id: str) -> ModelResponse:
    """Build one deterministic search request."""

    return ModelResponse(
        action=ToolCallAction(
            call_id=call_id,
            tool_name="search_evidence",
            arguments={"query": "tail latency"},
        ),
        usage=TokenUsage(input_tokens=10, output_tokens=3),
    )


def test_runner_completes_with_final_answer_and_usage() -> None:
    provider = ScriptedModelProvider(
        [
            ModelResponse(
                action=FinalAnswerAction(content="Grounded answer."),
                usage=TokenUsage(input_tokens=12, output_tokens=5),
            )
        ]
    )

    state = AgentRunner(provider).run("What did I learn?")

    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "Grounded answer."
    assert state.step_count == 1
    assert state.usage == TokenUsage(input_tokens=12, output_tokens=5)
    assert state.messages[-1].role is MessageRole.ASSISTANT


def test_runner_executes_tool_then_returns_final_answer() -> None:
    provider = ScriptedModelProvider(
        [
            tool_call("call-1"),
            ModelResponse(
                action=FinalAnswerAction(content="Answer from evidence."),
                usage=TokenUsage(input_tokens=20, output_tokens=4),
            ),
        ]
    )
    executor = RecordingToolExecutor()

    state = AgentRunner(provider, executor).run("Find evidence")

    assert state.status is AgentStatus.COMPLETED
    assert state.step_count == 2
    assert state.usage == TokenUsage(input_tokens=30, output_tokens=7)
    assert executor.actions[0].call_id == "call-1"
    assert provider.requests[1].messages[2].tool_call is not None
    assert provider.requests[1].messages[3].tool_call_id == "call-1"


def test_runner_enforces_step_limit_outside_model() -> None:
    provider = ScriptedModelProvider(
        [tool_call("call-1"), tool_call("call-2")]
    )
    executor = RecordingToolExecutor()

    state = AgentRunner(provider, executor).run(
        "Keep searching",
        max_steps=2,
    )

    assert state.status is AgentStatus.STEP_LIMIT_REACHED
    assert state.step_count == 2
    assert len(executor.actions) == 2
    assert state.error == "agent reached the maximum of 2 steps"


def test_runner_fails_safely_when_tool_executor_is_missing() -> None:
    provider = ScriptedModelProvider([tool_call("call-1")])

    state = AgentRunner(provider).run("Search")

    assert state.status is AgentStatus.FAILED
    assert state.step_count == 1
    assert state.error == (
        "RuntimeError: no tool executor is configured"
    )


def test_runner_traces_actions_observations_usage_and_latency() -> None:
    provider = ScriptedModelProvider(
        [
            tool_call("call-1"),
            ModelResponse(
                action=FinalAnswerAction(content="Grounded answer."),
                usage=TokenUsage(input_tokens=20, output_tokens=4),
            ),
        ]
    )
    times = iter([1.0, 1.01, 2.0, 2.025, 3.0, 3.02])

    state = AgentRunner(
        provider,
        RecordingToolExecutor(),
        clock=lambda: next(times),
    ).run("Find evidence")

    assert len(state.trace) == 2
    tool_step, answer_step = state.trace
    assert isinstance(tool_step.action, ToolCallAction)
    assert tool_step.tool_observation is not None
    assert tool_step.tool_observation.tool_call_id == "call-1"
    assert tool_step.usage == TokenUsage(
        input_tokens=10,
        output_tokens=3,
    )
    assert tool_step.model_duration_ms == pytest.approx(10)
    assert tool_step.tool_duration_ms == pytest.approx(25)
    assert isinstance(answer_step.action, FinalAnswerAction)
    assert answer_step.model_duration_ms == pytest.approx(20)


def test_runner_traces_provider_failure() -> None:
    provider = ScriptedModelProvider([tool_call("call-1")])

    state = AgentRunner(
        provider,
        RecordingToolExecutor(),
    ).run("Keep going")

    assert state.status is AgentStatus.FAILED
    assert len(state.trace) == 2
    assert state.trace[-1].action is None
    assert state.trace[-1].error == (
        "RuntimeError: scripted model responses exhausted"
    )


def test_runner_stops_before_model_call_when_context_is_too_large() -> None:
    provider = ScriptedModelProvider(
        [
            ModelResponse(
                action=FinalAnswerAction(content="Unused answer.")
            )
        ]
    )

    state = AgentRunner(provider).run(
        "This prompt exceeds a deliberately tiny budget.",
        max_context_chars=10,
    )

    assert state.status is AgentStatus.CONTEXT_LIMIT_REACHED
    assert state.step_count == 0
    assert provider.requests == []


def test_runner_stops_after_provider_reports_excessive_token_usage() -> None:
    provider = ScriptedModelProvider(
        [
            ModelResponse(
                action=FinalAnswerAction(content="Expensive answer."),
                usage=TokenUsage(input_tokens=12, output_tokens=5),
            )
        ]
    )

    state = AgentRunner(provider).run(
        "Question",
        max_total_tokens=16,
    )

    assert state.status is AgentStatus.TOKEN_LIMIT_REACHED
    assert state.final_answer is None
    assert state.usage == TokenUsage(input_tokens=12, output_tokens=5)
    assert state.trace[0].error is not None


def test_runner_keeps_tools_after_one_abstention_for_rewrite() -> None:
    abstained = json.dumps(
        {
            "ok": True,
            "result": {"abstained": True, "evidence": []},
        }
    )
    evidence = json.dumps(
        {
            "ok": True,
            "result": {
                "abstained": False,
                "evidence": [{"source": {"locator": "queue.md"}}],
            },
        }
    )
    provider = ScriptedModelProvider(
        [
            tool_call("call-1"),
            tool_call("call-2"),
            ModelResponse(
                action=FinalAnswerAction(content="Bounded queue answer."),
                usage=TokenUsage(input_tokens=20, output_tokens=4),
            ),
        ]
    )
    executor = RecordingToolExecutor(
        observations=[abstained, evidence],
    )

    state = AgentRunner(provider, executor).run("Paraphrase question")

    assert state.status is AgentStatus.COMPLETED
    assert len(executor.actions) == 2
    assert provider.requests[1].tools
    assert provider.requests[2].tools == ()


def test_runner_hides_tools_after_second_abstention() -> None:
    abstained = json.dumps(
        {
            "ok": True,
            "result": {"abstained": True, "evidence": []},
        }
    )
    provider = ScriptedModelProvider(
        [
            tool_call("call-1"),
            tool_call("call-2"),
            ModelResponse(
                action=FinalAnswerAction(
                    content="No supporting evidence.",
                    abstained=True,
                ),
                usage=TokenUsage(input_tokens=20, output_tokens=4),
            ),
        ]
    )
    executor = RecordingToolExecutor(
        observations=[abstained, abstained],
    )

    state = AgentRunner(provider, executor).run("Unsupported question")

    assert state.status is AgentStatus.COMPLETED
    assert len(executor.actions) == 2
    assert provider.requests[1].tools
    assert provider.requests[2].tools == ()
