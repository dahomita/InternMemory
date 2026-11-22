"""Model provider boundary and deterministic test implementation."""

from collections import deque
from collections.abc import Sequence
from typing import Protocol

from internmemory.agent_models import (
    AgentMessage,
    ModelRequest,
    ModelResponse,
    ToolDefinition,
)


class ModelProvider(Protocol):
    """Return one model decision for the supplied context and tools."""

    def generate(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        """Generate one validated action."""


class ScriptedModelProvider:
    """Return predetermined responses and record immutable requests."""

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        if not responses:
            raise ValueError("scripted provider requires responses")
        self._responses = deque(responses)
        self.requests: list[ModelRequest] = []

    def generate(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        """Return the next scripted response."""

        request = ModelRequest(
            messages=tuple(messages),
            tools=tuple(tools),
        )
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("scripted model responses exhausted")
        return self._responses.popleft()
