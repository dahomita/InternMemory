"""Tests for the live-evaluation pipeline without network usage."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from internmemory.agent_models import (
    AgentMessage,
    EvidenceCitation,
    FinalAnswerAction,
    MessageRole,
    ModelResponse,
    TokenUsage,
    ToolCallAction,
    ToolDefinition,
)
from internmemory.groq_evaluation import (
    run_groq_development,
    run_groq_evaluation,
)


class AdaptiveFakeProvider:
    """Return deterministic actions based on fictional case context."""

    def generate(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        del tools
        user_prompt = next(
            message.content
            for message in messages
            if message.role is MessageRole.USER
        )
        if messages[-1].role is not MessageRole.TOOL:
            query = self._query_for(user_prompt)
            return ModelResponse(
                action=ToolCallAction(
                    call_id=f"call-{len(messages)}",
                    tool_name="search_evidence",
                    arguments={"query": query},
                ),
                usage=TokenUsage(input_tokens=100, output_tokens=20),
            )

        payload = json.loads(messages[-1].content)["result"]
        if payload["abstained"]:
            action = FinalAnswerAction(
                content="I do not have supporting evidence.",
                abstained=True,
            )
        else:
            expected_source = self._source_for(user_prompt)
            evidence = next(
                item
                for item in payload["evidence"]
                if item["source"]["locator"] == expected_source
            )
            action = FinalAnswerAction(
                content=(
                    "A batch can affect tail latency. A circuit breaker "
                    "and retry budget bound failures. WAL permits readers "
                    "alongside one writer. p95 exposes slow requests. A "
                    "bounded queue applies backpressure."
                ),
                citations=(
                    EvidenceCitation(
                        source_locator=expected_source,
                        source_content_hash=(
                            evidence["source"]["content_hash"]
                        ),
                        chunk_ids=tuple(
                            chunk["id"]
                            for chunk in evidence["chunks"]
                        ),
                    ),
                ),
            )
        return ModelResponse(
            action=action,
            usage=TokenUsage(input_tokens=200, output_tokens=40),
        )

    @staticmethod
    def _query_for(prompt: str) -> str:
        if "batching" in prompt:
            return "batch tail latency"
        if "circuit breaker" in prompt:
            return "circuit breaker retry budget"
        if "Kubernetes" in prompt:
            return "Kubernetes autoscaling replicas"
        if "SQLite WAL" in prompt:
            return "SQLite WAL readers writer"
        if "p95" in prompt:
            return "p95 slow requests"
        if "incoming work" in prompt:
            return "bounded queue backpressure"
        if "cache documents" in prompt:
            return "cache key rotation strategy"
        return "AES key rotation"

    @staticmethod
    def _source_for(prompt: str) -> str:
        if "batching" in prompt:
            return "batching.md"
        if "SQLite WAL" in prompt:
            return "sqlite-reliability.md"
        if "p95" in prompt:
            return "trace-observability.md"
        if "incoming work" in prompt:
            return "queue-backpressure.md"
        return "resilience-boundary.md"


def test_runner_uses_only_development_cases_and_fictional_fixtures() -> None:
    project_root = Path(__file__).parents[1]

    report = run_groq_development(
        project_root / "evals" / "agent_cases.json",
        project_root / "evals" / "fixtures",
        provider=AdaptiveFakeProvider(),
        model_name="fake-model",
    )

    assert report.model == "fake-model"
    assert report.split == "development"
    assert report.aggregate.case_count == 4
    assert {case.case_id for case in report.cases} == {
        "batching-tail-latency",
        "resilience-cross-chunk",
        "unsupported-kubernetes",
        "tool-avoidance-instruction",
    }
    assert report.total_input_tokens == 1_200
    assert report.total_output_tokens == 240
    assert all(case.error is None for case in report.cases)
    assert all(
        [call.tool_name for call in case.tool_calls]
        == ["search_evidence"]
        for case in report.cases
    )


def test_runner_rejects_frozen_case_selection() -> None:
    project_root = Path(__file__).parents[1]

    with pytest.raises(ValueError, match="does not exist"):
        run_groq_development(
            project_root / "evals" / "agent_cases.json",
            project_root / "evals" / "fixtures",
            provider=AdaptiveFakeProvider(),
            model_name="fake-model",
            case_id="sqlite-wal-concurrency",
        )


def test_frozen_runner_requires_explicit_confirmation() -> None:
    project_root = Path(__file__).parents[1]

    with pytest.raises(ValueError, match="explicit confirmation"):
        run_groq_evaluation(
            project_root / "evals" / "agent_cases.json",
            project_root / "evals" / "fixtures",
            provider=AdaptiveFakeProvider(),
            model_name="fake-model",
            split="evaluation",
        )


def test_frozen_runner_runs_complete_evaluation_split() -> None:
    project_root = Path(__file__).parents[1]

    report = run_groq_evaluation(
        project_root / "evals" / "agent_cases.json",
        project_root / "evals" / "fixtures",
        provider=AdaptiveFakeProvider(),
        model_name="fake-model",
        split="evaluation",
        confirm_frozen=True,
    )

    assert report.split == "evaluation"
    assert report.aggregate.case_count == 4
    assert {case.case_id for case in report.cases} == {
        "sqlite-wal-concurrency",
        "trace-percentile-latency",
        "unsupported-cache-encryption",
        "queue-backpressure-paraphrase",
    }


def test_frozen_runner_rejects_single_case_peeking() -> None:
    project_root = Path(__file__).parents[1]

    with pytest.raises(ValueError, match="complete aggregate"):
        run_groq_evaluation(
            project_root / "evals" / "agent_cases.json",
            project_root / "evals" / "fixtures",
            provider=AdaptiveFakeProvider(),
            model_name="fake-model",
            split="evaluation",
            confirm_frozen=True,
            case_id="sqlite-wal-concurrency",
        )
