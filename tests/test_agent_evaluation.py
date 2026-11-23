"""Tests for component-level and overall agent evaluation."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from internmemory.agent_evaluation import (
    AgentEvalExpectation,
    AgentFailure,
    aggregate_agent_results,
    evaluate_agent_state,
    load_agent_cases,
)
from internmemory.agent_models import (
    AgentMessage,
    EvidenceCitation,
    FinalAnswerAction,
    MessageRole,
    ModelResponse,
    ToolCallAction,
    ToolDefinition,
)
from internmemory.agent_runner import AgentRunner
from internmemory.model_provider import ScriptedModelProvider


class FixedObservationExecutor:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    @property
    def definitions(self) -> Sequence[ToolDefinition]:
        return (
            ToolDefinition(
                name="search_evidence",
                description="Search evidence.",
                input_schema={"type": "object"},
            ),
        )

    def execute(self, action: ToolCallAction) -> AgentMessage:
        return AgentMessage(
            role=MessageRole.TOOL,
            content=json.dumps(self._payload),
            tool_call_id=action.call_id,
        )


def evidence_payload() -> dict[str, object]:
    return {
        "ok": True,
        "result": {
            "abstained": False,
            "evidence": [
                {
                    "source": {
                        "locator": "batching.md",
                        "content_hash": "source-hash",
                    },
                    "chunks": [
                        {
                            "id": 7,
                            "content": (
                                "A batch can increase tail latency."
                            ),
                        }
                    ],
                }
            ],
        },
    }


def run_with(
    final_action: FinalAnswerAction,
    payload: dict[str, object],
):
    provider = ScriptedModelProvider(
        [
            ModelResponse(
                action=ToolCallAction(
                    call_id="call-1",
                    tool_name="search_evidence",
                    arguments={"query": "tail latency"},
                )
            ),
            ModelResponse(action=final_action),
        ]
    )
    return AgentRunner(
        provider,
        FixedObservationExecutor(payload),
    ).run("What did I learn about batching?")


def answerable_expectation() -> AgentEvalExpectation:
    return AgentEvalExpectation(
        expected_tools=("search_evidence",),
        expected_source="batching.md",
        required_answer_concepts=(
            ("batch", "batching"),
            ("tail latency",),
        ),
    )


def test_all_dimensions_pass_for_grounded_cited_answer() -> None:
    state = run_with(
        FinalAnswerAction(
            content="Waiting for a batch can increase tail latency.",
            citations=(
                EvidenceCitation(
                    source_locator="batching.md",
                    source_content_hash="source-hash",
                    chunk_ids=(7,),
                ),
            ),
        ),
        evidence_payload(),
    )

    result = evaluate_agent_state(state, answerable_expectation())

    assert result.completed is True
    assert result.tool_selection_correct is True
    assert result.evidence_correct is True
    assert result.grounding_correct is True
    assert result.citation_correct is True
    assert result.abstention_correct is True
    assert result.overall_success is True
    assert result.primary_failure is None


def test_correct_evidence_does_not_hide_unsupported_answer() -> None:
    state = run_with(
        FinalAnswerAction(
            content="Kubernetes automatically solves this problem.",
            citations=(
                EvidenceCitation(
                    source_locator="batching.md",
                    source_content_hash="source-hash",
                    chunk_ids=(7,),
                ),
            ),
        ),
        evidence_payload(),
    )

    result = evaluate_agent_state(state, answerable_expectation())

    assert result.evidence_correct is True
    assert result.grounding_correct is False
    assert result.overall_success is False
    assert result.primary_failure is AgentFailure.GROUNDING


def test_repeated_correct_tool_is_separate_from_task_success() -> None:
    provider = ScriptedModelProvider(
        [
            ModelResponse(
                action=ToolCallAction(
                    call_id=f"call-{index}",
                    tool_name="search_evidence",
                    arguments={"query": "tail latency"},
                )
            )
            for index in range(3)
        ]
        + [
            ModelResponse(
                action=FinalAnswerAction(
                    content="A batch can increase tail latency.",
                    citations=(
                        EvidenceCitation(
                            source_locator="batching.md",
                            source_content_hash="source-hash",
                            chunk_ids=(7,),
                        ),
                    ),
                )
            )
        ]
    )
    state = AgentRunner(
        provider,
        FixedObservationExecutor(evidence_payload()),
    ).run("What did I learn about batching?")

    result = evaluate_agent_state(state, answerable_expectation())

    assert result.tool_selection_correct is True
    assert result.tool_efficiency_correct is False
    assert result.primary_failure is None
    assert result.overall_success is True


def test_invented_citation_fails_despite_expected_answer_terms() -> None:
    state = run_with(
        FinalAnswerAction(
            content="Waiting for a batch can increase tail latency.",
            citations=(
                EvidenceCitation(
                    source_locator="batching.md",
                    source_content_hash="invented-hash",
                    chunk_ids=(7,),
                ),
            ),
        ),
        evidence_payload(),
    )

    result = evaluate_agent_state(state, answerable_expectation())

    assert result.grounding_correct is True
    assert result.citation_correct is False
    assert result.overall_success is False
    assert result.primary_failure is AgentFailure.CITATION


def test_supported_abstention_passes_all_dimensions() -> None:
    state = run_with(
        FinalAnswerAction(
            content="I do not have supporting evidence.",
            abstained=True,
        ),
        {
            "ok": True,
            "result": {"abstained": True, "evidence": []},
        },
    )
    expectation = AgentEvalExpectation(
        expected_tools=("search_evidence",),
        expected_source=None,
        required_answer_concepts=(),
        expected_abstention=True,
    )

    result = evaluate_agent_state(state, expectation)

    assert result.overall_success is True


def test_missing_evidence_is_primary_over_downstream_failures() -> None:
    state = run_with(
        FinalAnswerAction(content="Unsupported answer."),
        {
            "ok": True,
            "result": {"abstained": True, "evidence": []},
        },
    )

    result = evaluate_agent_state(state, answerable_expectation())

    assert result.evidence_correct is False
    assert result.grounding_correct is False
    assert result.citation_correct is False
    assert result.primary_failure is AgentFailure.EVIDENCE


def test_expected_source_without_required_content_is_not_evidence() -> None:
    payload = evidence_payload()
    payload["result"]["evidence"][0]["chunks"][0]["content"] = (
        "This passage discusses an unrelated deployment."
    )
    state = run_with(
        FinalAnswerAction(
            content="I do not have enough evidence.",
            abstained=True,
        ),
        payload,
    )

    result = evaluate_agent_state(state, answerable_expectation())

    assert result.evidence_correct is False
    assert result.primary_failure is AgentFailure.EVIDENCE


def test_grounding_accepts_labeled_equivalent_phrases() -> None:
    state = run_with(
        FinalAnswerAction(
            content=(
                "Batching can make tail-latency outliers worse."
            ),
            citations=(
                EvidenceCitation(
                    source_locator="batching.md",
                    source_content_hash="source-hash",
                    chunk_ids=(7,),
                ),
            ),
        ),
        evidence_payload(),
    )

    result = evaluate_agent_state(state, answerable_expectation())

    assert result.grounding_correct is True


def test_grounding_accepts_hyphenated_concept_split() -> None:
    expectation = AgentEvalExpectation(
        expected_tools=("search_evidence",),
        expected_source="batching.md",
        required_answer_concepts=(
            ("batch", "batching"),
            ("backpressure", "back pressure"),
        ),
    )
    payload = evidence_payload()
    payload["result"]["evidence"][0]["chunks"][0]["content"] = (
        "A batch path applies backpressure under overload."
    )
    state = run_with(
        FinalAnswerAction(
            content=(
                "Use a batch with back-pressure when producers "
                "outpace consumers."
            ),
            citations=(
                EvidenceCitation(
                    source_locator="batching.md",
                    source_content_hash="source-hash",
                    chunk_ids=(7,),
                ),
            ),
        ),
        payload,
    )

    result = evaluate_agent_state(state, expectation)

    assert result.grounding_correct is True


def test_versioned_agent_dataset_has_balanced_frozen_split() -> None:
    project_root = Path(__file__).parents[1]

    cases = load_agent_cases(project_root / "evals" / "agent_cases.json")

    assert len(cases) == 9
    assert sum(case.split == "development" for case in cases) == 5
    assert sum(case.split == "evaluation" for case in cases) == 4
    assert any(
        case.category == "answer_without_evidence"
        for case in cases
    )
    assert any(
        case.case_id == "queue-backpressure-paraphrase"
        and case.split == "development"
        for case in cases
    )
    assert sum(
        case.expectation.expected_abstention for case in cases
    ) == 3


def test_agent_dataset_loader_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "agent_cases.json"
    case = {
        "id": "duplicate",
        "category": "no_evidence",
        "split": "development",
        "prompt": "Unsupported question",
        "expected_tools": ["search_evidence"],
        "expected_abstention": True,
    }
    dataset_path.write_text(
        json.dumps({"version": 2, "cases": [case, case]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_agent_cases(dataset_path)


def test_aggregate_report_exposes_component_bottlenecks() -> None:
    successful = evaluate_agent_state(
        run_with(
            FinalAnswerAction(
                content="A batch can increase tail latency.",
                citations=(
                    EvidenceCitation(
                        source_locator="batching.md",
                        source_content_hash="source-hash",
                        chunk_ids=(7,),
                    ),
                ),
            ),
            evidence_payload(),
        ),
        answerable_expectation(),
    )
    unsupported = evaluate_agent_state(
        run_with(
            FinalAnswerAction(
                content="An unsupported conclusion.",
                citations=(
                    EvidenceCitation(
                        source_locator="batching.md",
                        source_content_hash="source-hash",
                        chunk_ids=(7,),
                    ),
                ),
            ),
            evidence_payload(),
        ),
        answerable_expectation(),
    )
    invented_citation = evaluate_agent_state(
        run_with(
            FinalAnswerAction(
                content="A batch can increase tail latency.",
                citations=(
                    EvidenceCitation(
                        source_locator="batching.md",
                        source_content_hash="wrong-hash",
                        chunk_ids=(7,),
                    ),
                ),
            ),
            evidence_payload(),
        ),
        answerable_expectation(),
    )

    report = aggregate_agent_results(
        [successful, unsupported, invented_citation]
    )

    assert report.case_count == 3
    assert report.overall_success_rate == pytest.approx(1 / 3)
    assert report.evidence_rate == pytest.approx(1.0)
    assert report.grounding_rate == pytest.approx(2 / 3)
    assert report.citation_rate == pytest.approx(2 / 3)
    assert report.primary_failure_counts[AgentFailure.GROUNDING] == 1
    assert report.primary_failure_counts[AgentFailure.CITATION] == 1
