"""Deterministic component and end-to-end agent evaluation metrics."""

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from internmemory.agent_models import (
    AgentState,
    AgentStatus,
    EvidenceCitation,
    ToolCallAction,
)


AgentEvalSplit = Literal["development", "evaluation"]
_CONCEPT_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class AgentFailure(str, Enum):
    """Primary failure categories ordered by execution dependency."""

    EXECUTION = "execution"
    TOOL_SELECTION = "tool_selection"
    EVIDENCE = "evidence"
    ABSTENTION = "abstention"
    GROUNDING = "grounding"
    CITATION = "citation"


@dataclass(frozen=True, slots=True)
class AgentEvalExpectation:
    """Human-labeled requirements for one agent task."""

    expected_tools: tuple[str, ...]
    expected_source: str | None
    required_answer_concepts: tuple[tuple[str, ...], ...]
    expected_abstention: bool = False

    def __post_init__(self) -> None:
        if not self.expected_tools or any(
            not name.strip() for name in self.expected_tools
        ):
            raise ValueError("expected_tools must not be empty")
        if self.expected_abstention:
            if (
                self.expected_source is not None
                or self.required_answer_concepts
            ):
                raise ValueError(
                    "abstention cases cannot require evidence or terms"
                )
        elif (
            self.expected_source is None
            or not self.expected_source.strip()
            or not self.required_answer_concepts
            or any(
                not alternatives
                or any(
                    not term.strip() for term in alternatives
                )
                for alternatives in self.required_answer_concepts
            )
        ):
            raise ValueError(
                "answerable cases require a source and answer terms"
            )


@dataclass(frozen=True, slots=True)
class AgentEvalCase:
    """One versioned prompt and its human-labeled expectations."""

    case_id: str
    category: str
    split: AgentEvalSplit
    prompt: str
    expectation: AgentEvalExpectation

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.case_id,
                self.category,
                self.prompt,
            )
        ):
            raise ValueError("agent evaluation values must not be blank")
        if self.split not in ("development", "evaluation"):
            raise ValueError("unsupported agent evaluation split")


@dataclass(frozen=True, slots=True)
class AgentEvalResult:
    """Separate metrics that preserve the location of a failure."""

    completed: bool
    tool_selection_correct: bool
    tool_efficiency_correct: bool
    evidence_correct: bool
    grounding_correct: bool
    citation_correct: bool
    abstention_correct: bool
    overall_success: bool
    primary_failure: AgentFailure | None


@dataclass(frozen=True, slots=True)
class AgentEvaluationReport:
    """Aggregate quality rates and primary failure counts."""

    case_count: int
    overall_success_rate: float
    completion_rate: float
    tool_selection_rate: float
    tool_efficiency_rate: float
    evidence_rate: float
    grounding_rate: float
    citation_rate: float
    abstention_rate: float
    primary_failure_counts: dict[AgentFailure, int]


@dataclass(frozen=True, slots=True)
class _RetrievedEvidence:
    source_locator: str
    source_content_hash: str
    chunk_ids: frozenset[int]
    content: str


def load_agent_cases(dataset_path: str | Path) -> list[AgentEvalCase]:
    """Load and validate version 2 of the agent evaluation dataset."""

    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 2:
        raise ValueError("unsupported agent evaluation dataset version")
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("agent evaluation dataset must contain cases")

    cases: list[AgentEvalCase] = []
    seen_ids: set[str] = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("invalid agent evaluation case")
        try:
            raw_tools = raw_case["expected_tools"]
            raw_concepts = raw_case.get(
                "required_answer_concepts",
                [],
            )
            expected_source = raw_case.get("expected_source")
            expected_abstention = raw_case.get(
                "expected_abstention",
                False,
            )
            if (
                not all(
                    isinstance(raw_case.get(key), str)
                    for key in ("id", "category", "split", "prompt")
                )
                or not isinstance(raw_tools, list)
                or not all(isinstance(item, str) for item in raw_tools)
                or not isinstance(raw_concepts, list)
                or not all(
                    isinstance(alternatives, list)
                    and all(
                        isinstance(term, str)
                        for term in alternatives
                    )
                    for alternatives in raw_concepts
                )
                or (
                    expected_source is not None
                    and not isinstance(expected_source, str)
                )
                or not isinstance(expected_abstention, bool)
            ):
                raise TypeError
            case = AgentEvalCase(
                case_id=raw_case["id"],
                category=raw_case["category"],
                split=raw_case["split"],
                prompt=raw_case["prompt"],
                expectation=AgentEvalExpectation(
                    expected_tools=tuple(raw_tools),
                    expected_source=expected_source,
                    required_answer_concepts=tuple(
                        tuple(alternatives)
                        for alternatives in raw_concepts
                    ),
                    expected_abstention=expected_abstention,
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid agent evaluation case") from exc
        if case.case_id in seen_ids:
            raise ValueError(
                f"duplicate agent evaluation case: {case.case_id}"
            )
        seen_ids.add(case.case_id)
        cases.append(case)
    return cases


def evaluate_agent_state(
    state: AgentState,
    expectation: AgentEvalExpectation,
) -> AgentEvalResult:
    """Score one completed or failed run against human labels."""

    tool_names = tuple(
        trace.action.tool_name
        for trace in state.trace
        if isinstance(trace.action, ToolCallAction)
    )
    expected_tool_names = set(expectation.expected_tools)
    tool_selection_correct = (
        set(tool_names) == expected_tool_names
        and all(name in expected_tool_names for name in tool_names)
    )
    tool_efficiency_correct = tool_names == expectation.expected_tools
    retrieved, retrieval_abstained = _read_observations(state)
    completed = (
        state.status is AgentStatus.COMPLETED
        and state.final_action is not None
    )

    if expectation.expected_abstention:
        evidence_correct = retrieval_abstained and not retrieved
    else:
        expected_source_content = " ".join(
            evidence.content
            for evidence in retrieved
            if evidence.source_locator == expectation.expected_source
        ).casefold()
        evidence_correct = bool(
            expected_source_content
        ) and _matches_required_concepts(
            expected_source_content,
            expectation.required_answer_concepts,
        )

    final_action = state.final_action
    abstention_correct = (
        final_action is not None
        and final_action.abstained == expectation.expected_abstention
    )
    if final_action is None or expectation.expected_abstention:
        grounding_correct = (
            final_action is not None
            and final_action.abstained
            and retrieval_abstained
        )
    else:
        normalized_answer = final_action.content.casefold()
        grounding_correct = (
            not final_action.abstained
            and evidence_correct
            and _matches_required_concepts(
                normalized_answer,
                expectation.required_answer_concepts,
            )
        )

    if final_action is None:
        citation_correct = False
    elif expectation.expected_abstention:
        citation_correct = not final_action.citations
    else:
        citation_correct = any(
            _citation_matches(citation, retrieved)
            for citation in final_action.citations
        )

    checks = (
        completed,
        tool_selection_correct,
        evidence_correct,
        grounding_correct,
        citation_correct,
        abstention_correct,
    )
    primary_failure = _primary_failure(
        completed=completed,
        tool_selection_correct=tool_selection_correct,
        evidence_correct=evidence_correct,
        grounding_correct=grounding_correct,
        citation_correct=citation_correct,
        abstention_correct=abstention_correct,
    )
    return AgentEvalResult(
        completed=completed,
        tool_selection_correct=tool_selection_correct,
        tool_efficiency_correct=tool_efficiency_correct,
        evidence_correct=evidence_correct,
        grounding_correct=grounding_correct,
        citation_correct=citation_correct,
        abstention_correct=abstention_correct,
        overall_success=all(checks),
        primary_failure=primary_failure,
    )


def aggregate_agent_results(
    results: list[AgentEvalResult],
) -> AgentEvaluationReport:
    """Aggregate component rates without hiding the primary bottleneck."""

    if not results:
        raise ValueError("at least one agent result is required")
    count = len(results)
    failure_counts = {
        failure: sum(
            result.primary_failure is failure for result in results
        )
        for failure in AgentFailure
    }
    return AgentEvaluationReport(
        case_count=count,
        overall_success_rate=(
            sum(result.overall_success for result in results) / count
        ),
        completion_rate=(
            sum(result.completed for result in results) / count
        ),
        tool_selection_rate=(
            sum(result.tool_selection_correct for result in results)
            / count
        ),
        tool_efficiency_rate=(
            sum(result.tool_efficiency_correct for result in results)
            / count
        ),
        evidence_rate=(
            sum(result.evidence_correct for result in results) / count
        ),
        grounding_rate=(
            sum(result.grounding_correct for result in results) / count
        ),
        citation_rate=(
            sum(result.citation_correct for result in results) / count
        ),
        abstention_rate=(
            sum(result.abstention_correct for result in results) / count
        ),
        primary_failure_counts=failure_counts,
    )


def _matches_required_concepts(
    text: str,
    concepts: tuple[tuple[str, ...], ...],
) -> bool:
    """Require every concept while accepting labeled equivalent phrases."""

    normalized_text = f" {_normalize_concept_text(text)} "
    return all(
        any(
            f" {_normalize_concept_text(alternative)} "
            in normalized_text
            for alternative in alternatives
        )
        for alternatives in concepts
    )


def _normalize_concept_text(value: str) -> str:
    return " ".join(_CONCEPT_TOKEN_PATTERN.findall(value.casefold()))


def _primary_failure(
    *,
    completed: bool,
    tool_selection_correct: bool,
    evidence_correct: bool,
    grounding_correct: bool,
    citation_correct: bool,
    abstention_correct: bool,
) -> AgentFailure | None:
    checks = (
        (completed, AgentFailure.EXECUTION),
        (tool_selection_correct, AgentFailure.TOOL_SELECTION),
        (evidence_correct, AgentFailure.EVIDENCE),
        (abstention_correct, AgentFailure.ABSTENTION),
        (grounding_correct, AgentFailure.GROUNDING),
        (citation_correct, AgentFailure.CITATION),
    )
    return next(
        (failure for passed, failure in checks if not passed),
        None,
    )


def _citation_matches(
    citation: EvidenceCitation,
    retrieved: list[_RetrievedEvidence],
) -> bool:
    return any(
        citation.source_locator == evidence.source_locator
        and citation.source_content_hash
        == evidence.source_content_hash
        and set(citation.chunk_ids) <= evidence.chunk_ids
        for evidence in retrieved
    )


def _read_observations(
    state: AgentState,
) -> tuple[list[_RetrievedEvidence], bool]:
    retrieved: list[_RetrievedEvidence] = []
    abstained = False
    for trace in state.trace:
        observation = trace.tool_observation
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
        abstained = abstained or result.get("abstained") is True
        evidence_items = result.get("evidence")
        if not isinstance(evidence_items, list):
            continue
        for item in evidence_items:
            parsed = _parse_evidence(item)
            if parsed is not None:
                retrieved.append(parsed)
    return retrieved, abstained


def _parse_evidence(value: object) -> _RetrievedEvidence | None:
    if not isinstance(value, dict):
        return None
    source = value.get("source")
    chunks = value.get("chunks")
    if not isinstance(source, dict) or not isinstance(chunks, list):
        return None
    locator = source.get("locator")
    content_hash = source.get("content_hash")
    chunk_ids = {
        chunk.get("id")
        for chunk in chunks
        if isinstance(chunk, dict)
        and isinstance(chunk.get("id"), int)
        and chunk["id"] > 0
    }
    content = " ".join(
        chunk["content"]
        for chunk in chunks
        if isinstance(chunk, dict)
        and isinstance(chunk.get("content"), str)
    )
    if (
        not isinstance(locator, str)
        or not isinstance(content_hash, str)
        or not chunk_ids
        or not content
    ):
        return None
    return _RetrievedEvidence(
        source_locator=locator,
        source_content_hash=content_hash,
        chunk_ids=frozenset(chunk_ids),
        content=content,
    )
