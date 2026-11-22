"""Run Groq only against the fictional agent development split."""

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from internmemory.agent_evaluation import (
    AgentEvalSplit,
    AgentEvalResult,
    AgentEvaluationReport,
    aggregate_agent_results,
    evaluate_agent_state,
    load_agent_cases,
)
from internmemory.agent_models import AgentStatus
from internmemory.agent_models import JsonValue, ToolCallAction
from internmemory.agent_runner import AgentRunner
from internmemory.agent_tools import create_search_evidence_tool
from internmemory.chunk_repository import ChunkRepository
from internmemory.database import initialize_database
from internmemory.groq_provider import (
    DEFAULT_GROQ_MODEL,
    GroqModelProvider,
)
from internmemory.ingestion_service import IngestionService
from internmemory.model_provider import ModelProvider
from internmemory.retrieval_service import RetrievalService
from internmemory.source_repository import SourceRepository
from internmemory.tool_registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class AgentToolCallRecord:
    """One model-requested tool action for report diagnostics."""

    step_number: int
    tool_name: str
    arguments: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class AgentCaseRun:
    """One development case with execution and quality results."""

    case_id: str
    category: str
    status: AgentStatus
    step_count: int
    input_tokens: int
    output_tokens: int
    model_latency_ms: float
    tool_latency_ms: float
    tool_calls: tuple[AgentToolCallRecord, ...]
    final_answer: str | None
    error: str | None
    evaluation: AgentEvalResult


@dataclass(frozen=True, slots=True)
class GroqEvaluationReport:
    """Aggregate report for one model on one evaluation split."""

    model: str
    split: str
    aggregate: AgentEvaluationReport
    total_input_tokens: int
    total_output_tokens: int
    cases: list[AgentCaseRun]


def run_groq_development(
    dataset_path: str | Path,
    fixtures_root: str | Path,
    *,
    provider: ModelProvider,
    model_name: str,
    max_steps: int = 5,
    max_context_chars: int = 32_000,
    max_total_tokens: int = 20_000,
    case_id: str | None = None,
) -> GroqEvaluationReport:
    """Evaluate one provider without exposing the frozen split."""

    return run_groq_evaluation(
        dataset_path,
        fixtures_root,
        provider=provider,
        model_name=model_name,
        split="development",
        max_steps=max_steps,
        max_context_chars=max_context_chars,
        max_total_tokens=max_total_tokens,
        case_id=case_id,
    )


def run_groq_evaluation(
    dataset_path: str | Path,
    fixtures_root: str | Path,
    *,
    provider: ModelProvider,
    model_name: str,
    split: AgentEvalSplit = "development",
    confirm_frozen: bool = False,
    max_steps: int = 5,
    max_context_chars: int = 32_000,
    max_total_tokens: int = 20_000,
    case_id: str | None = None,
) -> GroqEvaluationReport:
    """Evaluate development or explicitly confirmed frozen cases."""

    if split == "evaluation" and not confirm_frozen:
        raise ValueError(
            "frozen evaluation requires explicit confirmation"
        )
    if split == "evaluation" and case_id is not None:
        raise ValueError(
            "frozen evaluation must run as one complete aggregate"
        )
    all_cases = load_agent_cases(dataset_path)
    cases = [case for case in all_cases if case.split == split]
    if case_id is not None:
        cases = [case for case in cases if case.case_id == case_id]
    if not cases:
        raise ValueError(
            f"requested {split} case does not exist"
            if case_id is not None
            else f"agent dataset has no {split} cases"
        )

    fixtures_path = Path(fixtures_root).resolve()
    source_paths = sorted(
        path
        for path in fixtures_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )
    if not source_paths:
        raise ValueError("fixtures directory has no supported documents")

    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = Path(temporary_directory) / "agent-evaluation.db"
        initialize_database(database_path)
        chunk_repository = ChunkRepository(database_path)
        ingestion_service = IngestionService(
            SourceRepository(database_path),
            chunk_repository,
            allowed_root=fixtures_path,
        )
        for source_path in source_paths:
            ingestion_service.ingest_file(source_path.name)

        registry = ToolRegistry(
            [
                create_search_evidence_tool(
                    RetrievalService(chunk_repository)
                )
            ]
        )
        case_runs: list[AgentCaseRun] = []
        results: list[AgentEvalResult] = []
        for case in cases:
            state = AgentRunner(provider, registry).run(
                case.prompt,
                max_steps=max_steps,
                max_context_chars=max_context_chars,
                max_total_tokens=max_total_tokens,
            )
            result = evaluate_agent_state(
                state,
                case.expectation,
            )
            results.append(result)
            case_runs.append(
                AgentCaseRun(
                    case_id=case.case_id,
                    category=case.category,
                    status=state.status,
                    step_count=state.step_count,
                    input_tokens=state.usage.input_tokens,
                    output_tokens=state.usage.output_tokens,
                    model_latency_ms=sum(
                        trace.model_duration_ms
                        for trace in state.trace
                    ),
                    tool_latency_ms=sum(
                        trace.tool_duration_ms or 0.0
                        for trace in state.trace
                    ),
                    tool_calls=tuple(
                        AgentToolCallRecord(
                            step_number=trace.step_number,
                            tool_name=trace.action.tool_name,
                            arguments=trace.action.arguments,
                        )
                        for trace in state.trace
                        if isinstance(
                            trace.action,
                            ToolCallAction,
                        )
                    ),
                    final_answer=state.final_answer,
                    error=state.error,
                    evaluation=result,
                )
            )

    return GroqEvaluationReport(
        model=model_name,
        split=split,
        aggregate=aggregate_agent_results(results),
        total_input_tokens=sum(
            case.input_tokens for case in case_runs
        ),
        total_output_tokens=sum(
            case.output_tokens for case in case_runs
        ),
        cases=case_runs,
    )


def main() -> None:
    """Run a bounded live evaluation using GROQ_API_KEY."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Groq on fictional InternMemory agent cases."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/agent_cases.json"),
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("evals/fixtures"),
    )
    parser.add_argument("--model", default=DEFAULT_GROQ_MODEL)
    parser.add_argument(
        "--split",
        choices=("development", "evaluation"),
        default="development",
    )
    parser.add_argument(
        "--confirm-frozen",
        action="store_true",
        help="Confirm the one aggregate run of the frozen evaluation split.",
    )
    parser.add_argument(
        "--case-id",
        help="Run one development case; evaluation must run in aggregate.",
    )
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=32_000,
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=20_000,
    )
    arguments = parser.parse_args()
    if arguments.split == "evaluation" and not arguments.confirm_frozen:
        parser.error(
            "--split evaluation requires --confirm-frozen"
        )
    if arguments.split == "evaluation" and arguments.case_id is not None:
        parser.error(
            "--case-id cannot be used with the frozen evaluation split"
        )

    provider = GroqModelProvider.from_environment(
        model=arguments.model,
    )
    report = run_groq_evaluation(
        arguments.dataset,
        arguments.fixtures,
        provider=provider,
        model_name=arguments.model,
        split=arguments.split,
        confirm_frozen=arguments.confirm_frozen,
        max_steps=arguments.max_steps,
        max_context_chars=arguments.max_context_chars,
        max_total_tokens=arguments.max_total_tokens,
        case_id=arguments.case_id,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
