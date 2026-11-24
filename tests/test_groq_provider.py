"""Tests for the Groq adapter without live network requests."""

import json
import ssl
from collections import deque
from collections.abc import Mapping
from io import BytesIO
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from internmemory.agent_models import (
    AgentMessage,
    FinalAnswerAction,
    MessageRole,
    TokenUsage,
    ToolCallAction,
    ToolDefinition,
)
from internmemory.groq_provider import (
    GROQ_CHAT_COMPLETIONS_URL,
    GroqModelProvider,
    ModelProviderError,
    UrllibJsonTransport,
)


class FakeJsonTransport:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = deque(responses)
        self.requests: list[dict[str, Any]] = []

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.requests.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_seconds": timeout_seconds,
            }
        )
        return self._responses.popleft()


def tool_response() -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "search_evidence",
                                "arguments": json.dumps(
                                    {"query": "tail latency", "limit": 5}
                                ),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def final_response(
    *,
    source_hash: str = "source-hash",
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "answer": (
                                "Batch waiting can increase tail latency."
                            ),
                            "abstained": False,
                            "citations": [
                                {
                                    "source_locator": "batching.md",
                                    "source_content_hash": source_hash,
                                    "chunk_ids": [7],
                                }
                            ],
                        }
                    ),
                }
            }
        ],
        "usage": {"prompt_tokens": 200, "completion_tokens": 40},
    }


def plain_final_response() -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        "Batch waiting can increase tail latency."
                    ),
                }
            }
        ],
        "usage": {"prompt_tokens": 200, "completion_tokens": 12},
    }


def evidence_observation() -> str:
    return json.dumps(
        {
            "ok": True,
            "result": {
                "abstained": False,
                "evidence": [
                    {
                        "source": {
                            "locator": "batching.md",
                            "content_hash": "source-hash",
                        },
                        "chunks": [{"id": 7}],
                    }
                ],
            },
        }
    )


def messages() -> list[AgentMessage]:
    return [
        AgentMessage(
            role=MessageRole.SYSTEM,
            content="Use grounded evidence.",
        ),
        AgentMessage(
            role=MessageRole.USER,
            content="What did I learn?",
        ),
    ]


def tools() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="search_evidence",
            description="Search grounded evidence.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]


def test_groq_provider_sends_bounded_tool_request() -> None:
    transport = FakeJsonTransport([tool_response()])
    provider = GroqModelProvider(
        "test-key",
        transport=transport,
        timeout_seconds=12,
        max_completion_tokens=256,
    )

    response = provider.generate(messages(), tools())

    assert response.action == ToolCallAction(
        call_id="call-1",
        tool_name="search_evidence",
        arguments={"query": "tail latency", "limit": 5},
    )
    assert response.usage == TokenUsage(
        input_tokens=100,
        output_tokens=20,
    )
    request = transport.requests[0]
    assert request["url"] == GROQ_CHAT_COMPLETIONS_URL
    assert request["timeout_seconds"] == 12
    assert request["headers"]["Authorization"] == "Bearer test-key"
    payload = request["payload"]
    assert payload["max_completion_tokens"] == 256
    assert "response_format" not in payload
    assert payload["tools"][0]["function"]["name"] == "search_evidence"


def test_groq_provider_uses_json_mode_without_tools() -> None:
    transport = FakeJsonTransport([final_response()])
    provider = GroqModelProvider("test-key", transport=transport)

    provider.generate(messages(), [])

    payload = transport.requests[0]["payload"]
    assert payload["response_format"] == {"type": "json_object"}
    assert "tools" not in payload


def test_groq_provider_parses_grounded_final_answer() -> None:
    provider = GroqModelProvider(
        "test-key",
        transport=FakeJsonTransport([final_response()]),
    )

    response = provider.generate(messages(), tools())

    assert isinstance(response.action, FinalAnswerAction)
    assert response.action.abstained is False
    assert response.action.citations[0].source_locator == "batching.md"
    assert response.action.citations[0].chunk_ids == (7,)
    assert response.usage == TokenUsage(
        input_tokens=200,
        output_tokens=40,
    )


def test_groq_provider_serializes_tool_call_history() -> None:
    transport = FakeJsonTransport([final_response()])
    provider = GroqModelProvider("test-key", transport=transport)
    call = ToolCallAction(
        call_id="call-1",
        tool_name="search_evidence",
        arguments={"query": "tail latency"},
    )
    conversation = messages() + [
        AgentMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_call=call,
        ),
        AgentMessage(
            role=MessageRole.TOOL,
            content='{"ok":true}',
            tool_call_id="call-1",
        ),
    ]

    provider.generate(conversation, tools())

    serialized = transport.requests[0]["payload"]["messages"]
    assert serialized[2]["tool_calls"][0]["id"] == "call-1"
    assert serialized[3] == {
        "role": "tool",
        "content": '{"ok":true}',
        "tool_call_id": "call-1",
    }
    assert "response_format" not in transport.requests[0]["payload"]


def test_groq_provider_flattens_tool_history_for_final_synthesis() -> None:
    transport = FakeJsonTransport([plain_final_response()])
    provider = GroqModelProvider("test-key", transport=transport)
    call = ToolCallAction(
        call_id="call-1",
        tool_name="search_evidence",
        arguments={"query": "tail latency"},
    )
    conversation = messages() + [
        AgentMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_call=call,
        ),
        AgentMessage(
            role=MessageRole.TOOL,
            content=evidence_observation(),
            tool_call_id="call-1",
        ),
    ]

    response = provider.generate(conversation, [])

    payload = transport.requests[0]["payload"]
    serialized = payload["messages"]
    assert [message["role"] for message in serialized] == [
        "system",
        "user",
        "user",
    ]
    assert "ordinary text" in serialized[0]["content"]
    assert "Use grounded evidence." not in serialized[0]["content"]
    assert "untrusted tool result" in serialized[-1]["content"]
    assert "write the concise final answer" in serialized[-1]["content"]
    assert "final JSON" not in serialized[-1]["content"]
    assert "tool_calls" not in serialized[-1]
    assert "response_format" not in payload
    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert isinstance(response.action, FinalAnswerAction)
    assert response.action.citations[0].source_locator == "batching.md"
    assert response.action.citations[0].chunk_ids == (7,)


def test_groq_provider_accepts_note_answer_while_tools_remain_available() -> None:
    transport = FakeJsonTransport(
        [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "answer": (
                                        "Your note says batching reduced "
                                        "p95 latency."
                                    ),
                                    "abstained": False,
                                    "citations": [
                                        {
                                            "note_id": 1,
                                            "work_date": "2026-11-22",
                                        }
                                    ],
                                }
                            ),
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 180,
                    "completion_tokens": 30,
                },
            }
        ]
    )
    provider = GroqModelProvider("test-key", transport=transport)
    call = ToolCallAction(
        call_id="call-1",
        tool_name="search_notes",
        arguments={"query": "batching"},
    )
    conversation = messages() + [
        AgentMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_call=call,
        ),
        AgentMessage(
            role=MessageRole.TOOL,
            content=json.dumps(
                {
                    "ok": True,
                    "result": {
                        "abstained": False,
                        "notes": [
                            {
                                "id": 1,
                                "title": "Batching",
                                "content": (
                                    "Batching reduced p95 latency."
                                ),
                            }
                        ],
                    },
                }
            ),
            tool_call_id="call-1",
        ),
    ]

    response = provider.generate(conversation, tools())

    assert isinstance(response.action, FinalAnswerAction)
    assert response.action.abstained is False
    assert response.action.citations == ()
    assert "p95 latency" in response.action.content
    assert '{"answer"' not in response.action.content


def test_groq_provider_strips_json_and_trailing_tool_markup() -> None:
    transport = FakeJsonTransport(
        [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                '{"answer":"Batching reduced p95 latency.",'
                                '"abstained":false,"citations":[]}\n'
                                "</parameter>\n</function>\n</tool_call>"
                            ),
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 160,
                    "completion_tokens": 28,
                },
            }
        ]
    )
    provider = GroqModelProvider("test-key", transport=transport)
    call = ToolCallAction(
        call_id="call-1",
        tool_name="search_notes",
        arguments={"query": "batching"},
    )
    conversation = messages() + [
        AgentMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_call=call,
        ),
        AgentMessage(
            role=MessageRole.TOOL,
            content=json.dumps(
                {
                    "ok": True,
                    "result": {
                        "abstained": False,
                        "notes": [
                            {
                                "id": 1,
                                "content": "Batching reduced p95 latency.",
                            }
                        ],
                    },
                }
            ),
            tool_call_id="call-1",
        ),
    ]

    response = provider.generate(conversation, tools())

    assert response.action.content == "Batching reduced p95 latency."
    assert "</tool_call>" not in response.action.content


def test_groq_provider_synthesizes_from_notes_without_document_citations() -> None:
    transport = FakeJsonTransport([plain_final_response()])
    provider = GroqModelProvider("test-key", transport=transport)
    call = ToolCallAction(
        call_id="call-1",
        tool_name="search_notes",
        arguments={"query": "batching"},
    )
    conversation = messages() + [
        AgentMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_call=call,
        ),
        AgentMessage(
            role=MessageRole.TOOL,
            content=json.dumps(
                {
                    "ok": True,
                    "result": {
                        "abstained": False,
                        "notes": [
                            {
                                "id": 1,
                                "title": "Batching",
                                "content": "Reduced p95 latency.",
                            }
                        ],
                    },
                }
            ),
            tool_call_id="call-1",
        ),
    ]

    response = provider.generate(conversation, [])

    assert transport.requests
    assert isinstance(response.action, FinalAnswerAction)
    assert response.action.abstained is False
    assert response.action.citations == ()


def test_groq_provider_abstains_locally_when_evidence_is_empty() -> None:
    transport = FakeJsonTransport([])
    provider = GroqModelProvider("test-key", transport=transport)
    call = ToolCallAction(
        call_id="call-1",
        tool_name="search_evidence",
        arguments={"query": "missing subject"},
    )
    conversation = messages() + [
        AgentMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_call=call,
        ),
        AgentMessage(
            role=MessageRole.TOOL,
            content='{"ok":true,"result":{"evidence":[]}}',
            tool_call_id="call-1",
        ),
    ]

    response = provider.generate(conversation, [])

    assert response.action == FinalAnswerAction(
        content=(
            "I couldn't find relevant notes or documents to answer that "
            "question. Add a note or ingest a source, then ask again."
        ),
        abstained=True,
    )
    assert response.usage == TokenUsage()
    assert transport.requests == []


def test_groq_provider_rejects_unverifiable_final_answer() -> None:
    response = final_response()
    content = response["choices"][0]["message"]["content"]
    payload = json.loads(content)
    payload["citations"] = []
    response["choices"][0]["message"]["content"] = json.dumps(payload)
    provider = GroqModelProvider(
        "test-key",
        transport=FakeJsonTransport([response]),
    )

    with pytest.raises(ModelProviderError, match="requires"):
        provider.generate(messages(), tools())


def test_groq_provider_rejects_blank_abstention_answer() -> None:
    response = final_response()
    content = response["choices"][0]["message"]["content"]
    payload = json.loads(content)
    payload["answer"] = " "
    payload["abstained"] = True
    payload["citations"] = []
    response["choices"][0]["message"]["content"] = json.dumps(payload)
    provider = GroqModelProvider(
        "test-key",
        transport=FakeJsonTransport([response]),
    )

    with pytest.raises(
        ModelProviderError,
        match="final answer must not be blank",
    ):
        provider.generate(messages(), [])


def test_groq_provider_requires_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqModelProvider.from_environment()


def test_standard_transport_uses_verified_certifi_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, _: int) -> bytes:
            return b'{"choices":[]}'

    def fake_urlopen(*args: object, **kwargs: Any) -> FakeResponse:
        captured["request"] = args[0]
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(
        "internmemory.groq_provider.urlopen",
        fake_urlopen,
    )

    result = UrllibJsonTransport().post_json(
        GROQ_CHAT_COMPLETIONS_URL,
        headers={"Authorization": "Bearer test-key"},
        payload={"model": "test"},
        timeout_seconds=7,
    )

    assert result == {"choices": []}
    assert captured["timeout"] == 7
    assert captured["context"].verify_mode == ssl.CERT_REQUIRED
    request = captured["request"]
    assert request.get_header("User-agent") == "InternMemory/0.1"
    assert request.get_header("Accept") == "application/json"


def test_standard_transport_classifies_tls_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_urlopen(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise URLError(
            ssl.SSLCertVerificationError(
                1,
                "certificate verify failed",
            )
        )

    monkeypatch.setattr(
        "internmemory.groq_provider.urlopen",
        fail_urlopen,
    )

    with pytest.raises(ModelProviderError, match="TLS"):
        UrllibJsonTransport().post_json(
            GROQ_CHAT_COMPLETIONS_URL,
            headers={"Authorization": "Bearer test-key"},
            payload={"model": "test"},
            timeout_seconds=7,
        )


def test_standard_transport_preserves_bounded_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_urlopen(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise HTTPError(
            GROQ_CHAT_COMPLETIONS_URL,
            403,
            "Forbidden",
            hdrs=None,
            fp=BytesIO(
                json.dumps(
                    {
                        "error": {
                            "message": (
                                "The selected model is not available."
                            )
                        }
                    }
                ).encode("utf-8")
            ),
        )

    monkeypatch.setattr(
        "internmemory.groq_provider.urlopen",
        fail_urlopen,
    )

    with pytest.raises(
        ModelProviderError,
        match="403: The selected model is not available",
    ):
        UrllibJsonTransport().post_json(
            GROQ_CHAT_COMPLETIONS_URL,
            headers={"Authorization": "Bearer test-key"},
            payload={"model": "test"},
            timeout_seconds=7,
        )
