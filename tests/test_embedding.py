"""Tests for the model-independent embedding adapter."""

import sys
from types import SimpleNamespace

import pytest

from internmemory.embedding import FastEmbedProvider


class FakeTextEmbedding:
    """Minimal FastEmbed stand-in with deterministic vectors."""

    def __init__(self, **_: object) -> None:
        pass

    def query_embed(self, query: str) -> list[list[float]]:
        return [[float(len(query)), 1.0]]

    def passage_embed(
        self,
        passages: list[str],
    ) -> list[list[float]]:
        return [
            [float(len(passage)), 2.0]
            for passage in passages
        ]


def test_fastembed_adapter_separates_query_and_passage_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter preserves asymmetric retrieval model behavior."""

    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        SimpleNamespace(TextEmbedding=FakeTextEmbedding),
    )
    provider = FastEmbedProvider(
        model_name="fictional-model",
        dimensions=2,
    )

    assert provider.embed_query("query") == (5.0, 1.0)
    assert provider.embed_passages(["one", "three"]) == [
        (3.0, 2.0),
        (5.0, 2.0),
    ]
    assert provider.dimensions == 2


def test_fastembed_adapter_rejects_blank_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blank text cannot become ambiguous semantic evidence."""

    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        SimpleNamespace(TextEmbedding=FakeTextEmbedding),
    )
    provider = FastEmbedProvider(
        model_name="fictional-model",
        dimensions=2,
    )

    with pytest.raises(ValueError, match="query must not be blank"):
        provider.embed_query(" ")
    with pytest.raises(ValueError, match="passages must not be blank"):
        provider.embed_passages(["valid", ""])
