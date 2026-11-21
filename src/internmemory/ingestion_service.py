"""Safe local Markdown and text document ingestion."""

import hashlib
from pathlib import Path

from internmemory.chunk_repository import ChunkRepository
from internmemory.chunking import chunk_text
from internmemory.models import Source, SourceType
from internmemory.source_repository import SourceRepository


SUPPORTED_SUFFIXES = {
    ".md": SourceType.MARKDOWN,
    ".markdown": SourceType.MARKDOWN,
    ".txt": SourceType.TEXT,
}


class IngestionService:
    """Read bounded files from one allowlisted root and store snapshots."""

    def __init__(
        self,
        source_repository: SourceRepository,
        chunk_repository: ChunkRepository,
        *,
        allowed_root: str | Path,
        max_bytes: int = 1_000_000,
        max_chunk_chars: int = 1_200,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be positive")

        self._source_repository = source_repository
        self._chunk_repository = chunk_repository
        self._allowed_root = Path(allowed_root).resolve()
        self._max_bytes = max_bytes
        self._max_chunk_chars = max_chunk_chars

    def ingest_file(self, relative_path: str | Path) -> Source:
        """Validate, read, hash, and persist one local document."""

        candidate = (self._allowed_root / relative_path).resolve(strict=True)
        try:
            locator = candidate.relative_to(self._allowed_root).as_posix()
        except ValueError as error:
            raise ValueError(
                "source path must remain inside the ingestion root"
            ) from error

        if not candidate.is_file():
            raise ValueError("source path must identify a regular file")

        source_type = SUPPORTED_SUFFIXES.get(candidate.suffix.lower())
        if source_type is None:
            raise ValueError("only .md, .markdown, and .txt files are allowed")

        with candidate.open("rb") as source_file:
            raw_content = source_file.read(self._max_bytes + 1)

        if len(raw_content) > self._max_bytes:
            raise ValueError(
                f"source exceeds the {self._max_bytes}-byte limit"
            )

        try:
            content = raw_content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("source must be valid UTF-8 text") from error

        if not content.strip():
            raise ValueError("source must not be empty")

        source = self._source_repository.upsert_source(
            source_type=source_type,
            locator=locator,
            title=candidate.stem,
            content=content,
            content_hash=hashlib.sha256(raw_content).hexdigest(),
            byte_size=len(raw_content),
        )
        self._chunk_repository.replace_for_source(
            source.id,
            chunk_text(content, max_chars=self._max_chunk_chars),
        )
        return source
