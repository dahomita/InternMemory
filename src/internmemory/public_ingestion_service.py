"""Controlled ingestion of allowlisted public documentation."""

import hashlib
import ssl
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import certifi

from internmemory.chunk_repository import ChunkRepository
from internmemory.chunking import chunk_text
from internmemory.models import Source, SourceType
from internmemory.source_repository import SourceRepository


SUPPORTED_CONTENT_TYPES = {
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "text/markdown",
}


@dataclass(frozen=True, slots=True)
class FetchedDocument:
    """Bounded bytes and metadata returned by an HTTP fetcher."""

    final_url: str
    content_type: str
    body: bytes


@dataclass(frozen=True, slots=True)
class PublicPageContent:
    """Sanitized visible text from one allowlisted public page."""

    url: str
    title: str
    content: str
    content_type: str


class DocumentFetcher(Protocol):
    """Fetch bounded bytes without coupling ingestion to one HTTP client."""

    def fetch(
        self,
        url: str,
        *,
        max_bytes: int,
        timeout_seconds: float,
    ) -> FetchedDocument:
        """Return one fetched public document."""


class UrlLibDocumentFetcher:
    """Fetch public documentation with bounded reads and a timeout."""

    def fetch(
        self,
        url: str,
        *,
        max_bytes: int,
        timeout_seconds: float,
    ) -> FetchedDocument:
        """Fetch one URL without executing page content."""

        request = Request(
            url,
            headers={"User-Agent": "InternMemory/0.1 documentation-ingester"},
        )
        tls_context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(
            request,
            timeout=timeout_seconds,
            context=tls_context,
        ) as response:
            body = response.read(max_bytes + 1)
            content_type = response.headers.get_content_type()
            final_url = response.geturl()
        return FetchedDocument(
            final_url=final_url,
            content_type=content_type,
            body=body,
        )


class PublicDocumentIngestionService:
    """Fetch, sanitize, snapshot, and chunk allowlisted public pages."""

    def __init__(
        self,
        source_repository: SourceRepository,
        chunk_repository: ChunkRepository,
        *,
        allowed_domains: set[str],
        max_bytes: int = 1_000_000,
        max_chunk_chars: int = 1_200,
        timeout_seconds: float = 10.0,
        fetcher: DocumentFetcher | None = None,
    ) -> None:
        if not allowed_domains:
            raise ValueError("at least one public domain must be allowed")
        if max_bytes <= 0 or max_chunk_chars <= 0:
            raise ValueError("size limits must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._source_repository = source_repository
        self._chunk_repository = chunk_repository
        self._allowed_domains = {
            domain.casefold().rstrip(".")
            for domain in allowed_domains
        }
        self._max_bytes = max_bytes
        self._max_chunk_chars = max_chunk_chars
        self._timeout_seconds = timeout_seconds
        self._fetcher = fetcher or UrlLibDocumentFetcher()

    def fetch_page(self, url: str) -> PublicPageContent:
        """Fetch and sanitize one page without writing to storage."""

        self._validate_url(url)
        fetched = self._fetcher.fetch(
            url,
            max_bytes=self._max_bytes,
            timeout_seconds=self._timeout_seconds,
        )
        self._validate_url(fetched.final_url)
        if len(fetched.body) > self._max_bytes:
            raise ValueError(
                f"source exceeds the {self._max_bytes}-byte limit"
            )

        content_type = fetched.content_type.casefold().split(";", 1)[0]
        if content_type not in SUPPORTED_CONTENT_TYPES:
            raise ValueError("unsupported public document content type")
        try:
            decoded = fetched.body.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                "public document must be valid UTF-8"
            ) from error

        if content_type in {"text/html", "application/xhtml+xml"}:
            extractor = _VisibleTextExtractor()
            extractor.feed(decoded)
            title = extractor.title or _title_from_url(fetched.final_url)
            content = extractor.text
        else:
            title = _title_from_url(fetched.final_url)
            content = decoded.strip()
        if not content:
            raise ValueError("public document contains no visible text")

        return PublicPageContent(
            url=fetched.final_url,
            title=title,
            content=content,
            content_type=content_type,
        )

    def ingest_page(self, page: PublicPageContent) -> Source:
        """Persist one already-fetched sanitized page into searchable storage."""

        content_bytes = page.content.encode("utf-8")
        source = self._source_repository.upsert_source(
            source_type=SourceType.TEXT,
            locator=page.url,
            title=page.title,
            content=page.content,
            content_hash=hashlib.sha256(content_bytes).hexdigest(),
            byte_size=len(content_bytes),
        )
        self._chunk_repository.replace_for_source(
            source.id,
            chunk_text(page.content, max_chars=self._max_chunk_chars),
        )
        return source

    def ingest_url(self, url: str) -> Source:
        """Ingest one HTTPS document after validating request and redirect."""

        return self.ingest_page(self.fetch_page(url))

    def _validate_url(self, url: str) -> None:
        """Require HTTPS and an exact allowlisted hostname."""

        parsed = urlparse(url)
        hostname = (parsed.hostname or "").casefold().rstrip(".")
        if (
            parsed.scheme != "https"
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
        ):
            raise ValueError("public URL must use standard HTTPS")
        if hostname not in self._allowed_domains:
            raise ValueError("public URL domain is not allowlisted")


class _VisibleTextExtractor(HTMLParser):
    """Extract visible text without executing or retaining active content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
        elif tag == "title" and self._ignored_depth == 0:
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        stripped = " ".join(data.split())
        if not stripped:
            return
        if self._in_title:
            self._title_parts.append(stripped)
        else:
            self._text_parts.append(stripped)

    @property
    def title(self) -> str:
        """Return normalized title text."""

        return " ".join(self._title_parts)

    @property
    def text(self) -> str:
        """Return normalized visible page text."""

        return "\n".join(self._text_parts)


def _title_from_url(url: str) -> str:
    """Derive a stable nonblank title from an allowed URL."""

    parsed = urlparse(url)
    path_name = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    return path_name or parsed.hostname or "public-document"
