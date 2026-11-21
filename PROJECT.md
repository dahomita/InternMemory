# InternMemory

## Vision

InternMemory is a local-first engineering knowledge engine for recording,
retrieving, and reasoning over projects, notes, decisions, and unfinished work.
It must remain useful without an LLM.

## Current Architecture

```text
FastAPI
├── Pydantic request/response validation
├── Application services
├── Repository layer
├── SQLite source-of-truth tables
└── SQLite FTS5 index with BM25 ranking
```

The future agent harness will call these capabilities through explicit tools.
Storage and retrieval remain independent of model providers.

## Current Capabilities

- Project creation and listing
- Note CRUD and date filtering
- FTS5 keyword search with BM25 ranking
- Structured task states and unfinished-task retrieval
- Timezone-aware deterministic morning recaps
- Bounded Markdown and text source ingestion with SHA-256 snapshots
- Deterministic source chunking with exact citation offsets
- FTS5/BM25 document-chunk retrieval with source citations
- Versioned fictional retrieval corpus and labeled evaluation runner
- Temporary-database integration tests

## Requirements

- Local-first operation with a target early cost of $0–$5 per month
- Typed Python and explicit validation
- Useful behavior without an LLM
- No secrets, personal notes, or confidential employer data in Git
- Confirmation before future agent-initiated writes
- Measured quality, latency, token usage, and cost where relevant

## Roadmap

1. Personal memory foundation
2. Document ingestion and retrieval evaluation
3. Model-independent agent harness and tools
4. Controlled browser research and reliability work

## Security Boundary

Development uses fictional or sanitized notes and public documentation.
Using employer information requires explicit authorization and approved
infrastructure; a local database alone does not establish compliance.
