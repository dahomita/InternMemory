# InternMemory

## Vision

InternMemory is a local-first engineering knowledge engine for recording,
retrieving, and reasoning over projects, notes, decisions, and unfinished work.
It must remain useful without an LLM.

## Current Architecture

```text
FastAPI
├── Minimal chat UI (/) over agent + notes APIs
├── Pydantic request/response validation
├── AgentService + POST /agent/run + confirm-note
│   └── Tool registry → notes, evidence, recap, confirmed save
├── Local and allowlisted-public ingestion services
├── Retrieval services
│   ├── SQLite FTS5/BM25
│   ├── Local BGE semantic baseline
│   └── Reciprocal-rank-fusion hybrid baseline
├── Repository layer
└── SQLite source-of-truth tables and FTS5 index
```

The agent harness calls application capabilities through explicit tools.
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
- Strict-first retrieval with broad fallback and coverage-gated evidence groups
- Optional model-independent embedding provider with local CPU inference
- Exact in-memory semantic retrieval and comparable evaluation reporting
- Reciprocal-rank-fusion hybrid retrieval evaluation
- Allowlisted HTTPS documentation ingestion with bounded fetches
- Versioned fictional retrieval corpus and labeled evaluation runner
- Temporary-database integration tests
- Bounded agent harness with typed tools and optional Groq provider
- Product `POST /agent/run` over the full tool registry
- Client-driven `POST /agent/confirm-note` for confirmation-gated writes
- Minimal same-origin chat UI for ask + confirm-note flows

## Requirements

- Local-first operation with a target early cost of $0–$5 per month
- Typed Python and explicit validation
- Useful behavior without an LLM
- No secrets, personal notes, or confidential employer data in Git
- Confirmation before future agent-initiated writes
- Measured quality, latency, token usage, and cost where relevant

## Roadmap

1. Personal memory foundation — complete
2. Document ingestion and retrieval evaluation — complete
3. Model-independent agent harness and tools — complete
3.5 Minimal chat + confirmation UI — complete
4. Controlled browser research and reliability work

## Security Boundary

Development uses fictional or sanitized notes and public documentation.
Using employer information requires explicit authorization and approved
infrastructure; a local database alone does not establish compliance.
