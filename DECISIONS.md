# Architecture Decisions

## ADR-001: Local-first Python backend

- **Problem:** Build a low-cost, inspectable personal knowledge system.
- **Options:** Cloud services, Django, FastAPI with SQLite.
- **Decision:** FastAPI, SQLite, and standard-library `sqlite3`.
- **Tradeoff:** Minimal cost and transparent behavior, but fewer ORM and
  migration conveniences.

## ADR-002: Separate API, service, and repository layers

- **Problem:** HTTP, orchestration, and SQL can become tightly coupled.
- **Decision:** Pydantic schemas validate HTTP data; application services
  orchestrate; repositories own persistence.
- **Tradeoff:** More files and interfaces, but reusable capabilities for future
  agent tools and easier isolated testing.

## ADR-003: Preserve project relationships

- **Problem:** Deleting a project could remove or detach engineering history.
- **Decision:** Project foreign keys use `ON DELETE RESTRICT`.
- **Tradeoff:** Deletion requires resolving dependent records; future project
  archiving will be safer than destructive deletion.

## ADR-004: FTS5/BM25 before embeddings

- **Problem:** Notes require useful retrieval without paid APIs.
- **Options:** `LIKE`, FTS5, or semantic embeddings.
- **Decision:** Use FTS5 candidate matching and BM25 ranking with title weight.
- **Tradeoff:** Fast, free, and explainable, but keyword search misses semantic
  equivalents. Embeddings require evaluation evidence before adoption.

## ADR-005: Database-triggered search synchronization

- **Problem:** Application code could forget to update the derived FTS index.
- **Decision:** SQLite insert, update, and delete triggers synchronize FTS5.
- **Tradeoff:** Stronger consistency, but schema evolution is more involved.

## ADR-006: Fixed task-state enum

- **Problem:** A completed boolean cannot represent blocked engineering work.
- **Decision:** Use `open`, `in_progress`, `blocked`, and `completed`, enforced
  by Python enums and database checks.
- **Tradeoff:** More state handling, but clearer workflows and recaps.

## ADR-007: Timezone-aware deterministic recaps

- **Problem:** Server-clock “yesterday” can select the wrong work date.
- **Decision:** Configure an IANA timezone, allow explicit date overrides, and
  inject the clock in tests.
- **Tradeoff:** Additional configuration, but accurate and reproducible dates.
