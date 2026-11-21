# Progress

## Current Status

Phase 1 has a working local functional baseline. Phase 2 retrieval work now
includes:

- FastAPI application factory and health endpoint
- SQLite projects, notes, and tasks
- Project creation/listing
- Complete note CRUD and date filtering
- FTS5/BM25 note search
- Structured unfinished-task tracking
- Deterministic morning recap
- Safe Markdown and text ingestion
- Deterministic no-overlap document chunks
- FTS5/BM25 chunk search with citation metadata
- Versioned retrieval cases and repeatable quality/latency reporting

## Verification

- Last full test run: `42 passed in 0.95s`
- IDE diagnostics: no lint errors
- Local SQLite version tested: `3.40.1` with FTS5 support
- Literal-query baseline: Recall@5 `1.0`, MRR `1.0`
- Prior raw-syntax baseline: Recall@5 `0.8333`, MRR `0.8333`
- Sample local latency: median `0.366 ms`, p95 `0.455 ms`

## Current Task

Expand evaluation coverage beyond exact keyword matches.

## Known Limitations

- No versioned database migration system
- FTS5 index rebuilds on each application startup
- Default timezone is configured in code
- Recaps return structured evidence but do not generate prose
- Source and chunk replacement do not yet share one transaction
- All query terms currently use AND semantics, which can over-constrain prose
- Retrieval corpus is too small for production-quality conclusions
- No custom frontend
- No LLM provider or agent harness

## Next Steps

1. Expand the corpus with prose, synonyms, and chunk-boundary failures.
2. Classify failures caused by tokenization versus missing semantics.
3. Compare chunk sizes and neighbor expansion.
4. Introduce semantic search only if measurements justify it.
