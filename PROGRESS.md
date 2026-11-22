# Progress

## Current Status

Phases 1 and 2 are complete. InternMemory now includes:

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
- Local BGE semantic retrieval and weighted-RRF hybrid baselines
- Allowlisted public documentation ingestion

## Verification

- Final Phase 2 test run: `69 passed in 1.64s`
- IDE diagnostics: no lint errors
- Local SQLite version tested: `3.40.1` with FTS5 support
- Retrieval corpus: 36 queries in balanced calibration, regression, and fresh
  evaluation splits; each split has 9 answerable and 3 no-answer cases
- Fresh held-out grounded keyword: Recall@5 `0.667`, MRR `0.667`, measured
  precision@5 `0.611`, no-answer accuracy `1.0`
- Fresh held-out semantic: Recall@5 `1.0`, MRR `0.870`, measured precision@5
  `0.200`, no-answer accuracy `0.0`
- Fresh held-out weighted hybrid: Recall@5 `1.0`, MRR `0.944`, measured
  precision@5 `0.200`, no-answer accuracy `0.0`
- Sample median/p95 query latency: keyword `1.16/3.18 ms`, semantic
  `4.17/4.94 ms`, hybrid `5.26/6.41 ms`
- Similarity-only threshold calibration was rejected after held-out no-answer
  accuracy fell to `0.333`
- Live public-ingestion verification captured Python's asyncio documentation
  into 4 chunks and returned 3 relevant keyword results

## Current Task

Phase 2 complete. Phase 3 agent-harness work has not started.

## Known Limitations

- No versioned database migration system
- FTS5 index rebuilds on each application startup
- Default timezone is configured in code
- Recaps return structured evidence but do not generate prose
- Source and chunk replacement do not yet share one transaction
- Production search still defaults to AND semantics, which can over-constrain
  prose; the evaluation runner can compare any-term OR candidates
- Retrieval corpus is too small for production-quality conclusions
- Semantic vectors are rebuilt in memory and brute-force scanned
- Hybrid retrieval has no reliable answerability/abstention policy and is not
  exposed as grounded API evidence
- Public HTML extraction is intentionally simple and limited to configured
  exact domains
- No custom frontend
- No LLM provider or agent harness

## Next Steps

1. Review and commit the completed Phase 2 milestone.
2. Define Phase 3 model and tool interfaces.
3. Build a bounded model-independent agent loop.
4. Expose grounded retrieval as a validated read-only agent tool.
