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

- Last full test run: `51 passed in 1.13s`
- IDE diagnostics: no lint errors
- Local SQLite version tested: `3.40.1` with FTS5 support
- Expanded 13-case strict baseline: Recall@5 `0.727`, MRR `0.682`,
  measured mean precision@5 `0.682`
- No-answer accuracy: `1.0` on 2 cases
- Exact/punctuation cases: all retrieved; one distractor moved relevant evidence
  from rank 1 to rank 2
- Synonym/abbreviation cases: 0 of 2 retrieved
- Cross-chunk case: 0 of 1 retrieved because strict AND found no candidate
- Any-term candidate experiment: Recall@5 `0.909`, MRR `0.773`; measured mean
  precision@5 fell to `0.500`
- Prioritized fallback preserved the same measured quality as OR on this
  corpus while executing two FTS queries
- Coverage-gated neighbor grouping combines cross-boundary evidence while
  preserving each chunk's citation offsets
- Grouped evaluation: Recall@5 `0.909`, MRR `0.773`, measured mean precision@5
  `0.515`
- Coverage reranking raised grouped MRR to `0.818` without changing measured
  recall or precision; `/search/evidence` now uses this ordering
- Batched source-chunk loading removes per-candidate reads; one sample reduced
  grouped median latency from `1.37 ms` to `0.99 ms`
- `/search/evidence` exposes grouped chunks without changing the existing
  `/search/documents` response contract
- Prior raw-syntax baseline: Recall@5 `0.8333`, MRR `0.8333`
- Sample expanded-run latency: median `0.290 ms`, p95 `0.346 ms`

## Current Task

Expand evaluation coverage before introducing local semantic embeddings.

## Known Limitations

- No versioned database migration system
- FTS5 index rebuilds on each application startup
- Default timezone is configured in code
- Recaps return structured evidence but do not generate prose
- Source and chunk replacement do not yet share one transaction
- Production search still defaults to AND semantics, which can over-constrain
  prose; the evaluation runner can compare any-term OR candidates
- Retrieval corpus is too small for production-quality conclusions
- No custom frontend
- No LLM provider or agent harness

## Next Steps

1. Add representative public-document and terminology cases.
2. Classify exact-keyword failures separately from semantic failures.
3. Select a small local embedding model and document resource costs.
4. Compare keyword, semantic, and hybrid retrieval on the fixed dataset.
