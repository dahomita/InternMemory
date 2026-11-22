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

- Last full test run: `53 passed in 1.21s`
- IDE diagnostics: no lint errors
- Local SQLite version tested: `3.40.1` with FTS5 support
- Expanded 20-case strict baseline: Recall@5 `0.647`, MRR `0.618`,
  measured mean precision@5 `0.618`
- Strict no-answer accuracy: `1.0` on 3 cases
- Exact/punctuation cases: all retrieved; one distractor moved relevant evidence
  from rank 1 to rank 2
- Synonym/abbreviation cases: 0 of 2 retrieved
- Semantic-paraphrase cases: 0 of 3 retrieved by strict AND
- Cross-chunk case: 0 of 1 retrieved because strict AND found no candidate
- Any-term candidate experiment: Recall@5 `0.941`, MRR `0.796`; measured mean
  precision@5 fell to `0.383`, no-answer accuracy to `0.667`
- Prioritized fallback preserved the same measured quality as OR on this
  corpus while executing two FTS queries
- Coverage-gated neighbor grouping combines cross-boundary evidence while
  preserving each chunk's citation offsets
- Grouped evaluation: Recall@5 `0.941`, MRR `0.799`, measured mean precision@5
  `0.394`
- Coverage-reranked grouping: Recall@5 `0.941`, MRR `0.828`, measured mean
  precision@5 `0.394`, no-answer accuracy `0.667`
- The partial-overlap no-answer case exposes false evidence from broad OR
  fallback
- A 50% coverage acceptance threshold raises measured precision to `0.588`
  and no-answer accuracy to `1.0`, while recall falls to `0.706`;
  `/search/evidence` uses this threshold after coverage reranking
- Batched source-chunk loading removes per-candidate reads; one sample reduced
  grouped median latency from `1.37 ms` to `0.99 ms`
- `/search/evidence` exposes grouped chunks without changing the existing
  `/search/documents` response contract
- Prior raw-syntax baseline: Recall@5 `0.8333`, MRR `0.8333`
- Sample expanded-run latency: median `0.290 ms`, p95 `0.346 ms`

## Current Task

Research a small local embedding baseline to recover semantic failures.

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

1. Select a small local embedding model and document resource costs.
2. Add representative public-document cases.
3. Implement semantic retrieval behind an explicit interface.
4. Compare keyword, semantic, and hybrid retrieval on the fixed dataset.
