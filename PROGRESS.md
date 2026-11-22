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

- Last full test run: `59 passed in 1.33s`
- IDE diagnostics: no lint errors
- Local SQLite version tested: `3.40.1` with FTS5 support
- Expanded 24-case strict baseline: Recall@5 `0.667`, MRR `0.639`,
  measured mean precision@5 `0.639`
- Strict no-answer accuracy: `1.0` on 6 cases
- Exact/punctuation cases: all retrieved; one distractor moved relevant evidence
  from rank 1 to rank 2
- Synonym/abbreviation cases: 0 of 2 retrieved
- Semantic-paraphrase cases: 0 of 3 retrieved by strict AND
- Cross-chunk case: 0 of 1 retrieved because strict AND found no candidate
- Any-term candidate experiment: Recall@5 `0.944`, MRR `0.807`; measured mean
  precision@5 fell to `0.381`, no-answer accuracy to `0.333`
- Prioritized fallback preserved the same measured quality as OR on this
  corpus while executing two FTS queries
- Coverage-gated neighbor grouping combines cross-boundary evidence while
  preserving each chunk's citation offsets
- Grouped evaluation: Recall@5 `0.944`, MRR `0.810`, measured mean precision@5
  `0.391`
- Coverage-reranked grouping: Recall@5 `0.944`, MRR `0.838`, measured mean
  precision@5 `0.391`, no-answer accuracy `0.333`
- The partial-overlap no-answer case exposes false evidence from broad OR
  fallback
- A 50% coverage acceptance threshold raises measured precision to `0.611`
  and no-answer accuracy to `1.0`, while recall falls to `0.722`;
  `/search/evidence` uses this threshold after coverage reranking
- Batched source-chunk loading removes per-candidate reads; one sample reduced
  grouped median latency from `1.37 ms` to `0.99 ms`
- `/search/evidence` exposes grouped chunks without changing the existing
  `/search/documents` response contract
- Optional FastEmbed adapter verified with local
  `BAAI/bge-small-en-v1.5` query and passage embeddings (384 dimensions)
- Calibration and held-out evaluation splits each contain 9 answerable and 3
  no-answer cases while sharing the complete corpus
- Untuned BGE Recall@5 is `1.0` on both splits; calibration MRR is `0.870` and
  held-out MRR is `0.828`; no-answer accuracy is `0.0` on both
- Prior raw-syntax baseline: Recall@5 `0.8333`, MRR `0.8333`
- Sample expanded-run latency: median `0.290 ms`, p95 `0.346 ms`

## Current Task

Calibrate semantic abstention using only the calibration split.

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

1. Separate calibration and final-evaluation cases.
2. Calibrate semantic abstention on the calibration split.
3. Add representative public-document cases.
4. Compare keyword, semantic, and hybrid retrieval on held-out cases.
