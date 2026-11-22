# Learning Log

## Concepts Learned

- FastAPI maps typed Python functions to validated HTTP endpoints.
- Pydantic performs runtime validation and serialization from type hints.
- Repository classes isolate persistence SQL from HTTP and application logic.
- Parameterized SQL keeps user values separate from executable SQL structure.
- Foreign keys preserve relationships; `RESTRICT` prevents destructive deletion.
- `PATCH` distinguishes omitted fields from explicit empty or null values.
- An inverted index maps terms to documents instead of scanning every document.
- FTS5 provides local full-text indexing and query matching.
- BM25 ranks matches using term rarity, frequency, and length normalization.
- SQLite FTS5 uses lower BM25 scores for stronger matches.
- Database triggers keep derived search indexes synchronized.
- Fixed task states provide more information than a completed boolean.
- Application services orchestrate repositories without coupling them to HTTP.
- Timezone-aware clocks make date behavior accurate and testable.
- Source snapshots and hashes preserve provenance and detect content revisions.
- Chunking changes retrieval granularity rather than guaranteeing faster search.
- BM25 ranks chunks against other chunks; it does not rank text within a chunk.
- Chunk boundaries trade precision against missing neighboring context.
- Exact offsets let a retrieved chunk cite and reconstruct its source passage.
- Recall@5 measures whether relevant evidence appears in the top five but not
  whether it ranks first.
- MRR rewards systems that place the first relevant result earlier.
- Raw FTS5 syntax can turn punctuation such as hyphens into query operators.
- Literal query normalization separates user text from FTS5 query syntax and
  should be evaluated against the same fixed ground-truth cases.
- A lower score after adding harder cases is broader measurement, not a system
  regression; comparisons require the same dataset.
- Keyword retrieval cannot infer synonyms or expanded abbreviations without
  explicit query expansion, indexed terminology, or semantic retrieval.
- Recall can improve by returning more candidates while precision and context
  quality worsen; no-answer cases require separate measurement.
- BM25 can favor a short keyword-dense distractor over a longer source with
  stronger supporting evidence.
- Strict AND retrieval fails when required query terms are split across chunks.
  Neighbor expansion cannot help unless candidate generation first finds an
  anchor chunk.
- Candidate generation usually favors recall so later stages have evidence to
  rerank. Precision measures the relevant fraction of returned results;
  latency measures retrieval speed.
- Any-term OR matching improved measured recall but reduced measured
  precision, demonstrating why broad candidates need strict-match priority or
  reranking before they become final context.
- A two-query strict-then-broad fallback makes ordering explicit but increases
  latency; aggregate retrieval metrics may remain unchanged when BM25 already
  orders strict matches first.
- Combined query-term coverage measures the unique query terms present across
  grouped chunks. Coverage-gated neighbors add context only when they
  contribute missing exact terms.
- N+1 queries occur when one initial retrieval is followed by a separate
  database read per result. Correct grouping improved measured precision, but
  its first implementation exposed this latency cost.
- Batch-loading related rows with one parameterized `IN` query bounds database
  round trips as candidate count grows. Dynamically generated `?` placeholders
  are safe when user values remain separate bound parameters.
- Group-level reranking can improve result order without changing recall.
  Combined exact-term coverage promoted complete cross-chunk evidence in the
  controlled case, but keyword-stuffed text remains a known limitation.
- A small evaluation set can hide failure modes and encourage overfitting.
  Adding paraphrases and partial-overlap no-answer cases exposed both semantic
  misses and broad-fallback false positives.
- Grounded retrieval should abstain when evidence is too weak. A lexical
  coverage threshold improved precision and no-answer behavior but sacrificed
  paraphrase recall, making the semantic-search requirement measurable.

## Clarifications and Corrections

- A health regression test is different from a load or stress test.
- Python f-strings do not have C `printf` format-string vulnerabilities; raw
  SQL interpolation is dangerous because input can become SQL syntax.
- `404` means a singular resource does not exist, not that the server failed
  to receive the request.
- Omit means leave a field out; clear changes its value; delete removes a
  resource.
- Current backend capabilities support a future harness but are not themselves
  the model-driven agent execution loop.

## Still to Study

- Larger, representative retrieval datasets and failure classification
- Chunk overlap, neighbor expansion, and parent-child retrieval
- Semantic embeddings and hybrid retrieval
- Agent tool schemas, execution loops, and termination
- Context budgets and evidence grounding
- Tracing, model routing, token accounting, and cost evaluation
- Browser automation and MCP
