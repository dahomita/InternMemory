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
