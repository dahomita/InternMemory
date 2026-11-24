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
- Model-independent agent domain types and provider interface
- Deterministic scripted model provider for harness testing
- Bounded agent execution with tool-call history and token accounting
- Typed tool registry with strict input validation and safe dispatch
- Read-only grounded evidence tool with citations and abstention
- In-memory per-step traces for actions, observations, usage, latency, and errors
- Scripted end-to-end conformance scenarios for abstention and call correction
- Independent step, context-character, token-usage, and tool-output budgets
- Component-level agent metrics and hash/chunk-validated citations
- Versioned eight-case agent dataset with development/evaluation splits
- Aggregate component rates and causal primary-failure classification
- Optional Groq adapter with bounded HTTPS and validated structured responses
- Development-only live Groq evaluation command using fictional fixtures

## Verification

- Latest test run: `151 passed in 2.31s`
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
- First live Groq development case completed with correct evidence, grounding,
  citation, and answer; three repeated searches used `4,814` input and `245`
  output tokens, exposing a tool-efficiency failure
- After instruction changes, the same case used two searches and `2,449` input
  tokens while preserving answer quality; task correctness now reports
  separately from ideal one-call efficiency
- The successful-tool stop policy reduced the resilience case from three
  repeated searches and `5,361` input tokens to one search and `617` input
  tokens; final synthesis then exposed Groq's JSON-mode/tool-history conflict
- Prompt-only JSON synthesis remained unreliable, so final synthesis now uses
  a clean evidence context with Groq JSON mode and strict local validation
- Resilience synthesis completed safely but abstained because the model queried
  only `retry budget`; evidence scoring now requires labeled content, not only
  the expected source locator
- GPT-OSS 120B passed the same cross-chunk case on every correctness dimension
  with one complete query, `1,312` input tokens, `508` output tokens, and about
  `1.77 s` model latency; output was verbose with minor spacing defects
- GPT-OSS also passed the batching development case with one search, `1,337`
  input tokens, `454` output tokens, and about `1.70 s` model latency; its
  visible answer was concise and clean
- On the Kubernetes no-evidence case, retrieval abstained correctly but the
  model returned a blank user-facing answer; the abstention contract now
  explicitly requires a brief non-empty explanation
- After the contract clarification, GPT-OSS passed the Kubernetes no-evidence
  case with one search, a clear abstention, `893` input tokens, `218` output
  tokens, and about `1.09 s` model latency
- GPT-OSS passed the tool-avoidance development case by searching despite the
  user instruction, then abstaining without a citation; it used `909` input
  tokens, `449` output tokens, and about `2.50 s` model latency
- Groq JSON mode, strict schema mode, and a synthetic final-answer tool all
  failed intermittently on citation-heavy synthesis
- Final synthesis now asks the model only for plain answer text; the adapter
  builds citations from validated retrieval output and handles empty-evidence
  abstention locally without a second API request
- The frozen GPT-OSS development candidate passed all four cases in one
  aggregate run: every component rate was `1.0`, with `3,503` input and `881`
  output tokens; one prior run had a transient remote disconnect
- The one-time frozen agent evaluation scored `0.5` overall across four cases:
  completion, tool selection, and tool efficiency were `1.0`; evidence and
  citation were `0.75`; grounding was `0.5`
- Frozen token usage was `3,077` input and `678` output tokens. SQLite WAL and
  unsupported cache-encryption passed
- The p95 case retrieved and cited the correct source, but the exact-term
  grounding grader rejected semantically equivalent wording, exposing an
  evaluation false negative
- The queue-backpressure paraphrase produced a literal paraphrase query,
  retrieved no evidence, and abstained; this is a genuine lexical-retrieval
  and query-formulation failure
- Agent dataset version 2 replaces exact required phrases with labeled concept
  groups: every concept remains required, but deterministic alternatives and
  punctuation-normalized forms can satisfy each concept
- Chose rewrite-then-lexical (A→B) instead of semantic evidence fallback: after
  one abstaining search, tools remain available for a single reformulation;
  tools hide after evidence or a second abstention
- Moved `queue-backpressure-paraphrase` into development and added a fresh
  held-out `batching-throughput-paraphrase` evaluation case
- Development aggregate after A→B scored `0.8` (4/5). Exact and cross-chunk
  cases stayed one-shot; no-evidence cases correctly used two searches then
  abstained. Queue paraphrase failed when the rewrite stayed conversational
  and omitted document nouns such as `backpressure`
- Strengthened reformulation instructions to prefer short systems nouns over
  restating the user paraphrase
- Post-instruction development aggregate scored `1.0` overall across five
  cases; tool efficiency was `0.6` because no-evidence cases used two searches
  before abstaining. Paraphrase passed in one search. Token use was `6,375`
  input and `1,599` output
- Refreshed frozen evaluation scored `0.75` overall (3/4). Completion, tool
  selection, evidence, citation, and abstention were `1.0`; grounding was
  `0.75`. Token use was `5,600` input and `1,194` output
- The only frozen miss was `batching-throughput-paraphrase`: rewrite found
  evidence, but the answer wrote `tail (p95) latency`, so the contiguous
  phrase grader rejected `tail latency`. Treat this split as consumed
- Added `search_notes`, `get_daily_summary`, and confirmation-gated `save_note`
  tools. Preview writes keep tools available until `confirmed=true`
- Multi-tool policy now allows up to two successful reads before forcing
  synthesis; confirmed writes still end the tool phase immediately
- Product `POST /agent/run` wires the full tool registry through `AgentService`;
  Groq loads from the environment when present, otherwise the endpoint is `503`
- `pending_confirmation` on agent runs plus `POST /agent/confirm-note` for
  client-driven `save_note` approval without a second model call
- Minimal chat UI at `/` for ask, pending-note confirmation, and recent notes
- `GET /agent/status` reports provider readiness without spending tokens
- Phase 4 (no MCP): `fetch_public_page` + confirm-gated `ingest_public_page`
- Richer UI: create/delete notes and allowlisted URL ingest
- `POST /agent/confirm-ingest` for client-driven public-page approval

## Current Task

Phase 4 tool/UI work without MCP is in place. Optional next: MCP adapter,
durable redacted traces, or broader allowlists after measurement.

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
- Chat UI has no multi-turn session store or signed confirmation tokens
- MCP adapter is intentionally deferred
- Agent traces are in memory only and have no redacted durable export
- Character context estimates are tokenizer-independent but not exact
- Token-usage limits are reactive until a real adapter applies output limits
- Deterministic answer-term checks cannot detect every unsupported claim
- The current harness supports one sequential tool call per model step
- Successful reads are capped at two per run; richer multi-hop planning still
  needs an explicit evidence-completeness policy
- Final synthesis citations currently cover retrieved evidence groups rather
  than mapping each individual answer claim to one passage
- A transient Groq disconnect is reported as an execution failure; automatic
  retries are intentionally absent until recurrence justifies duplicate-call
  and rate-limit risk

## Next Steps

1. Optional: MCP adapter over the existing tool contracts.
2. Optional: durable redacted agent-trace export.
3. Optional: signed confirmation tokens for draft integrity.
