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
- Embedding models map queries and passages into a shared vector space so
  semantic similarity can recover different wording. Query and passage
  encoding may be asymmetric and should remain explicit in the provider
  interface.
- Local ONNX inference avoids API cost and PyTorch, but still introduces model
  downloads, dependencies, CPU latency, vector storage, and model-version
  reproducibility concerns.
- Brute-force semantic retrieval embeds passages once, embeds each query, and
  compares it with every cached passage vector using cosine similarity. This
  is transparent for a small corpus but scales linearly with chunk count.
- Vector search always has a nearest neighbor even when no answer exists.
  Semantic retrieval therefore needs separately calibrated abstention rather
  than treating the top result as automatically relevant.
- Threshold selection is a form of fitting even when model weights remain
  unchanged. Category-balanced calibration and held-out query splits reduce
  evaluation leakage and ordering bias while sharing the same search corpus.
- A global semantic-similarity threshold may separate calibration examples yet
  fail on unseen domains. Embedding similarity captures relatedness, not
  whether retrieved text supports every requested detail.
- Once held-out failures influence a new design, that split is no longer
  untouched evidence for the revised system; refresh held-out cases before
  claiming the next generalization result.
- Reciprocal Rank Fusion combines result positions rather than incompatible
  BM25 and cosine scores. Agreement across retrievers raises a candidate's
  fused score, while optional weights express measured trust in each ranking.
- Hybrid retrieval improved held-out ranking quality, but ranking and
  abstention are separate problems: semantic and hybrid systems still returned
  evidence for every unsupported query.
- Safe public-document ingestion requires exact domain allowlists, HTTPS,
  redirect revalidation, timeouts, byte bounds, content-type checks, visible
  text extraction, and treating all downloaded content as untrusted data.
- An agent harness owns execution policy around a model: state, tools, limits,
  validation, termination, and observability. The model proposes actions; it
  does not learn permanently from one harness run.
- A provider protocol separates agent behavior from a specific model vendor.
  A deterministic scripted provider makes harness failures reproducible and
  testable without API cost or model variability.
- Immutable model requests preserve exact context snapshots for debugging and
  prevent later message mutations from rewriting execution history.
- Tool schemas constrain structure, types, and bounds before dispatch, while a
  registry restricts calls to explicitly available tools. Schema validation
  does not sanitize valid strings; parameterized SQL and safe query handling
  remain necessary downstream defenses.
- Read-only tools reduce blast radius while validating model behavior. Reusing
  the application retrieval service keeps API and agent grounding consistent;
  explicit abstention prevents an empty result from being mistaken for proof.
- A useful agent trace separates model input/action, tool arguments/output,
  timing, token usage, and errors. Good retrieved evidence followed by an
  unsupported answer indicates a model-grounding failure; missing evidence can
  instead indicate retrieval or tool-query failure.
- Traces can contain sensitive retrieved content. Keeping initial traces in
  memory avoids accidental external logging; durable traces later require
  explicit retention, access, and redaction policies.
- Scripted-model scenarios prove deterministic harness behavior such as
  bounded correction and abstention propagation. They do not measure whether
  a real model will select the right tool, form a useful query, follow
  evidence, or behave consistently across repeated runs.
- A step limit bounds model decisions, not context or token cost. Context grows
  across steps, and one call or tool result can still be very large, so useful
  harnesses need independent step, context, token, and tool-output budgets.
- Character counts provide a model-independent context estimate but do not
  equal tokens. Provider-reported token limits are also reactive; real model
  adapters should enforce maximum output tokens before generation.
- End-to-end task success should require correct tool use, evidence, grounding,
  citations, and abstention behavior. Reporting component metrics separately
  reveals whether retrieval or answer generation caused the failure.
- A citation is verifiable when it matches the locator, source snapshot hash,
  and retrieved chunk IDs. Required-answer-term checks are a reproducible
  baseline, not proof that every generated claim is supported.
- A no-evidence prompt and an answer-without-evidence failure are different:
  the first describes missing knowledge, while the second describes an agent
  that answered without grounding. Agent datasets should label expected
  behavior and classify the observed failure separately.
- Development cases can guide iteration, while a frozen evaluation split gives
  less-biased evidence for the final prompt/model configuration.
- Component rates identify which subsystem needs improvement. A causal primary
  failure order prevents one upstream miss from being counted again as several
  downstream root causes; for example, missing evidence precedes grounding and
  citation failures.
- A hosted provider adapter translates internal messages and tool schemas into
  an external API contract, then validates every returned action before the
  harness executes it. Timeouts, response-size limits, output-token caps, and
  environment-only credentials bound cost and failure impact.
- Free hosted inference still transmits selected evidence off-device. Zero
  Data Retention and no-training terms reduce provider retention risk but do
  not authorize sending confidential employer data.
- Live evaluations should be explicit commands, use fictional fixtures, and
  exclude frozen cases during iteration. Keeping network calls out of unit
  tests preserves deterministic, free local verification.
- A sub-second "request failed or timed out" result with zero provider tokens
  is not a model-quality failure. Distinguishing TLS, DNS, network, timeout,
  and HTTP errors prevents infrastructure defects from contaminating agent
  evaluation conclusions.
- Correct tool choice and efficient tool use are separate metrics. Repeating
  the right search can still produce a grounded answer while wasting model
  steps, context tokens, latency, and free-tier capacity.
- End-to-end correctness should not fail solely because a bounded corrective
  search was needed. Report efficiency separately so optimization pressure
  does not encourage the agent to answer from weak first-pass evidence.
- Provider capabilities can conflict: Groq JSON mode cannot be combined with
  tool calling for this model. Harness instructions plus strict local parsing
  preserve safety, but schema adherence becomes an evaluated model behavior.
- Repeated identical successful calls are a harness-policy problem when the
  model remains free to choose tools forever. Removing tool availability after
  a successful bounded result forces synthesis, while retaining tools after a
  validation error still permits correction.
- Rate-limit retries require backoff and a retry budget. Re-running an entire
  agent task immediately repeats earlier calls and can worsen token pressure;
  no automatic retry is safer during initial evaluation.
- Provider feature constraints can depend on conversation history, not only
  the current request's tool list. Groq JSON generation failed after prior tool
  messages even when tools were hidden on the final turn, so the adapter must
  validate instruction-produced JSON locally for tool-using conversations.
- Separating tool selection from final synthesis can simplify provider
  constraints. A clean synthesis context includes the original question and
  retrieved evidence as untrusted data, enabling JSON syntax enforcement
  without carrying incompatible tool-protocol history. The harness still
  validates answer fields and citations locally.
- Stronger constrained decoding is not automatically more reliable. GPT-OSS
  formed the complete search query but Groq strict-schema generation failed;
  JSON Object mode plus local validation is the more portable boundary.
- Retrieving the expected source does not prove that the necessary passage was
  retrieved. Evidence evaluation must verify required labeled content across
  returned chunks; otherwise a narrow query can receive false credit.
- Query formulation is part of agent quality. A lexical search query must
  preserve all essential concepts from the user question while removing
  conversational filler.
- Model selection should use component evidence, not general reputation.
  GPT-OSS preserved both cross-chunk concepts where Qwen omitted one, producing
  complete retrieval and a grounded answer under the same harness and dataset.
- Correctness, latency, tokens, and presentation quality are distinct. A
  grounded answer can pass task metrics while still being too verbose or
  containing formatting defects worth optimizing later.
- Structured syntax does not guarantee valid product semantics. JSON mode can
  produce a schema-shaped abstention with a blank answer, so local domain
  validation and explicit user-facing requirements remain necessary.
- Provider-enforced structure is not always the most reliable boundary. After
  JSON mode, strict schema mode, and a synthetic final-answer tool all failed
  intermittently, reducing the model's final responsibility to plain answer
  text removed fragile nested citation generation.
- Citation metadata can be constructed from the exact validated retrieval
  result rather than recopied by the model. This improves format reliability
  and provenance integrity, but broad evidence-group citations do not yet
  prove that every individual generated claim is supported.
- Empty retrieval is a deterministic state, so the harness can return a safe
  local abstention without paying for another model call or allowing the model
  to invent an unsupported answer.
- A transient remote disconnect is an infrastructure failure, not evidence of
  poor reasoning. Retries should be bounded and added only with explicit
  consideration of duplicate requests, token cost, and rate limits.
- Held-out component metrics distinguish architecture failures from grader
  failures. The p95 answer retrieved and cited the right evidence but failed an
  exact phrase check despite equivalent meaning; the backpressure case failed
  earlier because lexical retrieval received none of the source terminology.
- Exact required-term grading is reproducible but can produce false negatives
  on valid paraphrases. A stronger semantic or human-reviewed grader must still
  remain evidence-aware so fluent unsupported answers do not receive credit.
- A model cannot reliably translate every paraphrase into unknown document
  vocabulary. Semantic candidate generation can address this recall gap, but
  earlier no-answer experiments show that semantic similarity alone is not
  sufficient evidence for answering.
- After examining frozen failures, that split is consumed for the current
  design. It can support diagnosis, but measuring the next design fairly
  requires new held-out cases that did not influence the change.
- Deterministic concept groups are a middle ground between brittle exact
  phrases and an expensive model judge. Every labeled concept is still
  required, while human-approved alternatives and punctuation normalization
  allow equivalent expressions. Their quality still depends on complete,
  carefully reviewed labels.

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
- Frozen agent evaluation and broader model reliability measurement
- Context budgets and evidence grounding
- Tracing, model routing, token accounting, and cost evaluation
- Browser automation and MCP
