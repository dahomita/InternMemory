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

## ADR-008: Store bounded source snapshots

- **Problem:** File paths can move or point to changed content, breaking
  reproducibility and citations.
- **Decision:** Store UTF-8 snapshots, SHA-256 hashes, relative locators, and
  enforce an allowlisted root and size limit.
- **Tradeoff:** SQLite stores duplicate file content, but retrieval remains
  local, reproducible, and independent of the original file.

## ADR-009: Start with deterministic chunks without overlap

- **Problem:** Whole-document ranking is imprecise, while overlap duplicates
  indexed evidence and can distort rankings.
- **Decision:** Prefer paragraph and word boundaries, preserve exact offsets,
  enforce a character bound, and begin without overlap.
- **Tradeoff:** Cross-boundary context may be missed. Evaluate neighbor
  expansion or overlap against labeled failures before adding complexity.

## ADR-010: Version retrieval ground truth before optimization

- **Problem:** Search changes cannot be justified from demonstrations alone.
- **Decision:** Keep fictional fixture documents and human-labeled evidence
  cases in Git; report Recall@5, MRR, no-answer accuracy, median latency, and
  p95 latency. Answerable and no-answer cases use separate denominators.
- **Tradeoff:** Initial metrics reflect a very small corpus and must not be
  generalized. Known failures remain visible until a measured change fixes
  them.

## ADR-011: Treat user search input as literal text

- **Problem:** Raw FTS5 syntax interpreted punctuation such as the hyphen in
  `per-request` as an operator and caused query failure.
- **Decision:** Extract word/number tokens and build a quoted AND expression.
  Advanced FTS syntax may be added later only through an explicit mode.
- **Evidence:** On the same six labeled cases, Recall@5 and MRR improved from
  `0.8333` to `1.0` with essentially unchanged median latency.
- **Tradeoff:** AND semantics can over-constrain natural-language queries and
  do not address synonyms or semantic mismatch.

## ADR-012: Preserve chunk identity in coverage-gated evidence groups

- **Problem:** Relevant evidence can span adjacent chunks, but attaching every
  neighbor adds irrelevant context and token cost.
- **Options:** Always attach neighbors, concatenate synthetic chunks, or attach
  only same-source neighbors that add missing query terms.
- **Decision:** Build explicit evidence groups and attach an immediate neighbor
  only when combined exact query-term coverage improves.
- **Tradeoff:** Citations and offsets remain accurate and context growth is
  bounded, but exact-term coverage cannot recognize semantic relationships.

## ADR-013: Add a separate grouped-evidence endpoint

- **Problem:** Grouped results contain multiple chunks and cannot satisfy the
  existing single-chunk document-search response contract.
- **Options:** Replace `/search/documents`, add a mode parameter, or add a new
  endpoint.
- **Decision:** Add `/search/evidence` and preserve `/search/documents`.
- **Tradeoff:** Two search endpoints require clear naming and documentation,
  but existing clients remain compatible and each response has one shape.

## ADR-014: Rerank evidence groups by exact-term coverage

- **Problem:** BM25 ranks anchor chunks before neighboring evidence is
  attached, so a dense partial match can outrank a complete multi-chunk group.
- **Decision:** Order groups by descending combined query-term coverage, then
  prefer fewer chunks, then preserve candidate order. Do not compare raw BM25
  scores produced by different AND and OR expressions.
- **Evidence:** On the 13-case corpus, grouped MRR improved from `0.773` to
  `0.818`; Recall@5 and measured mean precision@5 remained unchanged.
- **Tradeoff:** The rule is deterministic and token-aware, but exact coverage
  can still reward keyword-heavy text without understanding its meaning.

## ADR-015: Abstain below minimum lexical evidence coverage

- **Problem:** Broad OR fallback returned a cache document for an unsupported
  encryption query because one of four query terms matched.
- **Decision:** `/search/evidence` requires evidence groups to cover at least
  50% of unique query terms after grouping.
- **Evidence:** On the 20-case corpus, measured mean precision improved from
  `0.394` to `0.588` and no-answer accuracy from `0.667` to `1.0`; Recall@5
  decreased from `0.941` to `0.706`.
- **Tradeoff:** The endpoint now abstains safely from weak lexical evidence,
  but semantic retrieval is required to recover paraphrases and synonyms.

## ADR-016: Use FastEmbed with BGE-small as the semantic baseline

- **Problem:** Semantic retrieval needs a local, affordable embedding model
  without coupling core retrieval code to one library.
- **Options:** Sentence Transformers with MiniLM, Model2Vec static retrieval,
  or FastEmbed with `BAAI/bge-small-en-v1.5`.
- **Decision:** Add FastEmbed as an optional dependency behind an
  `EmbeddingProvider` interface and begin with BGE-small on CPU.
- **Evidence:** FastEmbed lists a quantized model size of about 67 MB; the
  model produces 384-dimensional vectors, accepts up to 512 tokens, and is
  MIT-licensed. Local query and passage smoke tests succeeded.
- **Tradeoff:** ONNX avoids PyTorch and API charges, but adds local model
  downloads, CPU work, cache management, and several package dependencies.
- **References:** [FastEmbed supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/),
  [BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5).

## ADR-017: Start semantic evaluation with brute-force in-memory search

- **Problem:** We need a trustworthy semantic baseline before committing to a
  persistent vector format or approximate-nearest-neighbor index.
- **Options:** A vector database/extension, a persistent custom index, or
  cached passage vectors scanned in memory.
- **Decision:** Embed the small fixture corpus once and rank every passage by
  cosine similarity for each query.
- **Evidence:** BGE retrieved all labeled evidence in the 20-case corpus with
  MRR `0.840`; one sample measured median query latency at `3.45 ms`.
- **Tradeoff:** The implementation is simple and exact, but search cost grows
  linearly with chunks, vectors are rebuilt per run, and no-answer queries
  require separately calibrated abstention.

## ADR-018: Separate threshold calibration from held-out evaluation

- **Problem:** Selecting a similarity threshold on the same cases used for the
  final report would leak answers into the design and overstate quality.
- **Decision:** Version deterministic `calibration`, `regression`, and fresh
  `evaluation` labels. Each split contains 9 answerable and 3 no-answer
  queries, and all search the same complete fixture corpus.
- **Tradeoff:** The split is transparent and reproducible, but 12 queries per
  side remain too few for production conclusions and should grow by category.

## ADR-019: Reject a global semantic-only abstention threshold

- **Problem:** Vector search always returns related passages, including for
  unsupported questions, so semantic retrieval needs an abstention policy.
- **Experiment:** Choose the lowest BGE similarity threshold that gives `1.0`
  no-answer accuracy on calibration. The selected threshold was `0.5934`.
- **Evidence:** Calibration Recall@5 and no-answer accuracy were both `1.0`.
  Held-out Recall@5 was `0.889`, but no-answer accuracy fell to `0.333`.
- **Decision:** Do not expose or persist this threshold. Investigate hybrid
  evidence policies and expand domain-balanced calibration data.
- **Tradeoff:** Rejecting the threshold delays semantic API integration, but
  avoids presenting topical relatedness as grounded support.

## ADR-020: Use weighted RRF as the hybrid ranking baseline

- **Problem:** BM25 scores and cosine similarities have different scales and
  directions, so adding their raw values is not meaningful.
- **Decision:** Fuse keyword and semantic rank positions with Reciprocal Rank
  Fusion. Use semantic weight `2.0` after calibration/regression comparison;
  do not expose hybrid results as grounded API evidence yet.
- **Evidence:** On the fresh 12-case held-out split, hybrid Recall@5 was `1.0`
  and MRR `0.944`, versus semantic MRR `0.870` and grounded-keyword Recall@5
  and MRR `0.667`.
- **Tradeoff:** Hybrid ranking was strongest, but no-answer accuracy remained
  `0.0`; rank fusion does not prove evidential support.

## ADR-021: Ingest public documentation through a bounded allowlist

- **Problem:** Public technical documentation is useful retrieval evidence,
  but arbitrary URL fetching introduces SSRF, resource, and untrusted-content
  risks.
- **Decision:** Permit exact allowlisted HTTPS hosts, revalidate redirects,
  enforce timeout/byte/content-type limits, extract visible text without
  executing HTML, and store a hashed local snapshot with URL provenance.
- **Evidence:** A live temporary-database test ingested Python's asyncio page
  into 4 chunks and retrieved relevant terms; network-free tests cover active
  content removal and request/redirect rejection.
- **Tradeoff:** The standard-library HTML extractor is intentionally simple
  and may include navigation text or omit complex rendered documentation.

## ADR-022: Keep grounded keyword retrieval as the Phase 2 product path

- **Problem:** Semantic and hybrid retrieval improve recall and ranking but
  returned nearest neighbors for all held-out no-answer queries.
- **Decision:** Keep `/search/evidence` on coverage-thresholded keyword
  retrieval. Retain semantic and hybrid implementations as evaluated
  baselines for future agent candidate generation, not final grounded context.
- **Tradeoff:** Product recall remains lower, but unsupported evidence is less
  likely to be presented as authoritative. A later reranker or answerability
  check can revisit the decision with fresh held-out cases.

## ADR-023: Build and verify the harness without a real model first

- **Problem:** Model variability, API cost, and tool-selection errors can hide
  defects in the execution loop itself.
- **Decision:** Define a provider protocol, test the bounded loop with scripted
  responses, validate tool inputs through a typed registry, and expose only
  the existing read-only grounded retrieval service initially.
- **Evidence:** Deterministic tests cover final answers, tool round trips, step
  limits, token accounting, invalid arguments, unknown tools, citations, and
  retrieval abstention without any model API calls.
- **Tradeoff:** This isolates harness correctness and avoids cost, but cannot
  measure whether a real model chooses tools or uses evidence reliably. Those
  behaviors require a separate agent evaluation before model selection.

## ADR-024: Require component checks for overall agent success

- **Problem:** One aggregate result can hide cases where retrieval succeeds
  but the final answer ignores evidence or invents a citation.
- **Decision:** Score completion, tool selection, evidence retrieval, answer
  grounding, citation correctness, and abstention separately. Overall success
  requires every case-specific dimension to pass.
- **Evidence:** A deterministic test retrieves the labeled source but produces
  an unsupported answer; evidence passes while grounding and overall success
  fail. Citations are checked against retrieved locator, snapshot hash, and
  chunk IDs.
- **Tradeoff:** Required-answer-term checks are transparent and repeatable but
  cannot detect every unsupported generated claim. Human review or a validated
  model judge will be needed for richer semantic grading.

## ADR-025: Keep final synthesis text-only and attach citations locally

- **Problem:** Groq JSON mode, strict JSON Schema mode, and a forced synthetic
  final-answer tool each failed intermittently when generating nested citation
  payloads, even after retrieval and evidence selection succeeded.
- **Options:** Retry structured generation, accept prompt-produced JSON, use a
  simpler plain-text synthesis boundary, or generate the entire answer
  deterministically.
- **Decision:** After one successful retrieval, give the model a clean context
  and request only concise answer text. Construct citation metadata from the
  validated retrieval observation. If retrieval returns no evidence, return a
  deterministic local abstention without a second model call.
- **Evidence:** The resulting GPT-OSS candidate passed all four development
  cases in one aggregate run with every component rate at `1.0`, using `3,503`
  input and `881` output tokens. The previous approaches produced intermittent
  HTTP 400 formatting failures.
- **Tradeoff:** The model no longer controls citation identifiers, eliminating
  malformed or invented citation metadata. However, citations currently cover
  retrieved evidence groups rather than associating each generated claim with
  a specific passage.

## ADR-026: Treat the first frozen agent split as consumed

- **Problem:** The frozen run exposed both a real paraphrase-retrieval failure
  and an exact-term grader false negative. Changing the system in response and
  repeatedly reporting against the same cases would overstate generalization.
- **Decision:** Preserve the `0.5` frozen result as the first unbiased
  measurement. Use its failures for diagnosis, but require new development
  coverage and a fresh held-out split before evaluating a revised design.
- **Evidence:** All four cases completed with correct tool selection and
  one-call efficiency. SQLite WAL and unsupported cache encryption passed; the
  p95 case failed grounding despite correct evidence and equivalent wording;
  the queue paraphrase failed evidence retrieval.
- **Tradeoff:** Refreshing evaluation data adds labeling work and makes the
  headline score less convenient, but prevents test-set tuning from being
  mistaken for a broadly better agent.

## ADR-027: Prefer rewrite-then-lexical over semantic evidence fallback

- **Problem:** Zero-overlap paraphrases abstain under coverage-gated keyword
  search. Semantic nearest neighbors can surface the right passage, but using
  them as grounded evidence without a lexical gate recreates the held-out
  no-answer failure mode measured in Phase 2.
- **Options:** Semantic candidate union with the existing query-term gate;
  similarity fallback after keyword abstention; model query rewrite then the
  same lexical evidence path.
- **Decision:** Implement A→B harness policy. Keep tools available after one
  abstaining search so the model can reformulate into document vocabulary.
  Hide tools after real evidence or a second abstention. Do not promote
  semantic similarity into the grounded `/search/evidence` product path yet.
- **Evidence:** Query-term coverage cannot pass paraphrases that share no
  tokens with the source. Similarity-only abstention previously failed held-out
  no-answer checks. Deterministic tests cover one rewrite opportunity and a
  hard stop after two abstentions.
- **Tradeoff:** Adds at most one extra search on hard paraphrases and no-answer
  cases, increasing tokens. It preserves the lexical grounding contract and
  avoids trusting topical but unsupported neighbors.

## ADR-031: Confirm note writes via a client-driven endpoint

- **Problem:** `save_note` previews must not persist until a human approves, but
  a second LLM turn is expensive and can invent or alter draft fields.
- **Decision:** Surface unresolved previews as `pending_confirmation` on
  `/agent/run`, and persist only through `POST /agent/confirm-note` with
  `confirmed=true`. The confirm path writes through `NoteRepository` without a
  model call.
- **Evidence:** Scripted HTTP tests cover preview surfacing with no DB write,
  rejection of `confirmed=false`, and successful persistence after approval.
- **Tradeoff:** Clients must echo the preview fields. A signed confirmation
  token would bind draft integrity more tightly, but adds crypto/session state
  before a UI exists. Multi-turn LLM resume remains possible later for chat UX.

## ADR-030: Expose the product agent as POST /agent/run


- **Problem:** Tools and the harness existed only in evals and unit tests, so
  the product API could not ask grounded questions over notes and documents.
- **Decision:** Assemble a default tool registry in `AgentService`, wire it in
  `create_app`, and expose `POST /agent/run`. Inject a `ModelProvider` for
  tests; optionally load Groq from `GROQ_API_KEY`. Return `503` when no
  provider is configured so local-first storage still works without an LLM.
- **Evidence:** Scripted HTTP tests verify the four-tool registry, note search,
  trace serialization, blank-message rejection, and unavailable-provider
  behavior.
- **Tradeoff:** Single-turn HTTP keeps confirmation UX for later. A session
  store would enable multi-turn `save_note` approval but adds state complexity
  before the write path is productized.

## ADR-029: Cap successful reads at two per agent run


- **Problem:** Hiding tools after the first successful read prevents questions
  that need both personal notes and document evidence. Leaving tools forever
  reintroduces duplicate-search waste.
- **Decision:** Allow up to two successful read results per run, then hide
  tools and force synthesis. Abstentions still permit one rewrite and do not
  count as successes. Confirmed writes immediately end the tool phase.
  Pending confirmation previews keep tools available.
- **Evidence:** Deterministic tests cover two successful reads followed by
  tool removal, rewrite-after-abstention, and confirmation-gated writes.
- **Tradeoff:** A model can still spend a second read poorly, but the bound
  is explicit, cheap to evaluate, and unblocks mixed notes+docs questions.

## ADR-028: Keep the refreshed frozen agent score at 0.75

- **Problem:** The new held-out paraphrase case retrieved and cited the right
  evidence, but failed grounding because the answer wrote `tail (p95) latency`
  instead of the contiguous phrase `tail latency`.
- **Decision:** Record overall held-out success as `0.75` for this candidate.
  Do not change prompts or graders and re-report a higher score on the same
  frozen cases. Future label improvements belong on a new development set and
  a later held-out refresh.
- **Evidence:** SQLite WAL, p95 observability, and unsupported cache-encryption
  passed. Batching paraphrase had `evidence_correct` and `citation_correct`
  true with primary failure `grounding`.
- **Tradeoff:** Leaves a known evaluator brittleness visible, which is better
  than silently overfitting the held-out set.
