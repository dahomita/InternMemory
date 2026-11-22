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
