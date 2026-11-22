# Retrieval Evaluations

This directory contains fictional source documents and human-labeled retrieval
queries. It is safe to commit and must not contain personal or employer data.

Run the baseline:

```bash
python -m internmemory.evaluation
```

Compare broad any-term candidate generation:

```bash
python -m internmemory.evaluation --strategy any
```

Measure strict matches followed by broad fallback candidates:

```bash
python -m internmemory.evaluation --strategy prioritized
```

Measure coverage-gated adjacent evidence groups:

```bash
python -m internmemory.evaluation --strategy grouped
```

Compare group-level exact-term coverage reranking:

```bash
python -m internmemory.evaluation --strategy grouped_reranked
```

Measure grounded keyword retrieval with a 50% coverage threshold:

```bash
python -m internmemory.evaluation \
  --strategy grouped_reranked_thresholded
```

Each case identifies a failure category, query, relevant source, and required
evidence text. A retrieved chunk counts as relevant only when it comes from
that source and contains the normalized required text. A grouped result counts
as relevant when any of its member chunks meets those conditions.

Mean precision at `k` averages the labeled relevant fraction of results
actually returned for each answerable query, up to `k`. The dataset labels a
required passage rather than every passage a human might find useful, so this
is measured precision against the current labels.

No-answer cases omit relevance labels and set `expected_no_answer` to `true`.
They measure whether retrieval correctly returns no chunks for unsupported
queries; they are excluded from Recall@k and MRR.

The current corpus is intentionally small. Its purpose is to establish a
reproducible evaluation pipeline before expanding coverage. Do not tune search
only to these examples; add representative failures as new labeled cases.
