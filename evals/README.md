# Retrieval Evaluations

This directory contains fictional source documents and human-labeled retrieval
queries. It is safe to commit and must not contain personal or employer data.

Run the baseline:

```bash
python -m internmemory.evaluation
```

Each case identifies a query, the relevant source, and required evidence text.
A retrieved chunk counts as relevant only when it comes from that source and
contains the normalized required text.

The current corpus is intentionally small. Its purpose is to establish a
reproducible evaluation pipeline before expanding coverage. Do not tune search
only to these examples; add representative failures as new labeled cases.
