# Batching Tradeoffs

Batching combines multiple requests into one processing operation. Larger
batches can improve throughput by amortizing per-request overhead.

Waiting for a batch to fill can increase tail latency. A benchmark should
compare throughput with p50 and p95 latency rather than reporting only average
response time.
