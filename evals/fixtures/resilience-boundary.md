# Resilience Design

The request path uses a circuit breaker to stop repeated calls when a
dependency is unavailable. The breaker records recent failures, opens after a
configured threshold, and allows a small number of probes before closing.
Timeouts remain bounded so one slow dependency cannot occupy every worker.
Failures are returned as explicit errors instead of being hidden behind an
unbounded wait. The design also records state transitions for debugging and
keeps recovery behavior deterministic during tests. Operators can distinguish
an open breaker from a network timeout by inspecting structured events.
Fallback behavior is intentionally limited because stale data may be worse
than a visible failure. These safeguards protect capacity while the dependency
recovers and prevent a local incident from becoming a cascading outage.

A retry budget limits repeated attempts after transient failures. Each request
may consume only a small fixed number of retries, with exponential backoff and
jitter between attempts. When the budget is exhausted, the operation fails
visibly so callers can decide whether to defer the work. Retry outcomes are
recorded separately from breaker state so operators can identify which
mechanism stopped an operation and tune each policy independently.
