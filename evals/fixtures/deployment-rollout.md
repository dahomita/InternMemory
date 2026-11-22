# Deployment Rollout

A canary rollout sends a small fraction of production traffic to a new
version before broad deployment. Engineers compare error rate and latency
against the stable version and stop the rollout when guardrails regress.

An automated rollback restores the last known-good version when those
guardrails fail. Rollback automation should be idempotent and preserve
diagnostic evidence from the failed release.
