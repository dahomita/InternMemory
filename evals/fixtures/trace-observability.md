# Trace Observability

A distributed trace follows one request across service boundaries. Trace
context carries a trace identifier and parent span identifier so independently
recorded spans can be assembled into the same request path.

Percentile latency exposes slow requests that an arithmetic mean can hide. The
95th percentile, often written p95, is the latency at or below which 95 percent
of observations fall. Traces and metrics answer different questions and are
most useful when engineers can correlate them.
