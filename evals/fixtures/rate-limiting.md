# Rate Limiting

A token bucket allows short traffic bursts while enforcing a sustained request
rate. Tokens accumulate up to a fixed capacity and each accepted request
consumes one token. When the bucket is empty, the service rejects or delays
work until capacity becomes available.

Rate-limit responses should identify when clients may retry. Randomized client
backoff prevents every rejected caller from retrying at the same instant.
