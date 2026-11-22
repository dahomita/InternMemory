# Cache Reliability

A cache stampede occurs when many workers recompute the same expired value at
once. Request coalescing allows one worker to refresh the value while the
others wait for its result. A randomized time-to-live can spread expirations
across time instead of creating one synchronized burst.

Stale-while-revalidate can serve a bounded stale value during refresh when the
product permits it. Metrics should separate cache hits, misses, refreshes, and
refresh failures so operators can identify whether the cache is protecting or
overloading its dependency.
