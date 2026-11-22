# Queue Backpressure

A bounded queue applies backpressure when producers outpace consumers. The
producer must wait, reject work, or shed load instead of allowing memory usage
to grow without limit. Queue depth and time spent waiting reveal sustained
capacity problems earlier than process memory exhaustion.

Load shedding protects completed work by rejecting lower-priority requests
before every worker becomes saturated. Retry behavior must remain bounded
because immediate retries can amplify the overload.
