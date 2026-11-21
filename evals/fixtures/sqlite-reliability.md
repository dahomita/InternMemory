# SQLite Reliability

SQLite WAL mode allows readers to continue while a writer commits changes.
Applications should still keep transactions short.

A busy timeout can reduce immediate failures during temporary lock contention.
Bounded retries are safer than retrying database operations forever.
