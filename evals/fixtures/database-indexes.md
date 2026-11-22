# Database Indexes

A covering index contains every column needed by a query, allowing the
database to answer it without reading the base table. This can reduce random
I/O for frequent read paths.

Every additional index increases storage and adds work to inserts and updates.
Index design therefore balances read latency against write amplification
rather than indexing every available column.
