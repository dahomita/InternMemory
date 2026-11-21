"""SQLite connection and schema management."""

import sqlite3
from pathlib import Path


DATABASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL COLLATE NOCASE UNIQUE
        CHECK (length(trim(name)) BETWEEN 1 AND 100),
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY,
    project_id INTEGER,
    title TEXT NOT NULL DEFAULT ''
        CHECK (length(title) <= 200),
    content TEXT NOT NULL
        CHECK (length(trim(content)) > 0),
    work_date TEXT NOT NULL
        CHECK (work_date GLOB
            '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    created_at TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS notes_project_id_idx ON notes(project_id);
CREATE INDEX IF NOT EXISTS notes_work_date_idx ON notes(work_date);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY,
    project_id INTEGER,
    description TEXT NOT NULL
        CHECK (length(trim(description)) > 0),
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'in_progress', 'blocked', 'completed')),
    due_date TEXT
        CHECK (
            due_date IS NULL OR due_date GLOB
            '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
        ),
    completed_at TEXT,
    created_at TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE RESTRICT,
    CHECK (
        (status = 'completed' AND completed_at IS NOT NULL)
        OR (status != 'completed' AND completed_at IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS tasks_project_id_idx ON tasks(project_id);
CREATE INDEX IF NOT EXISTS tasks_status_idx ON tasks(status);
CREATE INDEX IF NOT EXISTS tasks_due_date_idx ON tasks(due_date);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    title,
    content,
    content='notes',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS notes_fts_insert
AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, title, content)
    VALUES (new.id, new.title, new.content);
END;

CREATE TRIGGER IF NOT EXISTS notes_fts_delete
AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content)
    VALUES ('delete', old.id, old.title, old.content);
END;

CREATE TRIGGER IF NOT EXISTS notes_fts_update
AFTER UPDATE OF title, content ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content)
    VALUES ('delete', old.id, old.title, old.content);
    INSERT INTO notes_fts(rowid, title, content)
    VALUES (new.id, new.title, new.content);
END;
"""


def connect_database(path: str | Path) -> sqlite3.Connection:
    """Open a configured SQLite connection."""

    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(path: str | Path) -> None:
    """Create the current database schema if it does not exist."""

    connection = connect_database(path)
    try:
        with connection:
            connection.executescript(DATABASE_SCHEMA)
            connection.execute(
                "INSERT INTO notes_fts(notes_fts) VALUES ('rebuild')"
            )
    finally:
        connection.close()
