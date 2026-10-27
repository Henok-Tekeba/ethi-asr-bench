"""SQLite connection management and versioned schema initialization."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

from ethi_asr_bench.errors import ErrorCode, StorageError

LATEST_SCHEMA_VERSION = 1
_SCHEMA_V1: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS repositories (
        repo_id TEXT PRIMARY KEY COLLATE NOCASE,
        repo_type TEXT NOT NULL,
        canonical_url TEXT NOT NULL,
        owner TEXT,
        name TEXT NOT NULL,
        first_seen_at TEXT NOT NULL,
        last_inspected_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS inspections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        repo_id TEXT NOT NULL COLLATE NOCASE,
        requested_revision TEXT,
        resolved_commit TEXT,
        retrieved_at TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('success', 'partial', 'failed')),
        schema_version TEXT NOT NULL,
        record_json TEXT NOT NULL,
        raw_json TEXT,
        provider TEXT NOT NULL,
        provider_version TEXT,
        error_code TEXT,
        error_message TEXT,
        FOREIGN KEY (repo_id) REFERENCES repositories(repo_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS ix_inspections_repo_retrieved
    ON inspections(repo_id, retrieved_at DESC, id DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS ix_inspections_commit
    ON inspections(resolved_commit)
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS ux_inspections_revision_commit
    ON inspections(repo_id, IFNULL(requested_revision, ''), resolved_commit)
    WHERE resolved_commit IS NOT NULL AND status IN ('success', 'partial')
    """,
)
_MIGRATIONS: dict[int, tuple[str, ...]] = {1: _SCHEMA_V1}


class Database:
    """Owns one SQLite connection and applies forward-only migrations."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._connection: sqlite3.Connection | None = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = self._connect()
        return self._connection

    def _connect(self) -> sqlite3.Connection:
        try:
            if str(self.path) != ":memory:":
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            connection = sqlite3.connect(str(self.path), timeout=10.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            if str(self.path) != ":memory:":
                connection.execute("PRAGMA journal_mode = WAL")
                _restrict_file_permissions(self.path)
            return connection
        except (OSError, sqlite3.Error) as exc:
            raise StorageError(
                ErrorCode.STORAGE_INITIALIZATION_FAILED,
                f"Could not open the database at {self.path}: {exc}",
            ) from exc

    def initialize(self) -> None:
        """Create or migrate the schema transactionally."""
        connection = self.connection
        try:
            row = connection.execute("PRAGMA user_version").fetchone()
            current = int(row[0]) if row is not None else 0
            if current > LATEST_SCHEMA_VERSION:
                raise StorageError(
                    ErrorCode.STORAGE_INITIALIZATION_FAILED,
                    f"Database schema version {current} is newer than this application "
                    f"supports ({LATEST_SCHEMA_VERSION}). Upgrade EthiASR Bench.",
                )
            for target in range(current + 1, LATEST_SCHEMA_VERSION + 1):
                statements = _MIGRATIONS[target]
                with connection:
                    for statement in statements:
                        connection.execute(statement)
                    connection.execute(f"PRAGMA user_version = {target}")
        except StorageError:
            raise
        except sqlite3.Error as exc:
            raise StorageError(
                ErrorCode.STORAGE_INITIALIZATION_FAILED,
                f"Could not initialize the database schema: {exc}",
            ) from exc

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> Database:
        self.initialize()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _restrict_file_permissions(path: Path) -> None:
    if sys.platform == "win32":
        return
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Permission tightening is defense in depth; a read-only or unusual
        # filesystem should not prevent the CLI from working.
        return
