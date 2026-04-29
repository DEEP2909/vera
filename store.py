import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KNOWN_SCOPES = {"category", "merchant", "customer", "trigger"}


class ContextStore:
    """SQLite-backed context and conversation store.

    Context rows are keyed as "{scope}:{context_id}" so the schema remains the
    compact form requested by the challenge while still supporting counts and
    lookups by context layer.
    """

    def __init__(self, db_path: str | Path = "vera_context.db") -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS contexts (
                    key TEXT PRIMARY KEY,
                    version INT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS suppressions (
                    key TEXT PRIMARY KEY,
                    sent_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conv_id TEXT PRIMARY KEY,
                    merchant_id TEXT,
                    customer_id TEXT,
                    trigger_id TEXT,
                    history TEXT NOT NULL
                )
                """
            )
            columns = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(conversations)").fetchall()
            }
            if "customer_id" not in columns:
                try:
                    self._conn.execute("ALTER TABLE conversations ADD COLUMN customer_id TEXT")
                except sqlite3.OperationalError as exc:
                    if "duplicate column name" not in str(exc).lower():
                        raise

    @staticmethod
    def _context_key(scope: str, context_id: str) -> str:
        scope = (scope or "").strip().lower()
        if not scope:
            raise ValueError("scope is required")
        if not context_id:
            raise ValueError("context_id is required")
        return f"{scope}:{context_id}"

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def upsert(
        self, scope: str, context_id: str, version: int, payload: dict[str, Any]
    ) -> tuple[bool, int]:
        """Insert or replace a context only when the incoming version is newer."""

        key = self._context_key(scope, context_id)
        encoded = json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT version FROM contexts WHERE key = ?", (key,)
            ).fetchone()
            if row is not None:
                current_version = int(row["version"])
                if version <= current_version:
                    return False, current_version

            self._conn.execute(
                """
                INSERT INTO contexts (key, version, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    version = excluded.version,
                    payload = excluded.payload
                """,
                (key, int(version), encoded),
            )
            return True, int(version)

    def get(self, scope: str, context_id: str) -> dict[str, Any] | None:
        key = self._context_key(scope, context_id)
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM contexts WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload"])

    def get_with_version(
        self, scope: str, context_id: str
    ) -> tuple[dict[str, Any] | None, int | None]:
        key = self._context_key(scope, context_id)
        with self._lock:
            row = self._conn.execute(
                "SELECT version, payload FROM contexts WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None, None
        return json.loads(row["payload"]), int(row["version"])

    def list_contexts(self, scope: str) -> list[tuple[str, dict[str, Any], int]]:
        prefix = f"{scope.strip().lower()}:"
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, version, payload FROM contexts WHERE key LIKE ?",
                (f"{prefix}%",),
            ).fetchall()
        contexts: list[tuple[str, dict[str, Any], int]] = []
        for row in rows:
            context_id = row["key"][len(prefix) :]
            contexts.append((context_id, json.loads(row["payload"]), int(row["version"])))
        return contexts

    def counts(self) -> dict[str, int]:
        counts = {scope: 0 for scope in sorted(KNOWN_SCOPES)}
        with self._lock:
            rows = self._conn.execute("SELECT key FROM contexts").fetchall()
        for row in rows:
            scope = str(row["key"]).split(":", 1)[0]
            counts[scope] = counts.get(scope, 0) + 1
        return counts

    def mark_suppressed(self, key: str) -> None:
        if not key:
            return
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO suppressions (key, sent_at)
                VALUES (?, ?)
                """,
                (key, self._utc_now()),
            )

    def is_suppressed(self, key: str) -> bool:
        if not key:
            return False
        try:
            ttl_seconds = int(os.getenv("SUPPRESSION_TTL_SECONDS", "86400"))
        except ValueError:
            ttl_seconds = 86400
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT sent_at FROM suppressions WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return False
            if ttl_seconds <= 0:
                return True
            try:
                sent_at = datetime.fromisoformat(str(row["sent_at"]).replace("Z", "+00:00"))
            except ValueError:
                return True
            if sent_at.tzinfo is None:
                sent_at = sent_at.replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - sent_at).total_seconds() > ttl_seconds:
                self._conn.execute("DELETE FROM suppressions WHERE key = ?", (key,))
                return False
            return True

    def create_conversation(
        self,
        conv_id: str,
        merchant_id: str | None,
        trigger_id: str | None,
        customer_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        first_message: dict[str, Any] | None = None,
    ) -> None:
        history: list[dict[str, Any]] = []
        if metadata:
            history.append(
                {
                    "role": "system",
                    "type": "metadata",
                    "at": self._utc_now(),
                    "metadata": metadata,
                }
            )
        if first_message:
            history.append(first_message)
        encoded = json.dumps(history, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO conversations
                    (conv_id, merchant_id, customer_id, trigger_id, history)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conv_id, merchant_id, customer_id, trigger_id, encoded),
            )

    def get_conversation(self, conv_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT conv_id, merchant_id, customer_id, trigger_id, history
                FROM conversations
                WHERE conv_id = ?
                """,
                (conv_id,),
            ).fetchone()
        if row is None:
            return None

        history = json.loads(row["history"] or "[]")
        metadata: dict[str, Any] = {}
        if history and history[0].get("type") == "metadata":
            metadata = dict(history[0].get("metadata") or {})
        return {
            "conv_id": row["conv_id"],
            "merchant_id": row["merchant_id"],
            "customer_id": row["customer_id"],
            "trigger_id": row["trigger_id"],
            "history": history,
            "metadata": metadata,
        }

    def append_history(
        self,
        conv_id: str,
        entry: dict[str, Any],
        merchant_id: str | None = None,
        customer_id: str | None = None,
        trigger_id: str | None = None,
    ) -> None:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT history, merchant_id, customer_id, trigger_id FROM conversations WHERE conv_id = ?",
                (conv_id,),
            ).fetchone()
            if row is None:
                history = [entry]
                self._conn.execute(
                    """
                    INSERT INTO conversations
                        (conv_id, merchant_id, customer_id, trigger_id, history)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        conv_id,
                        merchant_id,
                        customer_id,
                        trigger_id,
                        json.dumps(history, ensure_ascii=False, separators=(",", ":")),
                    ),
                )
                return

            history = json.loads(row["history"] or "[]")
            history.append(entry)
            next_merchant_id = merchant_id or row["merchant_id"]
            next_customer_id = customer_id or row["customer_id"]
            next_trigger_id = trigger_id or row["trigger_id"]
            self._conn.execute(
                """
                UPDATE conversations
                SET merchant_id = ?, customer_id = ?, trigger_id = ?, history = ?
                WHERE conv_id = ?
                """,
                (
                    next_merchant_id,
                    next_customer_id,
                    next_trigger_id,
                    json.dumps(history, ensure_ascii=False, separators=(",", ":")),
                    conv_id,
                ),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
