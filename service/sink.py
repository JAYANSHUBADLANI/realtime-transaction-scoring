"""Local SQLite sink for scored transactions and alerts.

This is the Phase 3 (local, GCP-free) storage backend, used to prove the
service's scoring and idempotency logic end to end before any real cloud
infrastructure is involved. The next build stage swaps this for a BigQuery
sink; the interface (``write_batch``, ``already_seen``) is small enough
that swap should not touch ``service/main.py``.

Idempotency matters here specifically because Pub/Sub's push delivery is
at-least-once, not exactly-once: the same message can legitimately arrive
twice. ``processed_transactions`` is keyed on ``transaction_id`` so a
redelivered batch does not double-count or double-score a transaction that
was already written.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Iterable, Set

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS scored_transactions (
    transaction_id TEXT PRIMARY KEY,
    step INTEGER,
    type TEXT,
    amount REAL,
    nameOrig TEXT,
    nameDest TEXT,
    orig_emptied INTEGER,
    balance_error_orig REAL,
    balance_error_dest REAL,
    zscore_flag INTEGER,
    zscore_score REAL,
    iqr_flag INTEGER,
    iqr_score REAL,
    iforest_flag INTEGER,
    iforest_score REAL,
    n_methods INTEGER,
    is_alert INTEGER,
    severity TEXT,
    severity_score REAL,
    scored_at_message_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_scored_is_alert
    ON scored_transactions (is_alert, severity_score DESC);
"""


class SQLiteSink:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def already_seen(self, transaction_ids: Iterable[str]) -> Set[str]:
        ids = list(transaction_ids)
        if not ids:
            return set()
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT transaction_id FROM scored_transactions "
                f"WHERE transaction_id IN ({placeholders})",
                ids,
            ).fetchall()
        return {r[0] for r in rows}

    def write_batch(self, scored: pd.DataFrame, message_id: str) -> int:
        """Insert new rows, silently skipping any transaction_id already
        present (belt-and-suspenders alongside the caller's own
        already_seen() pre-filter, so a race between two concurrent
        deliveries still cannot double-write).

        Returns the number of rows actually inserted.
        """
        if scored.empty:
            return 0

        records = scored.assign(scored_at_message_id=message_id).to_dict(
            orient="records"
        )
        columns = [
            "transaction_id",
            "step",
            "type",
            "amount",
            "nameOrig",
            "nameDest",
            "orig_emptied",
            "balance_error_orig",
            "balance_error_dest",
            "zscore_flag",
            "zscore_score",
            "iqr_flag",
            "iqr_score",
            "iforest_flag",
            "iforest_score",
            "n_methods",
            "is_alert",
            "severity",
            "severity_score",
            "scored_at_message_id",
        ]
        placeholders = ",".join("?" for _ in columns)
        sql = (
            f"INSERT OR IGNORE INTO scored_transactions ({','.join(columns)}) "
            f"VALUES ({placeholders})"
        )
        with self._connect() as conn:
            cur = conn.executemany(
                sql, [tuple(r.get(c) for c in columns) for r in records]
            )
            return cur.rowcount if cur.rowcount is not None else len(records)

    def read_all(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM scored_transactions", conn)

    def alert_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM scored_transactions WHERE is_alert = 1"
            ).fetchone()
        return int(row[0]) if row else 0
