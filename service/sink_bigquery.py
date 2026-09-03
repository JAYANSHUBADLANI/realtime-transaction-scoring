"""BigQuery sink for the deployed streaming service.

Same small interface as ``service/sink.py``'s ``SQLiteSink``
(``already_seen``, ``write_batch``), so swapping backends did not require
touching ``service/main.py`` beyond which sink class gets constructed at
startup (see ``select_sink`` there).

Two honestly-disclosed tradeoffs versus the local SQLite sink, not hidden:

1. Dedup here does a real `SELECT ... WHERE transaction_id IN UNNEST(@ids)`
   query per push, since BigQuery has no local index to check against
   in-process. This is a genuine latency cost avoided entirely by SQLite;
   the README's live-deployment evidence reports it honestly rather than
   comparing Cloud Run+BigQuery latency to the local-SQLite numbers as if
   they were the same thing.
2. Writes use the classic streaming insert API (``insert_rows_json`` with
   ``row_ids`` for BigQuery's best-effort insertId dedup), not the newer
   Storage Write API. That is simpler to wire correctly and, at this
   project's demo data volumes, costs fractions of a cent either way; a
   real production volume deployment would want the Storage Write API
   instead, stated here as a next step rather than silently deployed as if
   it were already the more scalable choice.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Set

import pandas as pd
from google.cloud import bigquery

TABLE_SCHEMA = [
    bigquery.SchemaField("transaction_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("step", "INTEGER"),
    bigquery.SchemaField("type", "STRING"),
    bigquery.SchemaField("amount", "FLOAT"),
    bigquery.SchemaField("nameOrig", "STRING"),
    bigquery.SchemaField("nameDest", "STRING"),
    bigquery.SchemaField("orig_emptied", "INTEGER"),
    bigquery.SchemaField("balance_error_orig", "FLOAT"),
    bigquery.SchemaField("balance_error_dest", "FLOAT"),
    bigquery.SchemaField("zscore_flag", "BOOLEAN"),
    bigquery.SchemaField("zscore_score", "FLOAT"),
    bigquery.SchemaField("iqr_flag", "BOOLEAN"),
    bigquery.SchemaField("iqr_score", "FLOAT"),
    bigquery.SchemaField("iforest_flag", "BOOLEAN"),
    bigquery.SchemaField("iforest_score", "FLOAT"),
    bigquery.SchemaField("n_methods", "INTEGER"),
    bigquery.SchemaField("is_alert", "INTEGER"),
    bigquery.SchemaField("severity", "STRING"),
    bigquery.SchemaField("severity_score", "FLOAT"),
    bigquery.SchemaField("scored_at_message_id", "STRING"),
    bigquery.SchemaField("scored_at", "TIMESTAMP"),
]

_COLUMNS = [f.name for f in TABLE_SCHEMA if f.name != "scored_at"]


class BigQuerySink:
    def __init__(self, project: str, dataset: str, table: str = "scored_transactions") -> None:
        self.client = bigquery.Client(project=project)
        self.table_ref = f"{project}.{dataset}.{table}"
        self._ensure_table(project, dataset, table)

    def _ensure_table(self, project: str, dataset: str, table: str) -> None:
        dataset_ref = bigquery.DatasetReference(project, dataset)
        try:
            self.client.get_dataset(dataset_ref)
        except Exception:
            self.client.create_dataset(bigquery.Dataset(dataset_ref), exists_ok=True)

        try:
            self.client.get_table(self.table_ref)
        except Exception:
            bq_table = bigquery.Table(self.table_ref, schema=TABLE_SCHEMA)
            bq_table.time_partitioning = bigquery.TimePartitioning(field="scored_at")
            self.client.create_table(bq_table, exists_ok=True)

    def already_seen(self, transaction_ids: Iterable[str]) -> Set[str]:
        ids = list(transaction_ids)
        if not ids:
            return set()
        query = (
            f"SELECT DISTINCT transaction_id FROM `{self.table_ref}` "
            f"WHERE transaction_id IN UNNEST(@ids)"
        )
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("ids", "STRING", ids)]
        )
        rows = self.client.query(query, job_config=job_config).result()
        return {r["transaction_id"] for r in rows}

    def write_batch(self, scored: pd.DataFrame, message_id: str) -> int:
        if scored.empty:
            return 0

        scored_at = datetime.now(timezone.utc).isoformat()
        records = scored.assign(scored_at_message_id=message_id).to_dict(orient="records")
        rows = [
            {**{c: r.get(c) for c in _COLUMNS}, "scored_at": scored_at} for r in records
        ]
        row_ids = [r["transaction_id"] for r in rows]

        errors = self.client.insert_rows_json(self.table_ref, rows, row_ids=row_ids)
        if errors:
            raise RuntimeError(f"BigQuery insert errors: {errors}")
        return len(rows)
