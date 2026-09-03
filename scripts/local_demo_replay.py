"""Local, no-GCP-dependency demonstration of the streaming service, run
against a live ``uvicorn`` process and the real reference artifact
``python main.py`` produced (the actual 463,152-row fit, not a synthetic
test fixture).

This directly POSTs Pub/Sub-push-shaped envelopes to a locally running
server, standing in for what real Pub/Sub delivery will do once this is
deployed. The service code (``service/main.py``) does not know or care
which one sent the request; the envelope contract is identical either way.

Usage:
    # terminal 1
    uvicorn service.main:app --port 8000

    # terminal 2
    python scripts/local_demo_replay.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICE_URL = os.environ.get("SERVICE_URL", "http://127.0.0.1:8000")
BATCH_SIZE = 10
N_TRANSACTIONS = int(os.environ.get("N_TRANSACTIONS", "2000"))

RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
]


def load_replay_rows(n: int) -> pd.DataFrame:
    """Real held-out transactions (step > the reference window main.py
    used), with the label columns stripped, exactly as a live feed would
    look. Sourced from the raw PaySim CSV, not reports/scored_transactions.csv,
    so this script exercises the service's own feature engineering rather
    than replaying values main.py already computed.
    """
    csv_path = os.path.join(PROJECT_ROOT, "data", "paysim_transactions.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run scripts/download_data.py first."
        )
    df = pd.read_csv(csv_path)
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
    df = df[df["step"] > 96].head(n).reset_index(drop=True)
    df = df[RAW_COLUMNS].copy()
    df["transaction_id"] = [f"demo-{i}" for i in range(len(df))]
    return df


def envelope_for(batch_rows: pd.DataFrame, message_id: str) -> dict:
    payload = {"transactions": batch_rows.to_dict(orient="records")}
    data = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return {"message": {"data": data, "messageId": message_id}}


def main() -> None:
    resp = requests.get(f"{SERVICE_URL}/health", timeout=5)
    resp.raise_for_status()
    print(f"Service healthy: {resp.json()}")

    rows = load_replay_rows(N_TRANSACTIONS)
    print(f"Replaying {len(rows):,} real held-out transactions in batches of {BATCH_SIZE}...")

    latencies_ms = []
    total_scored = 0
    total_alerts = 0
    start = time.perf_counter()

    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows.iloc[i : i + BATCH_SIZE]
        envelope = envelope_for(chunk, message_id=f"demo-msg-{i}")

        t0 = time.perf_counter()
        r = requests.post(f"{SERVICE_URL}/push", json=envelope, timeout=30)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        r.raise_for_status()
        body = r.json()
        total_scored += body["transactions_scored"]
        total_alerts += body["alerts_raised"]

    elapsed = time.perf_counter() - start
    latencies_ms.sort()
    n = len(latencies_ms)

    def pct(p: float) -> float:
        return latencies_ms[min(n - 1, int(n * p))]

    print(f"\n{total_scored:,} transactions scored across {n} requests in {elapsed:.1f}s"
          f" ({total_scored / elapsed:.1f} txn/s)")
    print(f"Alerts raised: {total_alerts:,} ({total_alerts / total_scored * 100:.1f}% of scored)")
    print(f"Per-request latency (batch of {BATCH_SIZE}): "
          f"mean={sum(latencies_ms)/n:.1f}ms  p50={pct(0.5):.1f}ms  "
          f"p95={pct(0.95):.1f}ms  p99={pct(0.99):.1f}ms")


if __name__ == "__main__":
    sys.exit(main())
