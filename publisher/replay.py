"""Real Google Cloud Pub/Sub publisher: replays held-out PaySim transactions
onto an actual Pub/Sub topic, batched, exactly as a live feed would arrive.

This is the true end-to-end path, distinct from
``scripts/local_demo_replay.py`` (which POSTs directly to a locally running
FastAPI process, standing in for Pub/Sub during local development without
needing any GCP infrastructure). Here, Pub/Sub itself is what delivers each
batch to the deployed Cloud Run push subscriber; this script never talks to
Cloud Run directly.

Usage (after infra/setup.sh has created the topic and deployed the service):
    python publisher/replay.py --project <your-gcp-project-id> \\
        --topic transaction-events --n 2000
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import pandas as pd
from google.cloud import pubsub_v1

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH_SIZE = 10

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


def load_replay_rows(n: int, reference_step_max: int) -> pd.DataFrame:
    csv_path = os.path.join(PROJECT_ROOT, "data", "paysim_transactions.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run scripts/download_data.py first."
        )
    df = pd.read_csv(csv_path)
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
    df = df[df["step"] > reference_step_max].head(n).reset_index(drop=True)
    df = df[RAW_COLUMNS].copy()
    df["transaction_id"] = [f"pubsub-demo-{i}" for i in range(len(df))]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--topic", default="transaction-events")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--reference-step-max", type=int, default=96)
    args = parser.parse_args()

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(args.project, args.topic)

    rows = load_replay_rows(args.n, args.reference_step_max)
    print(f"Publishing {len(rows):,} real held-out transactions to {topic_path}, "
          f"batched {BATCH_SIZE} per message...")

    start = time.perf_counter()
    futures = []
    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows.iloc[i : i + BATCH_SIZE]
        payload = {"transactions": chunk.to_dict(orient="records")}
        data = json.dumps(payload).encode("utf-8")
        futures.append(publisher.publish(topic_path, data))

    # Block until every publish is actually acknowledged by the Pub/Sub
    # service, not just handed to the client library's local buffer, so
    # the elapsed time reported below reflects real publish latency.
    for f in futures:
        f.result()

    elapsed = time.perf_counter() - start
    print(f"Published {len(futures)} messages ({len(rows):,} transactions) "
          f"in {elapsed:.1f}s ({len(rows) / elapsed:.1f} txn/s).")
    print("Pub/Sub delivers these to the deployed Cloud Run service "
          "asynchronously; check BigQuery or the service logs shortly after.")


if __name__ == "__main__":
    sys.exit(main())
