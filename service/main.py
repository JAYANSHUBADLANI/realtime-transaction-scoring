"""FastAPI push subscriber for real-time transaction scoring.

Run locally:
    uvicorn service.main:app --reload

Expects Google Cloud Pub/Sub's push subscription contract on ``POST /push``
(https://cloud.google.com/pubsub/docs/push): a JSON envelope with a
base64-encoded ``message.data`` field, which decodes to a JSON batch of
transactions (see ``service/schemas.py``). This is exactly what a real
push subscription will send once this service is deployed behind one; no
code here changes between local testing and that deployment, only which
process is sending the HTTP request (a test client / a local replay script
here, the managed Pub/Sub service after deployment).

Loads the frozen reference artifact from ``artifacts/`` (produced by
``python main.py``) once at startup and reuses it, unchanged, for every
request: this is the streaming side of the fit-once/score-many split in
``src/reference.py``.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from service.schemas import PubSubPushEnvelope, ScoreResponse
from service.scoring import score_transaction_batch
from service.sink import SQLiteSink
from src.load_data import PROJECT_ROOT, load_config
from src.reference import ReferenceBundle

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DB_PATH = os.environ.get(
    "SCORING_DB_PATH", os.path.join(PROJECT_ROOT, "reports", "streaming_scores.db")
)
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_PROJECT = os.environ.get("BQ_PROJECT")  # falls back to GOOGLE_CLOUD_PROJECT

state: Dict = {}


def select_sink():
    """BigQuery when the deployment sets BQ_DATASET (Cloud Run, Phase 4);
    local SQLite otherwise (unmodified local dev and tests, Phase 3). Kept
    as one function so main.py's route handlers never need to know which
    backend is active; both implement the same already_seen/write_batch
    interface.
    """
    if BQ_DATASET:
        from service.sink_bigquery import BigQuerySink

        project = BQ_PROJECT or os.environ.get("GOOGLE_CLOUD_PROJECT")
        return BigQuerySink(project=project, dataset=BQ_DATASET)
    return SQLiteSink(DB_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["config"] = load_config()
    state["reference"] = ReferenceBundle.load(ARTIFACTS_DIR)
    state["sink"] = select_sink()
    yield
    state.clear()


app = FastAPI(title="Real-Time Transaction Scoring", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    reference: ReferenceBundle = state["reference"]
    return {
        "status": "ok",
        "reference_fitted_on_rows": reference.fitted_on_rows,
        "reference_step_max": reference.reference_step_max,
    }


@app.post("/push", response_model=ScoreResponse)
def push(envelope: PubSubPushEnvelope) -> ScoreResponse:
    try:
        batch = envelope.message.decode_batch()
    except (ValueError, ValidationError) as exc:
        # Pub/Sub retries on any non-2xx response. A malformed payload will
        # never become well-formed on retry, so this is a 400 (do not keep
        # retrying), not a 500 (retry, this side might be broken).
        raise HTTPException(
            status_code=400, detail=f"could not decode message: {exc}"
        ) from exc

    if not batch.transactions:
        return ScoreResponse(
            messages_received=0,
            transactions_scored=0,
            duplicates_skipped=0,
            alerts_raised=0,
        )

    sink: SQLiteSink = state["sink"]
    incoming_ids = [t.transaction_id for t in batch.transactions]
    already = sink.already_seen(incoming_ids)
    fresh = [t for t in batch.transactions if t.transaction_id not in already]

    if not fresh:
        return ScoreResponse(
            messages_received=len(batch.transactions),
            transactions_scored=0,
            duplicates_skipped=len(batch.transactions),
            alerts_raised=0,
        )

    raw_df = pd.DataFrame([t.model_dump() for t in fresh])
    scored = score_transaction_batch(raw_df, state["config"], state["reference"])
    inserted = sink.write_batch(scored, message_id=envelope.message.messageId)

    return ScoreResponse(
        messages_received=len(batch.transactions),
        transactions_scored=inserted,
        duplicates_skipped=len(batch.transactions) - len(fresh),
        alerts_raised=int(scored["is_alert"].sum()),
    )
