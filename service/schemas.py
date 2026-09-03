"""Message contracts for the streaming scorer.

``IncomingTransaction`` is deliberately missing ``isFraud``/``isFlaggedFraud``:
a live transaction stream would never carry the answer key. Those two
columns exist only in the offline PaySim CSV used for the batch evaluation
in the main README, never in what actually gets published to Pub/Sub or
scored here.

``PubSubPushEnvelope`` matches the exact JSON shape Google Cloud Pub/Sub
sends to a push subscription's endpoint:
https://cloud.google.com/pubsub/docs/push . Modeling it precisely here is
what lets ``tests/test_service.py`` verify this service's logic completely
locally, with no real Pub/Sub topic or emulator involved, and still be
confident the same code will parse what real Pub/Sub delivers once this is
deployed behind an actual push subscription in the next build stage.
"""

from __future__ import annotations

import base64
import json
from typing import List, Optional

from pydantic import BaseModel, Field


class IncomingTransaction(BaseModel):
    """One PaySim-shaped transaction as it would arrive live, no label."""

    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float
    # Synthesized by the publisher, since PaySim has no native transaction
    # ID; used as the idempotency key (Pub/Sub push is at-least-once, so
    # the same message can be redelivered).
    transaction_id: str


class TransactionBatch(BaseModel):
    """What one Pub/Sub message actually carries: several transactions
    batched together, not one message per transaction.

    Batching exists to stay inside Pub/Sub's free tier, which bills a
    minimum of 1000 bytes per message regardless of how small the payload
    is, and inside Cloud Run's free tier, which counts one request per
    message delivered. See the README's cost-scoping note.
    """

    transactions: List[IncomingTransaction]


class PubSubMessage(BaseModel):
    data: str  # base64-encoded JSON, per the real Pub/Sub push contract
    messageId: str
    publishTime: Optional[str] = None
    attributes: Optional[dict] = None

    def decode_batch(self) -> TransactionBatch:
        raw = base64.b64decode(self.data)
        payload = json.loads(raw)
        return TransactionBatch(**payload)


class PubSubPushEnvelope(BaseModel):
    message: PubSubMessage
    subscription: Optional[str] = None


class ScoreResponse(BaseModel):
    messages_received: int = Field(description="Transactions in this push")
    transactions_scored: int = Field(description="Newly scored (post-dedup)")
    duplicates_skipped: int = Field(description="Already-seen transaction_ids")
    alerts_raised: int
