"""Integration tests for the FastAPI push subscriber.

Uses FastAPI's TestClient, which drives the app in-process over the same
ASGI interface a real HTTP request would use, so these tests exercise the
exact same request-parsing and scoring code path a deployed Cloud Run push
subscriber will run, with no real Pub/Sub topic, emulator, or network
socket involved. Each test gets its own tiny synthetic reference fitted
fresh (not the real 463K-row PaySim artifact main.py produces), so this
suite stays fast and independent of having run the batch pipeline first.
"""

import base64
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reference import fit_reference  # noqa: E402


def _config():
    return {
        "statistical": {
            "zscore": {"columns": ["amount", "balance_error_dest"], "threshold": 3.0},
            "iqr": {"columns": ["amount", "balance_error_dest"], "k": 1.5},
        },
        "model": {
            "isolation_forest": {
                "contamination": 0.05,
                "n_estimators": 30,
                "random_state": 42,
            }
        },
        "data": {
            "numeric_features": [
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
                "balance_error_orig",
                "balance_error_dest",
            ]
        },
        "alerting": {
            "rules": {
                "high_amount": {"column": "amount", "operator": ">", "value": 400_000},
                "orig_account_emptied": {
                    "column": "orig_emptied",
                    "operator": ">=",
                    "value": 1,
                },
                "high_zscore": {"value": 3.0},
            }
        },
        "prioritization": {
            "weights": {"deviation": 0.5, "agreement": 0.3, "rules": 0.2},
            "buckets": {"high": 0.66, "medium": 0.33},
        },
    }


def _reference_dir(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    ref_df = pd.DataFrame(
        {
            "amount": rng.gamma(2.0, 5000.0, size=n),
            "oldbalanceOrg": rng.gamma(2.0, 6000.0, size=n),
            "newbalanceOrig": rng.gamma(2.0, 4000.0, size=n),
            "oldbalanceDest": rng.gamma(2.0, 3000.0, size=n),
            "newbalanceDest": rng.gamma(2.0, 3500.0, size=n),
            "balance_error_orig": rng.normal(0.0, 100.0, size=n),
            "balance_error_dest": rng.gamma(1.5, 200.0, size=n),
        }
    )
    reference = fit_reference(ref_df, _config())
    directory = str(tmp_path / "artifacts")
    reference.save(directory)
    return directory


@pytest.fixture
def client(tmp_path, monkeypatch):
    ref_dir = _reference_dir(tmp_path)
    db_path = str(tmp_path / "test_scores.db")

    import service.main as service_main

    monkeypatch.setattr(service_main, "ARTIFACTS_DIR", ref_dir)
    monkeypatch.setattr(service_main, "DB_PATH", db_path)
    monkeypatch.setattr(service_main, "load_config", _config)

    from fastapi.testclient import TestClient

    with TestClient(service_main.app) as c:
        yield c


def _envelope(transactions, message_id="msg-1"):
    payload = {"transactions": transactions}
    data = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return {"message": {"data": data, "messageId": message_id}}


def _transaction(txn_id, amount=1000.0, orig_emptied=False):
    old_orig = amount if orig_emptied else amount * 3
    new_orig = 0.0 if orig_emptied else old_orig - amount
    return {
        "step": 100,
        "type": "TRANSFER",
        "amount": amount,
        "nameOrig": f"C{txn_id}",
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "nameDest": f"D{txn_id}",
        "oldbalanceDest": 0.0,
        "newbalanceDest": amount,
        "transaction_id": txn_id,
    }


def test_health_reports_reference_provenance(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["reference_fitted_on_rows"] == 400


def test_push_scores_and_persists_transactions(client):
    resp = client.post("/push", json=_envelope([_transaction("t1"), _transaction("t2")]))
    assert resp.status_code == 200
    body = resp.json()
    assert body["messages_received"] == 2
    assert body["transactions_scored"] == 2
    assert body["duplicates_skipped"] == 0

    import service.main as service_main

    stored = service_main.state["sink"].read_all()
    assert set(stored["transaction_id"]) == {"t1", "t2"}


def test_orig_emptied_transaction_raises_an_alert(client):
    # A cleanly-emptied origin account should trip the orig_account_emptied
    # rule regardless of how the statistical detectors score it.
    resp = client.post(
        "/push", json=_envelope([_transaction("t-empty", amount=5000.0, orig_emptied=True)])
    )
    assert resp.status_code == 200
    assert resp.json()["alerts_raised"] == 1

    import service.main as service_main

    stored = service_main.state["sink"].read_all()
    row = stored[stored["transaction_id"] == "t-empty"].iloc[0]
    assert row["is_alert"] == 1
    assert row["orig_emptied"] == 1


def test_redelivered_message_is_not_double_counted(client):
    envelope = _envelope([_transaction("t-dup")], message_id="msg-dup")

    first = client.post("/push", json=envelope)
    assert first.json()["transactions_scored"] == 1
    assert first.json()["duplicates_skipped"] == 0

    # Pub/Sub push is at-least-once: the identical envelope can arrive
    # again. The transaction must not be scored or stored twice.
    second = client.post("/push", json=envelope)
    assert second.json()["transactions_scored"] == 0
    assert second.json()["duplicates_skipped"] == 1

    import service.main as service_main

    stored = service_main.state["sink"].read_all()
    assert len(stored[stored["transaction_id"] == "t-dup"]) == 1


def test_malformed_base64_returns_400_not_500(client):
    resp = client.post(
        "/push", json={"message": {"data": "not-valid-base64!!!", "messageId": "bad-1"}}
    )
    assert resp.status_code == 400


def test_empty_transaction_list_is_a_no_op(client):
    resp = client.post("/push", json=_envelope([], message_id="msg-empty"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["transactions_scored"] == 0
    assert body["alerts_raised"] == 0
