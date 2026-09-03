"""Unit tests for severity scoring and incident bucketing."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prioritization import build_incident_queue, compute_severity  # noqa: E402


def _config():
    """Minimal config matching the structure prioritization expects."""
    return {
        "prioritization": {
            "weights": {"deviation": 0.5, "agreement": 0.3, "rules": 0.2},
            "buckets": {"high": 0.66, "medium": 0.33},
        }
    }


def _alerts():
    """Three alerts spanning a clearly severe, a middling, and a mild case."""
    return pd.DataFrame(
        {
            "TransactionID": ["A", "B", "C"],
            "zscore_flag": [True, True, False],
            "zscore_score": [10.0, 4.0, 0.5],
            "iqr_flag": [True, False, True],
            "iqr_score": [8.0, 0.0, 1.0],
            "iforest_flag": [True, True, False],
            "iforest_score": [0.9, 0.6, 0.1],
            "n_methods": [3, 2, 1],
            "rule_high_amount": [True, True, False],
            "rule_many_login_attempts": [True, False, False],
        }
    )


def test_severity_ordering_and_buckets():
    result = compute_severity(_alerts(), _config())
    # The most extreme alert (A, all three methods plus both rules) should rank
    # first and be High; the mildest (C) should be Low.
    assert result.iloc[0]["TransactionID"] == "A"
    assert result.iloc[0]["severity"] == "High"
    assert result.iloc[-1]["TransactionID"] == "C"
    assert result.iloc[-1]["severity"] == "Low"


def test_severity_scores_in_unit_range():
    result = compute_severity(_alerts(), _config())
    assert (result["severity_score"] >= 0).all()
    assert (result["severity_score"] <= 1).all()


def test_full_agreement_and_rules_gives_max_components():
    result = compute_severity(_alerts(), _config())
    row_a = result[result["TransactionID"] == "A"].iloc[0]
    # Alert A fired all three detectors, so agreement component is 1.0.
    assert row_a["agreement_component"] == 1.0
    # Alert A is the most extreme on every detector, so deviation is 1.0.
    assert row_a["deviation_component"] == 1.0
    # Alert A tripped both configured rules, so rules component is 1.0.
    assert row_a["rules_component"] == 1.0


def test_incident_queue_has_sequential_rank():
    queue = build_incident_queue(_alerts(), _config())
    assert list(queue["rank"]) == [1, 2, 3]


def test_empty_alerts_returns_empty_queue():
    empty = _alerts().iloc[0:0]
    queue = build_incident_queue(empty, _config())
    assert queue.empty
