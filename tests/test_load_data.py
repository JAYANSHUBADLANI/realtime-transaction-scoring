"""Unit tests for feature engineering and scoring-scope filtering in
load_data.py.

These use small, hand built PaySim-shaped frames so the domain logic (ledger
consistency checks, the type-based scope filter, the synthetic timestamp) is
verified independently of the real 6.36M-row dataset.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.load_data import engineer_features, load_transactions  # noqa: E402


def _raw_rows():
    return pd.DataFrame(
        {
            "step": [1, 1, 2, 2],
            "type": ["TRANSFER", "PAYMENT", "CASH_OUT", "TRANSFER"],
            "amount": [100.0, 50.0, 200.0, 75.0],
            "nameOrig": ["A", "B", "C", "D"],
            # Row 0: balance moved by exactly the amount, both sides
            # consistent, origin fully drained.
            "oldbalanceOrg": [100.0, 500.0, 200.0, 300.0],
            "newbalanceOrig": [0.0, 450.0, 0.0, 225.0],
            "nameDest": ["X", "Y", "Z", "W"],
            "oldbalanceDest": [0.0, 0.0, 1000.0, 40.0],
            "newbalanceDest": [100.0, 0.0, 1150.0, 115.0],
            "isFraud": [1, 0, 0, 0],
            "isFlaggedFraud": [0, 0, 0, 0],
        }
    )


def test_engineer_features_balance_error_orig():
    out = engineer_features(_raw_rows())
    # Row 0: 100 - 100 - 0 = 0 (perfectly consistent).
    assert out.loc[0, "balance_error_orig"] == 0.0
    # Row 1: 500 - 50 - 450 = 0.
    assert out.loc[1, "balance_error_orig"] == 0.0
    # Row 2: 200 - 200 - 0 = 0.
    assert out.loc[2, "balance_error_orig"] == 0.0
    # Row 3: 300 - 75 - 225 = 0.
    assert out.loc[3, "balance_error_orig"] == 0.0


def test_engineer_features_balance_error_dest():
    out = engineer_features(_raw_rows())
    # Row 2: 1000 + 200 - 1150 = 50, money did not fully land at the
    # destination.
    assert out.loc[2, "balance_error_dest"] == 50.0
    # Row 0: 0 + 100 - 100 = 0, fully consistent.
    assert out.loc[0, "balance_error_dest"] == 0.0


def test_engineer_features_orig_emptied():
    out = engineer_features(_raw_rows())
    # Rows 0 and 2 start positive and end at exactly zero.
    assert out.loc[0, "orig_emptied"] == 1
    assert out.loc[2, "orig_emptied"] == 1
    # Rows 1 and 3 still have a positive balance afterward.
    assert out.loc[1, "orig_emptied"] == 0
    assert out.loc[3, "orig_emptied"] == 0


def test_engineer_features_does_not_mutate_input():
    raw = _raw_rows()
    before_cols = list(raw.columns)
    engineer_features(raw)
    assert list(raw.columns) == before_cols


def _write_temp_csv(tmp_path):
    csv_path = tmp_path / "paysim.csv"
    _raw_rows().to_csv(csv_path, index=False)
    return str(csv_path)


def _config(csv_path):
    return {
        "data": {
            "csv_path": csv_path,
            "required_columns": [
                "step",
                "type",
                "amount",
                "nameOrig",
                "oldbalanceOrg",
                "newbalanceOrig",
                "nameDest",
                "oldbalanceDest",
                "newbalanceDest",
                "isFraud",
                "isFlaggedFraud",
            ],
            "scored_types": ["TRANSFER", "CASH_OUT"],
            "numeric_features": ["amount", "balance_error_orig", "balance_error_dest"],
            "step_column": "step",
            "synthetic_timestamp_column": "event_time",
        }
    }


def test_load_transactions_applies_scored_types_filter(tmp_path):
    csv_path = _write_temp_csv(tmp_path)
    df = load_transactions(config=_config(csv_path))
    # The single PAYMENT row (index 1 in the raw fixture) must be dropped.
    assert set(df["type"].unique()) == {"TRANSFER", "CASH_OUT"}
    assert len(df) == 3


def test_load_transactions_derives_synthetic_timestamp(tmp_path):
    csv_path = _write_temp_csv(tmp_path)
    df = load_transactions(config=_config(csv_path))
    assert "event_time" in df.columns
    # step=1 rows should be exactly one hour after step=0 would be, and
    # step=2 rows one hour after that.
    step1_time = df.loc[df["step"] == 1, "event_time"].iloc[0]
    step2_time = df.loc[df["step"] == 2, "event_time"].iloc[0]
    assert (step2_time - step1_time).total_seconds() == 3600.0


def test_load_transactions_engineers_features_after_filter(tmp_path):
    csv_path = _write_temp_csv(tmp_path)
    df = load_transactions(config=_config(csv_path))
    for col in ["balance_error_orig", "balance_error_dest", "orig_emptied"]:
        assert col in df.columns
