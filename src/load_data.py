"""Data and configuration loading utilities.

This module is the single entry point for reading the raw transaction CSV and
the project configuration. It validates that the expected columns are present
and fails with a clear, actionable message when the data file is missing, so
the rest of the pipeline can assume a clean, well typed frame.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd
import yaml


# Resolve paths relative to the project root (the parent of this src/ folder),
# so the code works no matter where it is invoked from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict:
    """Load the YAML configuration file into a dictionary.

    Parameters
    ----------
    config_path:
        Path to the YAML config. Defaults to the project level config.yaml.

    Returns
    -------
    dict
        Parsed configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found at '{config_path}'. "
            "Make sure config.yaml exists at the project root."
        )
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_csv_path(config: Dict) -> str:
    """Return an absolute path to the raw CSV based on the config."""
    csv_path = config["data"]["csv_path"]
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, csv_path)
    return csv_path


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-derived ledger features, computed with no access to the label.

    These come from double-entry bookkeeping logic on the PaySim ledger
    fields, not from ``isFraud``:

    - ``balance_error_orig``: ``oldbalanceOrg - amount - newbalanceOrig``.
      Zero when the sender's balance moved by exactly the transaction
      amount; nonzero balances reveal an inconsistency in the sender's
      recorded ledger.
    - ``balance_error_dest``: ``oldbalanceDest + amount - newbalanceDest``.
      Zero when the recipient's balance rose by exactly the transaction
      amount; a large positive value means money left the sender but did
      not fully land in the recipient's recorded balance, a real fraud
      fingerprint (money routed onward or the account emptied immediately).
    - ``orig_emptied``: 1 when the sender had a positive balance and it hit
      exactly zero after this transaction.

    Returns a copy of ``df`` with these three columns added.
    """
    out = df.copy()
    out["balance_error_orig"] = (
        out["oldbalanceOrg"] - out["amount"] - out["newbalanceOrig"]
    )
    out["balance_error_dest"] = (
        out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]
    )
    out["orig_emptied"] = (
        (out["oldbalanceOrg"] > 0) & (out["newbalanceOrig"] == 0)
    ).astype(int)
    return out


def load_transactions(
    config: Optional[Dict] = None,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> pd.DataFrame:
    """Load, filter, and lightly clean the transaction dataset.

    The function reads the CSV named in the config, validates the required
    columns, restricts rows to ``data.scored_types`` (fraud in this dataset
    is confirmed by EDA to be confined to those types, see config.yaml),
    engineers the ledger-consistency features via :func:`engineer_features`,
    derives a synthetic per-row timestamp from PaySim's hourly ``step``
    index so the existing EWMA control chart can aggregate by day, and
    coerces the declared numeric features to numeric dtype.

    Parameters
    ----------
    config:
        An already loaded config dict. If omitted, the config is read from
        ``config_path``.
    config_path:
        Path to the YAML config, used only when ``config`` is not given.

    Returns
    -------
    pandas.DataFrame
        The validated, filtered, feature-engineered transaction frame.

    Raises
    ------
    FileNotFoundError
        If the CSV is missing. The message explains how to download it.
    ValueError
        If any required column is absent from the file.
    """
    if config is None:
        config = load_config(config_path)

    csv_path = _resolve_csv_path(config)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Transaction data not found at '{csv_path}'.\n"
            "The raw data is not committed to the repo. Download it from Kaggle "
            "and place the CSV in the data/ folder. See data/README.md for the "
            "exact steps (kagglehub script, Kaggle API, or manual download)."
        )

    df = pd.read_csv(csv_path)

    required: List[str] = config["data"]["required_columns"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "The data file is missing required columns: "
            f"{missing}.\nColumns found: {list(df.columns)}.\n"
            "Update config.yaml 'data.required_columns' to match the real "
            "schema if the dataset uses different names."
        )

    scored_types = config["data"].get("scored_types")
    if scored_types:
        df = df[df["type"].isin(scored_types)].reset_index(drop=True)

    df = engineer_features(df)

    step_col = config["data"].get("step_column")
    ts_col = config["data"].get("synthetic_timestamp_column")
    if step_col and ts_col and step_col in df.columns:
        # Anchor arbitrarily at 2024-01-01; only relative spacing (one
        # simulation hour per step) matters for the daily control chart.
        df[ts_col] = pd.Timestamp("2024-01-01") + pd.to_timedelta(
            df[step_col], unit="h"
        )

    # Coerce declared numeric features so downstream math is safe.
    for col in config["data"].get("numeric_features", []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def data_available(config: Optional[Dict] = None) -> bool:
    """Return True if the raw CSV exists on disk.

    Useful for the Streamlit app, which should show download instructions
    instead of crashing when the data is absent.
    """
    if config is None:
        config = load_config()
    return os.path.exists(_resolve_csv_path(config))


if __name__ == "__main__":
    # Quick manual check: load the data and print a tiny summary.
    cfg = load_config()
    frame = load_transactions(cfg)
    print(f"Loaded {len(frame):,} rows and {frame.shape[1]} columns.")
    print("Columns:", list(frame.columns))
