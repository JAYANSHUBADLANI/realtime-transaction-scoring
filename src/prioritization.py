"""Incident prioritization.

Once alerts exist, an analyst needs to know what to look at first. This module
turns each alert into a single severity score on a 0 to 1 scale, blending three
ideas:

* deviation: how extreme the underlying detector scores are,
* agreement: how many independent detectors flagged the row,
* rules: how many hard business rules the row tripped.

The weights for each component come from ``config.yaml``. Alerts are then placed
into High, Medium, and Low buckets using configurable cut points and returned
as a ranked queue, most severe first.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _normalize(series: pd.Series) -> pd.Series:
    """Min-max scale a series to the 0 to 1 range.

    A constant series maps to all zeros, which is the sensible neutral value
    here since no point stands out.
    """
    s = series.astype(float)
    lo = s.min()
    hi = s.max()
    span = hi - lo
    if span == 0 or np.isnan(span):
        return pd.Series(0.0, index=series.index)
    return (s - lo) / span


def compute_severity(alerts: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add severity scoring and bucketing to the alert table.

    Parameters
    ----------
    alerts:
        Output of ``build_alerts``.
    config:
        Parsed configuration with ``prioritization.weights`` and
        ``prioritization.buckets``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``alerts`` with ``deviation_component``,
        ``agreement_component``, ``rules_component``, ``severity_score``, and a
        categorical ``severity`` column, sorted by severity descending.
    """
    out = alerts.copy()

    if out.empty:
        # Return an empty frame with the expected columns so callers do not
        # have to special case the no-alert situation.
        for col in [
            "deviation_component",
            "agreement_component",
            "rules_component",
            "severity_score",
            "severity",
        ]:
            out[col] = pd.Series(dtype="float64")
        out["severity"] = pd.Series(dtype="object")
        return out

    weights = config["prioritization"]["weights"]
    buckets = config["prioritization"]["buckets"]

    # Deviation: normalize each detector score to 0 to 1, then take the max so
    # an alert that is extreme on any single detector is treated as severe.
    deviation = pd.concat(
        [
            _normalize(out["zscore_score"]),
            _normalize(out["iqr_score"]),
            _normalize(out["iforest_score"]),
        ],
        axis=1,
    ).max(axis=1)

    # Agreement: share of the three detectors that fired.
    agreement = out["n_methods"].astype(float) / len(["zscore", "iqr", "iforest"])

    # Rules: share of the configured business rules that fired on the row.
    rule_cols = [c for c in out.columns if c.startswith("rule_")]
    if rule_cols:
        rules_component = out[rule_cols].sum(axis=1).astype(float) / len(rule_cols)
    else:
        rules_component = pd.Series(0.0, index=out.index)

    out["deviation_component"] = deviation
    out["agreement_component"] = agreement
    out["rules_component"] = rules_component

    out["severity_score"] = (
        weights["deviation"] * deviation
        + weights["agreement"] * agreement
        + weights["rules"] * rules_component
    )

    # Bucket using the configured cut points.
    high_cut = buckets["high"]
    medium_cut = buckets["medium"]

    def _bucket(score: float) -> str:
        if score >= high_cut:
            return "High"
        if score >= medium_cut:
            return "Medium"
        return "Low"

    out["severity"] = out["severity_score"].apply(_bucket)

    return out.sort_values("severity_score", ascending=False)


def build_incident_queue(alerts: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Return the ranked incident queue, most severe first.

    Thin wrapper around :func:`compute_severity` that also adds a 1 based
    ``rank`` column, which is convenient for reporting and dashboards.
    """
    ranked = compute_severity(alerts, config)
    ranked = ranked.reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked
