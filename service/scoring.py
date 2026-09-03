"""Score a small batch of incoming transactions against the frozen
reference, reusing the exact same alerting/severity code the batch
pipeline uses.

This module has no dependency on how the batch arrived (a Pub/Sub push
envelope, a direct test call, anything else); it only needs a raw
dataframe shaped like PaySim's live columns plus a ``transaction_id``.
Keeping it framework-agnostic is what lets ``tests/test_service.py`` (via
FastAPI's TestClient) and any future non-HTTP caller both exercise the
identical scoring path ``main.py``'s batch run already proved out.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from src.alerting import DETECTION_METHODS, build_alerts
from src.load_data import engineer_features
from src.prioritization import compute_severity
from src.reference import ReferenceBundle, score_with_reference


def score_transaction_batch(
    raw_df: pd.DataFrame, config: Dict, reference: ReferenceBundle
) -> pd.DataFrame:
    """Engineer features, score against the frozen reference, and attach
    alert/severity columns to every row (not just the alerted subset), so
    a full audit trail of every scored transaction can be written to the
    sink, matching what ``reports/scored_transactions.csv`` already does
    for the batch path.
    """
    engineered = engineer_features(raw_df)
    detector_scores = score_with_reference(engineered, config, reference)

    scored = engineered.copy()
    for col in detector_scores.columns:
        scored[col] = detector_scores[col]
    scored["n_methods"] = scored[DETECTION_METHODS].sum(axis=1)

    scored["is_alert"] = 0
    scored["severity"] = None
    scored["severity_score"] = 0.0

    zscore_result = scored[["zscore_flag", "zscore_score"]]
    iqr_result = scored[["iqr_flag", "iqr_score"]]
    iforest_result = scored[["iforest_flag", "iforest_score"]]
    alerts = build_alerts(engineered, zscore_result, iqr_result, iforest_result, config)

    if not alerts.empty:
        severity = compute_severity(alerts, config)
        scored.loc[severity.index, "is_alert"] = 1
        scored.loc[severity.index, "severity"] = severity["severity"]
        scored.loc[severity.index, "severity_score"] = severity["severity_score"]

    return scored
