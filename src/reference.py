"""Fit-once, score-many reference statistics for streaming-safe detection.

`main.py`'s batch pipeline calls `zscore_detection`/`iqr_detection` from
`statistical_methods.py`, which compute mean/std/quantiles from whatever
dataframe is passed in. That is fine for a one-shot batch run over a fixed,
already-complete dataset, but it cannot work for streaming: a transaction
arriving right now cannot wait for the mean of transactions that have not
happened yet, and computing a mean over "everything scored so far including
this row" makes every score implicitly depend on what else happened to be
in the same batch, which is not a property a single incoming transaction
can have.

This module freezes reference statistics once, from a fixed reference
window of historical data, and everything after that scores against the
frozen numbers. The `score_*` functions never look at anything other than
`df` and `reference`, which is the property that makes
`tests/test_streaming_equivalence.py` able to prove batch and one-row-at-a-
time streaming produce identical scores: neither row order nor which other
rows happen to be present in the same call can change a row's score.

The control chart is deliberately not part of this module. It needs a
completed window of days to mean anything (a single incoming transaction
has no "day" to compare against yet), so it stays a periodic batch-style
rollup over already-scored data rather than something in the per-transaction
hot path; see the README's project status section for that design choice.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def fit_zscore_reference(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
    """Freeze per-column mean/std from a reference window.

    A zero or NaN standard deviation (a constant reference column) is
    stored as 0.0, which `score_zscore` treats the same way
    `zscore_detection` always has: nothing on that column can be an
    outlier if the reference window never varied.
    """
    reference: Dict[str, Dict[str, float]] = {}
    for col in columns:
        series = df[col].astype(float)
        std = series.std(ddof=0)
        reference[col] = {
            "mean": float(series.mean()),
            "std": float(std) if std and not np.isnan(std) else 0.0,
        }
    return reference


def score_zscore(
    df: pd.DataFrame,
    columns: List[str],
    reference: Dict[str, Dict[str, float]],
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Score rows against a frozen z-score reference. No statistic here is
    computed from `df`; every mean/std comes from `reference`.
    """
    result = pd.DataFrame(index=df.index)
    z_frames = []
    for col in columns:
        series = df[col].astype(float)
        stats = reference[col]
        std = stats["std"]
        if std == 0:
            z = pd.Series(0.0, index=df.index)
        else:
            z = (series - stats["mean"]) / std
        result[f"{col}_zscore"] = z
        z_frames.append(z.abs())

    abs_z = pd.concat(z_frames, axis=1)
    result["zscore_score"] = abs_z.max(axis=1)
    result["zscore_flag"] = result["zscore_score"] > threshold
    return result


def fit_iqr_reference(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
    """Freeze per-column Q1/Q3/IQR from a reference window."""
    reference: Dict[str, Dict[str, float]] = {}
    for col in columns:
        series = df[col].astype(float)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        reference[col] = {"q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1)}
    return reference


def score_iqr(
    df: pd.DataFrame,
    columns: List[str],
    reference: Dict[str, Dict[str, float]],
    k: float = 1.5,
) -> pd.DataFrame:
    """Score rows against frozen IQR fences. No quantile here is computed
    from `df`; every fence comes from `reference`.
    """
    result = pd.DataFrame(index=df.index)
    distances = []
    for col in columns:
        series = df[col].astype(float)
        stats = reference[col]
        iqr = stats["iqr"]
        lower = stats["q1"] - k * iqr
        upper = stats["q3"] + k * iqr
        flag = (series < lower) | (series > upper)
        result[f"{col}_iqr_flag"] = flag

        if iqr == 0 or np.isnan(iqr):
            dist = pd.Series(0.0, index=df.index)
        else:
            below = (lower - series).clip(lower=0)
            above = (series - upper).clip(lower=0)
            dist = (below + above) / iqr
        distances.append(dist)

    all_dist = pd.concat(distances, axis=1)
    result["iqr_score"] = all_dist.max(axis=1)
    result["iqr_flag"] = result.filter(like="_iqr_flag").any(axis=1)
    return result


class ReferenceBundle:
    """All frozen reference state a streaming scorer needs: the JSON-safe
    z-score/IQR statistics plus the fitted Isolation Forest model.

    Saved as two files rather than one, since the statistics are plain
    JSON and the forest is a scikit-learn object: `reference.json` +
    `isolation_forest.joblib`, matching the training/serving artifact
    split already used elsewhere in this portfolio
    (credit-scorecard-service's frozen PSI reference).
    """

    def __init__(
        self,
        zscore: Dict[str, Dict[str, float]],
        iqr: Dict[str, Dict[str, float]],
        iforest_model: IsolationForest,
        iforest_columns: List[str],
        fitted_on_rows: int,
        reference_step_max: int,
    ) -> None:
        self.zscore = zscore
        self.iqr = iqr
        self.iforest_model = iforest_model
        self.iforest_columns = iforest_columns
        self.fitted_on_rows = fitted_on_rows
        self.reference_step_max = reference_step_max

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        stats = {
            "zscore": self.zscore,
            "iqr": self.iqr,
            "iforest_columns": self.iforest_columns,
            "fitted_on_rows": self.fitted_on_rows,
            "reference_step_max": self.reference_step_max,
        }
        with open(os.path.join(directory, "reference.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        joblib.dump(self.iforest_model, os.path.join(directory, "isolation_forest.joblib"))

    @classmethod
    def load(cls, directory: str) -> "ReferenceBundle":
        with open(os.path.join(directory, "reference.json"), "r", encoding="utf-8") as f:
            stats = json.load(f)
        model = joblib.load(os.path.join(directory, "isolation_forest.joblib"))
        return cls(
            zscore=stats["zscore"],
            iqr=stats["iqr"],
            iforest_model=model,
            iforest_columns=stats["iforest_columns"],
            fitted_on_rows=stats["fitted_on_rows"],
            reference_step_max=stats["reference_step_max"],
        )


def fit_reference(df: pd.DataFrame, config: Dict, reference_step_max: Optional[int] = None) -> ReferenceBundle:
    """Fit every reference statistic and model from one reference window in
    a single call.

    Parameters
    ----------
    df:
        The reference window (e.g. the first N simulated hours). Not the
        full dataset; the whole point is that this is a bounded, completed
        slice frozen before any streaming scoring happens.
    config:
        Parsed config.yaml.
    reference_step_max:
        The last `step` value included in the reference window, recorded
        in the bundle purely as provenance (so an artifact on disk states
        what it was trained on).
    """
    from src.models import fit_isolation_forest  # local import avoids a cycle

    zs_cfg = config["statistical"]["zscore"]
    iqr_cfg = config["statistical"]["iqr"]
    if_cfg = config["model"]["isolation_forest"]

    zscore_ref = fit_zscore_reference(df, zs_cfg["columns"])
    iqr_ref = fit_iqr_reference(df, iqr_cfg["columns"])
    iforest_columns = config["data"]["numeric_features"]
    iforest_model = fit_isolation_forest(
        df,
        iforest_columns,
        contamination=if_cfg["contamination"],
        n_estimators=if_cfg["n_estimators"],
        random_state=if_cfg["random_state"],
    )

    return ReferenceBundle(
        zscore=zscore_ref,
        iqr=iqr_ref,
        iforest_model=iforest_model,
        iforest_columns=iforest_columns,
        fitted_on_rows=int(len(df)),
        reference_step_max=int(reference_step_max) if reference_step_max is not None else -1,
    )


def score_with_reference(
    df: pd.DataFrame, config: Dict, reference: ReferenceBundle
) -> pd.DataFrame:
    """Score any dataframe (a 2.3M-row streaming-eval set or a single
    incoming transaction) against a frozen `ReferenceBundle`.

    This is the one function both the batch pipeline and the future
    streaming service call, which is what makes their outputs provably
    identical rather than merely similar: same code, same frozen
    reference, same math, regardless of how many rows arrive at once.
    """
    from src.models import score_isolation_forest  # local import avoids a cycle

    zs_cfg = config["statistical"]["zscore"]
    iqr_cfg = config["statistical"]["iqr"]

    zscore_result = score_zscore(df, zs_cfg["columns"], reference.zscore, zs_cfg["threshold"])
    iqr_result = score_iqr(df, iqr_cfg["columns"], reference.iqr, iqr_cfg["k"])
    iforest_result = score_isolation_forest(df, reference.iforest_columns, reference.iforest_model)

    out = pd.DataFrame(index=df.index)
    for col in ["zscore_score", "zscore_flag"]:
        out[col] = zscore_result[col]
    for col in ["iqr_score", "iqr_flag"]:
        out[col] = iqr_result[col]
    for col in ["iforest_score", "iforest_flag"]:
        out[col] = iforest_result[col]
    return out
