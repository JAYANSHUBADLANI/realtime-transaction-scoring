"""Model based anomaly detection using Isolation Forest.

Isolation Forest isolates observations by randomly partitioning the feature
space. Anomalies are easier to isolate, so they sit closer to the root of the
random trees and receive a higher anomaly score. This complements the
statistical methods, which look at one column at a time, because the forest can
catch rows that are unusual only in combination across several features.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def fit_isolation_forest(
    df: pd.DataFrame,
    columns: List[str],
    contamination: float = 0.02,
    n_estimators: int = 200,
    random_state: int = 42,
) -> IsolationForest:
    """Fit an Isolation Forest on a reference window and return the model.

    Split out from scoring so a streaming service can fit once (or
    periodically refit) on a reference window and reuse the same frozen
    model for every incoming transaction, instead of refitting on whatever
    happens to be in the current batch.

    Parameters
    ----------
    df:
        Reference data to fit on.
    columns:
        Numeric feature columns to feed the model.
    contamination:
        Expected proportion of anomalies. Drives the decision threshold.
    n_estimators:
        Number of trees in the forest.
    random_state:
        Seed for reproducibility.
    """
    features = df[columns].astype(float)
    features = features.fillna(features.median())

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(features)
    return model


def score_isolation_forest(
    df: pd.DataFrame,
    columns: List[str],
    model: IsolationForest,
) -> pd.DataFrame:
    """Score rows against an already-fitted Isolation Forest.

    Parameters
    ----------
    df:
        Data to score. Can be the full reference-scored population or a
        single transaction; the model itself never changes here.
    columns:
        Same feature columns the model was fit on, in the same order.
    model:
        A fitted :class:`~sklearn.ensemble.IsolationForest`, from
        :func:`fit_isolation_forest`.

    Returns
    -------
    pandas.DataFrame
        Same index as ``df`` with an ``iforest_score`` column (higher means
        more anomalous) and a boolean ``iforest_flag`` column.
    """
    # Median impute any missing values so the model always has complete rows.
    # Uses this dataframe's own median rather than the reference window's,
    # since a single streamed row has no median of its own to impute from;
    # in practice PaySim's numeric columns here have no missing values, so
    # this only guards against a column going missing in a future feed.
    features = df[columns].astype(float)
    features = features.fillna(features.median())

    raw = model.score_samples(features)
    predictions = model.predict(features)  # -1 anomaly, 1 normal

    result = pd.DataFrame(index=df.index)
    result["iforest_score"] = -raw
    result["iforest_flag"] = predictions == -1
    return result


def isolation_forest_detection(
    df: pd.DataFrame,
    columns: List[str],
    contamination: float = 0.02,
    n_estimators: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit and score in one call: batch convenience only.

    Kept for the one-shot "fit and score this exact dataset" use case
    (what the original unlabeled batch scaffold this project grew out of
    did). The streaming-safe path is :func:`fit_isolation_forest` +
    :func:`score_isolation_forest` with a frozen reference window, used by
    ``src/reference.py`` and, from this project's Phase 1 onward,
    ``main.py``.
    """
    model = fit_isolation_forest(df, columns, contamination, n_estimators, random_state)
    return score_isolation_forest(df, columns, model)


def agreement_rate(flag_a: pd.Series, flag_b: pd.Series) -> Dict[str, float]:
    """Compare two boolean flag series and report overlap statistics.

    Parameters
    ----------
    flag_a, flag_b:
        Boolean Series aligned on the same index.

    Returns
    -------
    dict
        ``n_a`` and ``n_b`` counts, the size of their intersection and union,
        the Jaccard index, and the overall agreement rate (the share of rows
        on which the two methods give the same verdict).
    """
    a = flag_a.astype(bool)
    b = flag_b.astype(bool)
    intersection = int((a & b).sum())
    union = int((a | b).sum())
    n_total = len(a)
    same_verdict = int((a == b).sum())

    return {
        "n_a": int(a.sum()),
        "n_b": int(b.sum()),
        "intersection": intersection,
        "union": union,
        "jaccard": (intersection / union) if union else 0.0,
        "agreement_rate": (same_verdict / n_total) if n_total else 0.0,
    }
