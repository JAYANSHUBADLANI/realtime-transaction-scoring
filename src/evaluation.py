"""Evaluation against real fraud labels.

The original scaffold this project grew out of used an unlabeled dataset, so
every detector could only be judged by how many rows it flagged, never by
whether it was right. PaySim carries a real ``isFraud`` label, so this module
adds the evaluation the unlabeled version could not do: precision, recall,
PR-AUC, ROC-AUC, and lift-at-k for every detector, plus a comparison against
the dataset's own built-in ``isFlaggedFraud`` rule (PaySim's naive baseline,
which fires on only 16 of 6.36M rows).

The label is used here only, never inside the detectors themselves, which
stay fully unsupervised. That separation matters: a detector that peeked at
``isFraud`` would not be evidence of anything.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_flag_and_score(
    y_true: pd.Series,
    flag: pd.Series,
    score: pd.Series = None,
) -> Dict[str, float]:
    """Precision/recall/F1 for a boolean flag, plus PR-AUC/ROC-AUC if a
    continuous score is also given.

    Returns ``None`` for ``pr_auc``/``roc_auc`` when ``score`` is omitted
    (used for PaySim's own binary ``isFlaggedFraud`` rule, which has no
    underlying continuous score) or when ``y_true`` has only one class.
    """
    y = y_true.astype(int)
    f = flag.astype(int)

    out: Dict[str, float] = {
        "n_flagged": int(f.sum()),
        "true_positives": int(((f == 1) & (y == 1)).sum()),
        "false_positives": int(((f == 1) & (y == 0)).sum()),
        "false_negatives": int(((f == 0) & (y == 1)).sum()),
        "precision": float(precision_score(y, f, zero_division=0)),
        "recall": float(recall_score(y, f, zero_division=0)),
        "f1": float(f1_score(y, f, zero_division=0)),
        "pr_auc": None,
        "roc_auc": None,
    }

    if score is not None and y.nunique() > 1:
        out["pr_auc"] = float(average_precision_score(y, score))
        out["roc_auc"] = float(roc_auc_score(y, score))

    return out


def lift_at_k(
    y_true: pd.Series,
    score: pd.Series,
    k_fractions: Iterable[float] = (0.001, 0.01, 0.05),
) -> Dict[str, Dict[str, float]]:
    """For each top-k fraction by score, report the fraud rate found there
    and the lift over the base rate.

    This answers the operational question a severity-ranked queue exists
    for: if an analyst can only review the top X% of scored transactions
    today, how much of the real fraud does that actually catch?
    """
    y = y_true.astype(int).to_numpy()
    s = score.to_numpy()
    n = len(y)
    base_rate = y.mean() if n else 0.0
    total_fraud = int(y.sum())

    order = np.argsort(-s)
    out: Dict[str, Dict[str, float]] = {}
    for frac in k_fractions:
        top_n = max(1, int(round(n * frac)))
        idx = order[:top_n]
        caught = int(y[idx].sum())
        rate = caught / top_n
        out[f"top_{frac}"] = {
            "n_reviewed": top_n,
            "fraud_caught": caught,
            "fraud_caught_share_of_all_fraud": (
                caught / total_fraud if total_fraud else 0.0
            ),
            "precision_at_k": rate,
            "lift_over_base_rate": (rate / base_rate) if base_rate else 0.0,
        }
    return out


def build_combined_score(
    zscore_score: pd.Series,
    iqr_score: pd.Series,
    iforest_score: pd.Series,
) -> pd.Series:
    """Max of the three min-max normalized detector scores, over the full
    scored population (not just the alert subset).

    Uses the same "take the worst signal" idea as
    :func:`src.prioritization.compute_severity`'s deviation component, but
    computed for every row so it can be used for full-population ranking
    metrics like PR-AUC and lift-at-k.
    """

    def _norm(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        lo, hi = s.min(), s.max()
        span = hi - lo
        if span == 0 or np.isnan(span):
            return pd.Series(0.0, index=s.index)
        return (s - lo) / span

    return pd.concat(
        [_norm(zscore_score), _norm(iqr_score), _norm(iforest_score)], axis=1
    ).max(axis=1)


def evaluate_all(
    scored: pd.DataFrame,
    label_col: str,
    baseline_flag_col: str,
) -> Dict:
    """Run the full evaluation suite used by main.py and return a JSON-safe
    dict: per-detector metrics, the combined-score lift table, and the
    baseline comparison.
    """
    y_true = scored[label_col]

    combined_score = build_combined_score(
        scored["zscore_score"], scored["iqr_score"], scored["iforest_score"]
    )

    detectors = {
        "zscore": evaluate_flag_and_score(
            y_true, scored["zscore_flag"], scored["zscore_score"]
        ),
        "iqr": evaluate_flag_and_score(
            y_true, scored["iqr_flag"], scored["iqr_score"]
        ),
        "isolation_forest": evaluate_flag_and_score(
            y_true, scored["iforest_flag"], scored["iforest_score"]
        ),
        "any_statistical_or_model": evaluate_flag_and_score(
            y_true,
            scored["zscore_flag"] | scored["iqr_flag"] | scored["iforest_flag"],
            combined_score,
        ),
        "orig_account_emptied_rule": evaluate_flag_and_score(
            y_true, scored["orig_emptied"] >= 1
        ),
        "paysim_builtin_flag_baseline": evaluate_flag_and_score(
            y_true, scored[baseline_flag_col] >= 1
        ),
    }

    return {
        "base_fraud_rate": float(y_true.mean()),
        "n_scored": int(len(scored)),
        "n_fraud": int(y_true.sum()),
        "detectors": detectors,
        "lift_at_k_combined_score": lift_at_k(y_true, combined_score),
    }
