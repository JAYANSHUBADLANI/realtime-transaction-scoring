"""Unit tests for the labeled-evaluation module.

Small, hand built label/score pairs where the right precision/recall/lift
numbers are obvious by inspection, so a regression here (an evaluation bug
producing a flattering wrong number) is caught immediately, not discovered
after it is already in a README.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import (  # noqa: E402
    build_combined_score,
    evaluate_all,
    evaluate_flag_and_score,
    lift_at_k,
)


def test_evaluate_flag_and_score_perfect_detector():
    y = pd.Series([0, 0, 0, 1, 1])
    flag = pd.Series([False, False, False, True, True])
    score = pd.Series([0.1, 0.1, 0.1, 0.9, 0.9])
    result = evaluate_flag_and_score(y, flag, score)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["true_positives"] == 2
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["pr_auc"] == 1.0


def test_evaluate_flag_and_score_no_flags():
    y = pd.Series([0, 0, 1, 1])
    flag = pd.Series([False, False, False, False])
    result = evaluate_flag_and_score(y, flag)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["n_flagged"] == 0


def test_evaluate_flag_and_score_without_continuous_score():
    y = pd.Series([0, 1, 1, 0])
    flag = pd.Series([False, True, False, True])
    result = evaluate_flag_and_score(y, flag)
    # No score given, so the AUC-style metrics must stay unset rather than
    # silently defaulting to something that looks like a real number.
    assert result["pr_auc"] is None
    assert result["roc_auc"] is None
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1


def test_lift_at_k_finds_all_fraud_at_top():
    # 10 rows, 2 fraud, both given the two highest scores. Reviewing the
    # top 20% should catch both.
    y = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = pd.Series([10, 9, 1, 1, 1, 1, 1, 1, 1, 1])
    result = lift_at_k(y, score, k_fractions=[0.2])
    top = result["top_0.2"]
    assert top["n_reviewed"] == 2
    assert top["fraud_caught"] == 2
    assert top["fraud_caught_share_of_all_fraud"] == 1.0
    assert top["precision_at_k"] == 1.0
    # Base rate is 0.2 and precision at k is 1.0, so lift is 5x.
    assert top["lift_over_base_rate"] == 5.0


def test_lift_at_k_random_score_gives_lift_near_one():
    # Fraud scattered independently of score order: reviewing the top half
    # should catch roughly half the fraud, lift near 1.
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
    score = pd.Series([8, 7, 6, 5, 4, 3, 2, 1])
    result = lift_at_k(y, score, k_fractions=[0.5])
    top = result["top_0.5"]
    assert top["fraud_caught"] == 2
    assert top["lift_over_base_rate"] == 1.0


def test_build_combined_score_takes_max_of_normalized():
    zscore = pd.Series([0.0, 10.0])
    iqr = pd.Series([5.0, 0.0])
    iforest = pd.Series([0.0, 0.0])
    combined = build_combined_score(zscore, iqr, iforest)
    # Row 0: normalized iqr=1.0 is the max. Row 1: normalized zscore=1.0.
    assert combined.iloc[0] == 1.0
    assert combined.iloc[1] == 1.0


def test_evaluate_all_end_to_end():
    scored = pd.DataFrame(
        {
            "isFraud": [0, 0, 0, 1, 1],
            "isFlaggedFraud": [0, 0, 0, 0, 1],
            "zscore_flag": [False, False, True, True, True],
            "zscore_score": [0.5, 0.5, 3.5, 4.0, 4.5],
            "iqr_flag": [False, False, False, True, True],
            "iqr_score": [0.1, 0.1, 0.1, 2.0, 3.0],
            "iforest_flag": [False, False, False, True, True],
            "iforest_score": [0.1, 0.1, 0.1, 0.8, 0.9],
            "orig_emptied": [0, 0, 1, 1, 1],
        }
    )
    result = evaluate_all(
        scored, label_col="isFraud", baseline_flag_col="isFlaggedFraud"
    )
    assert result["n_scored"] == 5
    assert result["n_fraud"] == 2
    assert result["base_fraud_rate"] == 0.4
    # Both fraud rows are caught by zscore.
    assert result["detectors"]["zscore"]["recall"] == 1.0
    # PaySim's own baseline only caught one of the two fraud rows here.
    assert result["detectors"]["paysim_builtin_flag_baseline"]["recall"] == 0.5
    assert "top_0.01" in result["lift_at_k_combined_score"]
