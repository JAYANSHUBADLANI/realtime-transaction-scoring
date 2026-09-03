"""Proof that batch and one-row-at-a-time streaming scoring are identical.

This is the property the whole reference-based refactor in
`src/reference.py` exists for: a transaction scored on its own, against a
frozen reference, must get exactly the score it would have gotten if it had
arrived as part of a 2.3M-row batch. If that were not true, a streaming
service scoring transactions one at a time would silently disagree with the
batch pipeline this project started from, and there would be no way to
trust either number.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reference import fit_reference, score_with_reference  # noqa: E402


def _config():
    return {
        "statistical": {
            "zscore": {"columns": ["amount", "balance_error_dest"], "threshold": 3.0},
            "iqr": {"columns": ["amount", "balance_error_dest"], "k": 1.5},
        },
        "model": {
            "isolation_forest": {
                "contamination": 0.05,
                "n_estimators": 50,
                "random_state": 42,
            }
        },
        "data": {
            "numeric_features": [
                "amount",
                "balance_error_orig",
                "balance_error_dest",
            ]
        },
    }


def _reference_window(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "amount": rng.gamma(shape=2.0, scale=5000.0, size=n),
            "balance_error_orig": rng.normal(0.0, 100.0, size=n),
            "balance_error_dest": rng.gamma(shape=1.5, scale=200.0, size=n),
        }
    )


def _incoming_batch(n=40, seed=1):
    rng = np.random.default_rng(seed)
    # Deliberately includes a few extreme rows so both detectors have
    # something real to flag, not just quiet, unremarkable transactions.
    df = pd.DataFrame(
        {
            "amount": rng.gamma(shape=2.0, scale=5000.0, size=n),
            "balance_error_orig": rng.normal(0.0, 100.0, size=n),
            "balance_error_dest": rng.gamma(shape=1.5, scale=200.0, size=n),
        }
    )
    df.loc[0, "amount"] = 500_000.0
    df.loc[1, "balance_error_dest"] = 50_000.0
    return df


def test_batch_and_row_by_row_scores_are_identical():
    config = _config()
    reference = fit_reference(_reference_window(), config)
    batch = _incoming_batch()

    batch_scores = score_with_reference(batch, config, reference)

    row_scores = []
    for i in range(len(batch)):
        single_row = batch.iloc[[i]].reset_index(drop=True)
        row_scores.append(score_with_reference(single_row, config, reference))
    row_scores = pd.concat(row_scores, ignore_index=True)

    for col in ["zscore_score", "iqr_score", "iforest_score"]:
        np.testing.assert_allclose(
            batch_scores[col].to_numpy(),
            row_scores[col].to_numpy(),
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"{col} differs between batch and row-by-row scoring",
        )
    for col in ["zscore_flag", "iqr_flag", "iforest_flag"]:
        assert (batch_scores[col].to_numpy() == row_scores[col].to_numpy()).all(), (
            f"{col} differs between batch and row-by-row scoring"
        )


def test_scoring_never_touches_the_reference_it_was_not_fit_on():
    # Fit on one reference window, score a batch. Then fit a second,
    # differently-shifted reference and confirm scores genuinely change:
    # this rules out score_with_reference silently ignoring `reference`
    # and falling back to recomputing statistics from the scored df itself
    # (which the whole point of this refactor is to prevent).
    config = _config()
    batch = _incoming_batch()

    reference_a = fit_reference(_reference_window(seed=0), config)
    reference_b = fit_reference(_reference_window(seed=99), config)

    scores_a = score_with_reference(batch, config, reference_a)
    scores_b = score_with_reference(batch, config, reference_b)

    assert not np.allclose(
        scores_a["zscore_score"].to_numpy(), scores_b["zscore_score"].to_numpy()
    ), "z-scores did not change with a different reference; scoring may be ignoring it"


def test_reference_stats_do_not_depend_on_scored_batch_size():
    # Score the same 5 rows twice: once alone, once padded with 200 extra
    # rows in the same call. If any statistic were still being computed
    # from the scored dataframe (the leakage this refactor removes), the
    # padding would change the first 5 rows' scores.
    config = _config()
    reference = fit_reference(_reference_window(), config)

    small_batch = _incoming_batch(n=5, seed=7)
    padded_batch = pd.concat(
        [small_batch, _incoming_batch(n=200, seed=8)], ignore_index=True
    )

    small_scores = score_with_reference(small_batch, config, reference)
    padded_scores = score_with_reference(padded_batch, config, reference)

    for col in ["zscore_score", "iqr_score", "iforest_score"]:
        np.testing.assert_allclose(
            small_scores[col].to_numpy(),
            padded_scores[col].to_numpy()[:5],
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"{col} depends on which other rows were scored alongside it",
        )
