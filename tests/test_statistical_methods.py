"""Unit tests for the statistical detection functions.

These use small, hand built inputs where the correct flags are obvious, so a
regression in the math is caught immediately.
"""

import os
import sys

import numpy as np
import pandas as pd

# Make the src package importable when running pytest from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.statistical_methods import (  # noqa: E402
    build_time_metric,
    control_chart_detection,
    iqr_detection,
    zscore_detection,
)


def test_zscore_flags_obvious_outlier():
    # A cluster of 50 points around 10 with one clear outlier at 30. Using
    # enough background points keeps the outlier from inflating the standard
    # deviation so much that it masks itself.
    cluster = [9, 10, 11, 10, 9, 11, 10, 9, 11, 10] * 5
    df = pd.DataFrame({"x": cluster + [30]})
    result = zscore_detection(df, ["x"], threshold=3.0)
    # Only the outlier row (the last one) should be flagged.
    assert result["zscore_flag"].sum() == 1
    assert bool(result["zscore_flag"].iloc[-1]) is True
    # The outlier should carry the highest score.
    assert result["zscore_score"].idxmax() == len(df) - 1


def test_zscore_constant_column_flags_nothing():
    # A constant column has zero std; nothing can be an outlier.
    df = pd.DataFrame({"x": [5, 5, 5, 5, 5]})
    result = zscore_detection(df, ["x"], threshold=3.0)
    assert result["zscore_flag"].sum() == 0


def test_iqr_flags_both_tails():
    # Symmetric body with one low and one high outlier.
    df = pd.DataFrame({"x": [-100, 10, 11, 12, 13, 14, 15, 16, 200]})
    result = iqr_detection(df, ["x"], k=1.5)
    assert bool(result["iqr_flag"].iloc[0]) is True   # low outlier
    assert bool(result["iqr_flag"].iloc[-1]) is True  # high outlier
    # The interior points should not be flagged.
    assert result["iqr_flag"].iloc[1:-1].sum() == 0


def test_iqr_score_is_zero_inside_fences():
    df = pd.DataFrame({"x": [10, 11, 12, 13, 14, 15]})
    result = iqr_detection(df, ["x"], k=1.5)
    # No outliers, so no flags and all distances are zero.
    assert result["iqr_flag"].sum() == 0
    assert np.allclose(result["iqr_score"].values, 0.0)


def test_control_chart_flags_spike():
    # A flat series with a single large spike in the middle.
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    values = np.full(20, 100.0)
    values[10] = 1000.0
    series = pd.Series(values, index=idx)
    result = control_chart_detection(series, method="ewma", span=5, num_std=3.0)
    # The spike must be flagged.
    assert bool(result.loc[idx[10], "cc_flag"]) is True
    assert result["cc_flag"].sum() >= 1


def test_build_time_metric_daily_sum():
    df = pd.DataFrame(
        {
            "TransactionDate": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-03"]
            ),
            "TransactionAmount": [100.0, 50.0, 200.0],
        }
    )
    ts = build_time_metric(
        df, "TransactionDate", "TransactionAmount", aggregation="sum", frequency="D"
    )
    # Jan 1 sums to 150, Jan 2 has no rows so should be filled with 0, Jan 3 is 200.
    assert ts.loc["2024-01-01"] == 150.0
    assert ts.loc["2024-01-02"] == 0.0
    assert ts.loc["2024-01-03"] == 200.0
