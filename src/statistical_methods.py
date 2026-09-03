"""Statistical anomaly detection methods.

This is the analytical core of the project. It implements three classic,
transparent techniques, each fully vectorized with pandas and numpy:

1. Z-score: flags points that sit more than ``threshold`` standard deviations
   from the column mean.
2. IQR (Tukey fences): flags points outside Q1 minus k * IQR and
   Q3 plus k * IQR.
3. Control chart: flags points on a time ordered metric that fall outside
   EWMA or rolling control limits.

Every function returns a tidy frame indexed like its input so results can be
joined back onto the original transactions, together with an interpretable
anomaly score.

``zscore_detection``/``iqr_detection`` below compute their statistics from
whatever dataframe is passed in (fit and score together), which is fine for
a one-shot batch report but not for streaming, where a single incoming
transaction cannot wait for the mean of transactions that have not happened
yet. ``main.py``'s pipeline uses the fit-once/score-many split in
``src/reference.py`` instead; these two functions are kept for the
one-shot batch case and are what ``tests/test_statistical_methods.py``
verifies the underlying math against. ``control_chart_detection`` has no
such split: it is a periodic rollup over a completed window of days, not
something a single transaction can be scored against on its own, so it
stays as-is and runs over the whole dataset in ``main.py``, independent of
the reference/scoring split; see the README's project status section.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def zscore_detection(
    df: pd.DataFrame,
    columns: List[str],
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Flag rows whose values are extreme on a z-score basis.

    For each column the standardized score ``z = (x - mean) / std`` is computed.
    A row is flagged when the maximum absolute z-score across the screened
    columns exceeds ``threshold``. The anomaly score is that maximum absolute
    z-score, so larger means more extreme.

    Parameters
    ----------
    df:
        Input data.
    columns:
        Numeric columns to screen.
    threshold:
        Number of standard deviations beyond which a point is anomalous.

    Returns
    -------
    pandas.DataFrame
        Same index as ``df`` with one ``<col>_zscore`` column per input column,
        a ``zscore_score`` column (max absolute z), and a boolean
        ``zscore_flag`` column.
    """
    result = pd.DataFrame(index=df.index)
    z_frames = []
    for col in columns:
        series = df[col].astype(float)
        std = series.std(ddof=0)
        # Guard against a zero standard deviation (a constant column).
        if std == 0 or np.isnan(std):
            z = pd.Series(0.0, index=df.index)
        else:
            z = (series - series.mean()) / std
        result[f"{col}_zscore"] = z
        z_frames.append(z.abs())

    abs_z = pd.concat(z_frames, axis=1)
    result["zscore_score"] = abs_z.max(axis=1)
    result["zscore_flag"] = result["zscore_score"] > threshold
    return result


def iqr_detection(
    df: pd.DataFrame,
    columns: List[str],
    k: float = 1.5,
) -> pd.DataFrame:
    """Flag rows outside the Tukey IQR fences.

    For each column the fences are ``Q1 - k * IQR`` and ``Q3 + k * IQR``.
    A row is flagged when any screened column lies outside its fences. The
    anomaly score is the largest distance beyond a fence expressed in IQR
    units, so a value exactly on a fence scores 0 and values further out score
    higher.

    Parameters
    ----------
    df:
        Input data.
    columns:
        Numeric columns to screen.
    k:
        Fence multiplier; 1.5 is the standard Tukey choice.

    Returns
    -------
    pandas.DataFrame
        Same index as ``df`` with one ``<col>_iqr_flag`` per input column, an
        ``iqr_score`` column, and a boolean ``iqr_flag`` column.
    """
    result = pd.DataFrame(index=df.index)
    distances = []
    for col in columns:
        series = df[col].astype(float)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        flag = (series < lower) | (series > upper)
        result[f"{col}_iqr_flag"] = flag

        # Distance beyond the nearest fence, normalized by IQR width.
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


def control_chart_detection(
    series: pd.Series,
    method: str = "ewma",
    span: int = 7,
    num_std: float = 3.0,
) -> pd.DataFrame:
    """Apply an EWMA or rolling control chart to a time ordered metric.

    The center line is either the exponentially weighted moving average
    (``method='ewma'``) or the simple rolling mean (``method='rolling'``).
    The control limits sit ``num_std`` standard deviations of the residuals
    (the deviations of each point from the center line) above and below the
    center. Using the residual spread, rather than a rolling window that would
    include the very spike being tested, keeps a single large outlier from
    inflating its own limits and masking itself.

    Parameters
    ----------
    series:
        A time indexed numeric series, for example daily total amount. It
        should already be sorted by time.
    method:
        ``'ewma'`` or ``'rolling'``.
    span:
        Span for the EWMA, or window length for the rolling mean.
    num_std:
        Width of the control limits in standard deviations.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``series`` with columns ``value``, ``center``, ``ucl``,
        ``lcl``, ``cc_score`` (absolute deviation from center in std units),
        and ``cc_flag``.
    """
    series = series.astype(float)
    out = pd.DataFrame(index=series.index)
    out["value"] = series

    if method == "ewma":
        center = series.ewm(span=span, adjust=False).mean()
    elif method == "rolling":
        # min_periods=1 so the early points still get a center line; the rolling
        # mean simply uses whatever history is available.
        center = series.rolling(window=span, min_periods=1).mean()
    else:
        raise ValueError("method must be 'ewma' or 'rolling'")

    # Sigma is the spread of the residuals about the center line. This is a
    # single, stable estimate of process noise, so one extreme point cannot
    # widen the limits enough to hide itself.
    residuals = series - center
    sigma = residuals.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        # Fall back to the overall series spread, then to 1, so the limits are
        # always well defined even for a perfectly flat series.
        fallback = series.std(ddof=0)
        sigma = fallback if fallback and not np.isnan(fallback) else 1.0

    out["center"] = center
    out["ucl"] = center + num_std * sigma
    out["lcl"] = center - num_std * sigma
    out["cc_score"] = (residuals.abs() / sigma).fillna(0.0)
    out["cc_flag"] = (series > out["ucl"]) | (series < out["lcl"])
    return out


def build_time_metric(
    df: pd.DataFrame,
    timestamp_col: str,
    metric_col: str,
    aggregation: str = "sum",
    frequency: str = "D",
) -> pd.Series:
    """Aggregate transactions into a regular time series for the control chart.

    Parameters
    ----------
    df:
        Transaction frame.
    timestamp_col:
        Name of the datetime column.
    metric_col:
        Column to aggregate (used for sum; ignored for count).
    aggregation:
        ``'sum'`` of ``metric_col`` per period, or ``'count'`` of rows.
    frequency:
        Pandas offset alias, for example ``'D'`` for daily.

    Returns
    -------
    pandas.Series
        Time indexed metric, sorted ascending, with gaps filled as 0.
    """
    ordered = df.dropna(subset=[timestamp_col]).set_index(timestamp_col).sort_index()
    grouper = pd.Grouper(freq=frequency)
    if aggregation == "sum":
        ts = ordered[metric_col].groupby(grouper).sum()
    elif aggregation == "count":
        ts = ordered[metric_col].groupby(grouper).count()
    else:
        raise ValueError("aggregation must be 'sum' or 'count'")
    return ts.asfreq(frequency, fill_value=0)
