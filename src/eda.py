"""Exploratory data analysis and data quality helpers.

These functions produce the numbers and figures used in the EDA notebook and
the README. They are deliberately small and pure so they can be reused from the
notebook, the pipeline, and the tests.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib

# Use a non interactive backend so the functions work in scripts and CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import pandas as pd


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for the numeric columns."""
    return df.describe(include="number").transpose()


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per column count and percentage of missing values."""
    missing = df.isna().sum()
    report = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_pct": (missing / len(df) * 100).round(3),
        }
    )
    return report.sort_values("missing_count", ascending=False)


def numeric_distribution_figure(
    df: pd.DataFrame,
    columns: List[str],
    out_path: str,
) -> str:
    """Save a grid of histograms for the given numeric columns.

    Returns the path written, so callers can record it.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = len(columns)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows))
    axes = axes.flatten()

    for ax, col in zip(axes, columns):
        ax.hist(df[col].dropna(), bins=40, color="#4C72B0", edgecolor="white")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    # Hide any unused subplot panels.
    for ax in axes[len(columns):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def categorical_counts(df: pd.DataFrame, columns: List[str]) -> Dict[str, pd.Series]:
    """Return value counts for each categorical column that exists."""
    return {col: df[col].value_counts() for col in columns if col in df.columns}


def daily_volume(
    df: pd.DataFrame, timestamp_col: str, amount_col: str = "amount"
) -> pd.DataFrame:
    """Aggregate transaction count and total amount per day.

    Returns an empty frame if the timestamp column is missing or unparsed.
    """
    if timestamp_col not in df.columns or amount_col not in df.columns:
        return pd.DataFrame()
    tmp = df.dropna(subset=[timestamp_col]).copy()
    if tmp.empty:
        return pd.DataFrame()
    tmp = tmp.set_index(timestamp_col).sort_index()
    daily = pd.DataFrame(
        {
            "transaction_count": tmp[amount_col].resample("D").count(),
            "total_amount": tmp[amount_col].resample("D").sum(),
        }
    )
    return daily


def time_series_figure(
    daily: pd.DataFrame,
    out_path: str,
) -> Optional[str]:
    """Save a two panel time series of daily count and total amount.

    Returns the path, or None if there is nothing to plot.
    """
    if daily.empty:
        return None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(daily.index, daily["transaction_count"], color="#4C72B0")
    axes[0].set_title("Daily transaction count")
    axes[0].set_ylabel("Count")
    axes[1].plot(daily.index, daily["total_amount"], color="#C44E52")
    axes[1].set_title("Daily total amount")
    axes[1].set_ylabel("Amount")
    axes[1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
