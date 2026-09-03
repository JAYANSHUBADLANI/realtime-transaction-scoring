"""Alerting engine.

This module combines the signals produced by the statistical detectors and the
Isolation Forest into a single alert table. On top of the data driven detectors
it also applies hard business rules loaded from ``config.yaml`` (for example
"amount over X" or "login attempts at or above Y"). Every alert records exactly
which methods and rules fired, so an analyst can see why a transaction
surfaced.
"""

from __future__ import annotations

import operator
from typing import Dict, List

import pandas as pd


# Map the operator strings allowed in config.yaml to real functions.
_OPERATORS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}

# The three data driven detectors whose agreement is counted.
DETECTION_METHODS = ["zscore_flag", "iqr_flag", "iforest_flag"]


def apply_rules(
    df: pd.DataFrame,
    zscore_result: pd.DataFrame,
    config: Dict,
) -> pd.DataFrame:
    """Evaluate the configured business rules against each transaction.

    Parameters
    ----------
    df:
        The original transaction frame.
    zscore_result:
        Output of ``zscore_detection``; used by the ``high_zscore`` rule.
    config:
        Parsed configuration.

    Returns
    -------
    pandas.DataFrame
        One boolean column per rule, named ``rule_<name>``, indexed like ``df``.
    """
    rules = config["alerting"]["rules"]
    out = pd.DataFrame(index=df.index)

    for name, spec in rules.items():
        if name == "high_zscore":
            # Special rule comparing the precomputed max absolute z-score.
            out[f"rule_{name}"] = zscore_result["zscore_score"] > spec["value"]
            continue

        column = spec["column"]
        op = _OPERATORS[spec["operator"]]
        out[f"rule_{name}"] = op(df[column], spec["value"])

    return out


def build_alerts(
    df: pd.DataFrame,
    zscore_result: pd.DataFrame,
    iqr_result: pd.DataFrame,
    iforest_result: pd.DataFrame,
    config: Dict,
) -> pd.DataFrame:
    """Assemble the combined alert table.

    A transaction becomes an alert if any detector flag or any business rule
    fires. The returned frame carries the identifying fields, the per method
    flags and scores, the rule flags, a human readable ``triggered_by`` list,
    and ``n_methods`` (how many of the three data driven detectors agreed).

    Parameters
    ----------
    df:
        Original transactions.
    zscore_result, iqr_result, iforest_result:
        Detector outputs aligned to ``df``.
    config:
        Parsed configuration.

    Returns
    -------
    pandas.DataFrame
        The alert table, one row per flagged transaction.
    """
    rule_flags = apply_rules(df, zscore_result, config)

    # Assemble everything on the shared index.
    combined = pd.DataFrame(index=df.index)

    # Identifying and context columns, kept if present, so an analyst
    # reading alerts.csv/incident_queue.csv can see which real transaction
    # each row is, not just its detector flags and scores.
    context_cols = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "orig_emptied",
        "balance_error_orig",
        "balance_error_dest",
    ]
    timestamp_col = config["data"].get("synthetic_timestamp_column")
    if timestamp_col:
        context_cols.append(timestamp_col)
    for col in context_cols:
        if col in df.columns:
            combined[col] = df[col]

    # Detector flags and scores.
    combined["zscore_flag"] = zscore_result["zscore_flag"]
    combined["zscore_score"] = zscore_result["zscore_score"]
    combined["iqr_flag"] = iqr_result["iqr_flag"]
    combined["iqr_score"] = iqr_result["iqr_score"]
    combined["iforest_flag"] = iforest_result["iforest_flag"]
    combined["iforest_score"] = iforest_result["iforest_score"]

    # Rule flags.
    for col in rule_flags.columns:
        combined[col] = rule_flags[col]

    # Count how many data driven detectors agree on each row.
    combined["n_methods"] = combined[DETECTION_METHODS].sum(axis=1)

    # Build the triggered_by label from the method and rule flags that fired.
    rule_cols = list(rule_flags.columns)
    flag_cols = DETECTION_METHODS + rule_cols

    def _label(row: pd.Series) -> str:
        fired = []
        for col in flag_cols:
            if bool(row[col]):
                # Make names readable: zscore_flag -> zscore, rule_high_amount.
                fired.append(col.replace("_flag", ""))
        return ", ".join(fired)

    combined["triggered_by"] = combined[flag_cols].apply(_label, axis=1)

    # An alert is any row where at least one signal fired.
    any_signal = combined[flag_cols].any(axis=1)
    alerts = combined[any_signal].copy()

    return alerts
