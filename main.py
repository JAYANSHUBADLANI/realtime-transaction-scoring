"""Pipeline entry point.

Run the full anomaly scoring workflow end to end:

    python main.py

The script loads the data, splits it into a reference window (used only to
fit the z-score/IQR/Isolation Forest statistics) and a scored population
(everything after the reference window, scored and evaluated against those
frozen statistics, never the other way around), persists the fitted
reference as an artifact, builds the combined alert table, prioritizes
incidents, writes all outputs to ``reports/``, and prints a concise summary.

This fit-once/score-many split (see ``src/reference.py``) is what a
streaming service can reuse unchanged: the artifact written to
``artifacts/`` here is exactly what a Cloud Run scorer would load at
startup and score single incoming transactions against.

Outputs written to reports/:
    scored_transactions.csv  every scored transaction with its detector flags/scores
    alerts.csv               only the flagged transactions
    incident_queue.csv       ranked queue with severity buckets
    control_chart.csv        the daily metric with control limits and flags
    evaluation_metrics.json  precision/recall/PR-AUC/lift vs the real label
    run_summary.json         headline numbers, also consumed by the app/README

Outputs written to artifacts/:
    reference.json            frozen z-score/IQR statistics, JSON
    isolation_forest.joblib   frozen Isolation Forest model
"""

from __future__ import annotations

import json
import os
from typing import Dict

import pandas as pd

from src import eda
from src.alerting import DETECTION_METHODS, build_alerts
from src.evaluation import evaluate_all
from src.load_data import PROJECT_ROOT, load_config, load_transactions
from src.models import agreement_rate
from src.prioritization import build_incident_queue
from src.reference import fit_reference, score_with_reference
from src.statistical_methods import build_time_metric, control_chart_detection

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")


def run_pipeline(config: Dict) -> Dict:
    """Execute the whole pipeline and return the headline summary dict."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1. Load data (already restricted to data.scored_types, feature
    # engineered, and given a synthetic event_time; see load_data.py).
    df = load_transactions(config)
    n_records = len(df)

    # 2. EDA figures over the whole dataset (saved for the README/notebook).
    eda.numeric_distribution_figure(
        df,
        config["data"]["numeric_features"],
        os.path.join(FIGURES_DIR, "numeric_distributions.png"),
    )
    daily = eda.daily_volume(
        df, config["data"]["synthetic_timestamp_column"], "amount"
    )
    eda.time_series_figure(daily, os.path.join(FIGURES_DIR, "daily_volume.png"))

    # Control chart stays a periodic rollup over the whole dataset, not a
    # per-transaction score: a single incoming row has no completed day to
    # compare against yet. See src/reference.py's module docstring.
    cc_cfg = config["statistical"]["control_chart"]
    metric_series = build_time_metric(
        df,
        config["data"]["synthetic_timestamp_column"],
        cc_cfg["metric"],
        cc_cfg["aggregation"],
        cc_cfg["frequency"],
    )
    control_chart = control_chart_detection(
        metric_series,
        method=cc_cfg["method"],
        span=cc_cfg["span"],
        num_std=cc_cfg["num_std"],
    )

    # 3. Split into a reference window (fit only) and a scored population
    # (score and evaluate only). The reference window is never scored or
    # reported on, so results reflect statistics applied to data strictly
    # after the window they were fit on, exactly like a streaming service
    # would see it.
    reference_step_max = config["data"]["reference_step_max"]
    step_col = config["data"]["step_column"]
    reference_df = df[df[step_col] <= reference_step_max].reset_index(drop=True)
    scoring_df = df[df[step_col] > reference_step_max].reset_index(drop=True)

    # 4. Fit the reference once, persist it, then score the (much larger)
    # remaining population against that frozen reference.
    reference = fit_reference(reference_df, config, reference_step_max=reference_step_max)
    reference.save(ARTIFACTS_DIR)

    detector_scores = score_with_reference(scoring_df, config, reference)
    scored = scoring_df.copy()
    for col in detector_scores.columns:
        scored[col] = detector_scores[col]
    scored["n_methods"] = scored[DETECTION_METHODS].sum(axis=1)

    # 5. Agreement between the union of statistical flags and the forest.
    statistical_flag = scored["zscore_flag"] | scored["iqr_flag"]
    overlap = agreement_rate(statistical_flag, scored["iforest_flag"])

    # 6. Build alerts and prioritize, over the scored population only.
    zscore_result = scored[["zscore_flag", "zscore_score"]]
    iqr_result = scored[["iqr_flag", "iqr_score"]]
    iforest_result = scored[["iforest_flag", "iforest_score"]]
    alerts = build_alerts(scoring_df, zscore_result, iqr_result, iforest_result, config)
    incident_queue = build_incident_queue(alerts, config)

    # 7. Evaluate every detector against the real isFraud label. The label
    # is used only here, never fed into fitting or scoring above.
    evaluation = evaluate_all(
        scored,
        label_col=config["data"]["label_column"],
        baseline_flag_col=config["data"]["baseline_flag_column"],
    )
    with open(
        os.path.join(REPORTS_DIR, "evaluation_metrics.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(evaluation, f, indent=2)

    # 8. Write outputs.
    scored.to_csv(os.path.join(REPORTS_DIR, "scored_transactions.csv"), index=False)
    alerts.to_csv(os.path.join(REPORTS_DIR, "alerts.csv"), index=False)
    incident_queue.to_csv(os.path.join(REPORTS_DIR, "incident_queue.csv"), index=False)
    control_chart.to_csv(os.path.join(REPORTS_DIR, "control_chart.csv"))

    severity_counts = (
        incident_queue["severity"].value_counts().to_dict()
        if not incident_queue.empty
        else {}
    )

    summary = {
        "n_records": int(n_records),
        "reference_rows": int(len(reference_df)),
        "scored_rows": int(len(scoring_df)),
        "anomalies": {
            "zscore": int(scored["zscore_flag"].sum()),
            "iqr": int(scored["iqr_flag"].sum()),
            "control_chart_periods": int(control_chart["cc_flag"].sum()),
            "isolation_forest": int(scored["iforest_flag"].sum()),
        },
        "statistical_union_flags": int(statistical_flag.sum()),
        "agreement": {
            "jaccard": round(overlap["jaccard"], 4),
            "agreement_rate": round(overlap["agreement_rate"], 4),
            "intersection": overlap["intersection"],
            "union": overlap["union"],
        },
        "n_alerts": int(len(alerts)),
        "severity_counts": {
            "High": int(severity_counts.get("High", 0)),
            "Medium": int(severity_counts.get("Medium", 0)),
            "Low": int(severity_counts.get("Low", 0)),
        },
        "control_chart_periods_total": int(len(control_chart)),
        "evaluation": evaluation,
    }

    with open(os.path.join(REPORTS_DIR, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def print_summary(summary: Dict) -> None:
    """Print a concise, human readable run summary."""
    print("\n" + "=" * 60)
    print("REAL-TIME TRANSACTION SCORING, BATCH RUN SUMMARY")
    print("=" * 60)
    print(f"Reference window rows (fit only):  {summary['reference_rows']:,}")
    print(f"Scored population rows:            {summary['scored_rows']:,}")
    print("\nAnomalies flagged by method:")
    print(f"  Z-score:                  {summary['anomalies']['zscore']:,}")
    print(f"  IQR:                      {summary['anomalies']['iqr']:,}")
    print(f"  Isolation Forest:         {summary['anomalies']['isolation_forest']:,}")
    print(
        f"  Control chart periods:    {summary['anomalies']['control_chart_periods']:,}"
        f" of {summary['control_chart_periods_total']:,} periods"
    )
    print(f"\nStatistical union flags:    {summary['statistical_union_flags']:,}")
    print(
        f"Statistical vs forest agreement rate: "
        f"{summary['agreement']['agreement_rate'] * 100:.2f}%"
        f" (Jaccard {summary['agreement']['jaccard']:.3f})"
    )
    print(f"\nTotal alerts:               {summary['n_alerts']:,}")
    print("Incidents by severity:")
    print(f"  High:                     {summary['severity_counts']['High']:,}")
    print(f"  Medium:                   {summary['severity_counts']['Medium']:,}")
    print(f"  Low:                      {summary['severity_counts']['Low']:,}")

    ev = summary["evaluation"]
    print(f"\nReal fraud rate in scored population: {ev['base_fraud_rate'] * 100:.3f}%"
          f"  ({ev['n_fraud']:,} of {ev['n_scored']:,})")
    print("Detector precision / recall / PR-AUC:")
    for name, d in ev["detectors"].items():
        pr_auc = f"{d['pr_auc']:.4f}" if d["pr_auc"] is not None else "n/a"
        print(
            f"  {name:<28} P={d['precision']:.4f}  R={d['recall']:.4f}  "
            f"F1={d['f1']:.4f}  PR-AUC={pr_auc}  (caught {d['true_positives']}"
            f" of {d['true_positives'] + d['false_negatives']} fraud)"
        )
    top1 = ev["lift_at_k_combined_score"].get("top_0.01")
    if top1:
        print(
            f"\nCombined score, top 1% reviewed: catches "
            f"{top1['fraud_caught_share_of_all_fraud'] * 100:.1f}% of all fraud, "
            f"{top1['lift_over_base_rate']:.1f}x lift over the base rate."
        )

    print("\nOutputs written to reports/, reference artifact written to artifacts/.")
    print("Run 'streamlit run app.py' to explore.")
    print("=" * 60 + "\n")


def main() -> None:
    config = load_config()
    summary = run_pipeline(config)
    print_summary(summary)


if __name__ == "__main__":
    main()
