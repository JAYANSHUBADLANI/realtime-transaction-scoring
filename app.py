"""Streamlit dashboard for the transaction anomaly detection system.

The app reads the processed outputs written by ``main.py`` (in ``reports/``) so
it loads quickly and does not recompute anything. If those outputs are missing,
for example on a fresh Streamlit Cloud deploy with no data, it shows friendly
instructions instead of crashing.

Sections:
    1. Overview KPIs
    2. Anomaly explorer with filters
    3. Time series view with flagged points highlighted
    4. Prioritized incident table with severity color coding
"""

from __future__ import annotations

import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

st.set_page_config(
    page_title="Transaction Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
)

SEVERITY_COLORS = {"High": "#C0392B", "Medium": "#E67E22", "Low": "#27AE60"}


def _path(name: str) -> str:
    return os.path.join(REPORTS_DIR, name)


def outputs_ready() -> bool:
    """True only if the key pipeline outputs exist on disk."""
    needed = ["run_summary.json", "scored_transactions.csv", "incident_queue.csv"]
    return all(os.path.exists(_path(n)) for n in needed)


@st.cache_data
def load_outputs():
    """Load the processed outputs, cached for snappy interaction."""
    with open(_path("run_summary.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)
    scored = pd.read_csv(_path("scored_transactions.csv"))
    incidents = pd.read_csv(_path("incident_queue.csv"))
    control_chart = (
        pd.read_csv(_path("control_chart.csv"), index_col=0, parse_dates=[0])
        if os.path.exists(_path("control_chart.csv"))
        else pd.DataFrame()
    )
    return summary, scored, incidents, control_chart


def show_missing_data_message() -> None:
    """Friendly screen shown when the pipeline has not been run yet."""
    st.title("🛡️ Transaction Anomaly Detection")
    st.warning("No processed outputs found yet.")
    st.markdown(
        """
This dashboard reads results produced by the analysis pipeline. To generate
them, follow these steps locally:

1. **Download the dataset** from Kaggle and place the CSV in `data/`.
   See `data/README.md` for the exact command.
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run the pipeline**: `python main.py`
4. **Launch the app**: `streamlit run app.py`

The pipeline writes its outputs to the `reports/` folder, which this app then
visualizes.
"""
    )
    st.info(
        "Dataset: Bank Transaction Dataset for Fraud Detection (valakhorasani) "
        "on Kaggle."
    )


def render_overview(summary: dict) -> None:
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records analyzed", f"{summary['n_records']:,}")
    c2.metric("Total alerts", f"{summary['n_alerts']:,}")
    c3.metric(
        "Stat vs forest agreement",
        f"{summary['agreement']['agreement_rate'] * 100:.1f}%",
    )
    c4.metric("High severity", f"{summary['severity_counts']['High']:,}")

    st.markdown("**Anomalies flagged by method**")
    methods = summary["anomalies"]
    method_df = pd.DataFrame(
        {
            "method": ["Z-score", "IQR", "Isolation Forest", "Control chart periods"],
            "count": [
                methods["zscore"],
                methods["iqr"],
                methods["isolation_forest"],
                methods["control_chart_periods"],
            ],
        }
    )
    fig = px.bar(
        method_df,
        x="method",
        y="count",
        text="count",
        color="method",
        title="Anomaly counts by detection method",
    )
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, use_container_width=True)


def render_explorer(scored: pd.DataFrame) -> None:
    st.subheader("Anomaly explorer")
    st.caption("Filter transactions and inspect which detectors flagged them.")

    col1, col2, col3 = st.columns(3)
    with col1:
        method_filter = st.selectbox(
            "Flagged by method",
            ["Any method", "Z-score", "IQR", "Isolation Forest", "No flag"],
        )
    with col2:
        if "TransactionType" in scored.columns:
            types = ["All"] + sorted(scored["TransactionType"].dropna().unique().tolist())
            type_filter = st.selectbox("Transaction type", types)
        else:
            type_filter = "All"
    with col3:
        if "Channel" in scored.columns:
            channels = ["All"] + sorted(scored["Channel"].dropna().unique().tolist())
            channel_filter = st.selectbox("Channel", channels)
        else:
            channel_filter = "All"

    view = scored.copy()
    if method_filter == "Z-score":
        view = view[view["zscore_flag"]]
    elif method_filter == "IQR":
        view = view[view["iqr_flag"]]
    elif method_filter == "Isolation Forest":
        view = view[view["iforest_flag"]]
    elif method_filter == "No flag":
        view = view[view["n_methods"] == 0]

    if type_filter != "All" and "TransactionType" in view.columns:
        view = view[view["TransactionType"] == type_filter]
    if channel_filter != "All" and "Channel" in view.columns:
        view = view[view["Channel"] == channel_filter]

    st.write(f"Showing {len(view):,} of {len(scored):,} transactions.")
    st.dataframe(view, use_container_width=True, height=420)

    # Amount distribution split by whether any detector flagged the row.
    if "TransactionAmount" in view.columns and not view.empty:
        view = view.copy()
        view["flagged"] = view["n_methods"] > 0
        fig = px.histogram(
            view,
            x="TransactionAmount",
            color="flagged",
            nbins=50,
            barmode="overlay",
            title="Transaction amount distribution (flagged vs not)",
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)


def render_timeseries(control_chart: pd.DataFrame) -> None:
    st.subheader("Time series control chart")
    if control_chart.empty:
        st.info("No time series available (timestamp column missing or unparsed).")
        return

    st.caption(
        "Daily metric with EWMA center line and control limits. Points outside "
        "the limits are flagged."
    )
    cc = control_chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cc.index, y=cc["value"], name="Value", mode="lines"))
    fig.add_trace(go.Scatter(x=cc.index, y=cc["center"], name="Center", mode="lines"))
    fig.add_trace(
        go.Scatter(x=cc.index, y=cc["ucl"], name="UCL", line=dict(dash="dash"))
    )
    fig.add_trace(
        go.Scatter(x=cc.index, y=cc["lcl"], name="LCL", line=dict(dash="dash"))
    )
    flagged = cc[cc["cc_flag"]]
    if not flagged.empty:
        fig.add_trace(
            go.Scatter(
                x=flagged.index,
                y=flagged["value"],
                name="Flagged",
                mode="markers",
                marker=dict(color="#C0392B", size=10, symbol="x"),
            )
        )
    fig.update_layout(height=460, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)


def _style_severity(row: pd.Series):
    color = SEVERITY_COLORS.get(row.get("severity", ""), "")
    return [f"background-color: {color}; color: white" if color else "" for _ in row]


def render_incidents(incidents: pd.DataFrame) -> None:
    st.subheader("Prioritized incident queue")
    if incidents.empty:
        st.info("No incidents in the queue.")
        return

    sev_filter = st.multiselect(
        "Severity",
        options=["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )
    view = incidents[incidents["severity"].isin(sev_filter)]
    st.write(f"Showing {len(view):,} incidents.")

    styled = view.style.apply(_style_severity, axis=1)
    st.dataframe(styled, use_container_width=True, height=460)

    st.download_button(
        "Download incident queue (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="incident_queue.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("🛡️ Transaction Anomaly Detection and Alerting")
    st.caption(
        "Statistics first monitoring of transaction streams: z-score, IQR, "
        "EWMA control charts, and Isolation Forest, with prioritized incidents."
    )

    if not outputs_ready():
        show_missing_data_message()
        return

    summary, scored, incidents, control_chart = load_outputs()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Anomaly explorer", "Time series", "Incident queue"]
    )
    with tab1:
        render_overview(summary)
    with tab2:
        render_explorer(scored)
    with tab3:
        render_timeseries(control_chart)
    with tab4:
        render_incidents(incidents)


if __name__ == "__main__":
    main()
