"""
CampusNet Guardian - Intelligent Network Monitoring & Anomaly Detection System
Run with:  streamlit run app.py

ACADEMIC PROTOTYPE: uses synthetic/demo network traffic only.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

import generate_data
from src import charts
from src.alerts import HIGH, WARNING, summarize_alerts
from src.anomaly_detection import analyze_dataframe, ensure_model
from src.data_processing import (REQUIRED_COLUMNS, DataValidationError,
                                 load_dataset, read_csv_bytes)

st.set_page_config(page_title="CampusNet Guardian", page_icon="🛡️", layout="wide")

DISCLAIMER = ("This project is an academic prototype that uses synthetic/demo network traffic. "
              "It is intended for educational purposes and does not represent real-time "
              "monitoring of a production campus network.")

TABLE_COLUMNS = ["timestamp", "source_ip", "destination_ip", "source_port",
                 "destination_port", "protocol", "packet_count", "bytes_transferred",
                 "duration", "anomaly_score", "status", "alert_level"]

CSS = """
<style>
.kpi-card {background:#111a2e; border:1px solid #1e2b47; border-radius:12px;
           padding:16px 18px; margin-bottom:8px;}
.kpi-label {color:#8fa3c7; font-size:0.85rem; letter-spacing:0.05em; text-transform:uppercase;}
.kpi-value {color:#e6edf7; font-size:2rem; font-weight:700; line-height:1.3;}
.app-title {font-size:2.2rem; font-weight:800; letter-spacing:0.12em; color:#00d4ff; margin-bottom:0;}
.app-subtitle {color:#8fa3c7; margin-top:0; margin-bottom:1rem;}
</style>
"""


# ---------------------------------------------------------------- loading ---
@st.cache_resource(show_spinner="Loading anomaly detection model...")
def get_bundle():
    """Load the saved model (trains it automatically if the file is missing)."""
    return ensure_model()


@st.cache_data(show_spinner="Analyzing network traffic...")
def get_default_results():
    """Analyze the built-in synthetic dataset (created automatically if missing)."""
    results, report = analyze_dataframe(load_dataset(), get_bundle())
    return results


@st.cache_data(show_spinner="Analyzing uploaded file...")
def analyze_uploaded(data):
    return analyze_dataframe(read_csv_bytes(data), get_bundle())


# --------------------------------------------------------------- UI helpers --
def kpi_card(label, value, accent="#00d4ff"):
    st.markdown(
        f'<div class="kpi-card" style="border-left:4px solid {accent}">'
        f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True)


def kpi_row(df):
    total = len(df)
    anomalies = int((df["status"] == "ANOMALOUS").sum())
    pct = (anomalies / total * 100) if total else 0
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Traffic Records", f"{total:,}")
    with c2:
        kpi_card("Normal Traffic", f"{total - anomalies:,}", "#00c896")
    with c3:
        kpi_card("Anomalies Detected", f"{anomalies:,}", "#ff4b6e")
    with c4:
        kpi_card("Anomaly Percentage", f"{pct:.2f}%", "#ffb020")


def show_alert(level, message):
    text = f"**{level}** — {message}"
    if level == HIGH:
        st.error(text, icon="🚨")
    elif level == WARNING:
        st.warning(text, icon="⚠️")
    else:
        st.success(text, icon="✅")


def human_bytes(value):
    value = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} TB"


def page_header(title):
    st.markdown('<p class="app-title">CAMPUSNET GUARDIAN</p>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">Intelligent Network Monitoring &amp; Anomaly '
                'Detection System</p>', unsafe_allow_html=True)
    st.header(title)


def chart_grid(df):
    """The six main charts, shown two per row."""
    left, right = st.columns(2)
    left.plotly_chart(charts.status_donut(df))
    right.plotly_chart(charts.traffic_over_time(df))
    left, right = st.columns(2)
    left.plotly_chart(charts.protocol_chart(df))
    right.plotly_chart(charts.top_ports_chart(df))
    left, right = st.columns(2)
    left.plotly_chart(charts.packet_hist(df))
    right.plotly_chart(charts.bytes_hist(df))


# -------------------------------------------------------------------- pages --
def page_dashboard():
    page_header("📊 Dashboard")
    df = get_default_results()
    kpi_row(df)
    st.caption("Data shown: built-in SYNTHETIC demo traffic.")
    chart_grid(df)


def page_traffic_analysis():
    page_header("🔎 Traffic Analysis")
    df = get_default_results()

    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        protocols = c1.multiselect("Protocol", sorted(df["protocol"].unique()),
                                   default=sorted(df["protocol"].unique()))
        statuses = c2.multiselect("Status", ["NORMAL", "ANOMALOUS"],
                                  default=["NORMAL", "ANOMALOUS"])
        port_text = c3.text_input("Destination port (optional, e.g. 443)")
        c4, c5 = st.columns(2)
        search = c4.text_input("Search by IP address (source or destination)")
        min_date, max_date = df["timestamp"].min().date(), df["timestamp"].max().date()
        date_range = c5.date_input("Date range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)

    mask = df["protocol"].isin(protocols) & df["status"].isin(statuses)
    if port_text.strip():
        if port_text.strip().isdigit():
            mask &= df["destination_port"] == int(port_text.strip())
        else:
            st.warning("Please enter the port as a number, for example 443.")
    if search.strip():
        term = search.strip()
        mask &= (df["source_ip"].str.contains(term, regex=False)
                 | df["destination_ip"].str.contains(term, regex=False))
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        mask &= (df["timestamp"].dt.date >= date_range[0]) & (df["timestamp"].dt.date <= date_range[1])

    view = df[mask]
    if view.empty:
        st.info("No records match the selected filters.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matching records", f"{len(view):,}")
    c2.metric("Avg packets", f"{view['packet_count'].mean():,.1f}")
    c3.metric("Avg data per record", human_bytes(view["bytes_transferred"].mean()))
    c4.metric("Avg duration (s)", f"{view['duration'].mean():,.2f}")

    st.subheader("Records")
    st.dataframe(view[TABLE_COLUMNS])
    st.subheader("Summary statistics")
    st.dataframe(view[["packet_count", "bytes_transferred", "duration"]].describe())


def page_anomaly_detection():
    page_header("🚨 Anomaly Detection")
    df = get_default_results()
    kpi_row(df)

    alerts = summarize_alerts(df)
    c1, c2 = st.columns(2)
    c1.metric("HIGH alerts", f"{alerts[HIGH]:,}")
    c2.metric("WARNING alerts", f"{alerts[WARNING]:,}")

    flagged = df[df["status"] == "ANOMALOUS"]
    if flagged.empty:
        st.success("No potential anomalies were found in this dataset.")
        return

    st.subheader("Highest-risk records")
    level = st.radio("Show", ["All flagged", HIGH, WARNING], horizontal=True)
    top_n = st.slider("Number of records", 5, 50, 15)
    pool = flagged if level == "All flagged" else flagged[flagged["alert_level"] == level]
    top = pool.sort_values("anomaly_score", ascending=False).head(top_n)
    st.dataframe(top[TABLE_COLUMNS + ["why_flagged"]])

    st.subheader("Why was this flagged?")
    if top.empty:
        st.info("No records at this alert level.")
    else:
        choice = st.selectbox(
            "Choose a record", top.index,
            format_func=lambda i: (f"#{i} | {top.loc[i, 'source_ip']} → "
                                   f"{top.loc[i, 'destination_ip']}:{top.loc[i, 'destination_port']} "
                                   f"| score {top.loc[i, 'anomaly_score']}"))
        row = top.loc[choice]
        show_alert(row["alert_level"], row["alert_message"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Packets", f"{row['packet_count']:,}")
        m2.metric("Data transferred", human_bytes(row["bytes_transferred"]))
        m3.metric("Duration (s)", f"{row['duration']:,.3f}")
        m4.metric("Anomaly score", row["anomaly_score"])
        for reason in row["why_flagged"].split("; "):
            st.markdown(f"- {reason}")
        st.caption("These reasons come from simple rules that compare the record with the "
                   "normal range. The Isolation Forest does not give an exact cause, so treat "
                   "them as hints, not proof.")

    left, right = st.columns(2)
    left.plotly_chart(charts.score_scatter(df))
    right.plotly_chart(charts.reason_bar(df))


def _start_live():
    st.session_state.live_running = True


def _stop_live():
    st.session_state.live_running = False


def _clear_live():
    st.session_state.live_history = []


def page_live_monitoring():
    page_header("📡 Live Monitoring (Simulation)")
    st.info("This is a SIMULATION. Records are randomly generated synthetic traffic — "
            "no real network traffic is captured or inspected.")
    bundle = get_bundle()

    st.session_state.setdefault("live_running", False)
    st.session_state.setdefault("live_history", [])

    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 2])
    c1.button("▶ Start Monitoring Simulation", on_click=_start_live, type="primary")
    c2.button("⏹ Stop", on_click=_stop_live)
    c3.button("🗑 Clear", on_click=_clear_live)
    c4.slider("Chance a new record is unusual (%)", 0, 50, 15, key="live_rate")

    running = st.session_state.live_running
    st.caption("Status: " + ("🟢 simulation running (new record every 1.5 seconds)"
                             if running else "⚪ stopped"))

    @st.fragment(run_every=1.5 if running else None)
    def live_panel():
        if st.session_state.live_running:
            record = generate_data.generate_live_record(st.session_state.live_rate / 100)
            result, _ = analyze_dataframe(record, bundle)
            history = st.session_state.live_history
            history.append(result.iloc[0].to_dict())
            st.session_state.live_history = history[-100:]

        history = st.session_state.live_history
        if not history:
            st.write("Press **Start Monitoring Simulation** to begin.")
            return

        hist = pd.DataFrame(history)
        latest = hist.iloc[-1]
        show_alert(latest["alert_level"], latest["alert_message"])

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Current status", latest["status"])
        m2.metric("Protocol", latest["protocol"])
        m3.metric("Packet count", f"{latest['packet_count']:,}")
        m4.metric("Bytes transferred", human_bytes(latest["bytes_transferred"]))
        m5.metric("Anomaly score", latest["anomaly_score"])
        st.write(f"**Source IP:** {latest['source_ip']}  →  "
                 f"**Destination IP:** {latest['destination_ip']}:{latest['destination_port']}")
        if latest["status"] == "ANOMALOUS":
            st.write(f"**Why flagged:** {latest['why_flagged']}")

        fig = px.line(hist.reset_index(), x="index", y="anomaly_score", markers=True,
                      title="Anomaly score of recent records (above 0 = potential anomaly)")
        fig.add_hline(y=0, line_dash="dash", line_color="#ffb020")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=300,
                          margin=dict(l=10, r=10, t=50, b=10))
        fig.update_xaxes(title_text="record number")
        st.plotly_chart(fig, key="live_chart")

        st.subheader("Latest records")
        st.dataframe(hist[TABLE_COLUMNS + ["alert_message"]].iloc[::-1].head(15))

    live_panel()


def page_upload():
    page_header("📤 Upload Dataset")
    st.write("Upload a CSV file with these columns: " + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS))
    sample = generate_data.generate_records(300, 0.10, seed=7).drop(columns="label")
    st.download_button("Download a sample CSV to try", sample.to_csv(index=False).encode("utf-8"),
                       file_name="sample_upload.csv", mime="text/csv")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is None:
        st.info("No file uploaded yet.")
        return

    try:
        results, report = analyze_uploaded(uploaded.getvalue())
    except DataValidationError as error:
        st.error(f"Could not analyze this file: {error}")
        return
    except Exception as error:   # last safety net so the app never crashes
        st.error(f"Something unexpected went wrong while analyzing the file: {error}")
        return

    st.success(f"Analysis complete: {report['rows_out']:,} of {report['rows_in']:,} rows analyzed.")
    removed = report["rows_in"] - report["rows_out"]
    if removed:
        st.warning(
            f"{removed:,} row(s) were removed during cleaning "
            f"(missing/unreadable values: {report['removed_missing_or_unreadable']}, "
            f"invalid ports: {report['removed_invalid_ports']}, "
            f"invalid numbers: {report['removed_invalid_numbers']}).")

    kpi_row(results)
    chart_grid(results)
    st.subheader("Analyzed results")
    export_columns = [c for c in results.columns if c not in ("prediction",)]
    st.dataframe(results[export_columns])
    st.download_button("Download Analysis Results",
                       results[export_columns].to_csv(index=False).encode("utf-8"),
                       file_name="campusnet_analysis_results.csv", mime="text/csv",
                       type="primary")


def page_about():
    page_header("ℹ️ About Project")
    st.markdown(f"""
**Project Title:** CampusNet Guardian: Intelligent Network Monitoring and Anomaly Detection System

**Problem Statement:** Campus networks carry a large amount of traffic. Manually checking it
for unusual behaviour (huge transfers, bursts of packets, strange ports) is slow and error-prone.

**Objective:** Build a simple dashboard that analyzes network traffic records, flags unusual
ones with machine learning, explains why, and shows clear alerts.

**How It Works**
1. Traffic records are loaded (synthetic data, or an uploaded CSV).
2. The data is cleaned and validated.
3. Extra features are calculated (average packet size, packets per second).
4. An Isolation Forest gives each record a score and a NORMAL / ANOMALOUS status.
5. Alerts and simple "why flagged" hints are generated.
6. Everything is shown in this Streamlit dashboard.

**Technology Stack:** Python, Streamlit, Pandas, NumPy, Scikit-learn, Plotly, Joblib

**Machine Learning Algorithm:** Isolation Forest — it builds many random trees; records that
are easy to separate from the rest (few splits) are treated as anomalies.

**Important Disclaimer**

> {DISCLAIMER}
""")


# --------------------------------------------------------------- navigation --
PAGES = {
    "📊 Dashboard": page_dashboard,
    "🔎 Traffic Analysis": page_traffic_analysis,
    "🚨 Anomaly Detection": page_anomaly_detection,
    "📡 Live Monitoring": page_live_monitoring,
    "📤 Upload Dataset": page_upload,
    "ℹ️ About Project": page_about,
}


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown("## 🛡️ CampusNet Guardian")
    choice = st.sidebar.radio("Navigation", list(PAGES.keys()))
    st.sidebar.caption("Academic prototype • Synthetic demo data")
    try:
        PAGES[choice]()
    except DataValidationError as error:
        st.error(f"Data problem: {error}")
    except FileNotFoundError as error:
        st.error(f"A required file was not found: {error}")


main()
