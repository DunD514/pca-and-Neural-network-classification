"""Streamlit Page 6: Inspection reports, telemetry querying, and PDF report export."""

import io
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Inspection Logs & Reports | Hull AI", layout="wide")

from app.ui import callout, inject_css, kpi_card, page_header
from src.reporting.report_generator import generate_pdf_inspection_report
from src.storage.parquet_reader import get_inspection_summary, list_available_inspections, read_inspection_log

inject_css()
page_header(
    "📄 Autonomous Mission Logs & Certification Export",
    "Query high-performance columnar Apache Parquet telemetry records, filter defect logs, and generate publication-grade PDF survey reports.",
    chips=["Apache Arrow Columnar Schema", "Snappy Compressed Parquet", "DNV / Lloyd's Compliant PDF"],
)


@st.cache_data(ttl=20, show_spinner=False)
def get_cached_inspections(logs_dir_str: str):
    return list_available_inspections(Path(logs_dir_str))


@st.cache_data(show_spinner=False)
def get_cached_summary(parquet_path_str: str):
    return get_inspection_summary(Path(parquet_path_str))


@st.cache_data(show_spinner=False)
def get_cached_df(parquet_path_str: str):
    return read_inspection_log(Path(parquet_path_str))


logs_dir = ROOT / "outputs" / "inspection_logs"
inspections = get_cached_inspections(str(logs_dir))

if not inspections:
    callout(
        "No mission telemetry logs archived yet. Navigate to <b>ROV Drone Simulation</b> and click 'Save Parquet' "
        "or execute <b>python scripts/run_simulation.py</b>.",
        "warn",
    )
    st.stop()

# Select inspection
insp_options = [
    f"{i['inspection_id']} (Telemetry Steps: {i['total_records']} &bull; Final Coverage: {i['final_coverage_pct']:.1f}%)"
    for i in inspections
]
selected_idx = st.selectbox("Select Archived Mission", range(len(insp_options)), format_func=lambda i: insp_options[i])
selected_insp = inspections[selected_idx]
parquet_path = Path(selected_insp["file_path"])

# Load summary and DataFrame
summary = get_cached_summary(str(parquet_path))
df = get_cached_df(str(parquet_path))

# KPI Cards row
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card(summary["inspection_id"].split("_")[-1], "Mission Sequence", icon="🛰️")
with c2:
    kpi_card(f"{summary['final_coverage_pct']:.1f}%", "Hull Coverage", icon="🗺️")
with c3:
    kpi_card(str(summary["total_suspected_defects"]), "Suspected Defects", icon="🛡️")
with c4:
    kpi_card(f"{summary['mean_confidence']*100:.1f}%", "Mean AI Confidence", icon="🎯")

if "ground_truth_defect" in df and df["predicted_defect_class"].ne("unprocessed").any():
    gt, pr = df["ground_truth_defect"].astype(bool), df["defect_detected"].astype(bool)
    tp, fp = int((gt & pr).sum()), int((~gt & pr).sum())
    fn, tn = int((gt & ~pr).sum()), int((~gt & ~pr).sum())
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        kpi_card(f"{(tp + tn) / len(df) * 100:.1f}%", "Ground Truth Accuracy", icon="🎯")
    with g2:
        kpi_card(f"{tp} / {tp + fn}", "Defect Frames Caught", icon="✅")
    with g3:
        kpi_card(f"{fp} / {fp + tn}", "False Alarms", icon="⚠️")
    with g4:
        kpi_card(str(fn), "Missed Defect Frames", icon="❌")

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# Download Buttons row
st.markdown("### 📥 Mission Artifact & Report Downloads")
col_d1, col_d2, col_d3 = st.columns(3)

# 1. Parquet Download
with open(parquet_path, "rb") as f:
    parquet_bytes = f.read()
with col_d1:
    st.download_button(
        "📥 Download Apache Parquet (.parquet)",
        data=parquet_bytes,
        file_name=parquet_path.name,
        mime="application/octet-stream",
        width="stretch",
    )

# 2. CSV Download
csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)
with col_d2:
    st.download_button(
        "📥 Download Raw Telemetry (.csv)",
        data=csv_buffer.getvalue(),
        file_name=f"{parquet_path.stem}.csv",
        mime="text/csv",
        width="stretch",
    )

# 3. PDF Report Generation & Download
pdf_path = ROOT / "outputs" / "reports" / f"{parquet_path.stem}_report.pdf"
try:
    generate_pdf_inspection_report(parquet_path, pdf_path)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    with col_d3:
        st.download_button(
            "📥 Download Formal Survey PDF (.pdf)",
            data=pdf_bytes,
            file_name=pdf_path.name,
            mime="application/pdf",
            width="stretch",
        )
except Exception as e:
    with col_d3:
        st.button("PDF Generation Error", disabled=True, width="stretch")

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Interactive Telemetry Query Table
st.markdown("### 🔍 Columnar Telemetry Query & Defect Filter")

# Filtering controls
f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    filter_defects = st.checkbox("Filter Flagged Defects Only", value=True)
with f_col2:
    all_zones = ["All Zones"] + sorted(list(df["zone_id"].unique()))
    selected_zone = st.selectbox("Filter by Hull Zone", all_zones)
with f_col3:
    min_conf = st.slider("Minimum Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

filtered_df = df.copy()
if filter_defects:
    filtered_df = filtered_df[filtered_df["defect_detected"] == True]
if selected_zone != "All Zones":
    filtered_df = filtered_df[filtered_df["zone_id"] == selected_zone]
filtered_df = filtered_df[filtered_df["confidence_score"] >= min_conf]

st.markdown(f"**Archived Records Matching Filter:** `{len(filtered_df)}` of `{len(df)}` total steps")

display_cols = [
    "step_id",
    "timestamp",
    "drone_x",
    "drone_y",
    "drone_z",
    "drone_heading_deg",
    "zone_id",
    "camera_image_id",
    "predicted_defect_class",
    "confidence_score",
    "defect_detected",
    "defect_x_min",
    "defect_y_min",
    "defect_x_max",
    "defect_y_max",
    "inspection_status",
    "coverage_pct",
]
st.dataframe(filtered_df[display_cols], width="stretch")

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
callout(
    """
<b>Maritime Regulatory Classification Notice:</b>
<ul>
<li><b>Scope:</b> Computer vision inspects optical surface phenomena (corrosion scale, oxidation, coating failure). Subsurface fatigue, plate thickness erosion, and metallurgical stress require Ultrasonic Testing (UTM) or Eddy Current sensor arrays.</li>
<li><b>Standard Classification:</b> Under Lloyd's Register & DNV marine classification guidelines, vision anomalies are registered as <i>'Suspected Defects'</i> pending certified non-destructive testing verification.</li>
</ul>
""",
    "warn",
)
