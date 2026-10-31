"""Home page: what the project does, headline results, and a guided tour of the dashboard."""

import sys
from pathlib import Path

import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Home Hub | Hull Inspection AI",
    page_icon="🌊",
    layout="wide",
)

from app.ui import inject_css, callout, kpi_card, page_header, load_metrics, missing_metrics_message

inject_css()

page_header(
    "🌊 AI Subsea ROV Hull Inspection System",
    "Autonomous remotely operated vehicle (ROV) inspecting commercial ship hulls for corrosion scale and oxidation. "
    "Featuring dual ML architecture (Residual U-Net + Handcrafted SVM), underwater optics simulation, and columnar Apache Parquet telemetry.",
    chips=[
        "MaVeCoDD: 76 Calibrated Vessel Images",
        "Dual-Model: Res U-Net (2.0M) & RBF-SVM",
        "Scene-Grouped Zero-Leakage Splits",
        "Apache Arrow & Parquet Telemetry",
        f"Compute: {'NVIDIA CUDA' if torch.cuda.is_available() else 'High-Performance CPU'}",
    ],
)

metrics = load_metrics("mavecodd")
logs_dir = ROOT / "outputs" / "inspection_logs"
num_missions = len(list(logs_dir.glob("*.parquet"))) if logs_dir.exists() else 0

c1, c2, c3, c4 = st.columns(4)
if metrics:
    b_test = metrics["baseline_model_a"]["test"]
    u_test = metrics["deep_learning_model_b"]["test"]
    up = u_test["patch_classification"]
    with c1:
        kpi_card(f"{u_test['mean_dice']:.3f}", "U-Net Test Dice (Overlap)", icon="🎯")
    with c2:
        kpi_card(f"{up['false_negatives']} vs {b_test['false_negatives']}", f"Missed Tiles (U-Net vs SVM / {up['n_defect_patches']})", icon="🛡️")
    with c3:
        kpi_card(f"{up['balanced_accuracy'] * 100:.1f}%", f"U-Net Balanced Acc (SVM: {b_test['balanced_accuracy']*100:.1f}%)", icon="⚖️")
else:
    for col in (c1, c2, c3):
        with col:
            kpi_card("—", "Awaiting Evaluation", icon="⏳")
with c4:
    kpi_card(str(num_missions), "Logged ROV Missions", icon="🛰️")

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
if not metrics:
    missing_metrics_message()

# Guided 3-Step Tour Section
st.markdown("### 🧭 Interactive Mission Tour")
t1, t2, t3 = st.columns(3)

with t1:
    with st.container(border=True):
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span class="step-num">1</span>
                <span style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:700; color:#e0f2fe;">Validation & Overfitting</span>
            </div>
            <p style="font-size:0.88rem; color:#94a3b8; line-height:1.55; min-height:64px;">
                Verify generalization across unseen vessel scenes. Inspect train vs validation vs test Dice bounds to confirm zero data leakage.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/5_Model_Evaluation.py", label="Open Overfitting Analysis →", icon=":material/monitoring:")

with t2:
    with st.container(border=True):
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span class="step-num">2</span>
                <span style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:700; color:#e0f2fe;">Live Visual Inspection</span>
            </div>
            <p style="font-size:0.88rem; color:#94a3b8; line-height:1.55; min-height:64px;">
                Inspect high-resolution test hull photographs. Compare ground-truth masks with U-Net predictions and color error heatmaps.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/2_Image_Inspection.py", label="Launch Image Inspector →", icon=":material/image_search:")

with t3:
    with st.container(border=True):
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span class="step-num">3</span>
                <span style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:700; color:#e0f2fe;">Subsea Drone Simulator</span>
            </div>
            <p style="font-size:0.88rem; color:#94a3b8; line-height:1.55; min-height:64px;">
                Fly an autonomous ROV over a 50m ship hull. Experience realistic Jaffe-McGlamery optical attenuation, turbidity, and live HUD inference.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/3_Drone_Simulation.py", label="Take Control of ROV →", icon=":material/sailing:")

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Architecture & Benchmark Section
left, right = st.columns([3, 2])
with left:
    st.markdown("### ⚙️ End-to-End System Architecture")
    steps = [
        ("📷", "Data Acquisition & Scene Grouping", "76 vessel photos partitioned into non-overlapping scene clusters to prevent video-burst contamination."),
        ("🧮", "Model A · Traditional ML Baseline", "47-dimensional color & LBP texture feature extraction per 256px tile paired with calibrated RBF-SVM."),
        ("🧠", "Model B · Deep Residual U-Net", "Compact 2.0M parameter convolutional neural network trained on combined BCE + Soft Dice loss."),
        ("🌊", "Optical Physics Simulation", "Jaffe-McGlamery underwater radiative transfer: red wavelength attenuation, marine snow, and forward scattering."),
        ("🗄️", "Columnar PyArrow & Parquet Storage", "Ultra-fast telemetry persistence logging 28 flight attributes per step for audit replay and PDF export."),
    ]
    for icon, name, text in steps:
        st.markdown(
            f"""
            <div style="background: rgba(10, 22, 42, 0.6); border: 1px solid rgba(0, 240, 255, 0.12); border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;">
                <span style="font-size: 1.15rem; margin-right: 8px;">{icon}</span>
                <span style="font-family:'Space Grotesk',sans-serif; font-weight:700; color:#38bdf8; font-size:0.95rem;">{name}</span>
                <div style="font-size: 0.86rem; color: #94a3b8; margin-top: 4px; padding-left: 28px;">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right:
    st.markdown("### 📊 Held-Out Benchmark Matrix")
    if metrics:
        of = metrics["overfitting_check"]
        st.markdown(
            f"""
| Metric Pipeline | Train Split | Validation Split | Test Split (95% CI) |
| :--- | :---: | :---: | :---: |
| **Model B U-Net Dice** | `{of['unet_dice']['train']:.3f}` | `{of['unet_dice']['val']:.3f}` | **`{of['unet_dice']['test']:.3f}`** |
| **Model B Tile Bal. Acc** | `{of['unet_patch_balanced_accuracy']['train']:.3f}` | `{of['unet_patch_balanced_accuracy']['val']:.3f}` | **`{of['unet_patch_balanced_accuracy']['test']:.3f}`** |
| **Model A SVM Bal. Acc** | `{of['baseline_patch_balanced_accuracy']['train']:.3f}` | `{of['baseline_patch_balanced_accuracy']['val']:.3f}` | **`{of['baseline_patch_balanced_accuracy']['test']:.3f}`** |
"""
        )
        st.caption("ℹ️ Stratified test split comprises 21 sound tiles and 144 defect tiles; bootstrap confidence intervals are ±0.08.")

    callout(
        "<b>⚠️ Marine Engineering & Regulatory Compliance:</b> Predictions are flagged as "
        "<b>'Suspected Surface Defects'</b> under DNV & Lloyd's Register rules requiring diver/surveyor visual corroboration. "
        "Optical cameras analyze visible surface phenomena only; internal metallurgical thinning requires ultrasonic thickness testing (UTM).",
        "warn",
    )
