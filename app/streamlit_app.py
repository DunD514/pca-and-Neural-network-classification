"""Autonomous Underwater ROV Ship Hull Inspection Dashboard - entry point and navigation."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Hull Inspection AI · Autonomous Subsea ROV",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.ui import inject_css, sidebar_status

inject_css()

# Futuristic Sidebar Header
st.sidebar.markdown(
    """
    <div style="padding: 12px 4px 18px 4px; border-bottom: 1px solid rgba(0, 240, 255, 0.15); margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="font-size: 1.8rem; background: linear-gradient(135deg, #0284c7 0%, #00f0ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🌊</div>
            <div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">HULL INSPECT AI</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.70rem; color: #38bdf8; letter-spacing: 1px; font-weight: 600;">SUBSEA ROV DRONE · V2.4</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

pages = {
    "Mission Command": [
        st.Page("pages/0_Home.py", title="Home Hub", icon=":material/home:", default=True),
        st.Page("pages/1_Overview.py", title="Project Overview", icon=":material/dashboard:"),
    ],
    "Subsea Operations": [
        st.Page("pages/2_Image_Inspection.py", title="Image Inspection", icon=":material/image_search:"),
        st.Page("pages/3_Drone_Simulation.py", title="ROV Drone Simulation", icon=":material/sailing:"),
    ],
    "Analytics & Validation": [
        st.Page("pages/5_Model_Evaluation.py", title="Model Benchmark", icon=":material/monitoring:"),
        st.Page("pages/4_Dataset_Explorer.py", title="Dataset Explorer", icon=":material/photo_library:"),
    ],
    "Reports & Telemetry": [
        st.Page("pages/6_Inspection_Reports.py", title="Inspection Logs & PDF", icon=":material/description:"),
    ],
}

nav = st.navigation(pages)
sidebar_status()
nav.run()
