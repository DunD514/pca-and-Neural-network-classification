"""Shared dashboard UI system: futuristic marine robotics glassmorphic styling and metrics."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* Global Font & Theme Styling */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    letter-spacing: -0.01em;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: -0.02em;
}

/* Custom Oceanic Scrollbar */
::-webkit-scrollbar {
    width: 7px;
    height: 7px;
}
::-webkit-scrollbar-track {
    background: #060b14;
}
::-webkit-scrollbar-thumb {
    background: #142844;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #00f0ff;
    box-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
}

/* Page Header Hero Banner */
.page-hero {
    background: linear-gradient(135deg, rgba(10, 24, 48, 0.95) 0%, rgba(6, 17, 34, 0.98) 100%);
    border: 1px solid rgba(0, 240, 255, 0.22);
    border-radius: 16px;
    padding: 26px 30px;
    margin: 0 0 24px 0;
    box-shadow: 0 12px 35px -5px rgba(0, 0, 0, 0.65), 0 0 25px rgba(0, 240, 255, 0.08);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(16px);
}
.page-hero::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, #00f0ff 30%, #3b82f6 70%, transparent 100%);
}
.page-hero .hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1.25;
    background: linear-gradient(90deg, #ffffff 0%, #d8f8ff 70%, #00f0ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.page-hero .hero-sub {
    font-size: 1.0rem;
    color: #94a3b8 !important;
    line-height: 1.5;
    max-width: 900px;
}
.page-hero .hero-chip {
    display: inline-flex;
    align-items: center;
    margin: 12px 8px 0 0;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(0, 240, 255, 0.08);
    color: #38bdf8 !important;
    border: 1px solid rgba(0, 240, 255, 0.25);
    transition: all 0.2s ease;
}
.page-hero .hero-chip:hover {
    background: rgba(0, 240, 255, 0.16);
    border-color: #00f0ff;
    box-shadow: 0 0 12px rgba(0, 240, 255, 0.3);
}

/* Futuristic KPI Card */
.kpi-card {
    background: linear-gradient(145deg, rgba(14, 27, 49, 0.75) 0%, rgba(8, 16, 30, 0.9) 100%);
    border: 1px solid rgba(0, 240, 255, 0.18);
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    min-height: 124px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: "";
    position: absolute;
    bottom: 0; left: 20%; right: 20%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 240, 255, 0.4), transparent);
}
.kpi-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0, 240, 255, 0.5);
    box-shadow: 0 12px 30px rgba(0, 240, 255, 0.18);
}
.kpi-card .kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.05rem;
    font-weight: 700;
    color: #00f0ff !important;
    text-shadow: 0 0 18px rgba(0, 240, 255, 0.45);
    line-height: 1.15;
}
.kpi-card .kpi-label {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    font-weight: 600;
    color: #94a3b8 !important;
    margin-top: 8px;
}
.kpi-card .kpi-icon {
    font-size: 1.25rem;
    margin-bottom: 4px;
    opacity: 0.9;
}

/* Callout Alert Boxes */
.callout {
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0 18px 0;
    font-size: 0.93rem;
    line-height: 1.6;
    backdrop-filter: blur(10px);
    position: relative;
}
.callout, .callout * {
    color: inherit !important;
}
.callout h4 {
    margin: 0 0 6px 0;
    font-size: 1.05rem;
    font-weight: 600;
}
.callout ul {
    margin: 6px 0 0 0;
    padding-left: 20px;
}
.callout.info {
    background: rgba(14, 165, 233, 0.09);
    border: 1px solid rgba(14, 165, 233, 0.28);
    border-left: 5px solid #0284c7;
    color: #bae6fd !important;
}
.callout.success {
    background: rgba(16, 185, 129, 0.09);
    border: 1px solid rgba(16, 185, 129, 0.28);
    border-left: 5px solid #10b981;
    color: #a7f3d0 !important;
}
.callout.warn {
    background: rgba(245, 158, 11, 0.09);
    border: 1px solid rgba(245, 158, 11, 0.28);
    border-left: 5px solid #f59e0b;
    color: #fde68a !important;
}
.callout.danger {
    background: rgba(239, 68, 68, 0.09);
    border: 1px solid rgba(239, 68, 68, 0.28);
    border-left: 5px solid #ef4444;
    color: #fecaca !important;
}
.callout.neutral {
    background: rgba(148, 163, 184, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-left: 5px solid #64748b;
    color: #cbd5e1 !important;
}

/* Modern Pill Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 12px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.3px;
    backdrop-filter: blur(6px);
}
.badge.good {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399 !important;
    border: 1px solid rgba(16, 185, 129, 0.4);
}
.badge.fair {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24 !important;
    border: 1px solid rgba(245, 158, 11, 0.4);
}
.badge.poor {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171 !important;
    border: 1px solid rgba(239, 68, 68, 0.4);
}
.badge.neutral {
    background: rgba(148, 163, 184, 0.12);
    color: #cbd5e1 !important;
    border: 1px solid rgba(148, 163, 184, 0.3);
}
.badge.cyber {
    background: rgba(0, 240, 255, 0.12);
    color: #00f0ff !important;
    border: 1px solid rgba(0, 240, 255, 0.35);
    box-shadow: 0 0 10px rgba(0, 240, 255, 0.18);
}

.legend-chip {
    display: inline-block;
    width: 14px;
    height: 14px;
    border-radius: 4px;
    margin: 0 6px -2px 10px;
    box-shadow: 0 0 8px rgba(0,0,0,0.4);
}

/* Animated Radar Pulse Dot */
@keyframes pulse-glow {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 240, 255, 0.65); }
    70% { transform: scale(1.05); box-shadow: 0 0 0 8px rgba(0, 240, 255, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 240, 255, 0); }
}
.status-beacon {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #00f0ff;
    animation: pulse-glow 2s infinite;
    margin-right: 8px;
    vertical-align: middle;
}
.status-beacon.green {
    background: #10b981;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-green {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.65); }
    70% { transform: scale(1.05); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* Sci-Fi HUD Cockpit Telemetry Box */
.hud-status {
    background: linear-gradient(180deg, rgba(6, 16, 32, 0.95) 0%, rgba(4, 10, 20, 0.98) 100%);
    color: #00f0ff !important;
    padding: 14px 18px;
    border-radius: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    border: 1px solid rgba(0, 240, 255, 0.3);
    box-shadow: inset 0 0 20px rgba(0, 240, 255, 0.08), 0 8px 24px rgba(0, 0, 0, 0.5);
    position: relative;
    overflow: hidden;
}
.hud-status::before {
    content: "● LIVE TELEMETRY";
    display: block;
    font-size: 0.68rem;
    letter-spacing: 1.5px;
    color: #38bdf8;
    margin-bottom: 6px;
    font-weight: 700;
}
.hud-status * {
    color: #00f0ff !important;
}

/* Glassmorphism Tour Cards */
.tour-card {
    background: linear-gradient(135deg, rgba(14, 28, 52, 0.7) 0%, rgba(7, 16, 32, 0.85) 100%);
    border: 1px solid rgba(0, 240, 255, 0.16);
    border-radius: 14px;
    padding: 22px 20px;
    height: 100%;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    position: relative;
    overflow: hidden;
}
.tour-card:hover {
    transform: translateY(-5px);
    border-color: rgba(0, 240, 255, 0.45);
    box-shadow: 0 14px 35px rgba(0, 240, 255, 0.15);
}
.tour-card .tour-step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.88rem;
    background: linear-gradient(135deg, #0284c7 0%, #00f0ff 100%);
    color: #040914 !important;
    margin-right: 10px;
}
.tour-card .tour-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #f1f5f9 !important;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
}
.tour-card .tour-desc {
    font-size: 0.90rem;
    color: #94a3b8 !important;
    line-height: 1.55;
    margin-bottom: 16px;
}

/* Streamlit Native Components Polishing */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(13, 27, 48, 0.65) 0%, rgba(7, 16, 30, 0.85) 100%) !important;
    border: 1px solid rgba(0, 240, 255, 0.16) !important;
    border-radius: 12px !important;
    padding: 12px 18px !important;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25) !important;
    backdrop-filter: blur(8px) !important;
    transition: all 0.25s ease !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(0, 240, 255, 0.35) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #00f0ff !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: #94a3b8 !important;
}

/* Polished Streamlit Buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    border: 1px solid rgba(0, 240, 255, 0.3) !important;
    background: linear-gradient(135deg, rgba(14, 30, 56, 0.8) 0%, rgba(9, 20, 38, 0.95) 100%) !important;
    color: #e0f2fe !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
}
.stButton > button:hover {
    border-color: #00f0ff !important;
    background: linear-gradient(135deg, rgba(18, 42, 80, 0.9) 0%, rgba(11, 26, 52, 1) 100%) !important;
    color: #00f0ff !important;
    box-shadow: 0 0 18px rgba(0, 240, 255, 0.35) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(1px) !important;
}

/* Polished Streamlit Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(8, 16, 30, 0.7);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid rgba(0, 240, 255, 0.12);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 18px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: rgba(0, 240, 255, 0.14) !important;
    border-bottom: 2px solid #00f0ff !important;
}

/* Styled Streamlit Containers with border=True */
div[data-testid="stVerticalBlock"] > div[style*="border: 1px solid"] {
    background: rgba(9, 18, 34, 0.55);
    border-color: rgba(0, 240, 255, 0.14) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(10px);
}

/* Custom styled step-num for backwards compatibility */
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.85rem;
    background: #0284c7;
    color: #ffffff !important;
    margin-right: 8px;
}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def callout(html: str, kind: str = "info") -> None:
    """Coloured message box with explicit theme-safe styling."""
    st.markdown(f'<div class="callout {kind}">{html}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", chips: Optional[list] = None) -> None:
    """Consistent, futuristic marine glassmorphic page hero banner."""
    chip_html = "".join(f'<span class="hero-chip">{c}</span>' for c in (chips or []))
    st.markdown(
        f'<div class="page-hero">'
        f'<div class="hero-title">{title}</div>'
        f'<div class="hero-sub">{subtitle}</div>'
        f'{chip_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def step_number(n: int) -> str:
    return f'<span class="step-num">{n}</span>'


def kpi_card(value: str, label: str, icon: str = "") -> None:
    icon_html = f'<div class="kpi-icon">{icon}</div>' if icon else ""
    st.markdown(
        f'<div class="kpi-card">{icon_html}'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def tour_card_html(step: int, title: str, description: str) -> str:
    return (
        f'<div class="tour-card">'
        f'<div class="tour-title"><span class="tour-step-badge">{step}</span> {title}</div>'
        f'<div class="tour-desc">{description}</div>'
        f'</div>'
    )


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="badge {kind}">{text}</span>'


def legend_chip(color: str, label: str) -> str:
    return f'<span class="legend-chip" style="background:{color};"></span>{label}'


def dice_quality(dice: float) -> Tuple[str, str]:
    """Plain-language rating of a segmentation Dice score."""
    if dice >= 0.70:
        return "Good match", "good"
    if dice >= 0.50:
        return "Fair match", "fair"
    return "Poor match", "poor"


def verdict_kind(verdict: str) -> str:
    return {"No meaningful overfitting": "good", "Mild overfitting": "fair"}.get(verdict, "poor")


def sidebar_status() -> None:
    """High-tech sidebar inspection status panel with live hardware & checkpoint indicators."""
    ckpt = ROOT / "models" / "checkpoints"
    unet_file = ckpt / "unet_mavecodd.pth"
    svm_file = ckpt / "baseline_mavecodd.joblib"
    unet_ok = unet_file.exists()
    svm_ok = svm_file.exists()
    unet_mb = f"{unet_file.stat().st_size / 1e6:.1f} MB" if unet_ok else "N/A"
    svm_mb = f"{svm_file.stat().st_size / 1e6:.2f} MB" if svm_ok else "N/A"

    device_str = "CUDA GPU" if torch.cuda.is_available() else "CPU Active"
    metrics = load_metrics("mavecodd")

    with st.sidebar.container(border=True):
        st.markdown(
            f'<div><span class="status-beacon green"></span><b>SYSTEM STATUS</b> '
            f'<span class="badge cyber" style="font-size:0.7rem; float:right;">{device_str}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"• **Model B (U-Net)**: {'🟢 Active' if unet_ok else '🔴 Missing'} `({unet_mb})`  \n"
            f"• **Model A (SVM)**: {'🟢 Active' if svm_ok else '🔴 Missing'} `({svm_mb})`"
        )
        if metrics:
            u = metrics["deep_learning_model_b"]["test"]
            up = u["patch_classification"]
            st.caption(
                f"🎯 **Test Dice**: `{u['mean_dice']:.3f}`  \n"
                f"🛡️ **Missed defect tiles**: `{up['false_negatives']}/{up['n_defect_patches']}`"
            )
        else:
            st.caption("Run training + evaluation to compile benchmarks.")


@st.cache_data(ttl=60, show_spinner=False)
def load_metrics(dataset: str = "mavecodd") -> Optional[Dict[str, Any]]:
    """Evaluation metrics written by scripts/evaluate_model.py (None if missing or outdated)."""
    f = ROOT / "outputs" / "metrics" / f"test_metrics_{dataset}.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text())
        return data if "overfitting_check" in data else None
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def load_training_summary(dataset: str = "mavecodd") -> Optional[Dict[str, Any]]:
    f = ROOT / "outputs" / "metrics" / f"training_summary_{dataset}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def load_split_lookup(dataset: str = "mavecodd") -> Dict[str, str]:
    """image_id -> 'train' / 'val' / 'test'."""
    f = ROOT / "data" / "processed" / f"split_manifest_{dataset}.json"
    if not f.exists():
        return {}
    try:
        data = json.loads(f.read_text())
        return {s["image_id"]: split for split, v in data.items() for s in v["samples"]}
    except Exception:
        return {}


def missing_metrics_message() -> None:
    callout(
        "<b>Evaluation results not found or outdated.</b> Run "
        "<b>python scripts/train_model.py --dataset mavecodd</b> and then "
        "<b>python scripts/evaluate_model.py --dataset mavecodd</b>.",
        "warn",
    )
