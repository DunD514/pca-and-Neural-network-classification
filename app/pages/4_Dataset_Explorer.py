"""Streamlit Page 4: Dataset explorer and leakage verification."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Dataset Explorer | Hull AI", layout="wide")

from app.ui import badge, inject_css, kpi_card, load_split_lookup, page_header
from src.data.dataset_loader import load_image_and_mask, scan_crack_samples, scan_mavecodd_samples
from src.data.split_dataset import load_split_manifest

inject_css()
page_header(
    "🗂️ Marine Dataset Explorer & Leakage Verification",
    "Inspect raw dry-dock ship hull imagery, underwater crack fissures, pixel-level masks, and scene-stratification matrices.",
    chips=["MaVeCoDD Corrosion", "Underwater Crack Dataset", "Stratified Split Manifests"],
)


@st.cache_data(show_spinner=False)
def get_cached_mavecodd(raw_dir_str: str, zip_fallback_str: str):
    return scan_mavecodd_samples(Path(raw_dir_str), zip_fallback=Path(zip_fallback_str) if zip_fallback_str else None)


@st.cache_data(show_spinner=False)
def get_cached_cracks(raw_dir_str: str, zip_fallback_str: str, limit: int = 100):
    return scan_crack_samples(Path(raw_dir_str), zip_fallback=Path(zip_fallback_str) if zip_fallback_str else None, limit=limit)


ds_tab = (
    st.segmented_control(
        "Dataset View",
        ["MaVeCoDD (Marine Vessel Corrosion)", "Underwater Crack Dataset", "Train/Val/Test Splits Manifest"],
        default="MaVeCoDD (Marine Vessel Corrosion)",
    )
    or "MaVeCoDD (Marine Vessel Corrosion)"
)

if ds_tab == "MaVeCoDD (Marine Vessel Corrosion)":
    st.subheader("🚢 MaVeCoDD: Commercial Vessel Corrosion in Dry-Dock")
    st.markdown("""
    - **Origin**: Mendeley Data (`dx.doi.org/10.17632/ry392rp8cj.1`), Kamzelis et al.
    - **Annotation Protocol**: High-precision segmentation masks (`*_label.jpg`). Corroded steel maintains color; sound background masked.
    - **Resolution Tiers**: HiRes (up to 4032 &times; 3024) and LoRes (1920 &times; 1080).
    """)

    mav_samples = get_cached_mavecodd(str(ROOT / "data/raw/mavecodd"), str(ROOT / "ry392rp8cj-1.zip"))

    if mav_samples:
        df_mav = pd.DataFrame(mav_samples)
        c1, c2 = st.columns([1, 2])
        with c1:
            kpi_card(str(len(mav_samples)), "Valid Paired Samples", icon="🚢")
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.write("##### Resolution Distribution")
            fig_pie = px.pie(
                df_mav,
                names="resolution_tier",
                title="MaVeCoDD Tier Composition",
                color_discrete_sequence=["#00f0ff", "#6366f1"],
            )
            fig_pie.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(6, 12, 24, 0.8)",
                plot_bgcolor="rgba(4, 9, 20, 0.85)",
                font=dict(family="Plus Jakarta Sans", color="#cbd5e1"),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_pie, width="stretch")

        with c2:
            st.write("##### Sample Gallery & Mask Inspector")
            sample_idx = st.slider("Select Sample Index", 0, len(mav_samples) - 1, 0)
            s = mav_samples[sample_idx]
            img, mask = load_image_and_mask(s)
            split_of = load_split_lookup("mavecodd").get(s["image_id"])
            kind = {"test": "good", "val": "fair", "train": "neutral"}.get(split_of, "neutral")
            st.markdown(
                f"**Image ID:** `{s['image_id'].replace('mavecodd_', '')}` &bull; Scene Sequence: `{s['scene_id']}` &bull; "
                f"Assigned Split: {badge((split_of or 'unknown').upper(), kind)}",
                unsafe_allow_html=True,
            )

            alpha = st.slider("Mask Overlay Blend Opacity", 0.0, 1.0, 0.45, 0.05)
            color_mask = np.zeros_like(img)
            color_mask[mask > 0] = [239, 68, 68]  # Red overlay
            blended = cv2.addWeighted(img, 1.0 - alpha, color_mask, alpha, 0)

            c_img1, c_img2 = st.columns(2)
            c_img1.image(img, caption=f"Raw Image: {s['image_id']} ({img.shape[1]}x{img.shape[0]}px)", width="stretch")
            c_img2.image(
                blended,
                caption=f"Ground-Truth Mask (Defect: {(np.sum(mask>0)/mask.size)*100:.2f}% of pixels)",
                width="stretch",
            )

elif ds_tab == "Underwater Crack Dataset":
    st.subheader("🌊 Underwater Concrete & Submerged Structure Cracks")
    st.markdown("""
    - **Origin**: Mendeley Data (`dx.doi.org/10.17632/g962rbm77b/1`), Gao et al.
    - **Subject**: Subsea pilings and submerged concrete piers with stress fracture fissures.
    - **Annotation Standard**: Binary PNG segmentation masks (Crack = 255, Sound Background = 0).
    """)

    crack_samples = get_cached_cracks(
        str(ROOT / "data/raw/underwater_crack"), str(ROOT / "under-crack-images.zip"), limit=100
    )
    kpi_card(str(len(crack_samples)), "Indexed Subsea Samples (Displaying Top 100)", icon="🌊")
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    if crack_samples:
        sample_idx = st.slider("Select Crack Sample Index", 0, len(crack_samples) - 1, 0)
        s = crack_samples[sample_idx]
        img, mask = load_image_and_mask(s)

        alpha = st.slider("Crack Mask Overlay Opacity", 0.0, 1.0, 0.45, 0.05)
        color_mask = np.zeros_like(img)
        color_mask[mask > 0] = [239, 68, 68]
        blended = cv2.addWeighted(img, 1.0 - alpha, color_mask, alpha, 0)

        c1, c2 = st.columns(2)
        c1.image(img, caption=f"Raw Underwater Frame: {s['image_id']} (512x512px)", width="stretch")
        c2.image(
            blended,
            caption=f"Ground-Truth Crack Annotation (Coverage: {(np.sum(mask>0)/mask.size)*100:.2f}%)",
            width="stretch",
        )

else:
    st.subheader("📊 Train / Validation / Test Scene Partitioning")
    st.markdown("""
    - **Zero-Leakage Grouping**: Consecutive video burst frames form one *scene group*. A whole scene is assigned exclusively to one split.
    - **Stratified Distribution**: HiRes and LoRes camera tiers are stratified evenly across training, validation, and test sets.
    - **Split Ratio**: Approximately 60% Train / 20% Validation / 20% Test.
    """)

    manifest_file = ROOT / "data" / "processed" / "split_manifest_mavecodd.json"
    if manifest_file.exists():
        splits = load_split_manifest(manifest_file)
        rows = []
        for name in ["train", "val", "test"]:
            items = splits[name]
            rows.append({
                "Split": name.upper(),
                "Images": len(items),
                "HiRes": sum(1 for s in items if s.get("resolution_tier") == "HiRes"),
                "LoRes": sum(1 for s in items if s.get("resolution_tier") == "LoRes"),
                "Scene Groups": len({s["scene_id"] for s in items}),
                "Group IDs": ", ".join(sorted({s["scene_id"] for s in items})),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        all_train = {s["scene_id"] for s in splits["train"]}
        overlap = all_train & ({s["scene_id"] for s in splits["val"]} | {s["scene_id"] for s in splits["test"]})

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if not overlap:
            st.markdown(
                f"**Split Integrity Audit:** {badge('VERIFIED · ZERO DATA LEAKAGE DETECTED', 'good')}",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"**Split Integrity Audit:** {badge(f'ALERT: OVERLAPPING SCENES {sorted(overlap)}', 'poor')}",
                unsafe_allow_html=True,
            )

        with st.expander("Explore Individual Image-to-Split Mapping"):
            st.dataframe(
                pd.DataFrame([
                    {"Image ID": s["image_id"], "Scene Group": s["scene_id"], "Assigned Split": name.upper()}
                    for name in ["train", "val", "test"]
                    for s in splits[name]
                ]),
                width="stretch",
                hide_index=True,
            )
    else:
        st.warning("Split manifest not detected. Run `python scripts/prepare_dataset.py`.")
