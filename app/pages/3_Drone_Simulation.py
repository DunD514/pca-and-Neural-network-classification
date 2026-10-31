"""Streamlit Page 3: Autonomous Underwater ROV 3D Simulation & 3D SLAM Mapping Console."""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="3D ROV Simulation & SLAM | Hull AI", layout="wide")

from app.ui import badge, callout, inject_css, page_header, kpi_card
from src.models.predict import HullInspectionPredictor
from src.simulation.environment_3d import ShipHull3DEnvironment
from src.simulation.rov_dynamics import ROVDynamics6DOF
from src.simulation.synthetic_sensors import SyntheticSonarArray, SyntheticCameraPipeline, SyntheticIMUSensor
from src.simulation.slam import UnderwaterSLAM3D
from src.simulation.autonomous_inspector import AutonomousHullInspector
from src.simulation.plain_english_explainer import PlainEnglishMissionExplainer
from src.simulation.webgl_visualizer import generate_persistent_threejs_html, generate_threejs_simulation_html
from src.simulation.sim_server import ensure_simulation_server_running, get_simulation_state_manager, rpy_to_quaternion
from src.storage.parquet_writer import write_inspection_log
from src.utils.reproducibility import get_device

# Start high-performance background simulation bridge server for persistent 60fps WebGL
SIM_SERVER_PORT = ensure_simulation_server_running(preferred_port=8502)

inject_css()
page_header(
    "🌊 3D Subsea ROV Simulation & Autonomous SLAM Mapping",
    "Real-time 6-DOF hydrodynamic simulation navigating beside a 50m vessel hull. "
    "Featuring synthetic multi-beam sonar ray-casting, live 3D SLAM voxel mapping, EKF pose estimation, and real-time U-Net corrosion inference.",
    chips=["Real-Time 3D WebGL", "6-DOF Hydrodynamics", "Scan-Matching 3D SLAM", "Autonomous Conformal Trajectory"],
)

U_PATH = ROOT / "models" / "checkpoints" / "unet_mavecodd.pth"

@st.cache_resource
def get_cached_predictor(mtime: float):
    if U_PATH.exists():
        return HullInspectionPredictor(unet_path=U_PATH, device=get_device("auto"))
    return None

predictor = get_cached_predictor(U_PATH.stat().st_mtime if U_PATH.exists() else 0.0)

# Initialize master simulation session state
def initialize_sim(zone_name: str = "BOW_STBD", standoff_m: float = 2.0):
    env = ShipHull3DEnvironment(length_m=50.0, height_m=10.0, beam_m=12.0)
    # Start beside the chosen zone strictly outside hull
    if "PORT" in zone_name:
        start_x = 1.5 if "BOW" in zone_name else (15.0 if "MID" in zone_name else 38.0)
        start_z = env.get_surface_z(start_x, 5.0, "port") - standoff_m
        start_y = 5.0
    elif "KEEL" in zone_name:
        start_x = 5.0
        start_y = max(env.seabed_y_m + 0.6, -standoff_m)
        start_z = 0.0
    else:
        start_x = 1.5 if "BOW" in zone_name else (15.0 if "MID" in zone_name else 38.0)
        start_z = env.get_surface_z(start_x, 5.0, "starboard") + standoff_m
        start_y = 5.0

    rov = ROVDynamics6DOF(initial_pos=(start_x, start_y, start_z), mass_kg=32.0)
    slam = UnderwaterSLAM3D(initial_pose=(start_x, start_y, start_z))
    sonar = SyntheticSonarArray(env=env)
    cam = SyntheticCameraPipeline(env=env)
    imu = SyntheticIMUSensor()

    inspector = AutonomousHullInspector(
        env=env,
        rov=rov,
        slam=slam,
        sonar=sonar,
        camera_pipeline=cam,
        imu=imu,
        predictor=predictor,
        standoff_m=standoff_m,
        min_safe_distance_m=1.0,
        max_inspection_distance_m=3.0,
        drone_radius_m=0.40,
        collision_buffer_m=0.45
    )
    inspector.plan_inspection_route(zone_name)
    explainer = PlainEnglishMissionExplainer()
    return inspector, explainer

if "inspector" not in st.session_state or st.session_state.get("cached_pred") is not predictor:
    inspector, explainer = initialize_sim("BOW_STBD", 2.0)
    st.session_state["inspector"] = inspector
    st.session_state["explainer"] = explainer
    st.session_state["cached_pred"] = predictor
    st.session_state["last_step_res"] = None
    st.session_state["cam_view_mode"] = "follow"
    st.session_state["sim_running"] = False
    st.session_state["sim_speed"] = 1.0
    st.session_state["last_wall_time"] = time.time()

inspector: AutonomousHullInspector = st.session_state["inspector"]
explainer: PlainEnglishMissionExplainer = st.session_state["explainer"]

# Sidebar controls
st.sidebar.markdown("### 🕹️ Simulation & Mission Settings")
ctrl_mode = st.sidebar.radio("Control Mode", ["Autonomous Route", "Manual Piloting"], horizontal=True)
inspector.mode = "AUTONOMOUS" if ctrl_mode == "Autonomous Route" else "MANUAL"

ZONES = {
    "Bow Starboard (Curve & Flare)": "BOW_STBD",
    "Bow Port (Symmetric Plating)": "BOW_PORT",
    "Midship Forward (Cargo Plating)": "MIDSHIP_FWD",
    "Midship Aft (Hull Seams)": "MIDSHIP_AFT",
    "Stern Starboard (Propeller Region)": "STERN_STBD",
    "Keel Bottom Plate (Draft 0-2.5m)": "KEEL_ZONE",
    "Complete Starboard Survey (50m)": "FULL_SURVEY"
}
zone_label = st.sidebar.selectbox("Inspection Target Zone", list(ZONES.keys()), index=0)
target_zone_key = ZONES[zone_label]

standoff = st.sidebar.slider("Hull Standoff Distance (m)", 1.0, 3.0, float(inspector.standoff_m), 0.1)
inspector.standoff_m = standoff

# Navigation Debugging Mode Toggle
nav_debug = st.sidebar.toggle("🛠️ Navigation & Collision Debug", value=inspector.debug_mode)
inspector.debug_mode = nav_debug

# Route Looping Toggle
loop_route = st.sidebar.toggle("Loop Route (Continuous Inspection)", value=inspector.loop_route)
inspector.loop_route = loop_route

# Manual piloting controls if selected
manual_forces = np.zeros(3)
manual_torques = np.zeros(3)
if inspector.mode == "MANUAL":
    st.sidebar.markdown("#### 🎮 Pilot Thruster Input")
    c_m1, c_m2 = st.sidebar.columns(2)
    surge = c_m1.slider("Surge (Fwd/Aft)", -50.0, 50.0, 0.0, 5.0)
    sway = c_m2.slider("Sway (Port/Stbd)", -50.0, 50.0, 0.0, 5.0)
    heave = c_m1.slider("Heave (Up/Down)", -50.0, 50.0, 0.0, 5.0)
    yaw_t = c_m2.slider("Yaw Torque", -20.0, 20.0, 0.0, 2.0)
    manual_forces = np.array([surge, heave, sway])
    manual_torques = np.array([0.0, yaw_t, 0.0])

# Change zone dynamically
if target_zone_key != inspector.selected_zone and inspector.mode == "AUTONOMOUS":
    inspector.plan_inspection_route(target_zone_key)
    inspector.clock.reset()
    st.session_state["last_step_res"] = inspector.advance_simulation(0.05)
    st.rerun()

# ---------------- SIMULATION CONTROL & SPEED RIBBON ----------------
with st.container(border=True):
    col_play, col_pause, col_step1, col_step10, col_reset, col_save = st.columns([1.2, 1.0, 1.0, 1.0, 1.1, 1.1])
    
    is_currently_running = st.session_state.get("sim_running", False)
    
    if is_currently_running:
        btn_start = col_play.button("Running...", icon=":material/play_circle:", disabled=True, type="primary", width="stretch")
        btn_pause = col_pause.button("Pause", icon=":material/pause_circle:", width="stretch")
    else:
        btn_start = col_play.button("Start Sim", icon=":material/play_arrow:", type="primary", width="stretch")
        btn_pause = col_pause.button("Paused", icon=":material/pause:", disabled=True, width="stretch")
        
    btn_step1 = col_step1.button("Step +1", icon=":material/skip_next:", width="stretch")
    btn_step10 = col_step10.button("Fast +10", icon=":material/fast_forward:", width="stretch")
    btn_reset = col_reset.button("Reset Mission", icon=":material/restart_alt:", width="stretch")
    btn_save = col_save.button("Save Parquet", icon=":material/save:", width="stretch")

    # Speed control slider & presets
    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
    c_spd_lbl, c_spd_preset, c_spd_slider = st.columns([1.2, 3.2, 2.6])
    c_spd_lbl.markdown("<div style='font-size: 0.9rem; font-weight: 600; color: #38bdf8; padding-top: 6px;'>⚡ Sim Speed:</div>", unsafe_allow_html=True)
    
    SPEED_PRESETS = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    curr_spd = st.session_state.get("sim_speed", 1.0)
    preset_idx = SPEED_PRESETS.index(curr_spd) if curr_spd in SPEED_PRESETS else 2
    
    selected_preset = c_spd_preset.segmented_control(
        "Speed Presets",
        options=["0.25x", "0.5x", "1x", "2x", "5x", "10x"],
        default=f"{curr_spd:g}x",
        label_visibility="collapsed"
    )
    if selected_preset:
        chosen_val = float(selected_preset.replace("x", ""))
        if chosen_val != curr_spd:
            st.session_state["sim_speed"] = chosen_val
            inspector.clock.set_speed(chosen_val)
            curr_spd = chosen_val

    speed_slider_val = c_spd_slider.slider("Fine Multiplier", 0.25, 10.0, float(curr_spd), 0.25, label_visibility="collapsed")
    if speed_slider_val != curr_spd:
        st.session_state["sim_speed"] = speed_slider_val
        inspector.clock.set_speed(speed_slider_val)
        curr_spd = speed_slider_val

    # Handle button events
    if btn_start:
        st.session_state["sim_running"] = True
        inspector.clock.is_paused = False
        st.session_state["last_wall_time"] = time.time()
        st.rerun()

    if btn_pause:
        st.session_state["sim_running"] = False
        inspector.clock.is_paused = True
        st.rerun()

    if btn_reset:
        inspector, explainer = initialize_sim(target_zone_key, standoff)
        inspector.clock.set_speed(curr_spd)
        st.session_state["inspector"] = inspector
        st.session_state["explainer"] = explainer
        st.session_state["last_step_res"] = None
        st.session_state["sim_running"] = False
        st.session_state["last_wall_time"] = time.time()
        st.rerun()

# Execute single manual steps if requested
if btn_step1:
    st.session_state["sim_running"] = False
    inspector.clock.is_paused = False
    current_res = inspector.advance_simulation(
        wall_dt_s=0.05 / max(0.1, curr_spd),
        manual_cmd=(manual_forces, manual_torques) if inspector.mode == "MANUAL" else None
    )
    st.session_state["last_step_res"] = current_res

elif btn_step10:
    st.session_state["sim_running"] = False
    inspector.clock.is_paused = False
    for _ in range(10):
        current_res = inspector.advance_simulation(
            wall_dt_s=0.05 / max(0.1, curr_spd),
            manual_cmd=(manual_forces, manual_torques) if inspector.mode == "MANUAL" else None
        )
    st.session_state["last_step_res"] = current_res

# Continuous Simulation Execution
now = time.time()
last_wall = st.session_state.get("last_wall_time", now)
dt_wall = min(0.20, max(0.015, now - last_wall))
st.session_state["last_wall_time"] = now

if st.session_state.get("sim_running", False):
    inspector.clock.is_paused = False
    inspector.clock.set_speed(curr_spd)
    current_res = inspector.advance_simulation(
        wall_dt_s=dt_wall,
        manual_cmd=(manual_forces, manual_torques) if inspector.mode == "MANUAL" else None
    )
    st.session_state["last_step_res"] = current_res
else:
    current_res = st.session_state.get("last_step_res")
    if current_res is None:
        current_res = inspector.advance_simulation(wall_dt_s=0.05)
        st.session_state["last_step_res"] = current_res

# Parquet Save
if btn_save:
    logs_dir = ROOT / "outputs" / "inspection_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    save_path = logs_dir / f"INSP_3D_{inspector.selected_zone}_{int(time.time())}.parquet"
    records = []
    for s in inspector.telemetry_history:
        if "pose" in s:
            p = s["pose"]
            c = s.get("camera_sensor", {})
            ml = s.get("ml_prediction") or {}
            records.append({
                "inspection_id": f"3D_SLAM_{inspector.selected_zone}",
                "step_id": len(records) + 1,
                "timestamp": str(c.get("timestamp_s", 0.0)),
                "drone_x": p["x_m"],
                "drone_y": p["y_m"],
                "drone_z": p["z_m"],
                "drone_heading_deg": p["yaw_deg"],
                "measured_x": s["slam"]["est_position"][0],
                "measured_y": s["slam"]["est_position"][1],
                "measured_z": s["slam"]["est_position"][2],
                "measured_heading_deg": p["yaw_deg"],
                "zone_id": s["current_zone"],
                "camera_image_id": f"frame_{c.get('frame_counter', 1)}",
                "dataset_source": "synthetic_3d_render",
                "ground_truth_defect": s.get("camera", {}).get("has_defect", False),
                "predicted_defect_class": ml.get("predicted_defect_class", "sound"),
                "confidence_score": ml.get("confidence_score", 0.0),
                "defect_detected": ml.get("defect_detected", False),
                "defect_bbox": str(ml.get("bounding_boxes", [])),
                "simulated_visibility": 0.85,
                "simulated_turbidity": 0.20,
                "inspection_status": "inspected",
                "coverage_pct": s["coverage_pct"],
                "battery_pct": s["avionics"]["battery_pct"],
                "lidar_min_dist_m": s["sonar_sensor"]["min_range_m"],
                "submerged_depth_m": s["depth_sensor"]["depth_measured_m"],
                "pressure_bar": s["depth_sensor"]["hydrostatic_pressure_bar"],
                "collision_status": "SAFE" if not inspector.rov.is_collided else "COLLISION"
            })
    if records:
        write_inspection_log(records, save_path)
        st.success(f"Archived {len(records)} 3D inspection telemetry records to `{save_path.name}`.")

# Top Headline Telemetry KPIs
s_slam = current_res.get("slam", {})
s_av = current_res.get("avionics", {})
s_sonar = current_res.get("sonar_sensor", current_res.get("sonar", {}))
s_depth = current_res.get("depth_sensor", {})
s_clk = current_res.get("clock", {})

# Simulation status banner
sim_status_str = "🟢 RUNNING" if st.session_state.get("sim_running", False) else ("🏁 ROUTE COMPLETE" if inspector.is_completed else "⏸️ PAUSED")
st.markdown(
    f"""
<div style="background: rgba(4, 14, 30, 0.85); border: 1px solid rgba(0, 240, 255, 0.3); border-radius: 8px; padding: 6px 16px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
    <span style="font-weight: 700; color: #00f0ff; font-family: 'JetBrains Mono', monospace;">STATUS: {sim_status_str}</span>
    <span style="color: #94a3b8; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">
        Sim Time: <b style="color:#e2e8f0;">{s_clk.get('sim_time_s', 0.0):.1f}s</b> | Wall Time: <b style="color:#e2e8f0;">{s_clk.get('wall_time_s', 0.0):.1f}s</b> | Speed: <b style="color:#38bdf8;">{curr_spd}x</b> | Waypoint: <b style="color:#fbbf24;">{current_res.get('motion', {}).get('current_wp_idx', 1)}/{current_res.get('motion', {}).get('total_wps', 1)}</b>
    </span>
</div>
""",
    unsafe_allow_html=True
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    kpi_card(f"{current_res.get('coverage_pct', 0.0):.1f}%", "Hull Coverage", icon="🗺️")
with k2:
    kpi_card(f"{s_depth.get('depth_measured_m', 5.0):.2f}m", "Submerged Depth", icon="⚓")
with k3:
    kpi_card(f"{s_sonar.get('nearest_hull_distance_m', 1.5):.2f}m", "Hull Standoff", icon="📏")
with k4:
    kpi_card(f"{s_slam.get('voxel_count', s_slam.get('map_points_count', 0))}", "SLAM 3D Voxels", icon="🛰️")
with k5:
    kpi_card(f"{s_slam.get('position_error_m', 0.0)*100:.1f} cm", "SLAM Pose Error", icon="🎯")
with k6:
    kpi_card(f"{current_res.get('confirmed_defects', 0)}", "Defects Verified", icon="🛡️")

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# ---------------- PUBLISH TO PERSISTENT 60 FPS SIMULATION BRIDGE ----------------
state_mgr = get_simulation_state_manager()
cur_wp_idx = min(inspector.current_wp_idx, len(inspector.waypoints) - 1) if inspector.waypoints else 0
act_wp = inspector.waypoints[cur_wp_idx] if inspector.waypoints else {"x": 1.5, "y": 5.0, "z": 8.0, "yaw_rad": 0.0}

state_mgr.update({
    "status": "running" if st.session_state.get("sim_running", False) else ("completed" if inspector.is_completed else "paused"),
    "sim_time_s": float(s_clk.get("sim_time_s", 0.0)),
    "wall_time_s": float(s_clk.get("wall_time_s", 0.0)),
    "speed_multiplier": float(curr_spd),
    "is_paused": not st.session_state.get("sim_running", False),
    "mode": inspector.mode,
    "rov": {
        "x": round(float(inspector.rov.pos[0]), 3),
        "y": round(float(inspector.rov.pos[1]), 3),
        "z": round(float(inspector.rov.pos[2]), 3),
        "roll_deg": round(float(np.degrees(inspector.rov.rpy[0])), 2),
        "pitch_deg": round(float(np.degrees(inspector.rov.rpy[1])), 2),
        "yaw_deg": round(float(np.degrees(inspector.rov.rpy[2])), 2),
        "heading_deg": round(float((np.degrees(inspector.rov.rpy[2]) + 360) % 360), 2),
        "speed_mps": round(float(np.linalg.norm(inspector.rov.vel_body)), 3),
        "quaternion": rpy_to_quaternion(inspector.rov.rpy[0], inspector.rov.rpy[1], inspector.rov.rpy[2])
    },
    "active_wp": act_wp,
    "current_wp_idx": inspector.current_wp_idx,
    "total_wps": len(inspector.waypoints),
    "waypoints": [[round(float(wp["x"]), 2), round(float(wp["y"]), 2), round(float(wp["z"]), 2)] for wp in inspector.waypoints],
    "traj_true": [[p["x"], p["y"], p["z"]] for p in inspector.slam.traj_true[-400:]],
    "traj_est": [[p["x"], p["y"], p["z"]] for p in inspector.slam.traj_est[-400:]],
    "voxels": [[round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2)] for p in inspector.slam.map.get_point_cloud()[-1500:]],
    "defects": inspector.env.defects,
    "standoff_m": round(float(s_sonar.get("nearest_hull_distance_m", 2.0)), 2),
    "depth_m": round(float(s_depth.get("depth_measured_m", 5.0)), 2),
    "ate_error_m": round(float(s_slam.get("position_error_m", 0.0)), 4),
    "collision_status": current_res.get("safety", {}).get("collision_status", "SAFE"),
    "clearance_m": round(float(current_res.get("safety", {}).get("clearance_m", inspector.last_clearance_m)), 2),
    "min_safe_dist_m": round(float(inspector.min_safe_distance_m), 2),
    "drone_radius_m": round(float(inspector.drone_radius_m), 2),
    "nearest_normal": current_res.get("safety", {}).get("outward_normal", [0.0, 0.0, 1.0]),
    "collision_warning": bool(inspector.collision_avoidance_active or inspector.rov.is_collided),
    "debug_mode": bool(inspector.debug_mode)
})

# Collision & Safety Alerts
if inspector.collision_avoidance_active or inspector.rov.is_collided:
    st.error(
        f"🛡️ **HARD COLLISION AVOIDANCE ACTIVE**: Proximity clearance ({current_res.get('safety', {}).get('clearance_m', 0.0):.2f}m) "
        f"reached safe threshold ({inspector.min_safe_distance_m:.1f}m). Drone trajectory was intercepted and deflected along the exterior hull normal.",
        icon="⚠️"
    )

if inspector.debug_mode:
    with st.expander("🛠️ 3D Navigation & Hull Collision Diagnostics", expanded=True):
        cd1, cd2, cd3, cd4 = st.columns(4)
        cd1.metric("Safety Status", current_res.get("safety", {}).get("collision_status", "SAFE"))
        cd2.metric("Exterior Clearance", f"{current_res.get('safety', {}).get('clearance_m', 0.0):.2f} m")
        cd3.metric("Avoidance Deflections", f"{inspector.avoidance_events_count}")
        normal_vec = current_res.get("safety", {}).get("outward_normal", [0, 0, 1])
        cd4.metric("Outward Normal", f"[{normal_vec[0]:.2f}, {normal_vec[1]:.2f}, {normal_vec[2]:.2f}]")

# ---------------- PRIMARY 3D SIMULATION VIEWPORT ----------------
st.markdown("### 🌐 Real-Time 3D Underwater Simulation Viewport")

# Static persistent Three.js HTML: generated ONCE, cached permanently so the WebGL iframe NEVER unmounts, reloads, or flashes black!
if "persistent_3d_html" not in st.session_state or st.session_state.get("sim_server_port") != SIM_SERVER_PORT:
    st.session_state["sim_server_port"] = SIM_SERVER_PORT
    st.session_state["persistent_3d_html"] = generate_persistent_threejs_html(server_port=SIM_SERVER_PORT, height_px=580)

components.html(st.session_state["persistent_3d_html"], height=580, scrolling=False)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ---------------- DUAL PANEL: LIVE CAMERA + SLAM STATE ----------------
col_cam, col_slam = st.columns([1, 1])

with col_cam:
    st.markdown("#### 📷 Forward Synthetic Inspection Camera")
    cam_data = current_res.get("camera", {})
    img_raw = cam_data.get("image")
    ml_pred = current_res.get("ml_prediction")

    if img_raw is not None:
        view_opt = st.segmented_control("Camera View Layer", ["AI Mask Overlay", "Raw Optical Feed", "Ground Truth Annotation"], default="AI Mask Overlay")

        if view_opt == "AI Mask Overlay" and ml_pred is not None:
            disp_img = HullInspectionPredictor.draw_overlay(img_raw, ml_pred, show_boxes=True)
            caption_str = f"Live U-Net Inference &bull; Defect Area: {ml_pred['defect_area_fraction']*100:.1f}% &bull; Confidence: {ml_pred['confidence_score']*100:.0f}%"
        elif view_opt == "Ground Truth Annotation" and cam_data.get("mask") is not None:
            disp_img = img_raw.copy()
            gm = cam_data["mask"] > 0
            disp_img[gm] = (0.45 * disp_img[gm] + 0.55 * np.array([255, 200, 0])).astype(np.uint8)
            caption_str = f"Ground Truth Mask &bull; Defect: {cam_data.get('ground_truth_defect_frac', 0)*100:.1f}%"
        else:
            disp_img = img_raw
            caption_str = f"Synthetic RGB Sensor &bull; Sector {cam_data.get('zone_id', 'HULL')}"

        st.image(disp_img, caption=caption_str, width="stretch")

        if ml_pred and ml_pred.get("defect_detected"):
            area_pct = ml_pred.get('defect_area_fraction', 0.0) * 100
            conf_pct = ml_pred.get('confidence_score', 0.0) * 100
            verdict_badge = badge(f"🚨 SUSPECTED CORROSION CONFIRMED (Area: {area_pct:.1f}% • Conf: {conf_pct:.0f}%)", "poor")
        elif cam_data.get("has_defect"):
            d_id = cam_data.get("defect_id", "DEF")
            verdict_badge = badge(f"⚠️ NEAR REGISTERED DEFECT ZONE ({d_id})", "fair")
        else:
            clear_m = current_res.get('safety', {}).get('clearance_m', inspector.last_clearance_m)
            verdict_badge = badge(f"✅ STEEL PLATING SOUND & CLEAN (Clearance: {clear_m:.2f}m)", "good")

        st.markdown(f"**Inspection Status:** {verdict_badge}", unsafe_allow_html=True)

with col_slam:
    st.markdown("#### 📡 3D SLAM State & Mapping Quality")
    p_true = current_res.get("pose", {})
    p_est = s_slam.get("est_position", [p_true.get("x_m", 1.5), p_true.get("y_m", 5.0), p_true.get("z_m", 8.0)])

    st.markdown(
        f"""
<div style="background: rgba(8, 20, 38, 0.85); border: 1px solid rgba(0, 240, 255, 0.25); border-radius: 12px; padding: 16px; margin-bottom: 12px;">
    <div style="font-family:'Space Grotesk',sans-serif; font-size:1.05rem; font-weight:700; color:#38bdf8; margin-bottom: 8px;">6-DOF Localization Comparison</div>
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.86rem; color:#cbd5e1; line-height:1.6;">
        • <b>Ground Truth Pose:</b> X={p_true.get('x_m', 0.0):.2f}m &bull; Y={p_true.get('y_m', 0.0):.2f}m &bull; Z={p_true.get('z_m', 0.0):.2f}m &bull; Yaw={p_true.get('yaw_deg', 0.0):.1f}°<br/>
        • <b>SLAM Estimated:</b> X={p_est[0]:.2f}m &bull; Y={p_est[1]:.2f}m &bull; Z={p_est[2]:.2f}m<br/>
        • <b>Absolute Trajectory Error (ATE):</b> <span style="color:#00f0ff; font-weight:700;">{s_slam.get('position_error_m', 0.0)*100:.1f} cm</span><br/>
        • <b>3D Voxel Count:</b> <span style="color:#38bdf8; font-weight:700;">{s_slam.get('voxel_count', 0)} voxels</span> &bull; <b>Trajectory Length:</b> {s_slam.get('traj_length_m', 0.0):.1f}m<br/>
        • <b>EKF Covariance Trace:</b> {s_slam.get('covariance_trace', 0.05):.4f} (High Confidence)
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    defects_found = inspector.detected_defect_records
    if defects_found:
        st.write("##### 📍 Discovered 3D Defect Landmarks")
        st.dataframe(pd.DataFrame(defects_found).iloc[::-1].head(5), width="stretch", hide_index=True)
    else:
        st.info("No defect landmarks identified in current sector yet.")

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# ---------------- DEDICATED LIVE SENSOR TELEMETRY PANEL ----------------
st.markdown("### 📊 Dedicated Live Sensor Telemetry Panel")

t_pose, t_mot, t_depth, t_imu, t_sonar, t_cam, t_slam = st.tabs([
    "🤖 Robot Pose",
    "🚀 Motion Dynamics",
    "⚓ Depth Sensor",
    "🧭 6-DOF IMU",
    "📡 Sonar Range",
    "📷 Camera Diagnostics",
    "🗺️ SLAM Quality"
])

with t_pose:
    st.markdown("##### 📍 Authoritative 6-DOF Robot Pose (World Frame)")
    c1, c2, c3, c4 = st.columns(4)
    p = current_res.get("pose", {})
    c1.metric("Position X (Longitudinal)", f"{p.get('x_m', 0.0):.3f} m")
    c2.metric("Position Y (Vertical)", f"{p.get('y_m', 0.0):.3f} m")
    c3.metric("Position Z (Lateral)", f"{p.get('z_m', 0.0):.3f} m")
    c4.metric("Heading (Compass)", f"{p.get('heading_deg', 0.0):.1f}°")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Roll Angle", f"{p.get('roll_deg', 0.0):+.2f}°")
    c6.metric("Pitch Angle", f"{p.get('pitch_deg', 0.0):+.2f}°")
    c7.metric("Yaw Angle", f"{p.get('yaw_deg', 0.0):+.2f}°")
    c8.metric("Coordinate Frame", "NED Underwater (X=Fwd, Y=Up, Z=Stbd)")

with t_mot:
    st.markdown("##### ⚡ Body Velocities & Waypoint Tracking")
    m = current_res.get("motion", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Surge Velocity (Fwd)", f"{m.get('surge_mps', 0.0):+.3f} m/s")
    c2.metric("Heave Velocity (Vert)", f"{m.get('heave_mps', 0.0):+.3f} m/s")
    c3.metric("Sway Velocity (Lat)", f"{m.get('sway_mps', 0.0):+.3f} m/s")
    c4.metric("Linear Speed", f"{m.get('linear_speed_mps', 0.0):.3f} m/s")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Angular Velocity", f"{m.get('angular_velocity_rps', 0.0):.3f} rad/s")
    c6.metric("Dist to Active Waypoint", f"{m.get('dist_to_wp_m', 0.0):.2f} m")
    c7.metric("Waypoint Progress", f"{m.get('current_wp_idx', 1)} / {m.get('total_wps', 1)}")
    c8.metric("Cruising Speed Limit", "1.20 m/s")

with t_depth:
    st.markdown("##### ⚓ Pressure-Derived Depth Measurements")
    d = current_res.get("depth_sensor", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Measured Depth", f"{d.get('depth_measured_m', 0.0):.3f} m")
    c2.metric("Ground Truth Depth", f"{d.get('depth_ground_truth_m', 0.0):.3f} m")
    c3.metric("Sensor Error", f"{d.get('measurement_error_m', 0.0)*100:.2f} cm")
    c4.metric("Hydrostatic Pressure", f"{d.get('hydrostatic_pressure_bar', 1.0):.2f} bar")
    st.caption("Coordinate Reference: Waterline at Y = 10.0m. Sensor Model: P = P_atm + ρgh (ρ = 1025 kg/m³, g = 9.80665 m/s²).")

with t_imu:
    st.markdown("##### 🧭 6-DOF Inertial Measurement Unit (Triaxial Accelerometer & Gyroscope)")
    imu = current_res.get("imu", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accel X", f"{imu.get('accel_x_mps2', 0.0):+.3f} m/s²")
    c2.metric("Accel Y", f"{imu.get('accel_y_mps2', 0.0):+.3f} m/s²")
    c3.metric("Accel Z", f"{imu.get('accel_z_mps2', 0.0):+.3f} m/s²")
    c4.metric("Accel Magnitude", f"{imu.get('accel_magnitude_mps2', 0.0):.3f} m/s²")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Gyro X (Roll Rate)", f"{imu.get('gyro_x_rps', 0.0):+.4f} rad/s")
    c6.metric("Gyro Y (Pitch Rate)", f"{imu.get('gyro_y_rps', 0.0):+.4f} rad/s")
    c7.metric("Gyro Z (Yaw Rate)", f"{imu.get('gyro_z_rps', 0.0):+.4f} rad/s")
    c8.metric("IMU Noise Std / Bias", f"σ={imu.get('noise_std_accel', 0.08)} | ||b||={imu.get('bias_norm', 0.02):.3f}")

with t_sonar:
    st.markdown("##### 📡 Acoustic Multi-Beam Sonar & Standoff Detection")
    s = current_res.get("sonar_sensor", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nearest Hull Range", f"{s.get('nearest_hull_distance_m', 1.5):.2f} m")
    c2.metric("Target Standoff", f"{inspector.standoff_m:.2f} m")
    c3.metric("Acoustic Beam Hits", f"{s.get('valid_returns_count', 0)} / {s.get('total_beams', 35)}")
    c4.metric("Acoustic Ping Rate", f"{s.get('update_rate_hz', 20.0):.0f} Hz")
    st.caption("Acoustic Sensor Array: 35 physical raycast beams (7 horizontal x 5 vertical) with surface normal reflection.")

with t_cam:
    st.markdown("##### 📷 Forward Camera & Vision Diagnostics")
    cam = current_res.get("camera_sensor", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Resolution", f"{cam.get('width_px', 480)} x {cam.get('height_px', 360)} px")
    c2.metric("Effective Camera FPS", f"{cam.get('effective_fps', 15.0):.1f} fps")
    c3.metric("Frame Counter", f"#{cam.get('frame_counter', 1)}")
    c4.metric("U-Net Latency", f"{cam.get('ai_latency_ms', 0.0):.1f} ms")

with t_slam:
    st.markdown("##### 🗺️ 3D SLAM Estimation & Map Density")
    slam = current_res.get("slam", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Absolute Trajectory Error", f"{slam.get('position_error_m', 0.0)*100:.1f} cm")
    c2.metric("Accumulated 3D Voxels", f"{slam.get('voxel_count', 0)}")
    c3.metric("Estimated Path Length", f"{slam.get('traj_length_m', 0.0):.2f} m")
    c4.metric("SLAM Algorithm", "Point-to-Plane ICP + 6-DOF EKF")

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# ---------------- LIVE TELEMETRY SCROLLING GRAPHS ----------------
st.markdown("### 📈 Live Telemetry Time-Series Plots")
history = inspector.telemetry_history

if len(history) > 2:
    plot_df = pd.DataFrame([
        {
            "Sim Time (s)": h["clock"]["sim_time_s"],
            "Submerged Depth (m)": h["depth_sensor"]["depth_measured_m"],
            "Linear Speed (m/s)": h["motion"]["linear_speed_mps"],
            "Hull Standoff (m)": h["sonar_sensor"]["nearest_hull_distance_m"],
            "SLAM ATE Error (m)": h["slam"]["position_error_m"],
            "IMU Accel (m/s²)": h["imu"]["accel_magnitude_mps2"]
        }
        for h in history
    ]).set_index("Sim Time (s)")

    cg1, cg2 = st.columns(2)
    with cg1:
        st.write("##### ⚓ Submerged Depth & Standoff Distance (m)")
        st.line_chart(plot_df[["Submerged Depth (m)", "Hull Standoff (m)"]], color=["#38bdf8", "#00f0ff"])

    with cg2:
        st.write("##### ⚡ Linear Speed (m/s) & SLAM Error (m)")
        st.line_chart(plot_df[["Linear Speed (m/s)", "SLAM ATE Error (m)"]], color=["#10b981", "#f43f5e"])

    with st.expander("📊 Additional Time-Series: IMU Acceleration Magnitude"):
        st.line_chart(plot_df[["IMU Accel (m/s²)"]], color=["#fbbf24"])
else:
    st.info("Collecting live telemetry time-series... Start simulation to stream scrolling sensor charts.")

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# ---------------- PLAIN-ENGLISH COMMENTARY SYSTEM ----------------
st.markdown("### 🎙️ Plain-English Mission Commentary")
commentary_points = explainer.generate_commentary(current_res)
for msg in commentary_points:
    st.markdown(
        f"""
<div style="background: rgba(10, 24, 48, 0.6); border-left: 4px solid #00f0ff; border-radius: 8px; padding: 10px 16px; margin-bottom: 6px; font-size:0.92rem; color:#e2e8f0;">
    {msg}
</div>
""",
        unsafe_allow_html=True
    )

# Continuous auto-rerun loop when simulation is running
if st.session_state.get("sim_running", False):
    delay_s = max(0.015, min(0.12, 0.08 / min(curr_spd, 4.0)))
    time.sleep(delay_s)
    st.rerun()
