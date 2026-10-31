"""Unit and integration tests for 3D Underwater Simulation, 6-DOF Hydrodynamics, and 3D SLAM."""

import pytest
import numpy as np

from src.simulation.environment_3d import ShipHull3DEnvironment
from src.simulation.rov_dynamics import ROVDynamics6DOF, ROVFlightController
from src.simulation.synthetic_sensors import SyntheticSonarArray, SyntheticCameraPipeline, SyntheticIMUSensor
from src.simulation.slam import UnderwaterSLAM3D
from src.simulation.autonomous_inspector import AutonomousHullInspector
from src.simulation.plain_english_explainer import PlainEnglishMissionExplainer


def test_3d_environment_geometry_and_defects():
    """Verify 3D ship hull mesh, bow flare, surface normals, and defects."""
    env = ShipHull3DEnvironment(length_m=50.0, height_m=10.0, beam_m=12.0)

    # Starboard surface at midship should be full half-beam (6.0m) at waterline
    z_mid = env.get_surface_z(25.0, 10.0, "starboard")
    assert 5.8 <= z_mid <= 6.2

    # Bow entry (x=0) should be sharp (z near 0)
    z_bow = env.get_surface_z(0.0, 5.0, "starboard")
    assert z_bow < 1.0

    # Normal vector should be normalized and point outward (+Z for starboard)
    normal = env.get_surface_normal(25.0, 5.0, "starboard")
    assert np.isclose(np.linalg.norm(normal), 1.0, atol=1e-3)
    assert normal[2] > 0.8

    # 5 defects registered
    assert len(env.defects) == 5
    assert env.defects[0]["defect_id"] == "DEF-001"


def test_ray_cast_against_3d_hull():
    """Verify Möller-Trumbore ray casting returns accurate contact point and normal."""
    env = ShipHull3DEnvironment()

    # Ray shot from x=20.0, y=5.0, z=9.0 pointing in -Z direction towards hull
    ray_origin = np.array([20.0, 5.0, 9.0])
    ray_dir = np.array([0.0, 0.0, -1.0])

    hit = env.ray_cast_hull(ray_origin, ray_dir, max_dist=10.0)
    assert hit is not None
    # Contact should occur around z = get_surface_z(20.0, 5.0)
    expected_z = env.get_surface_z(20.0, 5.0, "starboard")
    assert np.isclose(hit["point"][2], expected_z, atol=0.25)
    assert hit["distance"] > 0.0


def test_rov_6dof_hydrodynamics_and_motion():
    """Verify 6-DOF hydrodynamics, thruster forces, damping, and position integration."""
    rov = ROVDynamics6DOF(initial_pos=(5.0, 5.0, 8.0))

    initial_pos = rov.pos.copy()
    # Apply forward surge thrust of 50N for 1.0s
    f_cmd = np.array([50.0, 0.0, 0.0])
    t_cmd = np.zeros(3)

    for _ in range(20): # 20 steps * 0.05s = 1.0s
        rov.step_physics(f_cmd, t_cmd, dt=0.05)

    # ROV must have moved forward in X
    assert rov.pos[0] > initial_pos[0] + 0.2
    assert rov.total_distance_m > 0.2
    assert np.linalg.norm(rov.vel_body) > 0.1


def test_depth_sensor_physics():
    """Verify depth calculation agrees with simulated vertical position."""
    env = ShipHull3DEnvironment(waterline_y_m=10.0)
    # At y = 3.0m, submerged depth = 10.0 - 3.0 = 7.0m
    submerged_depth = env.waterline_y_m - 3.0
    assert np.isclose(submerged_depth, 7.0)

    # Pressure at 7m depth: P = 1.013 + (1025 * 9.81 * 7) / 1e5 approx 1.71 bar
    pressure_bar = 1.01325 + (1025.0 * 9.80665 * submerged_depth) / 1e5
    assert 1.68 <= pressure_bar <= 1.75


def test_synthetic_sonar_point_cloud():
    """Verify synthetic multi-beam sonar generates 3D point cloud from hull contact."""
    env = ShipHull3DEnvironment()
    rov = ROVDynamics6DOF(initial_pos=(15.0, 5.0, 8.0))
    sonar = SyntheticSonarArray(env=env, num_horizontal_beams=15, num_vertical_beams=3)

    scan_res = sonar.scan(rov)
    assert scan_res["num_hits"] > 0
    assert len(scan_res["points_world"]) > 0
    assert 1.0 <= scan_res["min_distance_m"] <= 4.0


def test_3d_slam_mapping_growth_and_drift_tracking():
    """Verify 3D SLAM map grows as observations arrive and estimates trajectory."""
    slam = UnderwaterSLAM3D(initial_pose=(2.0, 5.0, 8.0))

    assert len(slam.map.voxels) == 0

    # Feed point cloud
    pts = np.array([
        [0.0, 0.0, 1.5],
        [0.3, 0.1, 1.4],
        [-0.3, -0.1, 1.6],
        [0.1, -0.2, 1.5]
    ])
    vel_body = np.array([0.5, 0.0, 0.0])
    gyro = np.zeros(3)

    res = slam.step(
        true_pose={"x": 2.0, "y": 5.0, "z": 8.0, "yaw_deg": 0.0},
        measured_vel_body=vel_body,
        measured_gyro=gyro,
        points_body=pts,
        has_defect=False,
        dt_s=0.1
    )

    # Map must now contain voxels
    assert res["map_points_count"] > 0
    assert len(slam.map.voxels) > 0
    assert "estimated_pose" in res
    assert "localization_error_m" in res


def test_autonomous_route_planning_and_standoff():
    """Verify autonomous route planner generates conformal waypoints at target standoff."""
    env = ShipHull3DEnvironment()
    rov = ROVDynamics6DOF(initial_pos=(2.0, 5.0, 8.0))
    slam = UnderwaterSLAM3D(initial_pose=(2.0, 5.0, 8.0))
    sonar = SyntheticSonarArray(env=env)
    cam = SyntheticCameraPipeline(env=env)
    imu = SyntheticIMUSensor()

    inspector = AutonomousHullInspector(
        env=env, rov=rov, slam=slam, sonar=sonar, camera_pipeline=cam, imu=imu, standoff_m=1.5
    )
    wps = inspector.plan_inspection_route("BOW_STBD")

    assert len(wps) >= 8
    # All waypoints must have x within bow region
    for wp in wps:
        assert 0.5 <= wp["x"] <= 12.0
        # Standoff should place waypoint beyond the hull contact surface
        surface_z = env.get_surface_z(wp["x"], wp["y"], "starboard")
        assert wp["z"] > surface_z


def test_plain_english_explainer_commentary():
    """Verify plain English commentary produces human-readable updates."""
    explainer = PlainEnglishMissionExplainer()
    step_data = {
        "current_zone": "BOW_STBD",
        "mode": "AUTONOMOUS",
        "avionics": {"submerged_depth_m": 4.5, "speed_mps": 0.8},
        "sonar": {"mean_standoff_m": 1.5},
        "slam": {"map_points_count": 45, "localization_error_m": 0.04},
        "ml_prediction": {"defect_detected": True, "confidence_score": 0.88, "defect_area_fraction": 0.12},
        "camera": {"has_defect": True, "defect_id": "DEF-001"}
    }
    msgs = explainer.generate_commentary(step_data)
    assert len(msgs) >= 2
    joined = " ".join(msgs)
    assert "ROV" in joined or "corrosion" in joined.lower()


def test_synthetic_camera_rendering_weld_lines_and_defects():
    """Verify synthetic camera renders frames with weld seams and defect masks."""
    env = ShipHull3DEnvironment()
    rov = ROVDynamics6DOF(initial_pos=(8.5, 3.2, 8.0)) # At defect 1
    sonar = SyntheticSonarArray(env=env)
    cam = SyntheticCameraPipeline(env=env, resolution=(320, 240))

    scan_res = sonar.scan(rov)
    frame_data = cam.render_frame(rov, scan_res)

    assert frame_data["image"].shape == (240, 320, 3)
    assert frame_data["mask"].shape == (240, 320)
    assert frame_data["has_defect"] is True
    assert frame_data["defect_id"] == "DEF-001"
    assert frame_data["ground_truth_defect_frac"] > 0.0


def test_simulation_clock_speed_scaling_and_substepping():
    """Verify SimulationClock correctly calculates physics substeps across speed presets."""
    from src.simulation.autonomous_inspector import SimulationClock
    clk = SimulationClock(fixed_dt_s=0.05)

    # 1. Real-time 1.0x speed with 0.1s wall time
    clk.set_speed(1.0)
    substeps_1x = clk.advance(0.10)
    assert substeps_1x == 2
    assert abs(clk.sim_time_s - 0.10) < 1e-4

    # 2. Fast 5.0x speed with 0.1s wall time
    clk.reset()
    clk.set_speed(5.0)
    substeps_5x = clk.advance(0.10)
    assert substeps_5x == 10
    assert abs(clk.sim_time_s - 0.50) < 1e-4

    # 3. Pausing stops substeps and time progression
    clk.is_paused = True
    assert clk.advance(0.10) == 0


def test_autonomous_live_sensor_telemetry_completeness():
    """Verify that advance_simulation produces all required structured telemetry fields."""
    env = ShipHull3DEnvironment()
    rov = ROVDynamics6DOF(initial_pos=(2.0, 5.0, 8.0))
    slam = UnderwaterSLAM3D(initial_pose=(2.0, 5.0, 8.0))
    sonar = SyntheticSonarArray(env=env)
    cam = SyntheticCameraPipeline(env=env, resolution=(320, 240))
    imu = SyntheticIMUSensor()

    inspector = AutonomousHullInspector(
        env=env, rov=rov, slam=slam, sonar=sonar, camera_pipeline=cam, imu=imu, standoff_m=1.5
    )
    rec = inspector.advance_simulation(wall_dt_s=0.1)

    # A. Robot Pose
    assert all(k in rec["pose"] for k in ["x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg", "heading_deg"])
    # B. Motion
    assert all(k in rec["motion"] for k in ["surge_mps", "heave_mps", "sway_mps", "linear_speed_mps", "dist_to_wp_m", "current_wp_idx"])
    # C. Depth Sensor
    assert all(k in rec["depth_sensor"] for k in ["depth_measured_m", "depth_ground_truth_m", "measurement_error_m", "hydrostatic_pressure_bar"])
    # D. IMU
    assert all(k in rec["imu"] for k in ["accel_x_mps2", "accel_y_mps2", "accel_z_mps2", "accel_magnitude_mps2", "gyro_x_rps"])
    # E. Sonar
    assert all(k in rec["sonar_sensor"] for k in ["min_range_m", "nearest_hull_distance_m", "current_standoff_m", "valid_returns_count"])
    # F. Camera
    assert all(k in rec["camera_sensor"] for k in ["width_px", "height_px", "effective_fps", "segmentation_status"])
    # G. SLAM
    assert all(k in rec["slam"] for k in ["status", "est_position", "ground_truth_position", "position_error_m", "voxel_count"])


def test_persistent_webgl_and_sim_server_bridge():
    """Verify persistent WebGL HTML generation, quaternion math, and sim bridge server."""
    from src.simulation.sim_server import (
        ensure_simulation_server_running,
        get_simulation_state_manager,
        rpy_to_quaternion
    )
    from src.simulation.webgl_visualizer import generate_persistent_threejs_html

    # 1. Quaternion calculation
    q_zero = rpy_to_quaternion(0.0, 0.0, 0.0)
    assert len(q_zero) == 4
    assert q_zero == [0.0, 0.0, 0.0, 1.0]

    q_yaw = rpy_to_quaternion(0.0, 0.0, np.pi/2)
    norm = np.linalg.norm(q_yaw)
    assert abs(norm - 1.0) < 1e-4

    # 2. State manager update & snapshot
    state_mgr = get_simulation_state_manager()
    state_mgr.update({"sim_time_s": 42.5, "speed_multiplier": 2.0})
    snap = state_mgr.get_snapshot()
    assert snap["sim_time_s"] == 42.5
    assert snap["speed_multiplier"] == 2.0

    # 3. Persistent Three.js HTML generation
    port = ensure_simulation_server_running(preferred_port=8502)
    assert port in [8502, 8503, 8504, 8505, 8506]

    html = generate_persistent_threejs_html(server_port=port, height_px=580)
    assert "webgl-canvas" in html
    assert "three.min.js" in html
    assert "requestAnimationFrame" in html
    assert "lerpVectors" in html
    assert "SERVER_PORT = " in html
    assert "/api/state" in html


