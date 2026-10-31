"""Comprehensive Unit & Integration Tests for Hull-Aware 3D Navigation and Collision Avoidance.

Covers all 7 mandatory testing and validation scenarios:
1. Drone approaching the bow
2. Drone inspecting the port side
3. Drone inspecting the starboard side
4. Drone moving underneath the keel
5. Drone approaching the stern
6. Drone transitioning between upper and lower hull sections
7. Drone following sharply curved hull sections
8. Central Hard Safety Constraint (validate_movement) & straight-line hull penetration rejection
9. Manual thruster collision avoidance
"""

import pytest
import numpy as np

from src.simulation.environment_3d import ShipHull3DEnvironment
from src.simulation.rov_dynamics import ROVDynamics6DOF
from src.simulation.slam import UnderwaterSLAM3D
from src.simulation.synthetic_sensors import SyntheticSonarArray, SyntheticCameraPipeline, SyntheticIMUSensor
from src.simulation.autonomous_inspector import AutonomousHullInspector


@pytest.fixture
def test_setup():
    env = ShipHull3DEnvironment(length_m=50.0, height_m=10.0, beam_m=12.0)
    rov = ROVDynamics6DOF(initial_pos=(1.5, 5.0, 8.0))
    slam = UnderwaterSLAM3D(initial_pose=(1.5, 5.0, 8.0))
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
        standoff_m=2.0,
        min_safe_distance_m=1.0,
        max_inspection_distance_m=3.0,
        drone_radius_m=0.40,
        collision_buffer_m=0.45
    )
    return env, rov, inspector


def test_scenario_1_approaching_bow(test_setup):
    """Test 1: Drone approaching the bow along -X must remain outside and never penetrate bow stem."""
    env, rov, inspector = test_setup

    # Position drone 2.5m in front of the bow entry
    rov.pos = np.array([-2.5, 5.0, 0.0])
    rov.vel_body = np.array([1.5, 0.0, 0.0])  # surge velocity towards bow

    # Step physics with forward thrust towards bow
    f_cmd = np.array([80.0, 0.0, 0.0])
    t_cmd = np.zeros(3)

    for _ in range(25):  # 1.25s of thrust towards bow
        inspector.substep_physics(manual_cmd=(f_cmd, t_cmd), dt_s=0.05)
        # Must strictly remain outside solid hull
        assert not env.is_inside_hull(rov.pos)
        # Drone physical body boundary must not intersect hull
        s_pt, norm, dist, inside = env.get_closest_hull_point_and_normal(rov.pos)
        assert not inside
        assert dist >= inspector.drone_radius_m
        # Must not cross into the ship
        assert rov.pos[0] <= 0.0 or dist >= inspector.drone_radius_m


def test_scenario_2_inspecting_port_side(test_setup):
    """Test 2: Drone inspecting the port side (Z < 0) must remain outside and NEVER teleport to starboard."""
    env, rov, inspector = test_setup

    # Position on port side at 2.0m standoff
    z_port = env.get_surface_z(20.0, 5.0, "port")
    rov.pos = np.array([20.0, 5.0, z_port - 2.0])
    initial_z = rov.pos[2]
    assert initial_z < 0.0

    # Advance autonomous inspection or manual drift
    for _ in range(30):
        inspector.substep_physics(manual_cmd=(np.zeros(3), np.zeros(3)), dt_s=0.05)
        # Must remain strictly on the port side (no teleportation to starboard Z > 0)
        assert rov.pos[2] < 0.0
        assert not env.is_inside_hull(rov.pos)
        s_pt, norm, dist, inside = env.get_closest_hull_point_and_normal(rov.pos)
        assert not inside
        assert dist >= inspector.drone_radius_m


def test_scenario_3_inspecting_starboard_side(test_setup):
    """Test 3: Drone inspecting the starboard side (Z > 0) must remain outside and maintain standoff."""
    env, rov, inspector = test_setup

    z_stbd = env.get_surface_z(25.0, 5.0, "starboard")
    rov.pos = np.array([25.0, 5.0, z_stbd + 2.0])

    for _ in range(30):
        inspector.substep_physics(manual_cmd=(np.zeros(3), np.zeros(3)), dt_s=0.05)
        assert rov.pos[2] > 0.0
        assert not env.is_inside_hull(rov.pos)
        s_pt, norm, dist, inside = env.get_closest_hull_point_and_normal(rov.pos)
        assert not inside
        assert dist >= inspector.drone_radius_m
        assert dist <= inspector.max_inspection_distance_m + 0.5


def test_scenario_4_underneath_keel(test_setup):
    """Test 4: Drone moving underneath the keel (Y < 0) must remain outside and never penetrate bottom plate."""
    env, rov, inspector = test_setup

    # Position below keel bottom
    rov.pos = np.array([20.0, -1.8, 0.0])
    rov.vel_body = np.array([0.0, 1.0, 0.0])  # upward velocity towards keel plate

    f_upward = np.array([0.0, 60.0, 0.0])
    for _ in range(25):
        inspector.substep_physics(manual_cmd=(f_upward, np.zeros(3)), dt_s=0.05)
        assert not env.is_inside_hull(rov.pos)
        s_pt, norm, dist, inside = env.get_closest_hull_point_and_normal(rov.pos)
        assert not inside
        assert dist >= inspector.drone_radius_m
        # Must stay beneath or outside keel
        assert rov.pos[1] <= 0.0 or dist >= inspector.drone_radius_m


def test_scenario_5_approaching_stern(test_setup):
    """Test 5: Drone approaching stern counter (X >= 50m) must remain outside."""
    env, rov, inspector = test_setup

    # Position aft of stern transom
    rov.pos = np.array([52.5, 5.0, 0.0])
    f_fwd_to_stern = np.array([-70.0, 0.0, 0.0])  # thrust forward towards stern

    for _ in range(25):
        inspector.substep_physics(manual_cmd=(f_fwd_to_stern, np.zeros(3)), dt_s=0.05)
        assert not env.is_inside_hull(rov.pos)
        s_pt, norm, dist, inside = env.get_closest_hull_point_and_normal(rov.pos)
        assert not inside
        assert dist >= inspector.drone_radius_m


def test_scenario_6_upper_lower_transitions(test_setup):
    """Test 6: Drone transitioning between upper and lower hull sections (bilge curve) maintains outward standoff."""
    env, rov, inspector = test_setup

    # Waypoints traversing from keel (y=1.2) to upper topsides (y=8.5)
    wps = inspector.plan_inspection_route("MIDSHIP_FWD", step_x=4.0, step_y=1.5)
    assert len(wps) >= 6

    for wp in wps:
        pt = np.array([wp["x"], wp["y"], wp["z"]])
        assert not env.is_inside_hull(pt)
        s_pt, norm, dist, inside = env.get_closest_hull_point_and_normal(pt)
        assert not inside
        assert dist >= inspector.min_safe_distance_m


def test_scenario_7_sharply_curved_sections(test_setup):
    """Test 7: Sharply curved bow and stern tapering routes conform to exterior geometry."""
    env, rov, inspector = test_setup

    bow_wps = inspector.plan_inspection_route("BOW_STBD")
    for i in range(len(bow_wps) - 1):
        w1 = np.array([bow_wps[i]["x"], bow_wps[i]["y"], bow_wps[i]["z"]])
        w2 = np.array([bow_wps[i+1]["x"], bow_wps[i+1]["y"], bow_wps[i+1]["z"]])

        # Segment between consecutive waypoints must not penetrate the hull
        has_col, hit, min_clear = env.check_segment_collision(w1, w2, radius=inspector.drone_radius_m)
        assert not has_col, f"Trajectory segment from {w1} to {w2} intersected hull at {hit}!"


def test_central_safety_constraint_rejects_hull_penetration(test_setup):
    """Test 8: Central validate_movement strictly rejects trajectory segments passing straight through hull."""
    env, rov, inspector = test_setup

    # Proposed movement passing straight through the hull from starboard to port
    start_pos = np.array([25.0, 5.0, 7.5])
    bad_dest = np.array([25.0, 5.0, -7.5])

    val = inspector.validate_movement(start_pos, bad_dest)
    # Movement must be rejected
    assert val["is_valid"] is False
    assert val["reason"] in ["SEGMENT_HULL_INTERSECTION", "BODY_INTERIOR_PENETRATION"]
    # Safe position must remain outside the hull
    assert not env.is_inside_hull(val["safe_position"])
    assert np.linalg.norm(val["avoidance_force"]) > 0.0


def test_manual_piloting_hull_collision_prevention(test_setup):
    """Test 9: Full pilot thruster command into hull is halted by hard collision constraint."""
    env, rov, inspector = test_setup
    inspector.mode = "MANUAL"

    # Start 1.5m from starboard hull
    z_surf = env.get_surface_z(25.0, 5.0, "starboard")
    rov.pos = np.array([25.0, 5.0, z_surf + 1.5])

    # Command maximum thrust directly into the hull plate (-Z body thrust)
    # When facing hull (yaw = pi), surge forward drives toward hull (-Z world)
    f_into_hull = np.array([0.0, 0.0, -120.0])  # maximum lateral thrust into hull
    t_zero = np.zeros(3)

    for _ in range(40):  # 2.0s of continuous collision thrust
        inspector.substep_physics(manual_cmd=(f_into_hull, t_zero), dt_s=0.05)
        # The drone must NEVER penetrate the hull interior
        assert not env.is_inside_hull(rov.pos)
        s_pt, norm, dist, inside = env.get_closest_hull_point_and_normal(rov.pos)
        assert not inside
        assert dist >= inspector.drone_radius_m - 0.05

