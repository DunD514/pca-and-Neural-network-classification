"""Autonomous Hull Inspection Route Planner, Conformal Surface Tracking, Simulation Clock, and AI Defect Correlation."""

from typing import Dict, List, Tuple, Any, Optional
import time
import numpy as np

from src.simulation.environment_3d import ShipHull3DEnvironment
from src.simulation.rov_dynamics import ROVDynamics6DOF, ROVFlightController
from src.simulation.slam import UnderwaterSLAM3D
from src.simulation.synthetic_sensors import SyntheticSonarArray, SyntheticCameraPipeline, SyntheticIMUSensor
from src.models.predict import HullInspectionPredictor


class SimulationClock:
    """Simulation clock managing fixed-timestep physics substepping and variable speed multipliers."""

    def __init__(self, fixed_dt_s: float = 0.05):
        self.fixed_dt_s = fixed_dt_s
        self.sim_time_s = 0.0
        self.wall_time_s = 0.0
        self.speed_multiplier = 1.0
        self.accumulator_s = 0.0
        self.total_steps = 0
        self.is_paused = False

    def reset(self) -> None:
        self.sim_time_s = 0.0
        self.wall_time_s = 0.0
        self.accumulator_s = 0.0
        self.total_steps = 0
        self.is_paused = False

    def set_speed(self, multiplier: float) -> None:
        self.speed_multiplier = max(0.1, min(20.0, float(multiplier)))

    def advance(self, wall_dt_s: float) -> int:
        """Advance the clock by elapsed wall-clock time and return number of physics substeps to execute."""
        if self.is_paused:
            return 0

        self.wall_time_s += wall_dt_s
        scaled_dt = wall_dt_s * self.speed_multiplier
        self.accumulator_s += scaled_dt

        # Fixed substepping with floating-point tolerance
        substeps = int((self.accumulator_s + 1e-7) // self.fixed_dt_s)
        # Cap substeps per frame to prevent spiral of death on slower rendering
        substeps = min(substeps, 50)
        self.accumulator_s = max(0.0, self.accumulator_s - substeps * self.fixed_dt_s)
        self.sim_time_s += substeps * self.fixed_dt_s
        self.total_steps += substeps
        return substeps


class AutonomousHullInspector:
    """Master autonomous inspection controller driving the ROV along selected hull zones with full telemetry,
    incorporating central hard safety validation and 3D solid hull collision avoidance."""

    def __init__(
        self,
        env: ShipHull3DEnvironment,
        rov: ROVDynamics6DOF,
        slam: UnderwaterSLAM3D,
        sonar: SyntheticSonarArray,
        camera_pipeline: SyntheticCameraPipeline,
        imu: SyntheticIMUSensor,
        predictor: Optional[HullInspectionPredictor] = None,
        standoff_m: float = 2.0,
        min_safe_distance_m: float = 1.0,
        max_inspection_distance_m: float = 3.0,
        drone_radius_m: float = 0.40,
        collision_buffer_m: float = 0.45,
        **kwargs
    ):
        self.env = env
        self.rov = rov
        self.slam = slam
        self.sonar = sonar
        self.camera_pipeline = camera_pipeline
        self.imu = imu
        self.predictor = predictor

        # Configurable Navigation & Safety Parameters
        self.standoff_m = float(kwargs.get("standoff_m", standoff_m))
        self.min_safe_distance_m = float(kwargs.get("min_safe_distance_m", min_safe_distance_m))
        self.max_inspection_distance_m = float(kwargs.get("max_inspection_distance_m", max_inspection_distance_m))
        self.drone_radius_m = float(kwargs.get("drone_radius_m", drone_radius_m))
        self.collision_buffer_m = float(kwargs.get("collision_buffer_m", collision_buffer_m))
        self.debug_mode = False

        self.clock = SimulationClock(fixed_dt_s=0.05)
        self.controller = ROVFlightController(self.rov)
        self.mode = "AUTONOMOUS"  # "AUTONOMOUS" or "MANUAL"
        self.selected_zone = "BOW_STBD"

        self.waypoints: List[Dict[str, Any]] = []
        self.current_wp_idx = 0
        self.arrival_tolerance_m = 0.55
        self.loop_route = False
        self.is_completed = False
        self.is_paused = False

        # Collision avoidance state
        self.collision_avoidance_active = False
        self.last_clearance_m = float(standoff_m)
        self.last_outward_normal = np.array([0.0, 0.0, 1.0])
        self.avoidance_events_count = 0
        self.last_validation_reason = "SAFE"

        # Inspection metrics
        self.frames_analyzed = 0
        self.candidate_defects_count = 0
        self.confirmed_defects_count = 0
        self.surveyed_area_m2 = 0.0
        self.detected_defect_records: List[Dict[str, Any]] = []
        self.last_ai_latency_ms = 0.0
        self.telemetry_history: List[Dict[str, Any]] = []

        # Generate initial route
        self.plan_inspection_route(self.selected_zone)

    def validate_movement(
        self,
        current_pos: np.ndarray,
        proposed_pos: np.ndarray,
        min_clearance_m: Optional[float] = None
    ) -> Dict[str, Any]:
        """Central Hard Safety Constraint: validate proposed movement against 3D solid hull geometry.

        Checks:
        1. Drone body against hull collision geometry (interior penetration check).
        2. Continuous movement segment for hull intersection.
        3. Minimum clearance from exterior hull surface.

        If collision or unsafe clearance:
            Rejects movement, calculates alternative safe exterior position & avoidance vector.
        Otherwise:
            Allows movement.
        """
        required_clearance = min_clearance_m if min_clearance_m is not None else (self.drone_radius_m + 0.05)

        # 1. Check proposed position against solid hull interior
        s_pt, norm, dist, inside = self.env.get_closest_hull_point_and_normal(proposed_pos)
        clearance = 0.0 if inside else (dist - self.drone_radius_m)

        # 2. Check continuous segment from current to proposed
        has_seg_col, hit_pt, min_seg_clear = self.env.check_segment_collision(
            current_pos, proposed_pos, radius=self.drone_radius_m
        )

        is_valid = True
        reason = "SAFE"
        safe_pos = proposed_pos.copy()

        if inside or clearance < required_clearance or has_seg_col:
            is_valid = False
            self.avoidance_events_count += 1
            if inside:
                reason = "BODY_INTERIOR_PENETRATION"
            elif has_seg_col:
                reason = "SEGMENT_HULL_INTERSECTION"
            else:
                reason = "CLEARANCE_VIOLATION"

            # Compute safe exterior position projected along outward normal
            effective_offset = max(self.drone_radius_m + self.collision_buffer_m, self.min_safe_distance_m)
            safe_pos = s_pt + norm * effective_offset

            # Ensure safe_pos is strictly outside
            if self.env.is_inside_hull(safe_pos):
                safe_pos = s_pt + norm * (effective_offset + 0.5)

        # Repulsive avoidance force along outward normal
        avoidance_force = np.zeros(3)
        if clearance < self.min_safe_distance_m or inside or has_seg_col:
            penetration_depth = max(0.0, self.min_safe_distance_m - clearance) if not inside else (self.min_safe_distance_m + dist)
            repulsive_gain = 120.0
            avoidance_force = norm * (repulsive_gain * penetration_depth + 40.0)

        self.last_validation_reason = reason
        return {
            "is_valid": is_valid,
            "safe_position": safe_pos,
            "clearance_m": max(0.0, clearance),
            "surface_point": s_pt,
            "outward_normal": norm,
            "avoidance_force": avoidance_force,
            "reason": reason,
            "is_inside": inside
        }

    def plan_inspection_route(
        self,
        zone_name: str = "BOW_STBD",
        step_x: float = 2.2,
        step_y: float = 1.4
    ) -> List[Dict[str, Any]]:
        """Generate hull-aware 3D waypoints maintaining precise standoff normal to hull exterior.
        Validates every segment to guarantee no straight line cuts through ship geometry."""
        self.selected_zone = zone_name
        self.current_wp_idx = 0
        self.is_completed = False

        # Determine zone side and bounds
        side = "starboard"
        if zone_name in self.env.zones:
            z_info = self.env.zones[zone_name]
            x_min, x_max = z_info["x_min"] + 0.8, z_info["x_max"] - 0.8
            y_min, y_max = z_info["y_min"] + 1.0, z_info["y_max"] - 1.0
            side = z_info.get("side", "starboard")
        elif zone_name == "KEEL_ZONE":
            x_min, x_max = 2.0, 48.0
            y_min, y_max = 0.5, 2.2
            side = "bottom"
        elif zone_name == "FULL_SURVEY":
            x_min, x_max = 1.5, 48.5
            y_min, y_max = 1.2, 8.8
            side = "starboard"
        else:
            x_min, x_max = 1.0, 11.0
            y_min, y_max = 1.5, 8.5
            side = "starboard"

        xs = np.arange(x_min, x_max + 0.1, step_x)
        ys = np.arange(y_min, y_max + 0.1, step_y)

        raw_waypoints = []
        forward = True

        for iy, y in enumerate(ys):
            row_xs = xs if forward else xs[::-1]
            for x in row_xs:
                if side == "bottom":
                    # Keel inspection underneath hull
                    normal = np.array([0.0, -1.0, 0.0])
                    wp_pos = np.array([x, 0.0, 0.0]) + normal * self.standoff_m
                    target_yaw = 0.0
                else:
                    z_surf = self.env.get_surface_z(x, y, side)
                    normal = self.env.get_surface_normal(x, y, side)
                    # Conformal 3D waypoint offset strictly along validated outward normal
                    wp_pos = np.array([x, y, z_surf]) + normal * self.standoff_m
                    # Constrain to hull envelope to prevent overshooting past bow/stern stems
                    wp_pos[0] = np.clip(wp_pos[0], 0.5, self.env.length_m - 0.5)
                    # Orient front camera towards hull
                    target_yaw = np.arctan2(-normal[0], -normal[2])

                raw_waypoints.append({
                    "x": round(float(wp_pos[0]), 3),
                    "y": round(float(wp_pos[1]), 3),
                    "z": round(float(wp_pos[2]), 3),
                    "yaw_rad": float(target_yaw),
                    "zone": self.env.get_zone_at(x, y),
                    "row": iy
                })
            forward = not forward

        # Trajectory validation & hull contouring between waypoints
        validated_waypoints = []
        for i in range(len(raw_waypoints)):
            validated_waypoints.append(raw_waypoints[i])
            if i < len(raw_waypoints) - 1:
                w1 = np.array([raw_waypoints[i]["x"], raw_waypoints[i]["y"], raw_waypoints[i]["z"]])
                w2 = np.array([raw_waypoints[i+1]["x"], raw_waypoints[i+1]["y"], raw_waypoints[i+1]["z"]])

                # Check if straight line between w1 and w2 violates clearance
                has_col, hit_pt, min_clear = self.env.check_segment_collision(w1, w2, radius=self.drone_radius_m + 0.1)
                if has_col:
                    # Insert intermediate surface-conformal waypoint hugging exterior curvature
                    mid_pos = 0.5 * (w1 + w2)
                    s_pt, norm, dist, inside = self.env.get_closest_hull_point_and_normal(mid_pos)
                    inter_pos = s_pt + norm * self.standoff_m
                    inter_pos[0] = np.clip(inter_pos[0], 0.5, self.env.length_m - 0.5)
                    inter_yaw = float(np.arctan2(-norm[0], -norm[2]))
                    validated_waypoints.append({
                        "x": round(float(inter_pos[0]), 3),
                        "y": round(float(inter_pos[1]), 3),
                        "z": round(float(inter_pos[2]), 3),
                        "yaw_rad": inter_yaw,
                        "zone": raw_waypoints[i]["zone"],
                        "row": raw_waypoints[i]["row"]
                    })

        self.waypoints = validated_waypoints
        return self.waypoints

    def substep_physics(self, manual_cmd: Optional[Tuple[np.ndarray, np.ndarray]] = None, dt_s: float = 0.05) -> None:
        """Execute a single atomic physics substep (controller + 3D collision validation + 6-DOF Fossen integration)."""
        # 1. Flight Control / Guidance
        if self.mode == "AUTONOMOUS":
            if self.current_wp_idx < len(self.waypoints):
                target_wp = self.waypoints[self.current_wp_idx]
                target_pos = np.array([target_wp["x"], target_wp["y"], target_wp["z"]])
                target_yaw = target_wp["yaw_rad"]

                dist_to_wp = np.linalg.norm(self.rov.pos - target_pos)
                if dist_to_wp < self.arrival_tolerance_m:
                    self.current_wp_idx += 1
                    if self.current_wp_idx >= len(self.waypoints):
                        if self.loop_route:
                            self.current_wp_idx = 0
                        else:
                            self.is_completed = True

                if self.is_completed and not self.loop_route:
                    f_cmd = np.zeros(3)
                    t_cmd = np.zeros(3)
                else:
                    active_target_wp = self.waypoints[min(self.current_wp_idx, len(self.waypoints)-1)]
                    active_target_pos = np.array([active_target_wp["x"], active_target_wp["y"], active_target_wp["z"]])
                    active_target_yaw = active_target_wp["yaw_rad"]

                    # Obstruction check to active waypoint: replan detour if segment blocked
                    has_block, hit_pt, _ = self.env.check_segment_collision(
                        self.rov.pos, active_target_pos, radius=self.drone_radius_m + 0.15
                    )
                    if has_block:
                        mid = 0.5 * (self.rov.pos + active_target_pos)
                        s_pt, norm_detour, _, _ = self.env.get_closest_hull_point_and_normal(mid)
                        detour_target = s_pt + norm_detour * (self.standoff_m + 0.4)
                        f_cmd, t_cmd = self.controller.compute_control(detour_target, active_target_yaw, dt_s)
                    else:
                        f_cmd, t_cmd = self.controller.compute_control(active_target_pos, active_target_yaw, dt_s)
            else:
                if self.loop_route:
                    self.current_wp_idx = 0
                else:
                    self.is_completed = True
                f_cmd = np.zeros(3)
                t_cmd = np.zeros(3)
        else:
            if manual_cmd is not None:
                f_cmd, t_cmd = manual_cmd
            else:
                f_cmd, t_cmd = np.zeros(3), np.zeros(3)

        # 2. 3D Surface-Normal Repulsive Potential for Standoff Maintenance
        s_pt, norm, dist, inside = self.env.get_closest_hull_point_and_normal(self.rov.pos)
        clearance = 0.0 if inside else max(0.0, dist - self.drone_radius_m)
        self.last_clearance_m = clearance
        self.last_outward_normal = norm

        if clearance < self.min_safe_distance_m or inside:
            self.collision_avoidance_active = True
            penetration_depth = max(0.1, self.min_safe_distance_m - clearance) if not inside else (self.min_safe_distance_m + dist)
            # Apply repulsive force in body frame along outward normal
            R_world = self.rov.rotation_matrix()
            f_repulsive_world = norm * (90.0 * penetration_depth + 30.0)
            f_repulsive_body = R_world.T @ f_repulsive_world
            f_cmd += f_repulsive_body
        else:
            self.collision_avoidance_active = False

        # 3. Physics Integration (Fossen equations of motion)
        prev_pos = self.rov.pos.copy()
        self.rov.step_physics(f_cmd, t_cmd, dt_s)

        # 4. Central Hard Safety Constraint: validate movement
        val = self.validate_movement(prev_pos, self.rov.pos)
        if not val["is_valid"]:
            # Hard collision rejection: prevent penetrating the hull
            self.rov.pos = val["safe_position"]
            self.rov.is_collided = True
            self.collision_avoidance_active = True

            # Deflect velocity: cancel inward velocity component into hull
            R = self.rov.rotation_matrix()
            v_world = R @ self.rov.vel_body
            v_normal = float(np.dot(v_world, val["outward_normal"]))
            if v_normal < 0.0:
                v_deflected = v_world - v_normal * val["outward_normal"]
                self.rov.vel_body = R.T @ v_deflected
        else:
            self.rov.is_collided = False

        # 5. Boundary Enforcement for Seabed and Waterline
        if self.rov.pos[1] < self.env.seabed_y_m + 0.6:
            self.rov.pos[1] = self.env.seabed_y_m + 0.6
            self.rov.vel_body[1] = max(0.0, self.rov.vel_body[1])
        if self.rov.pos[1] > self.env.waterline_y_m - 0.2:
            self.rov.pos[1] = self.env.waterline_y_m - 0.2
            self.rov.vel_body[1] = min(0.0, self.rov.vel_body[1])

    def advance_simulation(
        self,
        wall_dt_s: float = 0.1,
        manual_cmd: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Advance the simulation clock, execute physics substeps, and produce full sensor telemetry."""
        # 1. Determine physics substeps from simulation clock
        substeps = self.clock.advance(wall_dt_s)
        if substeps == 0 and not self.clock.is_paused:
            substeps = 1
            self.clock.sim_time_s += self.clock.fixed_dt_s
            self.clock.total_steps += 1

        for _ in range(substeps):
            self.substep_physics(manual_cmd=manual_cmd, dt_s=self.clock.fixed_dt_s)

        # 2. Synthetic Sensor Suite Generation
        sonar_res = self.sonar.scan(self.rov)
        imu_res = self.imu.read(self.rov)
        cam_frame = self.camera_pipeline.render_frame(self.rov, sonar_res)

        # Depth sensor reading
        true_depth_m = max(0.0, self.env.waterline_y_m - self.rov.pos[1])
        depth_noise = float(np.random.normal(0, 0.02))
        measured_depth_m = max(0.0, true_depth_m + depth_noise)
        depth_error_m = abs(measured_depth_m - true_depth_m)
        hydrostatic_bar = 1.01325 + (1025.0 * 9.80665 * measured_depth_m) / 1e5

        # 3. 3D SLAM State Update (Point-to-Plane ICP + EKF)
        measured_vel = self.rov.vel_body + np.random.normal(0, 0.02, 3)
        measured_gyro = self.rov.omega_body + np.random.normal(0, 0.008, 3)

        effective_step_dt = substeps * self.clock.fixed_dt_s
        slam_res = self.slam.step(
            true_pose=self.rov.get_pose(),
            measured_vel_body=measured_vel,
            measured_gyro=measured_gyro,
            points_body=sonar_res["points_body"],
            has_defect=sonar_res["has_defect"],
            dt_s=max(0.02, effective_step_dt)
        )

        # 4. AI Defect Inspection on Camera Frame
        ml_prediction = None
        t_ai_start = time.perf_counter()
        if self.predictor is not None:
            self.frames_analyzed += 1
            ml_prediction = self.predictor.predict(cam_frame["image"], run_baseline=False)
            self.last_ai_latency_ms = round((time.perf_counter() - t_ai_start) * 1000.0, 1)

            if ml_prediction["defect_detected"]:
                self.candidate_defects_count += 1
                if ml_prediction["confidence_score"] > 0.65 and ml_prediction["defect_area_fraction"] > 0.06:
                    curr_pos = np.array([self.slam.est_pose[0], self.slam.est_pose[1], self.slam.est_pose[2]])
                    # Spatial deduplication: cluster detections within 1.5m
                    is_new_defect = True
                    for rec in self.detected_defect_records:
                        prev_pos = np.array(rec["world_pos"])
                        if np.linalg.norm(curr_pos - prev_pos) < 1.5:
                            is_new_defect = False
                            rec["confidence"] = max(rec["confidence"], round(ml_prediction["confidence_score"] * 100, 1))
                            rec["defect_area_pct"] = max(rec["defect_area_pct"], round(ml_prediction["defect_area_fraction"] * 100, 2))
                            break

                    if is_new_defect:
                        self.confirmed_defects_count += 1
                        self.detected_defect_records.append({
                            "frame": cam_frame["frame_number"],
                            "defect_id": f"AI-DET-{len(self.detected_defect_records)+1:03d}",
                            "world_pos": [round(float(self.slam.est_pose[0]), 2),
                                          round(float(self.slam.est_pose[1]), 2),
                                          round(float(self.slam.est_pose[2]), 2)],
                            "confidence": round(ml_prediction["confidence_score"] * 100, 1),
                            "zone": cam_frame["zone_id"],
                            "defect_area_pct": round(ml_prediction["defect_area_fraction"] * 100, 2),
                            "sim_time": round(self.clock.sim_time_s, 2)
                        })
        else:
            self.last_ai_latency_ms = 0.0

        # 5. Inspection Progress Calculations
        total_wps = max(1, len(self.waypoints))
        route_progress_pct = min(100.0, (self.current_wp_idx / total_wps) * 100.0)
        self.surveyed_area_m2 = len(self.slam.map.voxels) * (0.35**2)
        total_hull_area = self.env.length_m * self.env.height_m
        coverage_pct = min(100.0, (self.surveyed_area_m2 / (total_hull_area * 0.45)) * 100.0)

        # Waypoint metrics
        active_wp_idx = min(self.current_wp_idx, total_wps - 1)
        active_wp = self.waypoints[active_wp_idx] if self.waypoints else {"x": 0.0, "y": 0.0, "z": 0.0}
        dist_to_wp = float(np.linalg.norm(self.rov.pos - np.array([active_wp["x"], active_wp["y"], active_wp["z"]])))

        # Motion & Clearance metrics
        linear_speed = float(np.linalg.norm(self.rov.vel_body))
        angular_speed = float(np.linalg.norm(self.rov.omega_body))
        _, _, nearest_hull_dist, is_in = self.env.get_closest_hull_point_and_normal(self.rov.pos)
        current_standoff = round(float(nearest_hull_dist), 2)
        accel_x = imu_res.get("accel_x_mps2", 0.0)
        accel_y = imu_res.get("accel_y_mps2", 0.0)
        accel_z = imu_res.get("accel_z_mps2", 0.0)
        accel_mag = float(np.linalg.norm([accel_x, accel_y, accel_z]))

        # Build Comprehensive Telemetry Record
        record = {
            # Simulation Clock
            "clock": {
                "sim_time_s": round(self.clock.sim_time_s, 2),
                "wall_time_s": round(self.clock.wall_time_s, 2),
                "speed_multiplier": self.clock.speed_multiplier,
                "is_paused": self.clock.is_paused,
                "substeps_last_frame": substeps,
                "total_steps": self.clock.total_steps
            },
            # 1. Robot Pose
            "pose": {
                "x_m": round(float(self.rov.pos[0]), 3),
                "y_m": round(float(self.rov.pos[1]), 3),
                "z_m": round(float(self.rov.pos[2]), 3),
                "roll_deg": round(float(np.degrees(self.rov.rpy[0])), 2),
                "pitch_deg": round(float(np.degrees(self.rov.rpy[1])), 2),
                "yaw_deg": round(float(np.degrees(self.rov.rpy[2])), 2),
                "heading_deg": round(float((np.degrees(self.rov.rpy[2]) + 360) % 360), 2)
            },
            # 2. Motion
            "motion": {
                "surge_mps": round(float(self.rov.vel_body[0]), 3),
                "heave_mps": round(float(self.rov.vel_body[1]), 3),
                "sway_mps": round(float(self.rov.vel_body[2]), 3),
                "linear_speed_mps": round(linear_speed, 3),
                "angular_velocity_rps": round(angular_speed, 3),
                "angular_velocity_dps": round(float(np.degrees(angular_speed)), 2),
                "dist_to_wp_m": round(dist_to_wp, 3),
                "current_wp_idx": self.current_wp_idx + 1,
                "total_wps": total_wps,
                "current_target_wp": active_wp
            },
            # 3. Depth Sensor
            "depth_sensor": {
                "depth_below_surface_m": round(measured_depth_m, 2),
                "depth_measured_m": round(measured_depth_m, 3),
                "depth_ground_truth_m": round(true_depth_m, 3),
                "measurement_error_m": round(depth_error_m, 4),
                "hydrostatic_pressure_bar": round(hydrostatic_bar, 2),
                "timestamp_s": round(self.clock.sim_time_s, 2)
            },
            # 4. IMU
            "imu": {
                "accel_x_mps2": round(float(accel_x), 3),
                "accel_y_mps2": round(float(accel_y), 3),
                "accel_z_mps2": round(float(accel_z), 3),
                "accel_magnitude_mps2": round(accel_mag, 3),
                "gyro_x_rps": round(float(imu_res.get("gyro_x_rads", 0.0)), 4),
                "gyro_y_rps": round(float(imu_res.get("gyro_y_rads", 0.0)), 4),
                "gyro_z_rps": round(float(imu_res.get("gyro_z_rads", 0.0)), 4),
                "roll_deg": round(float(np.degrees(self.rov.rpy[0])), 2),
                "pitch_deg": round(float(np.degrees(self.rov.rpy[1])), 2),
                "yaw_deg": round(float(np.degrees(self.rov.rpy[2])), 2),
                "noise_std_accel": self.imu.accel_noise_std,
                "bias_norm": round(float(np.linalg.norm(self.imu.bias_accel)), 4),
                "timestamp_s": round(self.clock.sim_time_s, 2)
            },
            # 5. Range Sensor / Sonar
            "sonar_sensor": {
                "min_range_m": round(float(sonar_res["min_distance_m"]), 2),
                "max_range_m": round(float(np.max(sonar_res.get("ranges", [sonar_res["min_distance_m"]]))), 2),
                "nearest_hull_distance_m": round(float(sonar_res["min_distance_m"]), 2),
                "current_standoff_m": current_standoff,
                "valid_returns_count": int(sonar_res["num_hits"]),
                "total_beams": 35,
                "update_rate_hz": 20.0,
                "timestamp_s": round(self.clock.sim_time_s, 2)
            },
            # 6. Camera
            "camera_sensor": {
                "width_px": cam_frame["image"].shape[1],
                "height_px": cam_frame["image"].shape[0],
                "effective_fps": round(1.0 / max(0.01, wall_dt_s), 1),
                "frame_counter": cam_frame["frame_number"],
                "camera_pose": cam_frame["camera_pose"],
                "timestamp_s": round(self.clock.sim_time_s, 2),
                "ai_latency_ms": self.last_ai_latency_ms,
                "segmentation_status": "Defect Confirmed" if (ml_prediction and ml_prediction["defect_detected"]) else "Sound Hull"
            },
            # 7. Localization & SLAM
            "slam": {
                "status": "Active (3D Point-to-Plane ICP + EKF)",
                "est_position": [round(float(x), 3) for x in self.slam.est_pose[:3]],
                "ground_truth_position": [round(float(x), 3) for x in self.rov.pos],
                "position_error_m": round(float(slam_res.get("localization_error_m", 0.0)), 4),
                "voxel_count": len(self.slam.map.voxels),
                "update_rate_hz": 10.0,
                "traj_length_m": round(float(self.slam.traj_length_est), 2),
                "covariance_trace": round(float(np.trace(self.slam.cov_P)), 4),
                "timestamp_s": round(self.clock.sim_time_s, 2),
                "map_points_count": slam_res.get("map_points_count", len(self.slam.map.voxels)),
                "localization_error_m": slam_res.get("localization_error_m", 0.0)
            },
            # 8. Navigation & Collision Safety Telemetry
            "safety": {
                "collision_status": "AVOIDANCE_ACTIVE" if self.collision_avoidance_active else ("COLLISION" if self.rov.is_collided else "SAFE"),
                "clearance_m": round(self.last_clearance_m, 3),
                "min_safe_distance_m": round(self.min_safe_distance_m, 2),
                "preferred_standoff_m": round(self.standoff_m, 2),
                "drone_radius_m": round(self.drone_radius_m, 2),
                "avoidance_events_count": self.avoidance_events_count,
                "last_reason": self.last_validation_reason,
                "outward_normal": [round(float(x), 3) for x in self.last_outward_normal]
            },
            # Inspection Progress & Legacy Keys
            "mode": self.mode,
            "status": "completed" if self.is_completed else ("paused" if self.clock.is_paused else "inspecting"),
            "current_zone": cam_frame["zone_id"],
            "route_progress_pct": round(route_progress_pct, 1),
            "coverage_pct": round(coverage_pct, 1),
            "surveyed_area_m2": round(self.surveyed_area_m2, 1),
            "frames_analyzed": self.frames_analyzed,
            "candidate_defects": self.candidate_defects_count,
            "confirmed_defects": self.confirmed_defects_count,
            "detected_defect_records": self.detected_defect_records,
            "rov_true_pose": self.rov.get_pose(),
            "sonar": {
                "min_distance_m": sonar_res["min_distance_m"],
                "mean_standoff_m": sonar_res["mean_standoff_m"],
                "num_hits": sonar_res["num_hits"]
            },
            "avionics": {
                "submerged_depth_m": round(measured_depth_m, 2),
                "hydrostatic_pressure_bar": round(hydrostatic_bar, 2),
                "speed_mps": round(linear_speed, 2),
                "battery_pct": round(self.rov.battery_percent, 1)
            },
            "camera": cam_frame,
            "ml_prediction": ml_prediction
        }

        # Append to rolling history
        self.telemetry_history.append(record)
        if len(self.telemetry_history) > 300:
            self.telemetry_history.pop(0)

        return record

    def step(self, manual_cmd: Optional[Tuple[np.ndarray, np.ndarray]] = None, dt_s: float = 0.05) -> Dict[str, Any]:
        """Legacy atomic step function (used for single-step manual stepping)."""
        wall_dt = dt_s / max(0.1, self.clock.speed_multiplier)
        return self.advance_simulation(wall_dt_s=wall_dt, manual_cmd=manual_cmd)
