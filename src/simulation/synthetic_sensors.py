"""Synthetic Underwater Sensor Generation: Ray-Casting Sonar, Synthetic Camera, 6-DOF IMU, and Depth."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2

from src.simulation.environment_3d import ShipHull3DEnvironment
from src.simulation.rov_dynamics import ROVDynamics6DOF


class SyntheticSonarArray:
    """Multi-beam acoustic sonar profiler performing physical ray-casting against 3D ship hull."""

    def __init__(
        self,
        env: ShipHull3DEnvironment,
        num_horizontal_beams: int = 25,
        num_vertical_beams: int = 5,
        fov_h_deg: float = 70.0,
        fov_v_deg: float = 25.0,
        max_range_m: float = 8.0,
        noise_std_m: float = 0.015,
        random_seed: int = 42
    ):
        self.env = env
        self.num_h = num_horizontal_beams
        self.num_v = num_vertical_beams
        self.fov_h = np.radians(fov_h_deg)
        self.fov_v = np.radians(fov_v_deg)
        self.max_range = max_range_m
        self.noise_std = noise_std_m
        self.rng = np.random.default_rng(random_seed)

        # Precompute ray directions in sensor frame (forward = +Z, right = +X, up = +Y)
        h_angles = np.linspace(-self.fov_h / 2.0, self.fov_h / 2.0, self.num_h)
        v_angles = np.linspace(-self.fov_v / 2.0, self.fov_v / 2.0, self.num_v)

        self.sensor_rays = []
        for v in v_angles:
            for h in h_angles:
                # Direction vector in sensor frame pointing toward hull (-Z)
                dx = np.sin(h) * np.cos(v)
                dy = np.sin(v)
                dz = -np.cos(h) * np.cos(v)
                self.sensor_rays.append(np.array([dx, dy, dz]))
        self.sensor_rays = np.array(self.sensor_rays)

    def scan(self, rov: ROVDynamics6DOF) -> Dict[str, Any]:
        """Perform 3D ray-casting from current ROV pose against the hull geometry."""
        R_world = rov.rotation_matrix()
        origin = rov.pos.copy()

        # Ray origin offset to front sensor nose of the ROV pointing toward hull
        sensor_origin = origin + R_world @ np.array([0.0, 0.0, -0.2])

        points_body = []
        points_world = []
        ranges = []
        hit_defects = []

        # Transform rays to world frame and cast
        for ray_b in self.sensor_rays:
            ray_w = R_world @ ray_b
            hit = self.env.ray_cast_hull(sensor_origin, ray_w, self.max_range)

            if hit is not None:
                r_clean = hit["distance"]
                noise = float(self.rng.normal(0, self.noise_std))
                r_noisy = max(0.05, min(self.max_range, r_clean + noise))
                hit_pt_world = sensor_origin + ray_w * r_noisy
                hit_pt_body = R_world.T @ (hit_pt_world - origin)

                ranges.append(r_noisy)
                points_world.append(hit_pt_world)
                points_body.append(hit_pt_body)
                if hit["has_defect"]:
                    hit_defects.append(hit["defect"]["defect_id"])

        ranges_arr = np.array(ranges) if ranges else np.array([self.max_range])
        min_dist = float(np.min(ranges_arr))
        mean_standoff = float(np.median(ranges_arr))

        return {
            "num_beams": len(self.sensor_rays),
            "num_hits": len(points_world),
            "ranges": ranges,
            "min_distance_m": round(min_dist, 3),
            "mean_standoff_m": round(mean_standoff, 3),
            "points_body": np.array(points_body) if points_body else np.zeros((0, 3)),
            "points_world": np.array(points_world) if points_world else np.zeros((0, 3)),
            "detected_defects": list(set(hit_defects)),
            "has_defect": len(hit_defects) > 0
        }


class SyntheticCameraPipeline:
    """Simulated HD forward inspection camera rendering the vessel hull from ROV pose."""

    def __init__(self, env: ShipHull3DEnvironment, resolution: Tuple[int, int] = (640, 480)):
        self.env = env
        self.res = resolution
        self.frame_number = 0
        self.rng = np.random.default_rng(42)

    def render_frame(
        self,
        rov: ROVDynamics6DOF,
        sonar_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Render optical camera feed and ground-truth corrosion mask for current view."""
        self.frame_number += 1
        x, y, z = rov.pos
        zone = self.env.get_zone_at(x, y)

        # Determine if camera footprint intersects any ground-truth defects
        defect_hit = None
        defect_frac = 0.0
        for d in self.env.defects:
            dist = np.sqrt((x - d["x_m"])**2 + (y - d["y_m"])**2)
            if dist < d["radius_m"] + 0.5:
                defect_hit = d
                defect_frac = float(np.clip(1.0 - (dist / (d["radius_m"] + 0.5)), 0.1, 0.95))
                break

        # Generate realistic underwater image canvas
        w, h = self.res
        img = np.zeros((h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Base hull plate color (dark navy/grey primer with weld seam texture)
        plate_color = np.array([42, 58, 72], dtype=np.float32) # Starboard steel
        # Add steel plate grain noise
        noise = self.rng.normal(0, 8, (h, w, 3)).astype(np.float32)
        canvas = np.clip(plate_color + noise, 0, 255).astype(np.uint8)

        # Draw structural plate weld lines if within view (pure numpy for performance & robustness)
        # Vertical weld seams
        for wx in self.env.weld_lines_x:
            if abs(x - wx) < 1.8:
                px = int(w / 2 + (wx - x) * (w / 3.6))
                if 2 <= px < w - 2:
                    canvas[:, px-2:px+3] = (30, 40, 50)
                    canvas[:, px:px+1] = (85, 105, 120)

        # Horizontal weld seams
        for wy in self.env.weld_lines_y:
            if abs(y - wy) < 1.4:
                py = int(h / 2 - (wy - y) * (h / 2.8))
                if 2 <= py < h - 2:
                    canvas[py-2:py+3, :] = (30, 40, 50)
                    canvas[py:py+1, :] = (85, 105, 120)

        # Draw corrosion if defect in view
        if defect_hit is not None and defect_frac > 0.05:
            # Elliptical corrosion cluster
            cx = int(w/2 + self.rng.integers(-30, 30))
            cy = int(h/2 + self.rng.integers(-20, 20))
            ax = int(w * 0.35 * defect_frac)
            ay = int(h * 0.30 * defect_frac)

            # Ground truth mask
            cv2.ellipse(mask, (cx, cy), (ax, ay), 15, 0, 360, 255, -1)
            # Add texture to corrosion (rust oxide: red-brown)
            rust_color = np.array([160, 75, 38], dtype=np.float32)
            rust_patch = np.zeros_like(canvas)
            rust_patch[:] = rust_color
            rust_noise = self.rng.normal(0, 18, (h, w, 3)).astype(np.float32)
            rust_patch = np.clip(rust_patch + rust_noise, 0, 255).astype(np.uint8)

            m_bool = mask > 0
            canvas[m_bool] = (0.35 * canvas[m_bool] + 0.65 * rust_patch[m_bool]).astype(np.uint8)

        # Apply underwater optical attenuation (Jaffe-McGlamery light absorption)
        # Red wavelength absorption: reduce red channel (RGB channel 0) based on depth
        depth_m = max(0.1, self.env.waterline_y_m - y)
        red_scale = float(np.exp(-0.14 * depth_m))
        canvas[:, :, 0] = np.clip(canvas[:, :, 0] * red_scale, 0, 255).astype(np.uint8)
        # Blue/Green turbidity cast (RGB channels 1 and 2)
        canvas[:, :, 2] = np.clip(canvas[:, :, 2] * 1.05 + 8, 0, 255).astype(np.uint8)
        canvas[:, :, 1] = np.clip(canvas[:, :, 1] * 1.02 + 4, 0, 255).astype(np.uint8)

        # Marine snow particles
        num_snow = int(40 + 10 * depth_m)
        for _ in range(num_snow):
            sx = self.rng.integers(0, w)
            sy = self.rng.integers(0, h)
            sz = self.rng.integers(1, 3)
            cv2.circle(canvas, (sx, sy), sz, (180, 210, 230), -1)

        return {
            "image": canvas,
            "mask": mask,
            "has_defect": defect_hit is not None,
            "defect_id": defect_hit["defect_id"] if defect_hit else None,
            "defect_type": defect_hit["defect_type"] if defect_hit else "sound_hull",
            "ground_truth_defect_frac": round(float(np.mean(mask > 0)), 4),
            "zone_id": zone,
            "frame_number": self.frame_number,
            "timestamp": round(self.frame_number * 0.1, 2),
            "camera_pose": rov.get_pose()
        }


class SyntheticIMUSensor:
    """6-DOF Inertial Measurement Unit producing noisy acceleration, angular rates, and heading."""

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)
        self.accel_noise_std = 0.08
        self.gyro_noise_std = 0.015
        self.bias_accel = self.rng.normal(0, 0.02, 3)
        self.bias_gyro = self.rng.normal(0, 0.005, 3)

    def read(self, rov: ROVDynamics6DOF) -> Dict[str, Any]:
        """Compute accelerometer and gyroscope readings in body frame."""
        R = rov.rotation_matrix()
        g_world = np.array([0.0, -9.80665, 0.0])
        # Specific force: a_meas = R^T * (a_world - g_world)
        a_specific_body = R.T @ (-g_world) + self.bias_accel + self.rng.normal(0, self.accel_noise_std, 3)

        gyro_body = rov.omega_body + self.bias_gyro + self.rng.normal(0, self.gyro_noise_std, 3)

        return {
            "accel_x_mps2": round(float(a_specific_body[0]), 3),
            "accel_y_mps2": round(float(a_specific_body[1]), 3),
            "accel_z_mps2": round(float(a_specific_body[2]), 3),
            "gyro_x_rads": round(float(gyro_body[0]), 4),
            "gyro_y_rads": round(float(gyro_body[1]), 4),
            "gyro_z_rads": round(float(gyro_body[2]), 4),
            "pitch_deg": round(float(np.degrees(rov.rpy[1])), 2),
            "roll_deg": round(float(np.degrees(rov.rpy[0])), 2),
            "compass_heading_deg": round(float(np.degrees(rov.rpy[2])) % 360, 2)
        }
