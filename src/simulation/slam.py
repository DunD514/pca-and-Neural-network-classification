"""Underwater 3D SLAM (Simultaneous Localization and Mapping) Engine.

Implements:
1. Extended Kalman Filter (EKF) fusing noisy DVL odometry, IMU, and Scan Matching.
2. Iterative Closest Point (ICP) scan-to-map alignment against accumulated hull point cloud.
3. 3D Voxel Occupancy Grid & Point Cloud Map growing from scratch.
4. Trajectory Tracking: Ground-truth vs Estimated with localization error & 3-sigma covariance.
5. Defect 3D landmark registration and mapping quality metrics.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class VoxelGrid3D:
    """Discrete 3D voxel occupancy representation of the surveyed ship hull."""

    def __init__(self, voxel_size_m: float = 0.30):
        self.voxel_size = voxel_size_m
        # Dictionary mapping (vx, vy, vz) -> {"count": int, "centroid": np.ndarray, "has_defect": bool}
        self.voxels: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    def _to_voxel_coord(self, pt: np.ndarray) -> Tuple[int, int, int]:
        return (
            int(np.floor(pt[0] / self.voxel_size)),
            int(np.floor(pt[1] / self.voxel_size)),
            int(np.floor(pt[2] / self.voxel_size))
        )

    def insert_points(self, points: np.ndarray, has_defect: bool = False) -> int:
        """Insert a batch of 3D points into the voxel map."""
        new_voxels = 0
        for pt in points:
            key = self._to_voxel_coord(pt)
            if key not in self.voxels:
                self.voxels[key] = {
                    "count": 1,
                    "centroid": pt.copy(),
                    "has_defect": has_defect
                }
                new_voxels += 1
            else:
                v = self.voxels[key]
                v["count"] += 1
                # Running average of centroid
                alpha = 1.0 / v["count"]
                v["centroid"] = (1.0 - alpha) * v["centroid"] + alpha * pt
                if has_defect:
                    v["has_defect"] = True
        return new_voxels

    def get_point_cloud(self) -> np.ndarray:
        """Retrieve all occupied voxel centroids as a Nx3 array."""
        if not self.voxels:
            return np.zeros((0, 3))
        return np.array([v["centroid"] for v in self.voxels.values()])

    def get_defect_landmarks(self) -> List[Dict[str, Any]]:
        """Retrieve 3D clusters flagged as defect locations."""
        defects = []
        for key, v in self.voxels.items():
            if v["has_defect"]:
                defects.append({
                    "voxel": key,
                    "pos": v["centroid"].tolist(),
                    "observations": v["count"]
                })
        return defects


class UnderwaterSLAM3D:
    """Authentic 3D SLAM system with scan matching, state estimation, and live mapping."""

    def __init__(
        self,
        initial_pose: Tuple[float, float, float] = (1.0, 5.0, 7.5),
        initial_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        voxel_size_m: float = 0.35,
        random_seed: int = 42
    ):
        self.rng = np.random.default_rng(random_seed)

        # 6-DOF Estimated State vector: [x, y, z, roll, pitch, yaw]
        self.est_pose = np.array([
            initial_pose[0], initial_pose[1], initial_pose[2],
            np.radians(initial_rpy[0]), np.radians(initial_rpy[1]), np.radians(initial_rpy[2])
        ], dtype=float)

        # Estimation Error Covariance Matrix (6x6)
        self.P = np.diag([0.02, 0.02, 0.02, 0.005, 0.005, 0.01])**2

        # Process Noise Covariance (Q) per second
        self.Q = np.diag([0.05, 0.05, 0.04, 0.01, 0.01, 0.02])**2

        # Sensor Measurement Noise Covariance (R)
        self.R_dvl = np.diag([0.03, 0.03, 0.03])**2
        self.R_icp = np.diag([0.04, 0.04, 0.03, 0.02, 0.02, 0.02])**2

        # 3D Voxel Map
        self.map = VoxelGrid3D(voxel_size_m=voxel_size_m)

        # Trajectory History
        self.traj_est: List[Dict[str, Any]] = []
        self.traj_true: List[Dict[str, Any]] = []
        self.ate_history: List[float] = []

        # SLAM Performance Statistics
        self.total_icp_iterations = 0
        self.last_icp_residual_m = 0.0
        self.drift_error_m = 0.0
        self.step_count = 0

    @property
    def traj_length_est(self) -> float:
        """Accumulated length of estimated SLAM trajectory in meters."""
        if len(self.traj_est) < 2:
            return 0.0
        dist = 0.0
        for i in range(1, len(self.traj_est)):
            p1 = self.traj_est[i-1]
            p2 = self.traj_est[i]
            dist += float(np.linalg.norm([p2["x"]-p1["x"], p2["y"]-p1["y"], p2["z"]-p1["z"]]))
        return dist

    @property
    def cov_P(self) -> np.ndarray:
        return self.P

    def rotation_matrix(self, rpy: np.ndarray) -> np.ndarray:
        """Body to world rotation matrix."""
        cr, cp, cy = np.cos(rpy)
        sr, sp, sy = np.sin(rpy)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def predict_odometry(
        self,
        measured_vel_body: np.ndarray,
        measured_gyro: np.ndarray,
        dt_s: float
    ) -> None:
        """EKF Prediction Step: propagate state using noisy dead-reckoning."""
        rpy = self.est_pose[3:6]
        R = self.rotation_matrix(rpy)

        # Velocity in world frame
        v_world = R @ measured_vel_body

        # Update position and orientation
        self.est_pose[0:3] += v_world * dt_s
        self.est_pose[3:6] += measured_gyro * dt_s
        # Wrap yaw
        self.est_pose[5] = (self.est_pose[5] + np.pi) % (2.0 * np.pi) - np.pi

        # Propagate covariance P = F * P * F^T + Q * dt
        # Simplified Jacobian F approx I
        self.P += self.Q * dt_s

    def update_scan_matching(
        self,
        points_body: np.ndarray,
        has_defect: bool = False
    ) -> Dict[str, Any]:
        """Scan-to-Map Matching: Align current sonar scan with accumulated 3D map using ICP."""
        if len(points_body) == 0:
            return {"status": "no_points", "residual_m": 0.0}

        map_pts = self.map.get_point_cloud()

        # If map is sparse (< 30 points), initialize directly
        if len(map_pts) < 30:
            R_est = self.rotation_matrix(self.est_pose[3:6])
            pts_world = (R_est @ points_body.T).T + self.est_pose[0:3]
            self.map.insert_points(pts_world, has_defect)
            return {"status": "initializing_map", "residual_m": 0.0}

        # Iterative Closest Point (ICP) scan matching
        curr_p = self.est_pose[0:3].copy()
        curr_rpy = self.est_pose[3:6].copy()

        max_iter = 6
        residual = 0.0

        for it in range(max_iter):
            R = self.rotation_matrix(curr_rpy)
            scan_w = (R @ points_body.T).T + curr_p

            # Find nearest neighbor for each scan point in accumulated map
            dists = np.linalg.norm(scan_w[:, None, :] - map_pts[None, :, :], axis=2)
            closest_idx = np.argmin(dists, axis=1)
            min_dists = np.min(dists, axis=1)

            # Reject outlier correspondences (> 1.2m)
            valid = min_dists < 1.2
            if np.sum(valid) < 6:
                break

            src = scan_w[valid]
            dst = map_pts[closest_idx[valid]]
            residual = float(np.mean(min_dists[valid]))

            # Compute translation error correction
            delta_t = np.mean(dst - src, axis=0)
            curr_p += delta_t * 0.45 # Conservative gain for stability

        self.last_icp_residual_m = round(residual, 3)

        # EKF Measurement Update: Blend ICP pose correction with covariance
        icp_measurement = np.concatenate([curr_p, curr_rpy])
        y = icp_measurement - self.est_pose # Innovation
        y[3:6] = (y[3:6] + np.pi) % (2.0 * np.pi) - np.pi

        S = self.P + self.R_icp
        K = self.P @ np.linalg.inv(S)

        self.est_pose += K @ y
        self.est_pose[5] = (self.est_pose[5] + np.pi) % (2.0 * np.pi) - np.pi
        self.P = (np.eye(6) - K) @ self.P

        # Insert corrected points into accumulated map
        R_corrected = self.rotation_matrix(self.est_pose[3:6])
        corrected_pts_w = (R_corrected @ points_body.T).T + self.est_pose[0:3]
        self.map.insert_points(corrected_pts_w, has_defect)

        return {
            "status": "matched",
            "residual_m": round(residual, 3),
            "valid_correspondences": int(np.sum(valid)),
            "map_total_points": len(self.map.voxels)
        }

    def step(
        self,
        true_pose: Dict[str, Any],
        measured_vel_body: np.ndarray,
        measured_gyro: np.ndarray,
        points_body: np.ndarray,
        has_defect: bool = False,
        dt_s: float = 0.05
    ) -> Dict[str, Any]:
        """Perform a complete SLAM estimation cycle."""
        self.step_count += 1

        # 1. Prediction with dead-reckoning
        self.predict_odometry(measured_vel_body, measured_gyro, dt_s)

        # 2. Measurement update with 3D scan matching
        match_info = self.update_scan_matching(points_body, has_defect)

        # 3. Calculate Localization Error vs Simulation Ground Truth
        true_pos = np.array([true_pose["x"], true_pose["y"], true_pose["z"]])
        est_pos = self.est_pose[0:3]
        self.drift_error_m = float(np.linalg.norm(true_pos - est_pos))

        # Record trajectory
        pos_cov_trace = float(np.trace(self.P[0:3, 0:3]))
        std_3sigma = 3.0 * np.sqrt(max(1e-6, pos_cov_trace / 3.0))

        step_record_est = {
            "step": self.step_count,
            "x": round(float(est_pos[0]), 3),
            "y": round(float(est_pos[1]), 3),
            "z": round(float(est_pos[2]), 3),
            "yaw_deg": round(float(np.degrees(self.est_pose[5])), 2),
            "uncertainty_3sigma_m": round(float(std_3sigma), 3)
        }
        step_record_true = {
            "step": self.step_count,
            "x": round(float(true_pos[0]), 3),
            "y": round(float(true_pos[1]), 3),
            "z": round(float(true_pos[2]), 3),
            "yaw_deg": round(float(true_pose.get("yaw_deg", 0.0)), 2)
        }

        self.traj_est.append(step_record_est)
        self.traj_true.append(step_record_true)
        self.ate_history.append(self.drift_error_m)

        # Limit trajectory history to 600 steps to prevent memory growth
        if len(self.traj_est) > 600:
            self.traj_est.pop(0)
            self.traj_true.pop(0)
            self.ate_history.pop(0)

        return {
            "estimated_pose": step_record_est,
            "ground_truth_pose": step_record_true,
            "localization_error_m": round(self.drift_error_m, 3),
            "uncertainty_3sigma_m": round(float(std_3sigma), 3),
            "icp_residual_m": self.last_icp_residual_m,
            "map_points_count": len(self.map.voxels),
            "map_defects_count": len(self.map.get_defect_landmarks()),
            "scan_match_status": match_info.get("status", "ok")
        }
