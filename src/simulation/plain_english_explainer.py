"""Plain-English Mission Event Explainer for Non-Technical Users."""

from typing import Dict, Any, List


class PlainEnglishMissionExplainer:
    """Translates real-time robotics, SLAM, and AI telemetry into natural English commentary."""

    def __init__(self):
        self.last_zone = ""
        self.last_defect_count = 0
        self.last_slam_pts = 0
        self.last_error_warned = False

    def generate_commentary(self, step_data: Dict[str, Any]) -> List[str]:
        """Generate human-readable bullet points explaining current drone activity."""
        messages = []

        zone = step_data.get("current_zone", "HULL")
        mode = step_data.get("mode", "AUTONOMOUS")
        avionics = step_data.get("avionics", {})
        sonar = step_data.get("sonar", {})
        slam = step_data.get("slam", {})
        ml = step_data.get("ml_prediction")
        cam = step_data.get("camera", {})

        depth = avionics.get("submerged_depth_m", 0.0)
        speed = avionics.get("speed_mps", 0.0)
        standoff = sonar.get("mean_standoff_m", 1.5)
        map_pts = slam.get("map_points_count", 0)
        loc_err = slam.get("localization_error_m", 0.0)

        # 1. Navigation & Collision Avoidance status
        zone_clean = zone.replace("_", " ").title()
        safety = step_data.get("safety", {})
        if safety.get("collision_status") == "AVOIDANCE_ACTIVE":
            messages.append(f"🛡️ **Collision Avoidance Active:** Proximity to hull plate ({safety.get('clearance_m', 1.0):.2f}m) activated safety boundary. The ROV is actively maintaining standoff along the outward surface normal.")
        elif mode == "AUTONOMOUS":
            if speed > 0.08:
                messages.append(f"🤖 **Navigation:** The ROV is autonomously surveying the **{zone_clean}** at {speed:.1f} m/s, maintaining a steady {standoff:.1f}m standoff distance from the steel plate.")
            else:
                messages.append(f"⚓ **Navigation:** The ROV is holding position at **{zone_clean}**, depth {depth:.1f}m below the waterline.")
        else:
            messages.append(f"🕹️ **Manual Piloting:** Pilot controls active. The drone is maneuvering at depth {depth:.1f}m ({standoff:.1f}m clearance from the hull).")

        # 2. SLAM & Mapping activity
        if map_pts > self.last_slam_pts:
            pts_added = map_pts - self.last_slam_pts
            messages.append(f"🗺️ **3D SLAM Mapping:** The acoustic sonar scanned new surface area, integrating {pts_added} new 3D points into the growing structural map (total: {map_pts} surface voxels).")
            self.last_slam_pts = map_pts

        if loc_err > 0.15:
            messages.append(f"⚖️ **Pose Correction:** Estimated position drifted by {loc_err*100:.0f}cm from simulation ground truth. SLAM scan-matching is actively recalibrating the trajectory.")
        else:
            messages.append(f"🎯 **High-Precision Localization:** Estimated trajectory matches ground truth within {loc_err*100:.1f}cm.")

        # 3. AI Vision & Corrosion Detection
        if ml and ml.get("defect_detected"):
            conf = ml.get("confidence_score", 0.0) * 100
            area = ml.get("defect_area_fraction", 0.0) * 100
            messages.append(f"🚨 **Corrosion Alert:** The deep learning U-Net identified a suspected defect ({conf:.0f}% confidence, covering {area:.1f}% of camera view) on sector **{zone_clean}**.")
        elif cam.get("has_defect"):
            messages.append(f"🔍 **Visual Observation:** The camera is passing near registered defect zone **{cam.get('defect_id')}**.")
        else:
            messages.append("🛡️ **Hull Condition:** Steel plating in the current camera footprint appears sound and free of critical oxide scale.")

        return messages
