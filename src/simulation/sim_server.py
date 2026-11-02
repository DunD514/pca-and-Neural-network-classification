"""High-performance thread-safe HTTP bridge streaming simulation state to the persistent 3D WebGL viewport."""

import http.server
import json
import socketserver
import threading
import time
from typing import Dict, Any, Optional
import numpy as np


class ThreadSafeSimulationState:
    """Thread-safe snapshot buffer for simulation state shared with the WebGL client."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {
            "status": "ready",
            "sim_time_s": 0.0,
            "wall_time_s": 0.0,
            "speed_multiplier": 1.0,
            "is_paused": True,
            "mode": "AUTONOMOUS",
            "rov": {
                "x": 1.5, "y": 5.0, "z": 8.0,
                "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0,
                "heading_deg": 0.0, "speed_mps": 0.0,
                "quaternion": [0.0, 0.0, 0.0, 1.0]
            },
            "active_wp": {"x": 1.5, "y": 5.0, "z": 8.0, "yaw_rad": 0.0},
            "current_wp_idx": 0,
            "total_wps": 1,
            "waypoints": [],
            "traj_true": [],
            "traj_est": [],
            "voxels": [],
            "defects": [],
            "detected_defects": [],
            "standoff_m": 2.0,
            "depth_m": 5.0,
            "ate_error_m": 0.0,
            "collision_status": "SAFE",
            "clearance_m": 2.0,
            "min_safe_dist_m": 1.0,
            "drone_radius_m": 0.40,
            "nearest_normal": [0.0, 0.0, 1.0],
            "collision_warning": False,
            "debug_mode": False,
            "timestamp": time.time()
        }

    def update(self, new_state: Dict[str, Any]) -> None:
        with self._lock:
            self._state.update(new_state)
            self._state["timestamp"] = time.time()

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)


_GLOBAL_STATE = ThreadSafeSimulationState()


def get_simulation_state_manager() -> ThreadSafeSimulationState:
    return _GLOBAL_STATE


def rpy_to_quaternion(roll_rad: float, pitch_rad: float, yaw_rad: float) -> list:
    """Compute normalized [x, y, z, w] quaternion from Euler angles in Three.js Euler order (XYZ)."""
    cr = np.cos(roll_rad * 0.5)
    sr = np.sin(roll_rad * 0.5)
    cp = np.cos(pitch_rad * 0.5)
    sp = np.sin(pitch_rad * 0.5)
    cy = np.cos(yaw_rad * 0.5)
    sy = np.sin(yaw_rad * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm > 1e-9:
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    return [round(float(qx), 5), round(float(qy), 5), round(float(qz), 5), round(float(qw), 5)]


class SimulationRequestHandler(http.server.BaseHTTPRequestHandler):
    """CORS-enabled REST handler serving real-time telemetry snapshots."""

    def log_message(self, format, *args):
        # Silence per-request terminal output to keep logs clean
        pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/api/state"):
            state = _GLOBAL_STATE.get_snapshot()
            payload = json.dumps(state).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.path.startswith("/api/health"):
            payload = b'{"status": "ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()


class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


_SERVER_INSTANCE: Optional[ReusableTCPServer] = None
_SERVER_PORT: int = 8502
_SERVER_LOCK = threading.Lock()


def ensure_simulation_server_running(preferred_port: int = 8502) -> int:
    """Ensure background HTTP server is running on preferred_port or next available port."""
    global _SERVER_INSTANCE, _SERVER_PORT
    with _SERVER_LOCK:
        if _SERVER_INSTANCE is not None:
            return _SERVER_PORT

        for port in [preferred_port, 8503, 8504, 8505, 8506]:
            try:
                server = ReusableTCPServer(("0.0.0.0", port), SimulationRequestHandler)
                _SERVER_INSTANCE = server
                _SERVER_PORT = port
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                return _SERVER_PORT
            except OSError:
                continue

        # If all fail, fallback to 8502
        return preferred_port


def get_server_port() -> int:
    return _SERVER_PORT
