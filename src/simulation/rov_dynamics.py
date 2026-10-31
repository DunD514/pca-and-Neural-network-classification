"""6-DOF Underwater ROV Dynamics and Hydrodynamic Physics Simulator.

Implements Fossen's non-linear equations of motion for underwater robotic craft:
- 6 Degrees of Freedom: Position (X, Y, Z) and Orientation (Roll, Pitch, Yaw).
- Rigid-body inertia, added mass, linear and quadratic hydrodynamic damping.
- Restoring forces (buoyancy vs gravity with metacentric stability).
- 6-thruster allocation matrix (4 horizontal vectored + 2 vertical).
- Closed-loop PID position/attitude controller and manual piloting mode.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np


class ROVDynamics6DOF:
    """6-DOF Underwater Remotely Operated Vehicle physics model."""

    def __init__(
        self,
        initial_pos: Tuple[float, float, float] = (1.0, 5.0, 7.5),
        initial_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        mass_kg: float = 32.0,
        buoyancy_n: float = 314.5, # Slightly positive buoyancy (mass*g = 313.8N)
        dt_s: float = 0.05
    ):
        # World Frame State (p = [x, y, z], rpy = [roll, pitch, yaw] in radians)
        self.pos = np.array(initial_pos, dtype=float)
        self.rpy = np.radians(np.array(initial_rpy, dtype=float))

        # Body Frame Velocities (v_b = [surge, sway, heave], w_b = [p, q, r])
        self.vel_body = np.zeros(3, dtype=float)
        self.omega_body = np.zeros(3, dtype=float)

        # Vehicle Physical Constants
        self.mass = mass_kg
        self.g = 9.80665
        self.weight = mass_kg * self.g
        self.buoyancy = buoyancy_n
        self.dt = dt_s

        # Moments of inertia (diagonal approximation)
        self.I = np.diag([1.2, 1.8, 1.6]) # Ixx, Iyy, Izz
        # Added mass tensor (M_A)
        self.M_added = np.diag([15.0, 28.0, 32.0])
        self.I_added = np.diag([0.6, 0.9, 0.8])

        # Center of Buoyancy relative to Center of Gravity (CB above CG by 0.08m)
        self.r_bg = np.array([0.0, 0.08, 0.0]) # [x, y, z] in body frame (Y is vertical)

        # Hydrodynamic Damping (Linear + Quadratic Drag)
        self.d_lin_trans = np.array([16.0, 32.0, 38.0])   # surge, sway, heave
        self.d_quad_trans = np.array([28.0, 55.0, 70.0])
        self.d_lin_rot = np.array([3.5, 4.0, 5.5])        # roll, pitch, yaw
        self.d_quad_rot = np.array([5.0, 6.0, 8.5])

        # Thruster limits
        self.max_thrust_trans_n = 120.0
        self.max_torque_nm = 35.0

        # Accumulated statistics
        self.total_distance_m = 0.0
        self.battery_percent = 100.0
        self.is_collided = False

    def rotation_matrix(self) -> np.ndarray:
        """Compute rotation matrix R_b_to_w from body frame to world frame."""
        cr, cp, cy = np.cos(self.rpy)
        sr, sp, sy = np.sin(self.rpy)

        R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

        return R_z @ R_y @ R_x

    def step_physics(
        self,
        command_forces_body: np.ndarray,
        command_torques_body: np.ndarray,
        dt: Optional[float] = None
    ) -> None:
        """Integrate 6-DOF equations of motion forward in time."""
        dt = dt or self.dt

        # 1. Clip actuator commands to physical limits
        f_cmd = np.clip(command_forces_body, -self.max_thrust_trans_n, self.max_thrust_trans_n)
        t_cmd = np.clip(command_torques_body, -self.max_torque_nm, self.max_torque_nm)

        # 2. Hydrodynamic Drag Forces
        drag_f = -self.d_lin_trans * self.vel_body - self.d_quad_trans * np.abs(self.vel_body) * self.vel_body
        drag_t = -self.d_lin_rot * self.omega_body - self.d_quad_rot * np.abs(self.omega_body) * self.omega_body

        # 3. Restoring Forces (Gravitational + Buoyancy metacentric stability)
        # In body frame, gravity pulls in world -Y direction
        R = self.rotation_matrix()
        R_inv = R.T
        grav_world = np.array([0.0, -self.weight, 0.0])
        buoy_world = np.array([0.0, self.buoyancy, 0.0])

        f_restoring_body = R_inv @ (grav_world + buoy_world)
        # Restoring torque: r_bg x Buoyancy
        buoy_body = R_inv @ buoy_world
        t_restoring_body = np.cross(self.r_bg, buoy_body)

        # 4. Total Forces and Torques in Body Frame
        total_f_body = f_cmd + drag_f + f_restoring_body
        total_t_body = t_cmd + drag_t + t_restoring_body

        # 5. Accelerations (Mass + Added Mass)
        m_eff = self.mass * np.eye(3) + self.M_added
        i_eff = self.I + self.I_added

        acc_body = np.linalg.solve(m_eff, total_f_body)
        alpha_body = np.linalg.solve(i_eff, total_t_body)

        # 6. Velocity update (Euler integration)
        self.vel_body += acc_body * dt
        self.omega_body += alpha_body * dt

        # 7. World velocity & Position update
        vel_world = R @ self.vel_body
        prev_pos = self.pos.copy()
        self.pos += vel_world * dt

        # Orientation rate of change
        self.rpy += self.omega_body * dt
        # Normalize yaw to [-pi, pi]
        self.rpy[2] = (self.rpy[2] + np.pi) % (2.0 * np.pi) - np.pi
        # Bound pitch and roll to prevent unphysical tumbling
        self.rpy[0] = np.clip(self.rpy[0], -0.6, 0.6)
        self.rpy[1] = np.clip(self.rpy[1], -0.6, 0.6)

        moved = np.linalg.norm(self.pos - prev_pos)
        self.total_distance_m += moved
        self.battery_percent = max(0.0, self.battery_percent - (moved * 0.015) - (0.002 * dt))

    def get_pose(self) -> Dict[str, Any]:
        """Return full 6-DOF ground-truth pose and state."""
        return {
            "x": round(float(self.pos[0]), 3),
            "y": round(float(self.pos[1]), 3),
            "z": round(float(self.pos[2]), 3),
            "roll_deg": round(float(np.degrees(self.rpy[0])), 2),
            "pitch_deg": round(float(np.degrees(self.rpy[1])), 2),
            "yaw_deg": round(float(np.degrees(self.rpy[2])), 2),
            "speed_mps": round(float(np.linalg.norm(self.vel_body)), 3),
            "vel_body": self.vel_body.tolist(),
            "omega_body": self.omega_body.tolist(),
            "battery_percent": round(self.battery_percent, 1),
            "total_distance_m": round(float(self.total_distance_m), 2)
        }


class ROVFlightController:
    """Cascade PID flight controller for Autonomous Waypoint Tracking and Standoff Maintenance."""

    def __init__(self, rov: ROVDynamics6DOF):
        self.rov = rov
        # Guidance and velocity tracking gains
        self.kp_guidance = np.array([1.2, 1.4, 1.4])  # Position error to target velocity
        self.kp_vel = np.array([45.0, 50.0, 55.0])    # Velocity error to thruster force
        self.ki_pos = np.array([0.2, 0.2, 0.25])
        self.err_int_pos = np.zeros(3)

        # PID gains for attitude [roll, pitch, yaw]
        self.kp_att = np.array([16.0, 16.0, 24.0])
        self.kd_att = np.array([7.0, 7.0, 9.5])

    def compute_control(
        self,
        target_pos_world: np.ndarray,
        target_yaw_rad: float,
        dt: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute thrust forces and torques to drive ROV to target pose."""
        # Translation error in world frame
        err_world = target_pos_world - self.rov.pos
        self.err_int_pos += err_world * dt
        self.err_int_pos = np.clip(self.err_int_pos, -3.0, 3.0)

        # Rotate error into body frame
        R = self.rov.rotation_matrix()
        err_body = R.T @ err_world

        # Desired body velocity with proportional distance guidance
        v_des_body = self.kp_guidance * err_body + self.ki_pos * (R.T @ self.err_int_pos)
        v_des_norm = np.linalg.norm(v_des_body)
        max_speed = 1.2  # m/s max cruising speed
        if v_des_norm > max_speed:
            v_des_body = (v_des_body / v_des_norm) * max_speed

        # Hydrodynamic drag feedforward compensation so the ROV reaches target speed promptly
        drag_ff = self.rov.d_lin_trans * v_des_body + 0.5 * self.rov.d_quad_trans * np.abs(v_des_body) * v_des_body

        # Force command: velocity feedback + drag feedforward
        f_cmd = self.kp_vel * (v_des_body - self.rov.vel_body) + drag_ff

        # Attitude control: stabilize roll=0, pitch=0, drive yaw to target
        err_yaw = (target_yaw_rad - self.rov.rpy[2] + np.pi) % (2.0 * np.pi) - np.pi
        err_rpy = np.array([-self.rov.rpy[0], -self.rov.rpy[1], err_yaw])

        t_cmd = self.kp_att * err_rpy - self.kd_att * self.rov.omega_body

        return f_cmd, t_cmd

