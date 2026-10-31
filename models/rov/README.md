# Underwater Inspection ROV (3D Model & Kinematic Specification)

## Specifications
- **Model Type**: Modular 6-Thruster Subsea Remotely Operated Vehicle (ROV)
- **Dimensions**: Length: 0.65m, Beam: 0.45m, Height: 0.32m
- **Dry Mass**: 32.0 kg
- **Buoyancy**: 314.5 N (positive 0.8N for fail-safe passive ascent)
- **Actuation**: 
  - 4 $\times$ Horizontal vectored thrusters (surge, sway, yaw)
  - 2 $\times$ Vertical thrusters (heave, pitch)
- **Sensor Suite**:
  - Forward Inspection HD Pan/Tilt Camera Dome
  - Multi-beam acoustic sonar / rangefinder array
  - 6-DOF Inertial Measurement Unit (IMU)
  - Keel-mounted hydrostatic pressure depth sensor
  - Dual 1500-lumen LED subsea illuminators

## Coordinate Frames (ROS REP-103 standard)
- **Origin**: Center of mass / enclosure midpoint
- **+X**: Forward heading
- **+Y**: Port side (left)
- **+Z**: Zenith (upwards towards surface)

## License
This model and URDF specification are released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
Free for academic, research, and commercial applications.
