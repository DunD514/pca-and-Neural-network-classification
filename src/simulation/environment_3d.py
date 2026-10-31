"""3D Underwater Virtual Environment: Realistic Ship Hull, Seabed, Obstacles, and Ray Casting."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class ShipHull3DEnvironment:
    """Rigorous 3D underwater simulation world containing a 50m commercial vessel hull with full 3D solid collision geometry."""

    def __init__(
        self,
        length_m: float = 50.0,
        height_m: float = 10.0,
        beam_m: float = 12.0,
        seabed_y_m: float = -4.0,  # 14m below waterline
        waterline_y_m: float = 10.0
    ):
        self.length_m = length_m
        self.height_m = height_m
        self.beam_m = beam_m
        self.half_beam = beam_m / 2.0
        self.seabed_y_m = seabed_y_m
        self.waterline_y_m = waterline_y_m

        # Inspection structural zones
        self.zones = {
            "BOW_PORT": {"x_min": 0.0, "x_max": 12.0, "y_min": 0.0, "y_max": 10.0, "side": "port"},
            "BOW_STBD": {"x_min": 0.0, "x_max": 12.0, "y_min": 0.0, "y_max": 10.0, "side": "starboard"},
            "MIDSHIP_FWD": {"x_min": 12.0, "x_max": 25.0, "y_min": 0.0, "y_max": 10.0, "side": "starboard"},
            "MIDSHIP_AFT": {"x_min": 25.0, "x_max": 38.0, "y_min": 0.0, "y_max": 10.0, "side": "starboard"},
            "STERN_STBD": {"x_min": 38.0, "x_max": 50.0, "y_min": 0.0, "y_max": 10.0, "side": "starboard"},
            "KEEL_ZONE": {"x_min": 0.0, "x_max": 50.0, "y_min": 0.0, "y_max": 2.5, "side": "bottom"},
            "WATERLINE_ZONE": {"x_min": 0.0, "x_max": 50.0, "y_min": 7.5, "y_max": 10.0, "side": "starboard"}
        }

        # Registered Ground Truth Defects
        self.defects = [
            {"defect_id": "DEF-001", "x_m": 8.5, "y_m": 3.2, "radius_m": 1.0, "defect_type": "severe_corrosion", "zone": "BOW_STBD"},
            {"defect_id": "DEF-002", "x_m": 18.0, "y_m": 7.5, "radius_m": 1.2, "defect_type": "oxidation_scale", "zone": "MIDSHIP_FWD"},
            {"defect_id": "DEF-003", "x_m": 28.5, "y_m": 4.0, "radius_m": 0.9, "defect_type": "weld_seam_corrosion", "zone": "MIDSHIP_AFT"},
            {"defect_id": "DEF-004", "x_m": 38.0, "y_m": 6.8, "radius_m": 1.1, "defect_type": "pitting_corrosion", "zone": "STERN_STBD"},
            {"defect_id": "DEF-005", "x_m": 44.5, "y_m": 2.5, "radius_m": 0.8, "defect_type": "coating_failure", "zone": "STERN_STBD"}
        ]

        # Structural Plate Seams (longitudinal and vertical welds)
        self.weld_lines_x = [6.0, 12.0, 18.5, 25.0, 31.5, 38.0, 44.0]
        self.weld_lines_y = [2.5, 5.0, 7.5]

        # Generate full closed 3D collision mesh for physics & ray-tracing
        self._generate_mesh()

    def get_surface_z(self, x: float, y: float, side: str = "starboard") -> float:
        """Compute the lateral Z coordinate of the hull surface at (x, y).

        Coordinates:
        X: 0 = Bow, 50 = Stern
        Y: 0 = Keel bottom, 10 = Waterline
        Z: >0 Starboard shell plating, <0 Port shell plating
        """
        x_clamped = np.clip(x, 0.0, self.length_m)
        y_clamped = np.clip(y, 0.0, self.height_m)

        # Longitudinal profile
        if x_clamped < 12.0:
            # Bow curvature
            beam_factor = np.sin((x_clamped / 12.0) * (np.pi / 2.0))
        elif x_clamped > 38.0:
            # Stern tapering
            stern_ratio = (x_clamped - 38.0) / (self.length_m - 38.0)
            beam_factor = 1.0 - (0.45 * stern_ratio**1.5)
        else:
            beam_factor = 1.0

        # Vertical flare from keel (narrow) to waterline (full beam)
        y_ratio = y_clamped / self.height_m
        flare = 0.55 + 0.45 * (y_ratio ** 0.6)

        half_extent = self.half_beam * beam_factor * flare
        return half_extent if side == "starboard" else -half_extent

    def validate_outward_normal(self, surface_point: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Strictly guarantee normal vector points OUTWARD, away from the vessel interior centerline."""
        c_x = float(np.clip(surface_point[0], 0.5, self.length_m - 0.5))
        c_y = float(np.clip(surface_point[1], 0.5, self.height_m - 0.5))
        c_int = np.array([c_x, c_y, 0.0])  # Interior centerline anchor

        v_out = surface_point - c_int
        norm_v = np.linalg.norm(v_out)
        if norm_v > 1e-4:
            if np.dot(normal, v_out) < 0:
                normal = -normal

        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            return normal / norm
        return np.array([0.0, 0.0, 1.0 if surface_point[2] >= 0 else -1.0])

    def get_surface_normal(self, x: float, y: float, side: str = "starboard") -> np.ndarray:
        """Compute the unit normal vector pointing OUTWARD from the hull surface."""
        if side == "bottom":
            return np.array([0.0, -1.0, 0.0])

        eps = 0.05
        z_c = self.get_surface_z(x, y, side)
        z_dx = self.get_surface_z(x + eps, y, side)
        z_dy = self.get_surface_z(x, y + eps, side)

        # Tangent vectors
        t_x = np.array([eps, 0.0, z_dx - z_c])
        t_y = np.array([0.0, eps, z_dy - z_c])

        # Normal = t_x x t_y
        normal = np.cross(t_x, t_y)
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
        else:
            normal = np.array([0.0, 0.0, 1.0 if side == "starboard" else -1.0])

        if side == "port" and normal[2] > 0:
            normal = -normal
        elif side == "starboard" and normal[2] < 0:
            normal = -normal

        surface_pt = np.array([x, y, z_c])
        return self.validate_outward_normal(surface_pt, normal)

    def is_inside_hull(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """Determine whether 3D coordinate is strictly inside the solid hull interior (with optional margin)."""
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        if not (-margin <= x <= self.length_m + margin):
            return False
        if not (-margin <= y <= self.height_m + margin):
            return False
        hb = abs(self.get_surface_z(x, y, "starboard"))
        return abs(z) <= (hb + margin)

    def get_closest_hull_point_and_normal(
        self,
        point: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """Identify the closest valid point on the exterior hull surface and its validated outward normal.

        Returns:
            Tuple of (closest_surface_point, outward_normal, distance, is_inside)
        """
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        x_c = float(np.clip(x, 0.0, self.length_m))
        y_c = float(np.clip(y, 0.0, self.height_m))
        inside = self.is_inside_hull(point)

        candidates = []

        # 1. Starboard shell plating
        z_stbd = self.get_surface_z(x_c, y_c, "starboard")
        n_stbd = self.get_surface_normal(x_c, y_c, "starboard")
        candidates.append((np.array([x_c, y_c, z_stbd]), n_stbd, "starboard_shell"))

        # 2. Port shell plating
        z_port = self.get_surface_z(x_c, y_c, "port")
        n_port = self.get_surface_normal(x_c, y_c, "port")
        candidates.append((np.array([x_c, y_c, z_port]), n_port, "port_shell"))

        # 3. Keel bottom plate (y = 0.0)
        hb_keel = abs(self.get_surface_z(x_c, 0.0, "starboard"))
        z_keel = float(np.clip(z, -hb_keel, hb_keel))
        n_keel = np.array([0.0, -1.0, 0.0])
        candidates.append((np.array([x_c, 0.0, z_keel]), n_keel, "keel_bottom"))

        # 4. Bow stem / entry (x = 0.0)
        hb_bow = abs(self.get_surface_z(0.0, y_c, "starboard"))
        z_bow = float(np.clip(z, -hb_bow, hb_bow))
        n_bow = np.array([-1.0, 0.0, 0.0])
        candidates.append((np.array([0.0, y_c, z_bow]), n_bow, "bow_stem"))

        # 5. Stern transom counter (x = length_m)
        hb_stern = abs(self.get_surface_z(self.length_m, y_c, "starboard"))
        z_stern = float(np.clip(z, -hb_stern, hb_stern))
        n_stern = np.array([1.0, 0.0, 0.0])
        candidates.append((np.array([self.length_m, y_c, z_stern]), n_stern, "stern_transom"))

        # 6. Waterline / upper deck edge (y = height_m)
        hb_deck = abs(self.get_surface_z(x_c, self.height_m, "starboard"))
        z_deck = float(np.clip(z, -hb_deck, hb_deck))
        n_deck = np.array([0.0, 1.0, 0.0])
        candidates.append((np.array([x_c, self.height_m, z_deck]), n_deck, "waterline_deck"))

        best_pt = candidates[0][0]
        best_n = candidates[0][1]
        best_dist = float("inf")

        for pt, norm, name in candidates:
            d = float(np.linalg.norm(point - pt))
            if d < best_dist:
                best_dist = d
                best_pt = pt
                best_n = norm

        # Guarantee outward normal direction
        best_n = self.validate_outward_normal(best_pt, best_n)

        return best_pt, best_n, best_dist, inside

    def check_segment_collision(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        radius: float = 0.40
    ) -> Tuple[bool, Optional[np.ndarray], float]:
        """Continuous collision check (capsule / swept-sphere) along movement segment p1 -> p2.

        Returns:
            Tuple of (has_collision, hit_point, min_clearance_found)
        """
        length = float(np.linalg.norm(p2 - p1))
        steps = max(4, int(np.ceil(length / 0.10)))
        ts = np.linspace(0.0, 1.0, steps)

        min_clearance = float("inf")
        hit_point = None

        for t in ts:
            pt = p1 + t * (p2 - p1)
            s_pt, norm, dist, inside = self.get_closest_hull_point_and_normal(pt)
            clearance = 0.0 if inside else dist - radius
            if clearance < min_clearance:
                min_clearance = clearance

            if inside or dist < radius:
                hit_point = pt
                return True, hit_point, 0.0

        # Also ray-cast against mesh facets for thin/sharp feature intersection
        if length > 1e-4:
            ray_dir = (p2 - p1) / length
            mesh_hit = self.ray_cast_hull(p1, ray_dir, max_dist=length + radius)
            if mesh_hit is not None and mesh_hit["distance"] <= (length + radius):
                return True, mesh_hit["point"], 0.0

        return False, None, max(0.0, min_clearance)

    def _generate_mesh(self, nx: int = 40, ny: int = 20) -> None:
        """Pre-compute full closed 3D vertices and triangular facets for ray-casting and collision."""
        xs = np.linspace(0, self.length_m, nx)
        ys = np.linspace(0, self.height_m, ny)
        self.vertices = []
        self.triangles = []

        # 1. Starboard shell plating
        stbd_grid = np.zeros((ny, nx), dtype=int)
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                z = self.get_surface_z(x, y, "starboard")
                idx = len(self.vertices)
                self.vertices.append([x, y, z])
                stbd_grid[iy, ix] = idx

        for iy in range(ny - 1):
            for ix in range(nx - 1):
                p0 = stbd_grid[iy, ix]
                p1 = stbd_grid[iy, ix + 1]
                p2 = stbd_grid[iy + 1, ix]
                p3 = stbd_grid[iy + 1, ix + 1]
                self.triangles.append([p0, p1, p2])
                self.triangles.append([p1, p3, p2])

        # 2. Port shell plating
        port_grid = np.zeros((ny, nx), dtype=int)
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                z = self.get_surface_z(x, y, "port")
                idx = len(self.vertices)
                self.vertices.append([x, y, z])
                port_grid[iy, ix] = idx

        for iy in range(ny - 1):
            for ix in range(nx - 1):
                p0 = port_grid[iy, ix]
                p1 = port_grid[iy, ix + 1]
                p2 = port_grid[iy + 1, ix]
                p3 = port_grid[iy + 1, ix + 1]
                self.triangles.append([p0, p2, p1])
                self.triangles.append([p1, p2, p3])

        # 3. Keel bottom plate (iy = 0)
        for ix in range(nx - 1):
            s0 = stbd_grid[0, ix]
            s1 = stbd_grid[0, ix + 1]
            p0 = port_grid[0, ix]
            p1 = port_grid[0, ix + 1]
            self.triangles.append([s0, p0, s1])
            self.triangles.append([s1, p0, p1])

        # 4. Stern transom plate (ix = nx - 1)
        for iy in range(ny - 1):
            s0 = stbd_grid[iy, nx - 1]
            s1 = stbd_grid[iy + 1, nx - 1]
            p0 = port_grid[iy, nx - 1]
            p1 = port_grid[iy + 1, nx - 1]
            self.triangles.append([s0, s1, p0])
            self.triangles.append([s1, p1, p0])

        self.vertices = np.array(self.vertices, dtype=float)
        self.triangles = np.array(self.triangles, dtype=int)

    def ray_cast_hull(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        max_dist: float = 12.0
    ) -> Optional[Dict[str, Any]]:
        """Möller–Trumbore fast ray-triangle intersection against the complete closed hull collision mesh."""
        ray_dir = ray_direction / np.linalg.norm(ray_direction)
        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]

        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(ray_dir, edge2)
        a = np.einsum('ij,ij->i', edge1, h)

        # Parallel rays
        valid = np.abs(a) > 1e-7
        f = np.zeros_like(a)
        f[valid] = 1.0 / a[valid]

        s = ray_origin - v0
        u = f * np.einsum('ij,ij->i', s, h)
        valid = valid & (u >= 0.0) & (u <= 1.0)

        q = np.cross(s, edge1)
        v = f * np.einsum('j,ij->i', ray_dir, q)
        valid = valid & (v >= 0.0) & ((u + v) <= 1.0)

        t = f * np.einsum('ij,ij->i', edge2, q)
        valid = valid & (t > 0.05) & (t <= max_dist)

        indices = np.where(valid)[0]
        if len(indices) == 0:
            return None

        best_idx = indices[np.argmin(t[indices])]
        hit_dist = float(t[best_idx])
        hit_point = ray_origin + ray_dir * hit_dist

        # Compute validated outward normal at hit point
        s_pt, normal, _, _ = self.get_closest_hull_point_and_normal(hit_point)

        # Check if hit point is inside any registered defect
        defect_hit = None
        for d in self.defects:
            dist_to_center = np.sqrt((hit_point[0] - d["x_m"])**2 + (hit_point[1] - d["y_m"])**2)
            if dist_to_center <= d["radius_m"]:
                defect_hit = d
                break

        return {
            "distance": round(hit_dist, 3),
            "point": hit_point,
            "normal": normal,
            "defect": defect_hit,
            "has_defect": defect_hit is not None
        }

    def get_zone_at(self, x: float, y: float) -> str:
        """Query inspection zone identifier."""
        if y < 2.5:
            return "KEEL_ZONE"
        if y >= 7.5:
            return "WATERLINE_ZONE"
        if x < 12.0:
            return "BOW_STBD"
        elif x <= 25.0:
            return "MIDSHIP_FWD"
        elif x <= 38.0:
            return "MIDSHIP_AFT"
        else:
            return "STERN_STBD"
