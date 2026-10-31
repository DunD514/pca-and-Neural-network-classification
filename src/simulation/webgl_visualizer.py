"""Persistent 3D WebGL / Three.js Underwater Simulation Viewport with 60 FPS Render Loop and Pose Interpolation.

Eliminates black flashes, state-by-state teleportation, and canvas remounts by:
1. Creating ONE persistent Three.js scene, renderer, camera, ship hull, and ROV model that stays alive continuously.
2. Running an independent 60 FPS requestAnimationFrame render loop with Vector3 LERP and Quaternion SLERP pose interpolation.
3. Updating dynamic trajectory lines and point-cloud voxels in place using BufferAttribute (.needsUpdate = true).
4. Smooth camera follow with damping that never abruptly snaps.
5. Communicating asynchronously with the local simulation bridge without causing React/Streamlit iframe reloads.
"""

from typing import Dict, List, Any, Optional
import json


def generate_persistent_threejs_html(
    server_port: int = 8502,
    height_px: int = 580
) -> str:
    """Generate the static, persistent Three.js HTML string that never remounts or reloads."""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>3D Autonomous ROV Persistent Simulation</title>
    <style>
        body, html {{ margin: 0; padding: 0; overflow: hidden; background: #030a16; font-family: 'Segoe UI', system-ui, sans-serif; }}
        #webgl-canvas {{ width: 100vw; height: {height_px}px; display: block; }}
        #hud-overlay {{
            position: absolute; top: 10px; left: 14px;
            background: rgba(4, 12, 26, 0.88); border: 1px solid rgba(0, 240, 255, 0.35);
            border-radius: 6px; padding: 6px 14px; font-size: 11px; font-family: 'JetBrains Mono', monospace;
            color: #00f0ff; letter-spacing: 0.5px; z-index: 10; pointer-events: none;
            box-shadow: 0 4px 14px rgba(0,0,0,0.6);
        }}
        #cam-controls {{
            position: absolute; top: 10px; right: 14px; z-index: 10;
            display: flex; gap: 6px;
        }}
        .cam-btn {{
            background: rgba(8, 20, 40, 0.88); border: 1px solid rgba(0, 240, 255, 0.35);
            color: #e0f2fe; border-radius: 6px; padding: 6px 10px; font-size: 11px;
            font-weight: 600; cursor: pointer; transition: all 0.2s;
        }}
        .cam-btn:hover {{ background: rgba(0, 240, 255, 0.25); color: #00f0ff; }}
        .cam-btn.active {{ background: #00f0ff; color: #040914; font-weight: 700; }}
        #legend {{
            position: absolute; bottom: 12px; left: 14px;
            background: rgba(4, 12, 26, 0.88); border: 1px solid rgba(0, 240, 255, 0.25);
            border-radius: 6px; padding: 6px 12px; font-size: 10px; color: #94a3b8;
            display: flex; gap: 12px; pointer-events: none; z-index: 10;
        }}
        .leg-item {{ display: flex; align-items: center; gap: 4px; }}
        .leg-dot {{ width: 8px; height: 8px; border-radius: 50%; }}
        #conn-status {{
            position: absolute; bottom: 12px; right: 14px;
            font-size: 10px; font-family: monospace; color: #10b981;
            background: rgba(4, 12, 26, 0.85); padding: 4px 10px; border-radius: 4px;
            border: 1px solid rgba(16, 185, 129, 0.3); pointer-events: none; z-index: 10;
        }}
    </style>
    <!-- Three.js + OrbitControls -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="hud-overlay">
        3D SIMULATION ENGINE &bull; INITIALIZING PERSISTENT SCENE...
    </div>

    <div id="cam-controls">
        <button class="cam-btn active" id="btn-follow" onclick="setCam('follow')">Follow ROV</button>
        <button class="cam-btn" id="btn-free" onclick="setCam('free')">Free Orbit</button>
        <button class="cam-btn" id="btn-side" onclick="setCam('side')">Side Elevation</button>
        <button class="cam-btn" id="btn-top" onclick="setCam('top')">Top View</button>
        <button class="cam-btn" id="btn-front" onclick="setCam('front')">Bow Front</button>
        <button class="cam-btn" id="btn-reset" onclick="setCam('reset')">Reset View</button>
        <button class="cam-btn" id="btn-debug" onclick="toggleNavDebug()">Nav Debug</button>
    </div>

    <div id="legend">
        <div class="leg-item"><div class="leg-dot" style="background:#facc15;"></div> ROV Drone</div>
        <div class="leg-item"><div class="leg-dot" style="background:#fbbf24;"></div> Waypoint Route</div>
        <div class="leg-item"><div class="leg-dot" style="background:#10b981;"></div> True Trajectory</div>
        <div class="leg-item"><div class="leg-dot" style="background:#00f0ff;"></div> SLAM Path</div>
        <div class="leg-item"><div class="leg-dot" style="background:#38bdf8;"></div> Sonar Voxels</div>
        <div class="leg-item"><div class="leg-dot" style="background:#ef4444;"></div> Hull Defect</div>
        <div class="leg-item"><div class="leg-dot" style="background:#00f0ff; border:1px dotted #fff;"></div> Collision Boundary</div>
    </div>

    <div id="conn-status">● 60 FPS SYNC ACTIVE</div>

    <canvas id="webgl-canvas"></canvas>

    <script>
        const SERVER_PORT = {server_port};
        const API_URL = 'http://127.0.0.1:' + SERVER_PORT + '/api/state';

        // ---------------- 1. Persistent Three.js Scene Setup ----------------
        const canvas = document.getElementById('webgl-canvas');
        const renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias: true, powerPreference: 'high-performance' }});
        renderer.setSize(window.innerWidth, {height_px});
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.shadowMap.enabled = true;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x031024);
        scene.fog = new THREE.FogExp2(0x031024, 0.015);

        const camera = new THREE.PerspectiveCamera(50, window.innerWidth / {height_px}, 0.2, 220);
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;

        // Ambient & Underwater Lighting
        const ambLight = new THREE.AmbientLight(0x0a2a4a, 1.3);
        scene.add(ambLight);

        const sunLight = new THREE.DirectionalLight(0x70d8ff, 1.4);
        sunLight.position.set(25, 40, 20);
        scene.add(sunLight);

        // Water surface plane (Y = 10m)
        const waterGeo = new THREE.PlaneGeometry(120, 80);
        const waterMat = new THREE.MeshBasicMaterial({{
            color: 0x0077be, transparent: true, opacity: 0.35, side: THREE.DoubleSide
        }});
        const waterMesh = new THREE.Mesh(waterGeo, waterMat);
        waterMesh.rotation.x = Math.PI / 2;
        waterMesh.position.set(25, 10.0, 0);
        scene.add(waterMesh);

        // Seabed sandy floor (Y = -4m)
        const seabedGeo = new THREE.PlaneGeometry(120, 80, 20, 20);
        const posAttr = seabedGeo.attributes.position;
        for (let i = 0; i < posAttr.count; i++) {{
            const vx = posAttr.getX(i);
            const vz = posAttr.getY(i);
            posAttr.setZ(i, Math.sin(vx * 0.15) * 0.4 + Math.cos(vz * 0.15) * 0.3);
        }}
        seabedGeo.computeVertexNormals();
        const seabedMat = new THREE.MeshStandardMaterial({{ color: 0x1a2e26, roughness: 0.9 }});
        const seabedMesh = new THREE.Mesh(seabedGeo, seabedMat);
        seabedMesh.rotation.x = -Math.PI / 2;
        seabedMesh.position.set(25, -4.0, 0);
        scene.add(seabedMesh);

        // ---------------- 2. Parametric 3D Ship Hull Geometry ----------------
        function getHullZ(x, y) {{
            const xClamped = Math.max(0, Math.min(50.0, x));
            const yClamped = Math.max(0, Math.min(10.0, y));
            let halfBeam = 6.0;

            if (xClamped < 12.0) {{
                const s = xClamped / 12.0;
                halfBeam = 6.0 * (0.2 + 0.8 * Math.sin(s * Math.PI / 2));
                const flare = Math.pow(yClamped / 10.0, 1.6) * (1.0 - s) * 1.5;
                halfBeam += flare;
            }} else if (xClamped > 38.0) {{
                const s = (50.0 - xClamped) / 12.0;
                halfBeam = 6.0 * (0.35 + 0.65 * s);
            }}

            if (yClamped < 2.5) {{
                const keelFactor = Math.pow(yClamped / 2.5, 0.7);
                halfBeam *= keelFactor;
            }}
            return halfBeam;
        }}

        const nx = 48, ny = 20;
        const hullGeo = new THREE.BufferGeometry();
        const vertices = [], uvs = [], indices = [];

        // 1. Starboard shell plating (Z > 0)
        const stbdOffset = 0;
        for (let j = 0; j <= ny; j++) {{
            const y = (j / ny) * 10.0;
            for (let i = 0; i <= nx; i++) {{
                const x = (i / nx) * 50.0;
                const z = getHullZ(x, y);
                vertices.push(x, y, z);
                uvs.push(i / nx, j / ny);
            }}
        }}
        for (let j = 0; j < ny; j++) {{
            for (let i = 0; i < nx; i++) {{
                const a = stbdOffset + j * (nx + 1) + i;
                const b = a + 1;
                const c = stbdOffset + (j + 1) * (nx + 1) + i;
                const d = c + 1;
                indices.push(a, b, c);
                indices.push(b, d, c);
            }}
        }}

        // 2. Port shell plating (Z < 0)
        const portOffset = vertices.length / 3;
        for (let j = 0; j <= ny; j++) {{
            const y = (j / ny) * 10.0;
            for (let i = 0; i <= nx; i++) {{
                const x = (i / nx) * 50.0;
                const z = -getHullZ(x, y);
                vertices.push(x, y, z);
                uvs.push(i / nx, j / ny);
            }}
        }}
        for (let j = 0; j < ny; j++) {{
            for (let i = 0; i < nx; i++) {{
                const a = portOffset + j * (nx + 1) + i;
                const b = a + 1;
                const c = portOffset + (j + 1) * (nx + 1) + i;
                const d = c + 1;
                indices.push(a, c, b);
                indices.push(b, c, d);
            }}
        }}

        // 3. Keel bottom plate (Y = 0)
        for (let i = 0; i < nx; i++) {{
            const s0 = stbdOffset + i;
            const s1 = stbdOffset + i + 1;
            const p0 = portOffset + i;
            const p1 = portOffset + i + 1;
            indices.push(s0, p0, s1);
            indices.push(s1, p0, p1);
        }}

        // 4. Stern transom plate (X = 50.0)
        for (let j = 0; j < ny; j++) {{
            const s0 = stbdOffset + j * (nx + 1) + nx;
            const s1 = stbdOffset + (j + 1) * (nx + 1) + nx;
            const p0 = portOffset + j * (nx + 1) + nx;
            const p1 = portOffset + (j + 1) * (nx + 1) + nx;
            indices.push(s0, s1, p0);
            indices.push(s1, p1, p0);
        }}

        hullGeo.setIndex(indices);
        hullGeo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        hullGeo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
        hullGeo.computeVertexNormals();

        const hullMat = new THREE.MeshStandardMaterial({{
            color: 0x1e3a5f, roughness: 0.55, metalness: 0.35, side: THREE.DoubleSide
        }});
        const hullMesh = new THREE.Mesh(hullGeo, hullMat);
        scene.add(hullMesh);

        // Hull Plate Weld Seams & Collision Wireframe
        const wireMat = new THREE.LineBasicMaterial({{ color: 0x00f0ff, transparent: true, opacity: 0.16 }});
        const wireMesh = new THREE.LineSegments(new THREE.WireframeGeometry(hullGeo), wireMat);
        scene.add(wireMesh);

        // Hull Outward Surface Normal Arrows Group (Debug Mode)
        const normalGroup = new THREE.Group();
        normalGroup.visible = false;
        scene.add(normalGroup);

        // Populate sample surface normal arrows
        for (let ix = 2; ix < nx; ix += 6) {{
            const x = (ix / nx) * 50.0;
            for (let iy = 2; iy < ny; iy += 5) {{
                const y = (iy / ny) * 10.0;
                // Starboard normal
                const z_s = getHullZ(x, y);
                const n_s = new THREE.Vector3(0, 0, 1).normalize();
                const arrow_s = new THREE.ArrowHelper(n_s, new THREE.Vector3(x, y, z_s), 1.0, 0x38bdf8);
                normalGroup.add(arrow_s);

                // Port normal
                const z_p = -getHullZ(x, y);
                const n_p = new THREE.Vector3(0, 0, -1).normalize();
                const arrow_p = new THREE.ArrowHelper(n_p, new THREE.Vector3(x, y, z_p), 1.0, 0x38bdf8);
                normalGroup.add(arrow_p);
            }}
        }}

        let isNavDebug = false;
        function toggleNavDebug() {{
            isNavDebug = !isNavDebug;
            normalGroup.visible = isNavDebug;
            wireMat.opacity = isNavDebug ? 0.45 : 0.16;
            const btn = document.getElementById('btn-debug');
            if (btn) {{
                if (isNavDebug) btn.classList.add('active');
                else btn.classList.remove('active');
            }}
        }}
        window.toggleNavDebug = toggleNavDebug;

        // Registered Hull Defects
        const defectGroup = new THREE.Group();
        scene.add(defectGroup);

        function updateDefectLandmarks(defects) {{
            while (defectGroup.children.length > 0) {{
                const obj = defectGroup.children.pop();
                if (obj.geometry) obj.geometry.dispose();
            }}
            defects.forEach(d => {{
                const z = getHullZ(d.x_m, d.y_m) + 0.05;
                const dGeo = new THREE.CircleGeometry(d.radius_m, 16);
                const dMat = new THREE.MeshBasicMaterial({{
                    color: 0xef4444, side: THREE.DoubleSide, transparent: true, opacity: 0.85
                }});
                const dMesh = new THREE.Mesh(dGeo, dMat);
                dMesh.position.set(d.x_m, d.y_m, z);
                defectGroup.add(dMesh);

                const haloGeo = new THREE.RingGeometry(d.radius_m * 0.9, d.radius_m * 1.3, 16);
                const haloMat = new THREE.MeshBasicMaterial({{ color: 0xff0055, transparent: true, opacity: 0.5, side: THREE.DoubleSide }});
                const haloMesh = new THREE.Mesh(haloGeo, haloMat);
                haloMesh.position.set(d.x_m, d.y_m, z + 0.02);
                defectGroup.add(haloMesh);
            }});
        }}

        // ---------------- 3. Persistent 3D ROV Model ----------------
        const rovGroup = new THREE.Group();

        // Main cylindrical electronics housing
        const podGeo = new THREE.CylinderGeometry(0.14, 0.14, 0.55, 16);
        const podMat = new THREE.MeshStandardMaterial({{ color: 0x0c3058, roughness: 0.3, metalness: 0.7 }});
        const podMesh = new THREE.Mesh(podGeo, podMat);
        podMesh.rotation.z = Math.PI / 2;
        rovGroup.add(podMesh);

        // Buoyancy foam block
        const foamGeo = new THREE.BoxGeometry(0.55, 0.12, 0.40);
        const foamMat = new THREE.MeshStandardMaterial({{ color: 0xfacc15, roughness: 0.5 }});
        const foamMesh = new THREE.Mesh(foamGeo, foamMat);
        foamMesh.position.set(0, 0.14, 0);
        rovGroup.add(foamMesh);

        // Forward inspection camera dome
        const domeGeo = new THREE.SphereGeometry(0.08, 16, 16);
        const domeMat = new THREE.MeshPhysicalMaterial({{
            color: 0x38bdf8, transmission: 0.85, opacity: 0.9, transparent: true, roughness: 0.1
        }});
        const domeMesh = new THREE.Mesh(domeGeo, domeMat);
        domeMesh.position.set(0.30, 0.02, 0);
        rovGroup.add(domeMesh);

        // Forward heading vector arrow
        const arrowHelper = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0.35, 0.02, 0), 0.8, 0x00f0ff, 0.22, 0.12);
        rovGroup.add(arrowHelper);

        // Safety standoff halo
        const haloGeo = new THREE.RingGeometry(0.48, 0.54, 24);
        const haloMat = new THREE.MeshBasicMaterial({{ color: 0x00f0ff, transparent: true, opacity: 0.4, side: THREE.DoubleSide }});
        const safetyHaloMesh = new THREE.Mesh(haloGeo, haloMat);
        safetyHaloMesh.rotation.x = Math.PI / 2;
        safetyHaloMesh.position.set(0, -0.05, 0);
        rovGroup.add(safetyHaloMesh);

        // Twin LED headlights & volumetric light cones
        const lightMat = new THREE.MeshBasicMaterial({{ color: 0x00f0ff }});
        [-0.14, 0.14].forEach(zOffset => {{
            const lMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.05, 8), lightMat);
            lMesh.rotation.z = Math.PI / 2;
            lMesh.position.set(0.28, 0.06, zOffset);
            rovGroup.add(lMesh);

            const beamGeo = new THREE.ConeGeometry(0.35, 1.6, 16, 1, true);
            const beamMat = new THREE.MeshBasicMaterial({{
                color: 0x00f0ff, transparent: true, opacity: 0.22, side: THREE.DoubleSide
            }});
            const beamMesh = new THREE.Mesh(beamGeo, beamMat);
            beamMesh.rotation.z = -Math.PI / 2;
            beamMesh.position.set(1.05, 0.06, zOffset);
            rovGroup.add(beamMesh);
        }});

        // 6 Thrusters with rotating propellers
        const thrusterMat = new THREE.MeshStandardMaterial({{ color: 0x1f2937, metalness: 0.8 }});
        const propMat = new THREE.MeshBasicMaterial({{ color: 0x00f0ff }});
        const propellers = [];

        const thrusterPositions = [
            [0.20, 0.0, 0.18, Math.PI/4],
            [0.20, 0.0, -0.18, -Math.PI/4],
            [-0.20, 0.0, 0.18, -Math.PI/4],
            [-0.20, 0.0, -0.18, Math.PI/4]
        ];
        thrusterPositions.forEach(p => {{
            const tMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.10, 8), thrusterMat);
            tMesh.rotation.y = p[3];
            tMesh.position.set(p[0], p[1], p[2]);
            rovGroup.add(tMesh);

            // Propeller blade
            const pr = new THREE.Mesh(new THREE.BoxGeometry(0.015, 0.07, 0.01), propMat);
            pr.position.set(p[0], p[1], p[2]);
            pr.rotation.y = p[3];
            rovGroup.add(pr);
            propellers.push(pr);
        }});
        [-0.16, 0.16].forEach(zOffset => {{
            const vtMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.10, 8), thrusterMat);
            vtMesh.position.set(0, 0.0, zOffset);
            rovGroup.add(vtMesh);

            const pr = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.015, 0.01), propMat);
            pr.position.set(0, 0.05, zOffset);
            rovGroup.add(pr);
            propellers.push(pr);
        }});

        // Drone Physical Collision Boundary Sphere (for Visual Debugging)
        const colSphereGeo = new THREE.SphereGeometry(0.85, 16, 12);
        const colSphereMat = new THREE.MeshBasicMaterial({{
            color: 0x00f0ff, wireframe: true, transparent: true, opacity: 0.25
        }});
        const colSphere = new THREE.Mesh(colSphereGeo, colSphereMat);
        rovGroup.add(colSphere);

        rovGroup.position.set(1.5, 5.0, 8.0);
        scene.add(rovGroup);

        // ---------------- 4. Dynamic Trajectory & Map Buffers (Allocated ONCE) ----------------
        const MAX_TRAJ_PTS = 3000;
        
        // True Trajectory Line (Emerald)
        const trueTrajGeo = new THREE.BufferGeometry();
        const truePositions = new Float32Array(MAX_TRAJ_PTS * 3);
        trueTrajGeo.setAttribute('position', new THREE.BufferAttribute(truePositions, 3));
        const trueTrajMat = new THREE.LineBasicMaterial({{ color: 0x10b981, linewidth: 2 }});
        const trueTrajLine = new THREE.Line(trueTrajGeo, trueTrajMat);
        scene.add(trueTrajLine);

        // SLAM Estimated Trajectory Line (Cyan)
        const slamTrajGeo = new THREE.BufferGeometry();
        const slamPositions = new Float32Array(MAX_TRAJ_PTS * 3);
        slamTrajGeo.setAttribute('position', new THREE.BufferAttribute(slamPositions, 3));
        const slamTrajMat = new THREE.LineBasicMaterial({{ color: 0x00f0ff, linewidth: 2 }});
        const slamTrajLine = new THREE.Line(slamTrajGeo, slamTrajMat);
        scene.add(slamTrajLine);

        // 3D Voxel Point Cloud (Sky Blue)
        const MAX_VOXELS = 5000;
        const voxelGeo = new THREE.BufferGeometry();
        const voxelPositions = new Float32Array(MAX_VOXELS * 3);
        voxelGeo.setAttribute('position', new THREE.BufferAttribute(voxelPositions, 3));
        const voxelMat = new THREE.PointsMaterial({{ color: 0x38bdf8, size: 0.28, transparent: true, opacity: 0.85 }});
        const voxelPoints = new THREE.Points(voxelGeo, voxelMat);
        scene.add(voxelPoints);

        // Planned Waypoint Route Line
        const wpRouteGeo = new THREE.BufferGeometry();
        const wpPositions = new Float32Array(500 * 3);
        wpRouteGeo.setAttribute('position', new THREE.BufferAttribute(wpPositions, 3));
        const wpRouteMat = new THREE.LineDashedMaterial({{ color: 0xfbbf24, dashSize: 0.6, gapSize: 0.3, transparent: true, opacity: 0.75 }});
        const wpRouteLine = new THREE.Line(wpRouteGeo, wpRouteMat);
        scene.add(wpRouteLine);

        // Active Waypoint Glowing Beacon
        const beaconGeo = new THREE.SphereGeometry(0.22, 16, 16);
        const beaconMat = new THREE.MeshBasicMaterial({{ color: 0xfbbf24, transparent: true, opacity: 0.9 }});
        const beaconMesh = new THREE.Mesh(beaconGeo, beaconMat);
        scene.add(beaconMesh);

        // Sonar Rays
        const sonarRaysGeo = new THREE.BufferGeometry();
        const sonarRayPos = new Float32Array(21 * 2 * 3);
        sonarRaysGeo.setAttribute('position', new THREE.BufferAttribute(sonarRayPos, 3));
        const sonarRaysMat = new THREE.LineBasicMaterial({{ color: 0x00f0ff, transparent: true, opacity: 0.45 }});
        const sonarRaysMesh = new THREE.LineSegments(sonarRaysGeo, sonarRaysMat);
        scene.add(sonarRaysMesh);

        // ---------------- 5. Pose Interpolation Engine (Vector3 LERP & Quaternion SLERP) ----------------
        let statePrev = null;
        let stateNext = null;
        let lastReceivedTime = performance.now();
        let targetSpeed = 0.0;

        const currentPos = new THREE.Vector3(1.5, 5.0, 8.0);
        const currentQuat = new THREE.Quaternion();
        const p0 = new THREE.Vector3(1.5, 5.0, 8.0);
        const p1 = new THREE.Vector3(1.5, 5.0, 8.0);
        const q0 = new THREE.Quaternion();
        const q1 = new THREE.Quaternion();

        function onStateReceived(data) {{
            const now = performance.now();
            if (!stateNext) {{
                statePrev = data;
                stateNext = data;
                p0.set(data.rov.x, data.rov.y, data.rov.z);
                p1.set(data.rov.x, data.rov.y, data.rov.z);
                q0.fromArray(data.rov.quaternion);
                q1.fromArray(data.rov.quaternion);
                currentPos.copy(p1);
                currentQuat.copy(q1);
            }} else {{
                statePrev = stateNext;
                stateNext = data;
                p0.copy(currentPos);
                p1.set(data.rov.x, data.rov.y, data.rov.z);
                q0.copy(currentQuat);
                q1.fromArray(data.rov.quaternion);
            }}
            lastReceivedTime = now;
            targetSpeed = data.rov.speed_mps || 0.0;

            // Update Trajectories in-place without reallocation
            if (data.traj_true && data.traj_true.length > 0) {{
                const posArr = trueTrajGeo.attributes.position.array;
                const count = Math.min(MAX_TRAJ_PTS, data.traj_true.length);
                for (let i = 0; i < count; i++) {{
                    posArr[i*3] = data.traj_true[i][0];
                    posArr[i*3+1] = data.traj_true[i][1];
                    posArr[i*3+2] = data.traj_true[i][2];
                }}
                trueTrajGeo.setDrawRange(0, count);
                trueTrajGeo.attributes.position.needsUpdate = true;
            }}

            if (data.traj_est && data.traj_est.length > 0) {{
                const posArr = slamTrajGeo.attributes.position.array;
                const count = Math.min(MAX_TRAJ_PTS, data.traj_est.length);
                for (let i = 0; i < count; i++) {{
                    posArr[i*3] = data.traj_est[i][0];
                    posArr[i*3+1] = data.traj_est[i][1];
                    posArr[i*3+2] = data.traj_est[i][2];
                }}
                slamTrajGeo.setDrawRange(0, count);
                slamTrajGeo.attributes.position.needsUpdate = true;
            }}

            // Update Point Cloud in-place
            if (data.voxels && data.voxels.length > 0) {{
                const vArr = voxelGeo.attributes.position.array;
                const count = Math.min(MAX_VOXELS, data.voxels.length);
                for (let i = 0; i < count; i++) {{
                    vArr[i*3] = data.voxels[i][0];
                    vArr[i*3+1] = data.voxels[i][1];
                    vArr[i*3+2] = data.voxels[i][2];
                }}
                voxelGeo.setDrawRange(0, count);
                voxelGeo.attributes.position.needsUpdate = true;
            }}

            // Update Waypoints
            if (data.waypoints && data.waypoints.length > 0) {{
                const wpArr = wpRouteGeo.attributes.position.array;
                const count = Math.min(500, data.waypoints.length);
                for (let i = 0; i < count; i++) {{
                    wpArr[i*3] = data.waypoints[i][0];
                    wpArr[i*3+1] = data.waypoints[i][1];
                    wpArr[i*3+2] = data.waypoints[i][2];
                }}
                wpRouteGeo.setDrawRange(0, count);
                wpRouteGeo.attributes.position.needsUpdate = true;
                wpRouteLine.computeLineDistances();
            }}

            // Update Active Waypoint Beacon
            if (data.active_wp) {{
                beaconMesh.position.set(data.active_wp.x, data.active_wp.y, data.active_wp.z);
            }}

            // Update Sonar Rays
            const sRayArr = sonarRaysGeo.attributes.position.array;
            const contactZ = getHullZ(data.rov.x, data.rov.y);
            let idx = 0;
            for (let i = -10; i <= 10; i++) {{
                sRayArr[idx++] = data.rov.x;
                sRayArr[idx++] = data.rov.y;
                sRayArr[idx++] = data.rov.z;
                sRayArr[idx++] = data.rov.x + i * 0.16;
                sRayArr[idx++] = data.rov.y + (i%3)*0.12;
                sRayArr[idx++] = contactZ;
            }}
            sonarRaysGeo.setDrawRange(0, 42);
            sonarRaysGeo.attributes.position.needsUpdate = true;

            // Standoff Warning Halo & Collision Boundary Indicator
            const isHazard = (data.collision_warning === true) || (data.collision_status && data.collision_status !== 'SAFE') || (data.standoff_m < 1.0);
            if (isHazard) {{
                haloMat.color.setHex(0xef4444);
                haloMat.opacity = 0.95;
                colSphereMat.color.setHex(0xef4444);
                colSphereMat.opacity = 0.85;
            }} else {{
                haloMat.color.setHex(0x00f0ff);
                haloMat.opacity = 0.4;
                colSphereMat.color.setHex(0x00f0ff);
                colSphereMat.opacity = isNavDebug ? 0.45 : 0.20;
            }}

            // Defects landmarks
            if (data.defects && defectGroup.children.length === 0) {{
                updateDefectLandmarks(data.defects);
            }}

            // HUD string with live collision safety status
            const hudEl = document.getElementById('hud-overlay');
            if (hudEl) {{
                const pausedTxt = data.is_paused ? ' [PAUSED]' : '';
                const colStatus = data.collision_status || 'SAFE';
                const colColor = (colStatus === 'SAFE') ? '#10b981' : '#ef4444';
                const colBadge = `<span style="background:${{colColor}}25; border:1px solid ${{colColor}}; color:${{colColor}}; padding:1px 6px; border-radius:3px; font-weight:700;">HULL COLLISION: ${{colStatus}}</span>`;
                const clearanceVal = data.clearance_m !== undefined ? data.clearance_m : data.standoff_m;
                hudEl.innerHTML = `3D PERSISTENT ENGINE &bull; SIM TIME: ${{data.sim_time_s.toFixed(1)}}s${{pausedTxt}} &bull; ROV: (${{data.rov.x.toFixed(2)}}, ${{data.rov.y.toFixed(2)}}, ${{data.rov.z.toFixed(2)}}) &bull; CLEARANCE: ${{clearanceVal.toFixed(2)}}m &bull; ${{colBadge}}`;
            }}
        }}

        // Asynchronous High-Frequency State Poller (runs continuously)
        let isPolling = false;
        async function pollState() {{
            if (isPolling) return;
            isPolling = true;
            try {{
                const res = await fetch(API_URL);
                if (res.ok) {{
                    const data = await res.json();
                    onStateReceived(data);
                    document.getElementById('conn-status').innerText = '● 60 FPS SYNC ACTIVE';
                    document.getElementById('conn-status').style.color = '#10b981';
                }}
            }} catch(e) {{
                document.getElementById('conn-status').innerText = '○ CONNECTING...';
                document.getElementById('conn-status').style.color = '#f59e0b';
            }} finally {{
                isPolling = false;
            }}
        }}
        setInterval(pollState, 35); // 28 Hz state poll, render at 60 Hz

        // ---------------- 6. Camera View Modes & Persistence ----------------
        let currentMode = 'follow';

        function applyCameraMode(mode) {{
            currentMode = mode;
            document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
            const btn = document.getElementById('btn-' + mode);
            if (btn) btn.classList.add('active');

            if (mode === 'follow') {{
                camera.position.set(currentPos.x - 3.8, currentPos.y + 1.4, currentPos.z + 3.4);
                controls.target.set(currentPos.x + 1.0, currentPos.y, currentPos.z);
            }} else if (mode === 'side') {{
                camera.position.set(25.0, 5.0, 26.0);
                controls.target.set(25.0, 5.0, 6.0);
            }} else if (mode === 'top') {{
                camera.position.set(25.0, 32.0, 7.0);
                controls.target.set(25.0, 5.0, 7.0);
            }} else if (mode === 'front') {{
                camera.position.set(-8.0, 5.0, 8.0);
                controls.target.set(8.0, 5.0, 6.0);
            }} else if (mode === 'reset' || mode === 'free') {{
                camera.position.set(-6.0, 14.0, 22.0);
                controls.target.set(16.0, 5.0, 6.0);
            }}
            controls.update();
        }}
        window.setCam = applyCameraMode;
        applyCameraMode('follow');

        // ---------------- 7. Continuous 60 FPS Render Loop with Interpolation ----------------
        let lastFrameTime = performance.now();
        const desiredCamPos = new THREE.Vector3();
        const desiredTarget = new THREE.Vector3();

        function animate(now) {{
            requestAnimationFrame(animate);
            const dtSec = Math.min(0.1, (now - lastFrameTime) * 0.001);
            lastFrameTime = now;

            // Interpolation factor between last state and current state
            const elapsedSinceReceive = (now - lastReceivedTime) * 0.001;
            const alpha = Math.min(1.0, elapsedSinceReceive / 0.045);

            // Vector3 LERP for continuous position
            currentPos.lerpVectors(p0, p1, alpha);
            rovGroup.position.copy(currentPos);

            // Quaternion SLERP for orientation
            currentQuat.copy(q0).slerp(q1, alpha);
            rovGroup.quaternion.copy(currentQuat);

            // Propeller spin animation proportional to velocity
            const spinDelta = (targetSpeed + 0.1) * dtSec * 35.0;
            propellers.forEach(pr => {{
                pr.rotation.x += spinDelta;
            }});

            // Damped Follow Camera update (smooth, non-snapping)
            if (currentMode === 'follow') {{
                desiredTarget.set(currentPos.x + 1.2, currentPos.y, currentPos.z);
                desiredCamPos.set(currentPos.x - 3.8, currentPos.y + 1.4, currentPos.z + 3.4);

                camera.position.lerp(desiredCamPos, 0.08);
                controls.target.lerp(desiredTarget, 0.08);
            }}

            controls.update();
            renderer.render(scene, camera);
        }}
        requestAnimationFrame(animate);

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / {height_px};
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, {height_px});
        }});
    </script>
</body>
</html>
    """
    return html


def generate_threejs_simulation_html(*args, **kwargs) -> str:
    """Backwards-compatible alias for persistent Three.js HTML generation."""
    server_port = kwargs.get("server_port", 8502)
    height_px = kwargs.get("height_px", 580)
    return generate_persistent_threejs_html(server_port=server_port, height_px=height_px)

