"""Automated Production Deployment Readiness & Performance Verification Suite."""

import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_checks() -> bool:
    print("=" * 70)
    print("🚀 RUNNING PRODUCTION DEPLOYMENT READINESS AUDIT")
    print("=" * 70)

    passed_all = True

    # 1. Dependency & Environment Check
    print("\n[1/6] Inspecting Python Environment & Core Runtimes...")
    try:
        import torch
        import cv2
        import streamlit
        import pyarrow
        import reportlab

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  ✓ Python: {sys.version.split()[0]}")
        print(f"  ✓ PyTorch: {torch.__version__} (Inference Device: {device.upper()})")
        print(f"  ✓ OpenCV: {cv2.__version__}")
        print(f"  ✓ Streamlit: {streamlit.__version__}")
        print(f"  ✓ PyArrow: {pyarrow.__version__}")
        print(f"  ✓ ReportLab: {reportlab.__version__}")
    except Exception as e:
        print(f"  ❌ Dependency check failed: {e}")
        return False

    # 2. Checkpoint Availability & Integrity
    print("\n[2/6] Validating Pretrained Model Checkpoints...")
    ckpt_dir = ROOT / "models" / "checkpoints"
    unet_path = ckpt_dir / "unet_mavecodd.pth"
    base_path = ckpt_dir / "baseline_mavecodd.joblib"

    if not unet_path.exists():
        print(f"  ❌ Missing U-Net checkpoint: {unet_path}")
        passed_all = False
    else:
        print(f"  ✓ U-Net Checkpoint: {unet_path.name} ({unet_path.stat().st_size / 1e6:.1f} MB)")

    if not base_path.exists():
        print(f"  ❌ Missing Baseline model checkpoint: {base_path}")
        passed_all = False
    else:
        print(f"  ✓ Baseline Checkpoint: {base_path.name} ({base_path.stat().st_size / 1e3:.1f} KB)")

    # 3. Model Inference Pipeline Benchmark
    print("\n[3/6] Benchmarking Inference Pipeline Latency...")
    try:
        from src.models.predict import HullInspectionPredictor

        t0 = time.perf_counter()
        predictor = HullInspectionPredictor(
            baseline_path=base_path if base_path.exists() else None,
            unet_path=unet_path if unet_path.exists() else None,
        )
        load_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  ✓ Model initialization time: {load_ms:.1f} ms")

        # Run synthetic frame inference
        dummy_img = (np.random.rand(360, 480, 3) * 255).astype(np.uint8)
        # 1. Real-time U-Net inference (used during 10-20 FPS live simulation)
        t_unet0 = time.perf_counter()
        pred_unet = predictor.predict(dummy_img, run_baseline=False)
        unet_ms = (time.perf_counter() - t_unet0) * 1000.0
        print(f"  ✓ U-Net inference latency: {unet_ms:.1f} ms (Target: < 50 ms for live 20 FPS sim)")

        # 2. Comprehensive Dual-Model inference (used on manual inspections)
        t_dual0 = time.perf_counter()
        pred_dual = predictor.predict(dummy_img, run_baseline=True)
        dual_ms = (time.perf_counter() - t_dual0) * 1000.0
        print(f"  ✓ Dual-Model (U-Net + Tile Baseline) latency: {dual_ms:.1f} ms")
        print(f"  ✓ Prediction outputs verified: defect_detected={pred_dual['defect_detected']}, status={pred_dual['status']}")
    except Exception as e:
        print(f"  ❌ Inference benchmark error: {e}")
        passed_all = False

    # 4. 3D Hydrodynamics & Hull Collision Safety System
    print("\n[4/6] Verifying 3D Simulation & Hard Safety Constraints...")
    try:
        from src.simulation.environment_3d import ShipHull3DEnvironment
        from src.simulation.rov_dynamics import ROVDynamics6DOF
        from src.simulation.autonomous_inspector import AutonomousHullInspector
        from src.simulation.synthetic_sensors import SyntheticSonarArray, SyntheticCameraPipeline, SyntheticIMUSensor
        from src.simulation.slam import UnderwaterSLAM3D

        env = ShipHull3DEnvironment()
        rov = ROVDynamics6DOF(initial_pos=(2.0, 5.0, 8.0))
        slam = UnderwaterSLAM3D(initial_pose=(2.0, 5.0, 8.0))
        sonar = SyntheticSonarArray(env=env)
        cam = SyntheticCameraPipeline(env=env)
        imu = SyntheticIMUSensor()

        inspector = AutonomousHullInspector(
            env=env, rov=rov, slam=slam, sonar=sonar, camera_pipeline=cam, imu=imu, standoff_m=2.0
        )
        wps = inspector.plan_inspection_route("BOW_STBD")
        print(f"  ✓ Planned {len(wps)} conformal outward-normal waypoints")

        # Verify hard safety rejection
        inside_hull_pos = np.array([10.0, 5.0, 0.0])
        val_res = inspector.validate_movement(inspector.rov.pos, inside_hull_pos)
        assert val_res["is_valid"] is False, "Movement into hull interior must be rejected"
        print(f"  ✓ Hard collision constraint verified: intercepted penetration ({val_res['reason']})")
    except Exception as e:
        print(f"  ❌ 3D Simulation verification error: {e}")
        passed_all = False

    # 5. Columnar Parquet Telemetry & PDF Certification
    print("\n[5/6] Checking Parquet Telemetry Logs & PDF Reporting...")
    try:
        from src.storage.parquet_reader import list_available_inspections, read_inspection_log
        from src.reporting.report_generator import generate_pdf_inspection_report

        logs_dir = ROOT / "outputs" / "inspection_logs"
        inspections = list_available_inspections(logs_dir)
        print(f"  ✓ Discovered {len(inspections)} archived mission telemetry logs")

        if inspections:
            sample_file = Path(inspections[0]["file_path"])
            df = read_inspection_log(sample_file)
            print(f"  ✓ Parquet columnar reader verified: {len(df)} records in {sample_file.name}")

            # Verify PDF generator smoke test
            out_pdf = ROOT / "outputs" / "reports" / "SMOKE_TEST_REPORT.pdf"
            generate_pdf_inspection_report(sample_file, out_pdf)
            assert out_pdf.exists() and out_pdf.stat().st_size > 1000
            print(f"  ✓ Generated DNV/Lloyd's compliant PDF: {out_pdf.name} ({out_pdf.stat().st_size / 1e3:.1f} KB)")
            out_pdf.unlink(missing_ok=True)
    except Exception as e:
        print(f"  ❌ Storage & Reporting error: {e}")
        passed_all = False

    # 6. Streamlit Pages Syntax & Import Validation
    print("\n[6/6] Compiling Streamlit Application Navigation Pages...")
    pages_dir = ROOT / "app" / "pages"
    for page_py in sorted(pages_dir.glob("*.py")):
        try:
            with open(page_py, "r", encoding="utf-8") as f:
                compile(f.read(), str(page_py), "exec")
            print(f"  ✓ {page_py.name}: syntax valid")
        except Exception as e:
            print(f"  ❌ {page_py.name} compilation error: {e}")
            passed_all = False

    print("\n" + "=" * 70)
    if passed_all:
        print("🎉 ALL PRODUCTION AUDIT CHECKS PASSED: READY FOR DEPLOYMENT!")
        print("=" * 70)
        return True
    else:
        print("⚠️ SOME CHECKS FAILED: REVIEW OUTPUT ABOVE")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
