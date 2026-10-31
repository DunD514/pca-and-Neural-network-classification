"""Dataset preparation script: extraction, synthetic sample generation, and stratified splitting."""

import os
import sys
import zipfile
from pathlib import Path

# Add repository root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed
from src.data.dataset_loader import (
    scan_mavecodd_samples,
    scan_crack_samples,
    generate_synthetic_samples
)
from src.data.split_dataset import create_leakage_free_splits, save_split_manifest


from PIL import Image

def validate_and_enrich_samples(samples: list) -> tuple[list, list]:
    valid_samples = []
    errors = []
    for s in samples:
        img_path = s.get("image_path")
        mask_path = s.get("mask_path")
        if not img_path or not os.path.exists(img_path):
            err = f"Missing image: {img_path}"
            errors.append({"image_id": s.get("image_id"), "error": err})
            continue
        if not mask_path or not os.path.exists(mask_path):
            err = f"Missing mask: {mask_path}"
            errors.append({"image_id": s.get("image_id"), "error": err})
            continue

        try:
            with Image.open(img_path) as im:
                iw, ih = im.size
                channels = len(im.getbands())
            with Image.open(mask_path) as mk:
                mw, mh = mk.size

            if (iw, ih) != (mw, mh):
                err = f"Dimension mismatch: image ({iw}x{ih}) vs mask ({mw}x{mh})"
                errors.append({"image_id": s.get("image_id"), "error": err})
                continue

            s["image_width"] = iw
            s["image_height"] = ih
            s["image_shape"] = [ih, iw, channels]
            s["mask_shape"] = [mh, mw]
            s["is_valid"] = True
            s["validation_error"] = None
            valid_samples.append(s)
        except Exception as e:
            errors.append({"image_id": s.get("image_id"), "error": str(e)})

    return valid_samples, errors


def main():
    print("=" * 70)
    print("AI-POWERED UNDERWATER HULL INSPECTION DRONE - DATASET PREPARATION")
    print("=" * 70)

    config = load_config()
    set_seed(config.get("random_seed", 42))
    cfg_paths = config["paths"]

    # 1. Extract zip archives if not extracted
    raw_mav_dir = ROOT / cfg_paths["raw_mavecodd"]
    zip_mav = ROOT / cfg_paths["zip_mavecodd"]
    if zip_mav.exists() and not (raw_mav_dir / "HiRes").exists():
        print(f"Extracting {zip_mav.name} to {raw_mav_dir}...")
        raw_mav_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_mav, "r") as z:
            z.extractall(raw_mav_dir)
        print("Extracted MaVeCoDD dataset.")
    elif (raw_mav_dir / "HiRes").exists():
        print(f"MaVeCoDD dataset found at: {raw_mav_dir}")
    else:
        print(f"MaVeCoDD dataset NOT found (expected {zip_mav.name} in repo root or files in {raw_mav_dir}).")

    raw_crack_dir = ROOT / cfg_paths["raw_underwater_crack"]
    zip_crack = ROOT / cfg_paths["zip_underwater_crack"]
    if zip_crack.exists() and not (raw_crack_dir / "images").exists():
        print(f"Extracting {zip_crack.name} to {raw_crack_dir}...")
        raw_crack_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_crack, "r") as z:
            z.extractall(raw_crack_dir)
        print("Extracted Underwater Crack dataset.")
    elif (raw_crack_dir / "images").exists():
        print(f"Underwater Crack dataset found at: {raw_crack_dir}")
    else:
        print(f"Underwater Crack dataset NOT found (expected {zip_crack.name} in repo root).")

    # 2. Generate Synthetic Sample Dataset (Quick-start mode)
    sample_dir = ROOT / cfg_paths["sample_dir"]
    print(f"Generating synthetic quick-start samples at {sample_dir}...")
    sample_items = generate_synthetic_samples(
        output_dir=sample_dir,
        num_samples=config["dataset"].get("sample_count", 40),
        image_size=tuple(config["dataset"].get("image_size", [256, 256])),
        random_seed=config.get("random_seed", 42)
    )
    print(f"Generated {len(sample_items)} synthetic hull samples.")

    # 3. Create Split Manifests with Leakage Prevention & Validation
    processed_dir = ROOT / cfg_paths["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    summary_report = {
        "datasets": {}
    }

    # 3a. MaVeCoDD splits
    mav_raw_samples = scan_mavecodd_samples(
        raw_mav_dir, zip_fallback=zip_mav,
        scene_block_size=config["dataset"].get("scene_block_size", 5)
    )
    mav_samples, mav_errors = validate_and_enrich_samples(mav_raw_samples)
    print(f"MaVeCoDD validation: {len(mav_samples)} valid pairs, {len(mav_errors)} errors.")
    if mav_samples:
        mav_splits = create_leakage_free_splits(
            mav_samples,
            train_ratio=config["dataset"]["train_split"],
            val_ratio=config["dataset"]["val_split"],
            test_ratio=config["dataset"]["test_split"],
            group_key="scene_id",
            random_seed=config.get("random_seed", 42),
            stratify_key="resolution_tier"
        )
        save_split_manifest(mav_splits, processed_dir / "split_manifest_mavecodd.json")
        for name, items in mav_splits.items():
            groups = sorted({s["scene_id"] for s in items})
            print(f"MaVeCoDD {name:5s}: {len(items):2d} images in {len(groups)} scene groups -> {', '.join(groups)}")
        summary_report["datasets"]["mavecodd"] = {
            "total_valid_pairs": len(mav_samples),
            "validation_errors": mav_errors,
            "splits": {k: len(v) for k, v in mav_splits.items()}
        }

    # 3b. Underwater Crack splits
    crack_raw_samples = scan_crack_samples(raw_crack_dir, zip_fallback=zip_crack)
    crack_samples, crack_errors = validate_and_enrich_samples(crack_raw_samples)
    print(f"Underwater Crack validation: {len(crack_samples)} valid pairs, {len(crack_errors)} errors.")
    if crack_samples:
        crack_splits = create_leakage_free_splits(
            crack_samples,
            train_ratio=config["dataset"]["train_split"],
            val_ratio=config["dataset"]["val_split"],
            test_ratio=config["dataset"]["test_split"],
            group_key="scene_id",
            random_seed=config.get("random_seed", 42)
        )
        save_split_manifest(crack_splits, processed_dir / "split_manifest_crack.json")
        print(f"Underwater Crack Splits -> Train: {len(crack_splits['train'])}, Val: {len(crack_splits['val'])}, Test: {len(crack_splits['test'])}")
        summary_report["datasets"]["underwater_crack"] = {
            "total_valid_pairs": len(crack_samples),
            "validation_errors": crack_errors,
            "splits": {k: len(v) for k, v in crack_splits.items()}
        }

    # 3c. Synthetic sample splits
    sample_items, sample_errors = validate_and_enrich_samples(sample_items)
    sample_splits = create_leakage_free_splits(
        sample_items,
        train_ratio=config["dataset"]["train_split"],
        val_ratio=config["dataset"]["val_split"],
        test_ratio=config["dataset"]["test_split"],
        group_key="scene_id",
        random_seed=config.get("random_seed", 42)
    )
    save_split_manifest(sample_splits, processed_dir / "split_manifest_sample.json")
    print(f"Synthetic Sample Splits -> Train: {len(sample_splits['train'])}, Val: {len(sample_splits['val'])}, Test: {len(sample_splits['test'])}")

    # Save summary report
    import json
    with open(processed_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print("=" * 70)
    print("Dataset preparation completed successfully!")


if __name__ == "__main__":
    main()
