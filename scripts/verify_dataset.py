"""
verify_dataset.py

Verifies that a Roboflow-exported YOLO-format ISL alphabet dataset is
structurally sound before training. Checks:

  1. data.yaml exists and parses correctly.
  2. train/valid/test splits referenced in data.yaml exist on disk.
  3. Each split's images/ and labels/ directories exist and are non-empty.
  4. Every image has a matching label file (and vice versa), unless the
     split has zero labels by design (rare for detection datasets).
  5. Label files contain valid YOLO-format rows: `class_id cx cy w h`
     with class_id within [0, nc) and normalized coords within [0, 1].
  6. Reports the class list / count and flags if it deviates from the
     expected 26 letters (A-Z) of ISL fingerspelling.

Usage:
    python verify_dataset.py --data /path/to/isl_alphabets/data.yaml
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASSES = 26  # A-Z static fingerspelling


def load_data_yaml(data_yaml_path: Path) -> dict:
    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")
    with open(data_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    for key in ("names",):
        if key not in cfg:
            raise ValueError(f"data.yaml is missing required key: '{key}'")
    return cfg


def resolve_split_dir(data_yaml_path: Path, split_value: str) -> Path:
    """Roboflow data.yaml paths are typically relative to the yaml's folder."""
    p = Path(split_value)
    if not p.is_absolute():
        p = (data_yaml_path.parent / p).resolve()
    return p


def find_images_and_labels_dirs(split_path: Path) -> tuple[Path, Path]:
    """
    Roboflow splits usually look like:
        train/images/*.jpg
        train/labels/*.txt
    but data.yaml's split value sometimes points directly at the images dir.
    Handle both cases.
    """
    if split_path.name == "images":
        images_dir = split_path
        labels_dir = split_path.parent / "labels"
    else:
        images_dir = split_path / "images"
        labels_dir = split_path / "labels"
    return images_dir, labels_dir


def check_split(name: str, images_dir: Path, labels_dir: Path, nc: int) -> dict:
    report = {
        "split": name,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "num_images": 0,
        "num_labels": 0,
        "images_without_labels": [],
        "labels_without_images": [],
        "bad_rows": [],  # (label_file, line_no, reason)
        "class_counts": Counter(),
        "ok": True,
        "errors": [],
    }

    if not images_dir.is_dir():
        report["ok"] = False
        report["errors"].append(f"Missing images dir: {images_dir}")
        return report
    if not labels_dir.is_dir():
        report["ok"] = False
        report["errors"].append(f"Missing labels dir: {labels_dir}")
        return report

    images = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    labels = {p.stem: p for p in labels_dir.iterdir() if p.suffix.lower() == ".txt"}

    report["num_images"] = len(images)
    report["num_labels"] = len(labels)

    if not images:
        report["ok"] = False
        report["errors"].append("No images found.")

    report["images_without_labels"] = sorted(set(images) - set(labels))
    report["labels_without_images"] = sorted(set(labels) - set(images))

    if report["images_without_labels"]:
        report["ok"] = False
    if report["labels_without_images"]:
        report["ok"] = False

    # Validate label file contents
    for stem, label_path in labels.items():
        with open(label_path, "r") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    report["bad_rows"].append((str(label_path), line_no, f"expected 5 fields, got {len(parts)}"))
                    report["ok"] = False
                    continue
                try:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    report["bad_rows"].append((str(label_path), line_no, "non-numeric field"))
                    report["ok"] = False
                    continue

                if not (0 <= cls_id < nc):
                    report["bad_rows"].append(
                        (str(label_path), line_no, f"class_id {cls_id} out of range [0,{nc})")
                    )
                    report["ok"] = False
                    continue

                if not all(0.0 <= c <= 1.0 for c in coords):
                    report["bad_rows"].append(
                        (str(label_path), line_no, f"un-normalized coords {coords}")
                    )
                    report["ok"] = False
                    continue

                report["class_counts"][cls_id] += 1

    return report


def main():
    parser = argparse.ArgumentParser(description="Verify ISL YOLO dataset structure")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    args = parser.parse_args()

    data_yaml_path = Path(args.data).resolve()
    print(f"Loading data.yaml: {data_yaml_path}")
    cfg = load_data_yaml(data_yaml_path)

    names = cfg["names"]
    if isinstance(names, dict):
        # Ultralytics allows {0: 'A', 1: 'B', ...}
        names_list = [names[k] for k in sorted(names, key=int)]
    else:
        names_list = list(names)
    nc = cfg.get("nc", len(names_list))

    print(f"Classes declared (nc={nc}): {names_list}")
    if nc != EXPECTED_CLASSES or len(names_list) != EXPECTED_CLASSES:
        print(
            f"  WARNING: expected {EXPECTED_CLASSES} classes (A-Z fingerspelling), "
            f"found {len(names_list)}."
        )

    letters = set(c.upper() for c in names_list)
    missing_letters = sorted(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - letters)
    if missing_letters:
        print(f"  WARNING: missing expected letters: {missing_letters}")

    overall_ok = True
    splits_present = {}
    for split_key in ("train", "val", "valid", "test"):
        if split_key in cfg and cfg[split_key]:
            splits_present[split_key] = cfg[split_key]

    if "train" not in splits_present:
        print("  ERROR: 'train' split missing from data.yaml")
        overall_ok = False
    if not any(k in splits_present for k in ("val", "valid")):
        print("  WARNING: no validation split ('val'/'valid') found in data.yaml")

    print("\n--- Split checks ---")
    for split_name, split_value in splits_present.items():
        split_path = resolve_split_dir(data_yaml_path, split_value)
        images_dir, labels_dir = find_images_and_labels_dirs(split_path)
        report = check_split(split_name, images_dir, labels_dir, nc)

        status = "OK" if report["ok"] else "FAILED"
        print(f"\n[{split_name}] -> {status}")
        print(f"  images_dir: {report['images_dir']} ({report['num_images']} images)")
        print(f"  labels_dir: {report['labels_dir']} ({report['num_labels']} labels)")

        if report["errors"]:
            for e in report["errors"]:
                print(f"  ERROR: {e}")
            overall_ok = False
            continue

        if report["images_without_labels"]:
            overall_ok = False
            print(f"  ERROR: {len(report['images_without_labels'])} images missing labels "
                  f"(e.g. {report['images_without_labels'][:5]})")
        if report["labels_without_images"]:
            overall_ok = False
            print(f"  ERROR: {len(report['labels_without_images'])} labels missing images "
                  f"(e.g. {report['labels_without_images'][:5]})")
        if report["bad_rows"]:
            overall_ok = False
            print(f"  ERROR: {len(report['bad_rows'])} malformed label rows, e.g.:")
            for lf, ln, reason in report["bad_rows"][:5]:
                print(f"    {lf}:{ln} -> {reason}")

        # Per-class instance counts for this split
        counts = report["class_counts"]
        if counts:
            print("  Per-class instance counts:")
            for cls_id in sorted(counts):
                cls_name = names_list[cls_id] if cls_id < len(names_list) else f"id{cls_id}"
                print(f"    {cls_name:>3}: {counts[cls_id]}")
            zero_classes = [names_list[i] for i in range(len(names_list)) if counts.get(i, 0) == 0]
            if zero_classes:
                print(f"  WARNING: classes with zero instances in this split: {zero_classes}")

    print("\n=====================================")
    if overall_ok:
        print("Dataset verification PASSED.")
        sys.exit(0)
    else:
        print("Dataset verification FAILED. Fix the errors above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()