"""
infer_sanity_check.py

Quick sanity check that trained weights actually detect ISL letters
correctly, before porting to the C++ deployment app. Supports a single
image, a folder of images, a video file, or a live webcam feed.

Usage:
    # single image
    python infer_sanity_check.py --weights best.pt --source test_image.jpg

    # folder of images
    python infer_sanity_check.py --weights best.pt --source ./test_images/

    # video file
    python infer_sanity_check.py --weights best.pt --source clip.mp4

    # webcam (device index 0)
    python infer_sanity_check.py --weights best.pt --source 0 --show
"""

import argparse
import sys
import time
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="ISL YOLOv8 inference sanity check")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt weights")
    parser.add_argument("--source", type=str, required=True,
                         help="Image path, folder, video path, or webcam index (e.g. '0')")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="runs/isl_alphabet/sanity_check")
    parser.add_argument("--show", action="store_true",
                         help="Open a live display window (webcam/video)")
    return parser.parse_args()


def resolve_source(source_str: str):
    # Webcam index like "0", "1"
    if source_str.isdigit():
        return int(source_str)
    return source_str


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        print(f"ERROR: weights file not found: {weights_path}")
        sys.exit(1)

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    source = resolve_source(args.source)

    predict_kwargs = dict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save=True,
        project=str(Path(args.save_dir).parent),
        name=Path(args.save_dir).name,
        exist_ok=True,
        stream=isinstance(source, int) or (isinstance(source, str) and source.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        )),
        show=args.show,
    )
    if args.device is not None:
        predict_kwargs["device"] = args.device

    print(f"Running inference on source: {source}")
    t0 = time.time()
    results_generator = model.predict(**predict_kwargs)

    total_frames = 0
    total_detections = 0
    per_class_counts = {}

    for result in results_generator:
        total_frames += 1
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        total_detections += len(boxes)
        for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            cname = result.names[int(cls_id)]
            per_class_counts.setdefault(cname, []).append(conf)

        # Print a compact per-frame summary for the first few frames/images
        if total_frames <= 20:
            det_str = ", ".join(
                f"{result.names[int(c)]}:{conf:.2f}"
                for c, conf in zip(boxes.cls.tolist(), boxes.conf.tolist())
            )
            print(f"  frame {total_frames}: {det_str}")

    elapsed = time.time() - t0
    print(f"\nProcessed {total_frames} frame(s)/image(s) in {elapsed:.2f}s "
          f"({total_frames / elapsed:.1f} FPS)" if elapsed > 0 else "")
    print(f"Total detections: {total_detections}")

    if per_class_counts:
        print("\n=== Detected classes summary ===")
        for cname, confs in sorted(per_class_counts.items()):
            print(f"  {cname}: count={len(confs)}  avg_conf={sum(confs)/len(confs):.3f}")
    else:
        print("\nWARNING: no detections were made at all. Check --conf threshold, "
              "weights path, and that the source contains a clear, in-frame hand sign.")

    print(f"\nAnnotated output saved under: {Path(args.save_dir).resolve()}")
    print("If detections look correct here, the weights are ready to export/port to C++.")


if __name__ == "__main__":
    main()