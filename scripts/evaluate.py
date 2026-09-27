"""
evaluate.py

Runs model.val() on the ISL alphabet dataset, prints the confusion matrix,
and identifies which letter classes are most frequently confused with
one another (useful for spotting visually-similar ISL handshapes, e.g.
M/N/S or U/V).

Usage:
    python evaluate.py --data /path/to/data.yaml \
        --weights runs/isl_alphabet/yolov8n_isl/weights/best.pt \
        --split test \
        --top-k 10 \
        --save-plot confusion_matrix.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ISL YOLOv8 model + confusion analysis")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt weights")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                         help="Which split to evaluate on")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for val")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold for val")
    parser.add_argument("--top-k", type=int, default=10,
                         help="How many most-confused class pairs to report")
    parser.add_argument("--save-plot", type=str, default=None,
                         help="Optional path to save a confusion matrix heatmap PNG")
    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        print(f"ERROR: weights file not found: {weights_path}")
        sys.exit(1)

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"Running validation on split='{args.split}' ...")
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        plots=True,
    )

    print("\n=== Overall metrics ===")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision (mean): {metrics.box.mp:.4f}")
    print(f"  Recall (mean):    {metrics.box.mr:.4f}")

    names = metrics.names  # {class_id: name}
    class_ids_sorted = sorted(names.keys())
    class_names_sorted = [names[i] for i in class_ids_sorted]

    cm_obj = metrics.confusion_matrix
    if cm_obj is None or not hasattr(cm_obj, "matrix"):
        print("\nERROR: confusion matrix not available from this val() call.")
        sys.exit(1)

    cm = cm_obj.matrix  # shape (nc+1, nc+1); last row/col = background/FP-FN
    nc = len(class_names_sorted)

    print(f"\n=== Confusion matrix ({nc} classes + background) ===")
    header = "true\\pred".rjust(10) + "".join(f"{n:>6}" for n in class_names_sorted) + f"{'bg':>6}"
    print(header)
    row_labels = class_names_sorted + ["bg"]
    for i in range(cm.shape[0]):
        row_label = row_labels[i] if i < len(row_labels) else "bg"
        row_vals = "".join(f"{int(cm[i, j]):>6}" for j in range(cm.shape[1]))
        print(f"{row_label:>10}{row_vals}")

    # ---- Identify most-confused class pairs (off-diagonal, excluding background) ----
    print(f"\n=== Top {args.top_k} most-confused class pairs (true -> predicted) ===")
    confusions = []
    for i in range(nc):
        for j in range(nc):
            if i == j:
                continue
            count = cm[i, j]
            if count > 0:
                confusions.append((count, class_names_sorted[i], class_names_sorted[j]))

    confusions.sort(key=lambda x: x[0], reverse=True)
    if not confusions:
        print("  No off-diagonal confusions found — model is cleanly separating all classes.")
    else:
        for count, true_name, pred_name in confusions[: args.top_k]:
            print(f"  true='{true_name}'  predicted='{pred_name}'   count={int(count)}")

    # ---- Per-class recall / precision derived from confusion matrix (sanity cross-check) ----
    print("\n=== Per-class summary (from confusion matrix) ===")
    print(f"{'Class':>6} {'TP':>6} {'FN(->bg)':>10} {'Confused-with-other-letters':>30}")
    for i in range(nc):
        tp = cm[i, i]
        fn_to_bg = cm[i, -1] if cm.shape[1] > nc else 0
        confused_total = cm[i, :].sum() - tp - fn_to_bg
        print(f"{class_names_sorted[i]:>6} {int(tp):>6} {int(fn_to_bg):>10} {int(confused_total):>30}")

    if args.save_plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(max(10, nc * 0.4), max(8, nc * 0.4)))
            sns.heatmap(
                cm, annot=False, cmap="Blues", xticklabels=row_labels, yticklabels=row_labels
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"ISL Alphabet Confusion Matrix ({args.split} split)")
            plt.tight_layout()
            plt.savefig(args.save_plot, dpi=150)
            print(f"\nSaved confusion matrix heatmap to: {args.save_plot}")
        except Exception as e:
            print(f"\nWARNING: could not save plot ({e}). "
                  f"Ultralytics also auto-saves one under the val run directory.")


if __name__ == "__main__":
    main()