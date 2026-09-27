"""
train.py

Fine-tunes a pretrained YOLOv8 model (transfer learning) on the ISL
alphabet dataset and logs mAP50, mAP50-95, and per-class precision/recall
after training completes.

Usage:
    python train.py --data /path/to/data.yaml \
        --model yolov8n.pt \
        --epochs 100 \
        --imgsz 640 \
        --batch 16 \
        --patience 20 \
        --project runs/isl_alphabet \
        --name yolov8n_isl

Notes:
    - `--model yolov8n.pt` downloads (or reuses a cached) COCO-pretrained
      checkpoint; Ultralytics handles the transfer-learning head swap
      automatically based on `nc` in data.yaml.
    - `--patience` enables early stopping: training halts if val metrics
      don't improve for that many epochs.
    - Results, weights (best.pt/last.pt), and plots are written under
      <project>/<name>/.
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on ISL alphabet dataset")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="Pretrained checkpoint to start from (transfer learning)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20,
                         help="Early stopping patience (epochs with no improvement)")
    parser.add_argument("--device", type=str, default=None,
                         help="e.g. '0' for GPU 0, 'cpu' for CPU. Default: auto-detect.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs/isl_alphabet")
    parser.add_argument("--name", type=str, default="yolov8n_isl")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt in project/name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Freeze first N layers (backbone) for lighter fine-tuning; 0 = train all layers",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"ERROR: data.yaml not found at {data_path}")
        sys.exit(1)

    print(f"Loading pretrained model: {args.model}")
    model = YOLO(args.model)

    train_kwargs = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        resume=args.resume,
        exist_ok=True,
        # Sensible defaults for a static hand-sign detection task:
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        val=True,
        plots=True,
        save=True,
        save_period=-1,  # only save best/last, not every epoch
        verbose=True,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.freeze > 0:
        train_kwargs["freeze"] = args.freeze

    print("Starting training with config:")
    for k, v in train_kwargs.items():
        print(f"  {k}: {v}")

    results = model.train(**train_kwargs)

    run_dir = Path(results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")

    # ---- Post-training validation + metric logging ----
    print("\nRunning final validation on best weights...")
    best_model = YOLO(str(best_weights))
    metrics = best_model.val(data=str(data_path), imgsz=args.imgsz, split="val")

    print("\n=== Overall metrics ===")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision (mean): {metrics.box.mp:.4f}")
    print(f"  Recall (mean):    {metrics.box.mr:.4f}")

    names = metrics.names  # dict {class_id: name}
    print("\n=== Per-class metrics ===")
    print(f"{'Class':>6} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}")
    # metrics.box.p, .r, .ap50, .ap are arrays aligned by class index (per-class)
    try:
        p_per_class = metrics.box.p
        r_per_class = metrics.box.r
        ap50_per_class = metrics.box.ap50
        ap_per_class = metrics.box.ap
        for i, cls_id in enumerate(metrics.ap_class_index):
            cname = names.get(int(cls_id), str(cls_id))
            print(f"{cname:>6} {p_per_class[i]:>10.4f} {r_per_class[i]:>10.4f} "
                  f"{ap50_per_class[i]:>10.4f} {ap_per_class[i]:>10.4f}")
    except Exception as e:
        print(f"  (Per-class arrays unavailable in this Ultralytics version: {e})")

    print(f"\nRun artifacts saved under: {run_dir}")
    print(f"Best weights for deployment: {best_weights}")


if __name__ == "__main__":
    main()