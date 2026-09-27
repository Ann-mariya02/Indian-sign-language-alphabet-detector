# ISL Alphabet Detection — Python Training Pipeline (Part 1)

YOLOv8-based detection of Indian Sign Language (ISL) static fingerspelling
letters (A–Z), trained on the Roboflow "ISL_Alphabets" dataset. This is
Part 1 (Python data + training). A C++ deployment app for real-time
webcam inference consumes the exported weights from this pipeline.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the dataset

Download the dataset in **YOLOv8** export format from Roboflow Universe:
https://universe.roboflow.com/divya-wn2ga/isl_alphabets-ogisd

Unzip it so you have a structure like:

## 3. Verify dataset structure

```bash
python scripts/verify_dataset.py --data /path/to/isl_alphabets/data.yaml
```

## 4. Train

```bash
python scripts/train.py \
    --data /path/to/isl_alphabets/data.yaml \
    --model yolov8n.pt \
    --epochs 150 \
    --imgsz 640 \
    --batch 16 \
    --patience 25 \
    --project runs/isl_alphabet \
    --name yolov8n_isl
```

## 5. Evaluate + confusion analysis

```bash
python scripts/evaluate.py \
    --data /path/to/isl_alphabets/data.yaml \
    --weights runs/isl_alphabet/yolov8n_isl/weights/best.pt \
    --split test \
    --top-k 10 \
    --save-plot confusion_matrix.png
```

## 6. Inference sanity check (before C++ porting)

```bash
python scripts/infer_sanity_check.py --weights runs/isl_alphabet/yolov8n_isl/weights/best.pt --source 0 --show
```

## Next step (Part 2, not included here)

Export `best.pt` to ONNX for the C++ deployment app:

```bash
yolo export model=runs/isl_alphabet/yolov8n_isl/weights/best.pt format=onnx opset=12 imgsz=640
```