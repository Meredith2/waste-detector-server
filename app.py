import os
import sys
import gc
import torch
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Промпты (для world-модели, если используется)
WORLD_CLASSES = [
    "plastic bottle, transparent, with cap, cylindrical shape, PET",
    "plastic bottle, crushed, empty, beverage container, recyclable",
    "plastic bottle, colored, personal care product, squeezable",
    "metal can, aluminum, soda can, cylindrical, with pull tab",
    "tin can, steel, food can, cylindrical, with lid, recyclable",
    "aluminum can, crushed, beverage container, shiny",
    "paper, white sheet, A4, printer paper, flat, recyclable",
    "newspaper, printed text, grayscale pages, folded",
    "magazine, glossy cover, bound pages, colorful",
    "paper, crumpled, waste paper, notebook page",
    "cardboard box, corrugated, brown, shipping box, flattened",
    "cardboard, packaging material, folded, corrugated fiberboard",
    "cardboard sheet, flat, brown paperboard, recyclable",
    "glass bottle, transparent, green glass, wine bottle, long neck",
    "glass bottle, clear, beer bottle, with cap, recyclable",
    "glass jar, transparent, wide mouth, with lid, food container"
]

# Укажите путь к папке с OpenVINO моделью (должна быть в корне репозитория)
MODEL_PATH = 'yolov8n_openvino_model'

try:
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH, task='detect')
    # Если модель поддерживает set_classes (world-версия)
    if hasattr(model, 'set_classes'):
        model.set_classes(WORLD_CLASSES)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

@app.route('/detect', methods=['POST'])
def detect():
    gc.collect()
    if 'image' not in request.files:
        return jsonify({"error": "No image file"}), 400
    file = request.files['image']
    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    img = cv2.resize(frame, (320, 320))
    results = model(img, conf=0.15, iou=0.5, verbose=False)

    if not results or len(results[0].boxes) == 0:
        return jsonify({"error": "No objects detected"}), 404

    boxes = results[0].boxes
    best_idx = int(boxes.conf.argmax())
    confidence = float(boxes.conf[best_idx])
    class_id = int(boxes.cls[best_idx])
    class_name = results[0].names[class_id].lower().strip()
    xyxy = boxes.xyxy[best_idx].tolist()
    return jsonify({
        "class": class_name,
        "confidence": confidence,
        "bbox": xyxy
    })

@app.route('/ping', methods=['GET'])
def ping():
    return "OK", 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=False)
