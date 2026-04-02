from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Загружаем модель (можно заменить на yolov8n-worldv2.pt для скорости)
model = YOLO('yolov8l-worldv2.pt')

# ⚠️ ВАШ СПИСОК ПРОМПТОВ (скопируйте из waste_classifier.py)
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

model.set_classes(WORLD_CLASSES)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file"}), 400
    file = request.files['image']
    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    results = model(frame, conf=0.15, iou=0.5)
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
