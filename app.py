import os
import torch
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Загрузка локальной модели
model = YOLO('yolov8n.pt')  # или yolov8n.pt

# Если используете world-версию, установите классы
if 'world' in str(type(model)):
    WORLD_CLASSES = [
        "plastic bottle", "metal can", "paper", "cardboard", "glass bottle"
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

    # Уменьшаем размер для экономии памяти
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=False)
