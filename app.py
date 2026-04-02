import os
import torch
import gc  # Модуль для сборки мусора
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

# --- НАЧАЛО ОПТИМИЗАЦИИ ПАМЯТИ ---
# 1. Ограничиваем количество потоков PyTorch одним ядром CPU
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
# 2. Устанавливаем переменную для оптимизации работы с памятью
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
# ---------------------------------

app = Flask(__name__)

# 3. Загружаем самую легкую модель YOLOv8-Nano-World
print("Loading model...")
model = YOLO('yolov8n-worldv2.pt')
# 4. Переводим модель в режим оценки (inference)
model.model.eval()
print("Model loaded and set to eval mode.")

# Установка классов (скопируйте ваш список WORLD_CLASSES)
WORLD_CLASSES = [
    "plastic bottle, transparent, with cap, cylindrical shape, PET",
    "plastic bottle, crushed, empty, beverage container, recyclable",
    # ... остальные ваши промпты ...
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

    # 5. Уменьшаем размер входного изображения (пример: до 320x320)
    # Это сильно снижает нагрузку на память
    img = cv2.resize(frame, (320, 320))
    results = model(img, conf=0.15, iou=0.5, verbose=False)

    if not results or len(results[0].boxes) == 0:
        return jsonify({"error": "No objects detected"}), 404

    boxes = results[0].boxes
    best_idx = int(boxes.conf.argmax())
    confidence = float(boxes.conf[best_idx])
    class_id = int(boxes.cls[best_idx])
    class_name = results[0].names[class_id].lower().strip()
    # Координаты нужно будет пересчитать, если вы изменили размер img
    xyxy = boxes.xyxy[best_idx].tolist()
    return jsonify({
        "class": class_name,
        "confidence": confidence,
        "bbox": xyxy
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
