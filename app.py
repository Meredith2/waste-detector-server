import os

model_path = 'yolov8m-worldv2_openvino_model'
if not os.path.isdir(model_path):
    print(f"❌ Model directory '{model_path}' not found!")
    sys.exit(1)

try:
    model = YOLO(model_path, task='detect')
    # Проверяем, что model – это объект, а не строка
    if isinstance(model, str):
        raise Exception(f"YOLO returned string: {model}")
    model.set_classes(WORLD_CLASSES)
    print("✅ OpenVINO model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load OpenVINO model: {e}")
    sys.exit(1)
