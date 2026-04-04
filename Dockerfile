FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей (включая curl и unzip)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Скачивание модели из GitHub Release
ARG MODEL_URL=https://github.com/Meredith2/waste-detector-server/releases/download/v1.0-s-model/yolov8s-worldv2_openvino_model.zip
RUN curl -L -o model.zip $MODEL_URL && unzip model.zip && rm model.zip

# Установка Python зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
