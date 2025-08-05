import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from ultralytics import YOLO
from collections import Counter
from tqdm import tqdm

# Rutas
MODELO_LSTM_PATH = "modelo_lstm_final.keras"
LABELS_NPZ_PATH = "lstm_dataset_final_30frames_10step.npz"
YOLO_MODEL_PATH = "yolov8n-pose.pt"

# Cargar modelo LSTM
modelo_lstm = load_model(MODELO_LSTM_PATH)

# Cargar etiquetas
npz_data = np.load(LABELS_NPZ_PATH)
labels = npz_data["labels"]

# Cargar modelo YOLO Pose
modelo_pose = YOLO(YOLO_MODEL_PATH)


def analizar_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("No se pudo abrir el video.")

    secuencia_actual = []
    predicciones = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Analizando video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = modelo_pose.predict(source=frame, conf=0.1, verbose=False)[0]

        if result.keypoints is not None and result.keypoints.data is not None and result.keypoints.data.shape[0] > 0:
            kpts = result.keypoints.data[0]
            kpts_np = kpts.cpu().numpy().flatten()
            secuencia_actual.append(kpts_np)

            if len(secuencia_actual) == 30:
                X_input = np.expand_dims(np.array(secuencia_actual), axis=0)
                pred = modelo_lstm.predict(X_input, verbose=0)
                predicted_class = str(labels[np.argmax(pred)])  # Convertir a str
                predicciones.append(predicted_class)
                secuencia_actual = secuencia_actual[10:]

        pbar.update(1)

    pbar.close()
    cap.release()

    if not predicciones:
        return {"error": "No se pudieron obtener predicciones."}

    conteo = Counter(predicciones)
    resultados = {
        str(clase): round((count / len(predicciones)) * 100, 2)
        for clase, count in conteo.items()
    }
    mas_detectado = str(max(conteo, key=conteo.get))
    predicciones_str = [str(p) for p in predicciones]

    return {
        "porcentajes": resultados,
        "movimiento_mas_detectado": mas_detectado,
        "total_predicciones": len(predicciones),
        "raw": predicciones_str,
    }
