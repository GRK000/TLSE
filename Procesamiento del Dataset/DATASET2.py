"""
DATASET.py — Extracción profesional de landmarks para entrenamiento

Genera un archivo data.pickle con:
    - data: lista de vectores de landmarks normalizados
    - labels: lista de strings con la clase correspondiente
    - metadata: clases disponibles, nº muestras por clase, etc.

Compatible y consistente con el sistema de predicción en tiempo real.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np


# ==========================
# CONFIGURACIÓN
# ==========================

DATA_DIR = Path(r"./Data2")                 # Carpeta con subcarpetas por clase
OUTPUT_PATH = Path("./data.pickleNew")      # Salida del dataset
MIN_DETECTION_CONFIDENCE = 0.5              # Mejor que 0.3 para calidad de detección
STATIC_MODE = True                          # Para imágenes estáticas: True


# ==========================
# SETUP MEDIAPIPE
# ==========================

mp_hands = mp.solutions.hands


# ==========================
# NORMALIZACIÓN (MISMA QUE EN RUNTIME)
# ==========================

def normalize_landmarks(hand_landmarks) -> List[float]:
    """
    Normalización EXACTAMENTE igual que en el runtime (PM_V1.4.8.py).
    - Traslación al origen restando (x_min, y_min)
    - SIN normalización por escala (porque el modelo actual fue entrenado así)

    Devuelve un vector de 42 floats (21 landmarks × 2 coords).
    """

    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    x_min, y_min = min(xs), min(ys)

    data_aux: List[float] = []
    for x, y in zip(xs, ys):
        data_aux.append(x - x_min)
        data_aux.append(y - y_min)

    return data_aux


# ==========================
# FUNCIÓN PRINCIPAL DE EXTRACCIÓN
# ==========================

def extract_dataset(path: Path) -> Tuple[List[List[float]], List[str], Dict]:
    """
    Extrae landmarks de todas las imágenes en subdirectorios.
    Devuelve:
        - data: lista de vectores
        - labels: lista de strings
        - metadata: dict con clases y número de muestras
    """

    if not path.exists():
        raise FileNotFoundError(f"La carpeta {path} no existe.")

    data: List[List[float]] = []
    labels: List[str] = []
    metadata = {
        "classes": [],
        "samples_per_class": {},
        "total_samples": 0,
    }

    with mp_hands.Hands(
        static_image_mode=STATIC_MODE,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE
    ) as hands:

        # Iteramos sobre carpetas de clase
        for label_dir in sorted(p for p in path.iterdir() if p.is_dir()):
            label = label_dir.name
            metadata["classes"].append(label)
            metadata["samples_per_class"][label] = 0

            print(f"\nProcesando clase: {label}")

            image_paths = sorted(label_dir.iterdir())

            for img_path in image_paths:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"[WARN] No se pudo leer: {img_path}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if not results.multi_hand_landmarks:
                    print(f"[WARN] No se detectó mano en: {img_path}")
                    continue

                # Solo usamos la primera mano detectada
                hand_landmarks = results.multi_hand_landmarks[0]

                vector = normalize_landmarks(hand_landmarks)
                if len(vector) != 42:
                    print(f"[ERROR] Vector incorrecto en {img_path}. Se ignora.")
                    continue

                data.append(vector)
                labels.append(label)
                metadata["samples_per_class"][label] += 1
                metadata["total_samples"] += 1

    print("\nResumen del dataset:")
    print("----------------------")
    print(f"Clases encontradas: {metadata['classes']}")
    print("Muestras por clase:")
    for lbl, count in metadata["samples_per_class"].items():
        print(f"   {lbl}: {count}")
    print(f"Total muestras: {metadata['total_samples']}")

    return data, labels, metadata


# ==========================
# GUARDAR PICKLE
# ==========================

def save_pickle(
    data: List[List[float]],
    labels: List[str],
    metadata: Dict,
    output_path: Path
) -> None:
    """
    Guarda el dataset en formato pickle con estructura profesional.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = {
        "data": data,
        "labels": labels,
        "metadata": metadata,
    }

    with output_path.open("wb") as f:
        pickle.dump(dataset, f)

    print(f"\nDataset guardado en: {output_path.absolute()}")
    print("Contenido guardado: data, labels, metadata")


# ==========================
# MAIN
# ==========================

def main() -> None:
    print("=== EXTRACCIÓN DE DATASET ===")
    data, labels, metadata = extract_dataset(DATA_DIR)
    print("\nGuardando dataset...")
    save_pickle(data, labels, metadata, OUTPUT_PATH)


if __name__ == "__main__":
    main()
