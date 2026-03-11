import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


# ============================================================
#           NORMALIZACIÓN DE LANDMARKS (47 FEATURES)
# ============================================================

def normalize_landmarks(landmarks):
    """
    Normaliza los 21 landmarks de MediaPipe: traslación, rotación, escala L2
    y depth features (Z relativas al wrist y normalizadas con la misma escala XY).
    Devuelve un vector de 47 features: 42 XY + 5 Z tips.
    """

    wrist_x = landmarks[0].x
    wrist_y = landmarks[0].y
    wrist_z = landmarks[0].z

    pts = np.array([(lm.x - wrist_x, lm.y - wrist_y) for lm in landmarks], dtype=np.float32)
    zs = np.array([lm.z - wrist_z for lm in landmarks], dtype=np.float32)

    base_x, base_y = pts[9]
    angle = math.atan2(base_y, base_x)

    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]], dtype=np.float32)

    pts = pts @ R.T

    scale = np.sqrt(np.mean(pts[:,0]**2 + pts[:,1]**2))
    if scale < 1e-6:
        scale = 1e-6

    pts /= scale

    tip_idx = [4, 8, 12, 16, 20]
    depth_features = (zs[tip_idx] / scale).astype(np.float32)

    flat_xy = pts.flatten()
    return np.concatenate([flat_xy, depth_features])


# ============================================================
#                    CONFIGURACIÓN
# ============================================================

DATA_DIR = "Data2"
SAVE_PATH = "Pickles/model_depth47.p"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=1,
                       min_detection_confidence=0.5)


# ============================================================
#           CARGA Y PROCESADO DEL DATASET
# ============================================================

data = []
labels = []
class_names = []

print("=== Procesando dataset ===")

for class_name in sorted(os.listdir(DATA_DIR)):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    class_names.append(class_name)
    print(f"Procesando clase: {class_name}")

    for file_name in tqdm(os.listdir(class_path)):
        file_path = os.path.join(class_path, file_name)

        img = cv2.imread(file_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if not result.multi_hand_landmarks:
            continue

        lm = result.multi_hand_landmarks[0].landmark
        vec = normalize_landmarks(lm)

        data.append(vec)
        labels.append(class_name)

data = np.array(data)
labels = np.array(labels)

print("\nDataset cargado:")
print("Features:", data.shape)
print("Clases:", len(class_names))


# ============================================================
#                TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.20, random_state=42, stratify=labels
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")


# ============================================================
#                    RANDOM FOREST + SEARCH
# ============================================================

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

print("\n=== INICIANDO BÚSQUEDA DE HIPERPARÁMETROS ===")

rs = RandomizedSearchCV(
    RandomForestClassifier(),
    param_grid,
    n_iter=20,
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

rs.fit(X_train, y_train)

print("Mejores hiperparámetros:", rs.best_params_)

model = rs.best_estimator_


# ============================================================
#            EVALUACIÓN FINAL
# ============================================================

print("\n=== REPORT POR CLASE ===")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("=== MATRIZ DE CONFUSIÓN ===")
print(confusion_matrix(y_test, y_pred))


# ============================================================
#                    GUARDAR MODELO
# ============================================================

pickle.dump(
    {"model": model, "class_names": class_names},
    open(SAVE_PATH, "wb")
)

print(f"\nModelo guardado como {SAVE_PATH}")