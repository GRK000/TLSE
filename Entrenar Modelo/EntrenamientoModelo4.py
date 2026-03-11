import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix

mp_hands = mp.solutions.hands

# ============================================================
# NORMALIZACIÓN PRO (TRASLACIÓN + ROTACIÓN + ESCALA)
# ============================================================
def normalize_landmarks(landmarks):
    # 1) Traslación (WRIST → origen)
    wrist_x = landmarks[0].x
    wrist_y = landmarks[0].y
    pts = np.array([(lm.x - wrist_x, lm.y - wrist_y) for lm in landmarks], dtype=np.float32)

    # 2) Rotación (alinear WRIST → MCP medio con eje X)
    base_x, base_y = pts[9]   # MCP dedo medio
    angle = math.atan2(base_y, base_x)
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)

    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]], dtype=np.float32)

    pts = pts @ R.T

    # 3) Escala (norma RMS: robusta y estable)
    scale = np.sqrt(np.mean(pts[:,0]**2 + pts[:,1]**2))
    if scale < 1e-6:
        scale = 1e-6

    pts /= scale

    # 4) Flatten
    return pts.flatten()



# ============================================================
# CARGA DE IMÁGENES Y GENERACIÓN DE DATASET
# ============================================================
DATA_DIR = "Data2"
data = []
labels = []

hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=1,
                       min_detection_confidence=0.5)

for class_name in sorted(os.listdir(DATA_DIR)):
    folder = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(folder):
        continue

    print(f"Procesando clase: {class_name}")

    for file in os.listdir(folder):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARN] No se pudo leer: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            print(f"[WARN] No se detectó mano en: {img_path}")
            continue

        landmarks = results.multi_hand_landmarks[0].landmark
        vec = normalize_landmarks(landmarks)

        data.append(vec)
        labels.append(class_name)

hands.close()

# ============================================================
# ENTRENAMIENTO
# ============================================================
data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Parámetros para RandomizedSearch
param_dist = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

rf = RandomForestClassifier()

search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    verbose=1,
    n_jobs=-1
)

print("=== INICIANDO BÚSQUEDA DE HIPERPARÁMETROS ===")
search.fit(X_train, y_train)
print("=== BÚSQUEDA FINALIZADA ===")

best_model = search.best_estimator_
print("Mejores hiperparámetros:", search.best_params_)

# ============================================================
# EVALUACIÓN
# ============================================================
y_pred = best_model.predict(X_test)
print("\n=== REPORT POR CLASE ===")
print(classification_report(y_test, y_pred))

print("=== MATRIZ DE CONFUSIÓN ===")
print(confusion_matrix(y_test, y_pred))

# ============================================================
# GUARDAR MODELO
# ============================================================
output = {
    "model": best_model,
    "class_names": sorted(set(labels)),
    "normalization": "translation+rotation+scale"
}

with open("model_escalas.p", "wb") as f:
    pickle.dump(output, f)

print("Modelo guardado como model_escalas.p")