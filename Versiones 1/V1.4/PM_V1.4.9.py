import cv2
import mediapipe as mp
import numpy as np
import math
import pickle
from typing import Tuple


# ============================================================
#           NORMALIZACIÓN DE LANDMARKS (47 FEATURES)
# ============================================================

def normalize_landmarks(landmarks):
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

    return np.concatenate([pts.flatten(), depth_features])


# ============================================================
#                    RECONOCEDOR EN TIEMPO REAL
# ============================================================

class RealTimeSignRecognizer:

    def __init__(self, model_path="Pickles/model_depth47.p"):
        print("[INFO] Cargando modelo:", model_path)
        obj = pickle.load(open(model_path, "rb"))
        self.model = obj["model"]
        self.classes = obj["class_names"]

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def _predict(self, data_aux: np.ndarray) -> Tuple[str, float]:
        data_aux = data_aux.reshape(1, -1)
        probs = self.model.predict_proba(data_aux)[0]
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark
                vec = normalize_landmarks(lm)
                pred, conf = self._predict(vec)

                cv2.putText(frame, f"{pred}  {conf:.2f}",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0,255,0), 3)

                self.mp_draw.draw_landmarks(
                    frame,
                    result.multi_hand_landmarks[0],
                    self.mp_hands.HAND_CONNECTIONS
                )

            cv2.imshow("Real-Time Sign Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# ============================================================
#                        MAIN
# ============================================================

if __name__ == "__main__":
    recognizer = RealTimeSignRecognizer()
    recognizer.run()
