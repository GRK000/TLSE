import threading
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
import math

import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3 as tts
from pynput import keyboard as kb


# ==========================
# CONFIGURACIÓN GLOBAL
# ==========================

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "Pickles" / "model_escalas.p"   # Igual que en tu versión anterior :contentReference[oaicite:0]{index=0}
# MODEL_PATH = ROOT / "Pickles" / "model_normalized.p"


CAMERA_INDEX = 0
WINDOW_NAME = "Traductor LSE - Realtime"

# MediaPipe config
MIN_DETECTION_CONFIDENCE = 0.35
MIN_TRACKING_CONFIDENCE = 0.6
MAX_NUM_HANDS = 1

# Smoothing / debouncing
PREDICTION_WINDOW_SIZE = 5      # Nº de frames en la ventana deslizante
STABILITY_FRAMES = 2            # Nº mínimo de frames iguales para aceptar cambio de letra
MIN_CONFIDENCE = 0.6            # Umbral mínimo de probabilidad para considerar una predicción

# Diccionario etiquetas → caracteres
LABELS_DICT: Dict[str, str] = {
    "A": "A", "B": "B", "C": "C", "D": "D",
    "E": "E", "F": "F", "I": "I", "K": "K",
    "L": "L", "M": "M", "N": "N", "None": " ",
    "O": "O", "P": "P", "Q": "Q", "R": "R",
    "S": "S", "T": "T", "U": "U", "Y": "Y",
}


# ==========================
# UTILIDADES MATEMÁTICAS
# ==========================

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

    # 3) Escala 
    scale = np.sqrt(np.mean(pts[:,0]**2 + pts[:,1]**2))
    if scale < 1e-6:
        scale = 1e-6

    pts /= scale

    # 4) Flatten
    return pts.flatten()


# ==========================
# CLASE PRINCIPAL
# ==========================

class RealTimeSignRecognizer:
    def __init__(self) -> None:
        # Modelo
        self.model = self._load_model(MODEL_PATH)
        self.classes = self.model.classes_

        # Estado compartido
        self.current_char: Optional[str] = None       # Letra “estable” actual
        self.current_confidence: float = 0.0
        self.prediction_buffer: Deque[str] = deque(maxlen=PREDICTION_WINDOW_SIZE)

        self.text: str = ""                           # Frase construida
        self.text_lock = threading.Lock()             # Para acceso seguro

        self.stop_event = threading.Event()           # Señal de parada global

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # TTS
        self.tts_engine = tts.init()
        rate = self.tts_engine.getProperty("rate")
        self.tts_engine.setProperty("rate", rate - 100)

    @staticmethod
    def _load_model(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {path}")
        with path.open("rb") as f:
            model_dicc = pickle.load(f)
        return model_dicc["model"]

    # ==========================
    # BUCLE PRINCIPAL DE VÍDEO
    # ==========================

    def run(self) -> None:
        """
        Inicia:
          - un hilo para el teclado
          - el bucle principal de vídeo (en el hilo actual)
        """
        # Hilo teclado
        keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        keyboard_thread.start()

        # Captura de vídeo
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: no se pudo abrir la cámara.")
            return

        hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

        print(
            "\nTeclas funcionales:\n"
            "  S  → Añadir letra actual a la frase\n"
            "  B  → Borrar el último carácter\n"
            "  T  → Borrar toda la frase\n"
            "  L  → Leer la frase en voz alta\n"
            "  ESC→ Salir (y leer la frase)\n"
        )

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("Error al capturar fotograma de la cámara.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    # Solo usamos la primera mano detectada
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # Dibujamos mano
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    # Normalizamos y predecimos
                    data_aux = normalize_landmarks(hand_landmarks.landmark)
                    top_label, top_conf = self._predict(data_aux)

                    # Actualizamos suavizado temporal
                    self._update_smoothing(top_label, top_conf)

                    # Dibujamos bbox y letra
                    self._draw_prediction(frame, hand_landmarks, top_label, top_conf)

                # Pintar texto actual en la parte inferior
                self._draw_text(frame)

                cv2.imshow(WINDOW_NAME, frame)

                # Pequeño waitKey para permitir cerrar ventana con botón X
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # Te dejo una salida rápida con 'q', por si acaso
                    self.stop_event.set()

        finally:
            # Liberamos recursos SIEMPRE
            hands.close()
            cap.release()
            cv2.destroyAllWindows()

            # Leer frase al salir
            with self.text_lock:
                final_text = self.text

            if final_text.strip():
                print("\nFrase final:", final_text)
                self._speak(final_text)

    # ==========================
    # PREDICCIÓN Y SUAVIZADO
    # ==========================

    def _predict(self, features: np.ndarray) -> Tuple[str, float]:
        # Asegurar batch
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Predecir probabilidades
        probs = self.model.predict_proba(features)[0]

        # Top-1
        idx = int(np.argmax(probs))

        # Seguridad: si por lo que sea hay mismatch de clases
        if idx >= len(self.classes):
            return "UNKNOWN", 0.0

        return self.classes[idx], float(probs[idx])


    def _update_smoothing(self, label: str, confidence: float) -> None:
        """
        Suavizado temporal + debouncing.
        - Añade label a la ventana
        - Calcula la moda
        - Acepta cambio solo si:
          * el label con más frecuencia se repite ≥ STABILITY_FRAMES
          * la confianza de la predicción actual ≥ MIN_CONFIDENCE
        """
        if confidence < MIN_CONFIDENCE:
            # aunque no se añada al buffer, sí actualizamos la letra temporal (actual)
            self.current_char = LABELS_DICT.get(label, " ")
            self.current_confidence = confidence
            return

        self.prediction_buffer.append(label)

        # Contamos frecuencia en ventana
        counts = Counter(self.prediction_buffer)
        most_common_label, freq = counts.most_common(1)[0]

        if freq >= STABILITY_FRAMES:
            # Aceptamos cambio de letra estable
            self.current_char = LABELS_DICT.get(most_common_label, " ")
            self.current_confidence = confidence

    # ==========================
    # DIBUJO EN PANTALLA
    # ==========================

    def _draw_prediction(
        self,
        frame: np.ndarray,
        hand_landmarks,
        top_label: str,
        top_conf: float,
    ) -> None:
        h, w, _ = frame.shape
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]

        x1 = int(min(xs) * w) - 20
        y1 = int(min(ys) * h) - 20
        x2 = int(max(xs) * w) + 20
        y2 = int(max(ys) * h) + 20

        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

        char_display = LABELS_DICT.get(top_label, " ")
        cv2.putText(
            frame,
            f"{char_display} ({top_conf*100:.1f}%)",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    def _draw_text(self, frame: np.ndarray) -> None:
        """
        Pinta la frase actual en la parte inferior de la imagen.
        """
        h, w, _ = frame.shape
        with self.text_lock:
            text_display = self.text

        # Fondo semitransparente para el texto
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(
            frame,
            text_display[-60:],  # mostramos solo las últimas ~60 chars
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # ==========================
    # TECLADO / INPUT USUARIO
    # ==========================

    def _keyboard_listener(self) -> None:
        """
        Hilo separado para escuchar teclas.
        Un solo Listener, un solo hilo. Nada de spawnear listeners en bucle como en tu versión anterior :contentReference[oaicite:1]{index=1}
        """
        with kb.Listener(on_press=self._on_key_press) as listener:
            listener.join()

    def _on_key_press(self, key) -> None:
        try:
            if key == kb.Key.esc:
                # Salir: marcamos evento y dejamos que el bucle principal cierre todo.
                self.stop_event.set()
                return

            if not hasattr(key, "char"):
                return  # teclas raras que no nos interesan

            ch = key.char.lower()

            if ch == "s":
                # Añadir letra actual
                if self.current_char is None:
                    return
                with self.text_lock:
                    self.text += self.current_char
                print(f"Letra añadida: {self.current_char}")

            elif ch == "b":
                # Borrar último carácter
                with self.text_lock:
                    if self.text:
                        borrado = self.text[-1]
                        self.text = self.text[:-1]
                        print(f"Último carácter eliminado: {borrado}")

            elif ch == "t":
                # Borrar todo
                with self.text_lock:
                    self.text = ""
                    print("Frase borrada.")

            elif ch == "l":
                # Leer frase actual
                with self.text_lock:
                    frase = self.text
                if frase.strip():
                    print("Leyendo frase:", frase)
                    self._speak(frase)

        except AttributeError:
            # Algunas teclas especiales saltan aquí, las ignoramos.
            pass

    # ==========================
    # TEXT-TO-SPEECH
    # ==========================

    def _speak(self, text: str) -> None:
        """
        Reproduce la frase en voz alta. Bloqueante a propósito, para no
        tener diez audios solapados si te emocionas dándole a 'L'.
        """
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()


# ==========================
# PUNTO DE ENTRADA
# ==========================

if __name__ == "__main__":
    recognizer = RealTimeSignRecognizer()
    recognizer.run()
