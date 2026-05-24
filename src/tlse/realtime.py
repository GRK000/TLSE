import argparse
from collections import Counter, deque
from pathlib import Path
from typing import Tuple

import numpy as np

from tlse.features import normalize_landmarks


class PredictionSmoother:
    def __init__(self, window_size: int) -> None:
        self.buffer: deque[str] = deque(maxlen=window_size)

    def smooth(self, prediction: str) -> str:
        self.buffer.append(prediction)
        return Counter(self.buffer).most_common(1)[0][0]


class RealTimeSignRecognizer:
    def __init__(self, model_path: Path, camera_index: int, confidence_threshold: float, smoothing_window: int) -> None:
        import mediapipe as mp
        from tlse.evaluate import load_model_artifact

        obj = load_model_artifact(model_path)

        self.model = obj["model"]
        self.classes = obj["class_names"]
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.smoother = PredictionSmoother(smoothing_window)
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        probs = self.model.predict_proba(features.reshape(1, -1))[0]
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])

    def run(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.hands.process(rgb)
                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    pred, conf = self.predict(normalize_landmarks(hand_landmarks.landmark))
                    accepted_pred = pred if conf >= self.confidence_threshold else "?"
                    smoothed_pred = self.smoother.smooth(accepted_pred)
                    cv2.putText(frame, f"{smoothed_pred} {conf:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                cv2.imshow("TLSE real-time recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-time webcam recognition.")
    parser.add_argument("--model", type=Path, default=Path("models/model_depth47.p"))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--smoothing-window", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    RealTimeSignRecognizer(args.model, args.camera_index, args.confidence_threshold, args.smoothing_window).run()


if __name__ == "__main__":
    main()
