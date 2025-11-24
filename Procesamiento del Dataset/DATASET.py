"""Dataset landmark extraction."""

import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp

DATA_DIR = Path("./Data")
OUTPUT_PATH = Path("data.pickle")
MIN_DETECTION_CONFIDENCE = 0.3

mp_hands = mp.solutions.hands


def extract_landmarks(data_dir: Path) -> Tuple[List[List[float]], List[str]]:
    """Return normalized landmark coordinates and labels for every image in subfolders."""
    data: List[List[float]] = []
    labels: List[str] = []

    with mp_hands.Hands(static_image_mode=True, min_detection_confidence=MIN_DETECTION_CONFIDENCE) as hands:
        for label_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            for img_path in sorted(label_dir.iterdir()):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                if not results.multi_hand_landmarks:
                    continue

                for hand_landmarks in results.multi_hand_landmarks:
                    xs = [landmark.x for landmark in hand_landmarks.landmark]
                    ys = [landmark.y for landmark in hand_landmarks.landmark]
                    x_min, y_min = min(xs), min(ys)

                    data_aux: List[float] = []
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - x_min)
                        data_aux.append(landmark.y - y_min)

                    data.append(data_aux)
                    labels.append(label_dir.name)

    return data, labels


def save_pickle(data: List[List[float]], labels: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump({"data": data, "labels": labels}, file)


def main() -> None:
    data, labels = extract_landmarks(DATA_DIR)
    save_pickle(data, labels, OUTPUT_PATH)


if __name__ == "__main__":
    main()
