"""Dataset builder with sorted/normalized landmarks."""

import pickle
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import mediapipe as mp

DATA_DIR = Path("./Data2")
OUTPUT_PATH = Path("data4.pickle")
MIN_DETECTION_CONFIDENCE = 0.3
MAX_POINTS = 21

mp_hands = mp.solutions.hands


def normalize_landmarks(xs: Sequence[float], ys: Sequence[float]) -> List[float]:
    """Sort by x, then normalize into a flat list [x1,y1,x2,y2,...]."""
    xy_sorted = sorted(zip(xs, ys), key=lambda point: point[0])
    x_sorted, y_sorted = zip(*xy_sorted)

    x_min, x_max = min(x_sorted), max(x_sorted)
    y_min, y_max = min(y_sorted), max(y_sorted)
    x_norm = [(x - x_min) / (x_max - x_min) if x_max != x_min else 0.0 for x in x_sorted]
    y_norm = [(y - y_min) / (y_max - y_min) if y_max != y_min else 0.0 for y in y_sorted]

    data_aux: List[float] = []
    for x_val, y_val in zip(x_norm, y_norm):
        data_aux.extend([x_val, y_val])
    return data_aux


def extract_landmarks(data_dir: Path) -> Tuple[List[List[float]], List[str]]:
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

                    if len(xs) >= MAX_POINTS and len(ys) >= MAX_POINTS:
                        data.append(normalize_landmarks(xs, ys))
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
