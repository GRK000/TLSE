import argparse
import pickle
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from tlse.features import normalize_landmarks


def build_landmark_dataset(data_dir: Path, min_detection_confidence: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    mp_hands = mp.solutions.hands
    data: list[np.ndarray] = []
    labels: list[str] = []
    class_names: list[str] = []

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
    ) as hands:
        for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
            class_names.append(class_dir.name)
            for image_path in tqdm(sorted(class_dir.iterdir()), desc=class_dir.name):
                img = cv2.imread(str(image_path))
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img_rgb)
                if not result.multi_hand_landmarks:
                    continue

                data.append(normalize_landmarks(result.multi_hand_landmarks[0].landmark))
                labels.append(class_dir.name)

    return np.asarray(data), np.asarray(labels), class_names


def save_dataset(output_path: Path, data: np.ndarray, labels: np.ndarray, class_names: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump({"data": data, "labels": labels, "class_names": class_names}, file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert labeled images into 47-feature landmark vectors.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/landmarks.pickle"))
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data, labels, class_names = build_landmark_dataset(args.data_dir, args.min_detection_confidence)
    save_dataset(args.output, data, labels, class_names)
    print(f"Saved {len(labels)} samples across {len(class_names)} classes to {args.output}")


if __name__ == "__main__":
    main()
