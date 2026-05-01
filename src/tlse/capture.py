import argparse
import time
from pathlib import Path

import cv2


def capture_dataset(
    output_dir: Path,
    num_classes: int,
    images_per_class: int,
    delay_seconds: float,
    camera_index: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)

    try:
        for class_idx in range(num_classes):
            class_dir = output_dir / str(class_idx)
            class_dir.mkdir(parents=True, exist_ok=True)

            print(f"Class {class_idx}: press q when the hand is ready.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Could not read from camera.")
                cv2.imshow("TLSE capture", frame)
                if cv2.waitKey(25) == ord("q"):
                    break

            for image_idx in range(images_per_class):
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow("TLSE capture", frame)
                cv2.waitKey(25)
                cv2.imwrite(str(class_dir / f"{image_idx:04d}.jpg"), frame)
                time.sleep(delay_seconds)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture webcam images into data/raw.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--images-per-class", type=int, default=100)
    parser.add_argument("--delay-seconds", type=float, default=0.25)
    parser.add_argument("--camera-index", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    capture_dataset(
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        images_per_class=args.images_per_class,
        delay_seconds=args.delay_seconds,
        camera_index=args.camera_index,
    )


if __name__ == "__main__":
    main()
