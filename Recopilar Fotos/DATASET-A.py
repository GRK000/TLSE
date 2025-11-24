"""Capture images for label A."""

from pathlib import Path

import cv2

DATA_DIR = Path("./Data")
LABEL = "A"
IMAGES_PER_LABEL = 100
WINDOW_NAME = "frame"


def capture_label(label: str) -> None:
    label_dir = DATA_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    try:
        print(f"Capturando muestras para la etiqueta {label} (pulsa q para iniciar)...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer la camara.")
                return
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(25) == ord("q"):
                break

        for counter in range(IMAGES_PER_LABEL):
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(25)
            cv2.imwrite(str(label_dir / f"{counter}.jpg"), frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    capture_label(LABEL)


if __name__ == "__main__":
    main()
