"""Capture dataset images by subfolder labels."""

from pathlib import Path

import cv2

DATA_DIR = Path("./BORRAR")
NUM_FOLDERS = 20
IMAGES_PER_FOLDER = 100
WINDOW_NAME = "frame"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def capture_dataset() -> None:
    ensure_dir(DATA_DIR)
    cap = cv2.VideoCapture(0)

    for folder_idx in range(NUM_FOLDERS):
        folder_path = DATA_DIR / str(folder_idx)
        ensure_dir(folder_path)

        print(f"Capturando muestras para la carpeta {folder_idx} (pulsa q para empezar)...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(25) == ord("q"):
                break

        for counter in range(IMAGES_PER_FOLDER):
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(25)
            cv2.imwrite(str(folder_path / f"{counter}.jpg"), frame)

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    capture_dataset()


if __name__ == "__main__":
    main()
