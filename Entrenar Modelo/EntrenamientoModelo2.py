"""Train a RandomForest model on normalized landmark data."""

import pickle
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data4.pickle")
MODEL_OUTPUT_PATH = Path("model3.p")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def pad_sequences(sequences: Iterable[List[float]]) -> np.ndarray:
    """Pad all sequences with zeros so they share the same length."""
    seq_list = list(sequences)
    max_len = max(len(seq) for seq in seq_list)
    padded = [seq + [0.0] * (max_len - len(seq)) for seq in seq_list]
    return np.asarray(padded)


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as file:
        data_dict = pickle.load(file)
    data = pad_sequences(data_dict["data"])
    labels = np.asarray(data_dict["labels"])
    return data, labels


def train_model(data: np.ndarray, labels: np.ndarray) -> Tuple[RandomForestClassifier, float]:
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=TEST_SIZE,
        shuffle=True,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    accuracy = accuracy_score(y_predict, y_test)
    return model, accuracy


def save_model(model: RandomForestClassifier, path: Path) -> None:
    with path.open("wb") as file:
        pickle.dump({"model": model}, file)


def main() -> None:
    data, labels = load_dataset(DATA_PATH)
    model, accuracy = train_model(data, labels)
    print(f"El {accuracy * 100:.2f}% de las imagenes fueron clasificadas correctamente")
    save_model(model, MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()
