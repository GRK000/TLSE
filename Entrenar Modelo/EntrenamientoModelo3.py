"""
entrenamientoModelo.py — Entrenamiento profesional del modelo de signos

- Carga data.pickleNew (data, labels, metadata)
- Split train/test estratificado
- RandomForest + búsqueda de hiperparámetros (RandomizedSearchCV)
- Métricas en consola
- Guarda modelo en Pickles/model.p en formato:
    {"model": best_model, "metadata": {...}}

Compatible con PM_V1.4.8.py, que espera un dict con clave "model".
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV


# ==========================
# CONFIGURACIÓN
# ==========================

# Fichero generado por tu DATASET.py
DATA_PATH = Path("data.pickleNew")

# Ruta donde PM_V1.4.8.py busca el modelo:
# ROOT / "Pickles" / "model.p"
# Si ejecutas este script desde la carpeta TR, esto funciona.
MODEL_OUTPUT_PATH = Path("Pickles/modelNew.p")

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_JOBS = -1  # Usa todos los núcleos disponibles (o cámbialo si tu portátil llora)


# ==========================
# UTILIDADES
# ==========================

def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Carga el dataset desde un pickle.

    Soporta dos formatos:
    - dict con "data", "labels", "metadata"
    - tu formato antiguo: (data, labels)
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {path}")

    with path.open("rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "data" in obj and "labels" in obj:
        data = obj["data"]
        labels = obj["labels"]
        metadata = obj.get("metadata", {})
    elif isinstance(obj, (list, tuple)) and len(obj) == 2:
        data, labels = obj
        metadata = {}
    else:
        raise ValueError("Formato de data.pickle no reconocido.")

    X = np.asarray(data, dtype=np.float32)
    y = np.asarray(labels)

    print("=== INFO DATASET ===")
    print(f"Shape X: {X.shape}  (samples x features)")
    print(f"Shape y: {y.shape}")
    if metadata:
        classes = metadata.get("classes")
        samples_per_class = metadata.get("samples_per_class")
        total = metadata.get("total_samples")
        if classes is not None:
            print(f"Clases: {classes}")
        if samples_per_class is not None:
            print("Muestras por clase:")
            for c in classes:
                print(f"  {c}: {samples_per_class.get(c, '??')}")
        if total is not None:
            print(f"Total muestras: {total}")
    else:
        print("No hay metadata disponible (formato antiguo).")
    print("====================\n")

    return X, y, metadata


def build_base_model() -> RandomForestClassifier:
    """
    Crea un RandomForest base decente para gestos:
    - muchos árboles
    - profundidad sin limitar (de entrada)
    - class_weight balanced por si el dataset no es perfecto
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )


def hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """
    Búsqueda aleatoria de hiperparámetros para RandomForest.

    No nos vamos a tirar 3 horas de GridSearch, pero tampoco vamos a dejarlo
    en los valores por defecto como animales.
    """

    base_model = RandomForestClassifier(
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )

    param_dist = {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,               # 20 combinaciones aleatorias
        scoring="accuracy",
        cv=3,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    print("=== INICIANDO BÚSQUEDA DE HIPERPARÁMETROS (RandomizedSearchCV) ===")
    search.fit(X_train, y_train)
    print("=== BÚSQUEDA TERMINADA ===")
    print(f"Mejores hiperparámetros: {search.best_params_}")
    print(f"Mejor accuracy (CV): {search.best_score_:.4f}\n")

    return search.best_estimator_


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Split train/test, búsqueda de hiperparámetros, evaluación y devolución
    del mejor modelo + métricas.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # MUY importante con clases equilibradas
    )

    print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    # 1) Búsqueda de hiperparámetros
    best_model = hyperparameter_search(X_train, y_train)

    # 2) Entrenar modelo final con los mejores hiperparámetros
    best_model.fit(X_train, y_train)

    # 3) Evaluación
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== RESULTADOS EN TEST ===")
    print(f"Accuracy global: {acc * 100:.2f}%")

    print("\n=== REPORT POR CLASE ===")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
    print("=== MATRIZ DE CONFUSIÓN (filas = reales, columnas = predichas) ===")
    print("Clases (orden columnas/filas):")
    print(best_model.classes_)
    print(cm)
    print("===========================================\n")

    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classes": list(best_model.classes_),
    }

    return best_model, metrics


def save_model(
    model: RandomForestClassifier,
    metadata: Dict[str, Any],
    path: Path,
) -> None:
    """
    Guarda el modelo junto con metadatos en un pickle compatible con PM_V1.4.8.py.

    Estructura:
        {
            "model": model,
            "metadata": {
                ... lo que quieras ...
            }
        }
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    to_save = {
        "model": model,
        "metadata": metadata,
    }

    with path.open("wb") as f:
        pickle.dump(to_save, f)

    print(f"Modelo guardado en: {path.resolve()}")


# ==========================
# MAIN
# ==========================

def main() -> None:
    print("=== ENTRENAMIENTO MODELO LSE — RandomForest ===")

    # 1) Cargar dataset
    X, y, ds_metadata = load_dataset(DATA_PATH)

    # 2) Entrenar y evaluar
    model, metrics = train_and_evaluate(X, y)

    # 3) Combinar metadatos del dataset + métricas
    full_metadata: Dict[str, Any] = {
        "dataset": ds_metadata,
        "training": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "accuracy": metrics["accuracy"],
            "classes": metrics["classes"],
        },
    }

    # 4) Guardar modelo
    save_model(model, full_metadata, MODEL_OUTPUT_PATH)

    print("\nEntrenamiento completado.")
    print(
        f"Accuracy final en test: {metrics['accuracy'] * 100:.2f}% "
        f"con {len(metrics['classes'])} clases."
    )


if __name__ == "__main__":
    main()
