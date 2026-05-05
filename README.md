# TLSE - Static Sign/Gesture Alphabet Recognizer

TLSE is a computer-vision experiment for recognizing static hand signs and gesture-like alphabet poses from webcam frames. It is not a full Spanish Sign Language translator: it does not model grammar, sentence context, signer variation, or dynamic movement. The current scope is closer to a real-time sign/gesture alphabet recognizer with MediaPipe hand landmarks and a Random Forest classifier.

## What Is In The Repo

The maintained code lives in `src/tlse`:

| Path | Purpose |
| --- | --- |
| `src/tlse/capture.py` | Capture labeled webcam images into `data/raw/<class>/`. |
| `src/tlse/preprocess.py` | Convert images into normalized 47-feature MediaPipe landmark vectors. |
| `src/tlse/train.py` | Train a Random Forest and write model + metrics artifacts. |
| `src/tlse/realtime.py` | Run webcam inference with landmark overlay, prediction, and confidence. |
| `src/tlse/demo.py` | Create a deterministic synthetic dataset for smoke tests without webcam access. |

Historical scripts are still present in folders such as `Versiones 1/`, `Versiones 2/`, `Entrenar Modelo/`, `Procesamiento del Dataset/`, and `Recopilar Fotos/`. They are useful for portfolio history, but new work should target the package layout above.

## Environment

Python 3.9-3.11 is recommended because MediaPipe wheels may lag behind the newest Python releases.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Dependencies are also declared in `pyproject.toml`, so the environment is freezeable with:

```bash
pip freeze > requirements.lock.txt
```

## Reproducible Demo Without Webcam

The demo dataset is synthetic 47-feature landmark data. It proves the package, CLI, training, model serialization, and metrics path work in a clean environment. It is not a sign-language benchmark.

```bash
python scripts/make_demo_dataset.py --output demo/sample_data/landmarks_demo.pickle
python -m tlse.train --dataset demo/sample_data/landmarks_demo.pickle --model-output models/demo_model.p --report-output reports/demo_metrics.json --n-iter 2 --cv 2 --n-jobs 1 --split-strategy group
```

Expected smoke-test result on the generated demo dataset:

| Dataset | Classes | Samples | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| `demo/sample_data/landmarks_demo.pickle` | 4 | 160 | 1.00 | 1.00 |

The perfect score is expected because the synthetic classes are deliberately separable. Use the real image workflow below for meaningful model quality.

## Real Dataset Workflow

Use a consistent path convention for new runs:

```bash
python -m tlse.capture --output-dir data/raw --num-classes 20 --images-per-class 100
python -m tlse.preprocess --data-dir data/raw --output data/processed/landmarks.pickle
python -m tlse.train --dataset data/processed/landmarks.pickle --model-output models/model_depth47.p --report-output reports/metrics.json
python -m tlse.realtime --model models/model_depth47.p
```

If you already have the legacy local dataset in `Data2/`, preprocess it into the new convention:

```bash
python -m tlse.preprocess --data-dir Data2 --output data/processed/data2_landmarks.pickle
python scripts/evaluate_real.py
```

For webcam captures, prefer grouped splits by session, person, or capture batch. Consecutive frames are often near-duplicates; a random `train_test_split` can leak almost identical frames into both train and test. Store `groups` or `group_ids` in the processed dataset and train with:

```bash
python -m tlse.train --dataset data/processed/landmarks.pickle --split-strategy group
```

## Metrics

`tlse.train` writes `reports/metrics.json` with:

- accuracy and macro F1
- class count and sample count
- train/test split sizes
- best Random Forest hyperparameters
- full `classification_report`
- confusion matrix
- split strategy metadata
- model artifact metadata

Current local benchmark from the legacy processed dataset:

| Dataset/model | Features | Classes | Samples | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Pickles/data.pickle` + `Pickles/model.p` | 42 | 20 | 2000 | 0.993 | 0.993 |

Main observed confusion: the `None` class has lower recall than the signed classes. In the local run, 12 of 100 `None` samples were predicted as another class, especially `K`, `E`, `L`, `C`, `P`, and `U`. Treat this as a legacy in-sample evaluation because the committed pickle does not include the original train/test split metadata.

The newer `Data2/` + `Pickles/model_depth47.p` 47-feature model needs OpenCV and MediaPipe to regenerate comparable processed features. Generate `reports/data2_eval_metrics.json` with the command above before publishing a stronger portfolio claim.

Use separate evaluation commands depending on what is being measured:

```bash
python scripts/evaluate_demo.py
python scripts/evaluate_legacy.py
python scripts/evaluate_real.py
```

`evaluate-demo` is a smoke test. `evaluate-legacy` is an in-sample compatibility check. `evaluate-real` should be used with held-out or group-split real captures. Testing on data already seen during training can report high scores while failing on new users, sessions, or lighting conditions.

To compare model families:

```bash
python scripts/benchmark_models.py --dataset data/processed/landmarks.pickle --split-strategy group
```

The benchmark includes `DummyClassifier`, `RandomForestClassifier`, `SVC`, `KNeighborsClassifier`, and `MLPClassifier`, reporting accuracy, macro F1, fit time, and prediction time.

## Model Artifacts

Saved model files keep compatibility with `.p` and `.pickle`, but they now include metadata:

```python
{
    "model": model,
    "class_names": class_names,
    "metadata": {
        "feature_version": "depth47-v1",
        "python_version": "...",
        "sklearn_version": "...",
        "mediapipe_version": "...",
        "numpy_version": "...",
        "dataset_hash": "...",
        "created_at": "...",
        "metrics_path": "reports/metrics.json"
    }
}
```

Only load pickle artifacts from trusted sources. Python pickle, joblib, and cloudpickle can execute arbitrary code during loading, and scikit-learn does not guarantee reliable loading across different library versions. For a more security-conscious format, evaluate `skops.io`.

## CI And Tests

GitHub Actions runs on every push and pull request via `.github/workflows/ci.yml`:

```bash
python -m compileall src scripts
pytest -q
python scripts/make_demo_dataset.py
python -m tlse.train --dataset demo/sample_data/landmarks_demo.pickle --n-iter 2 --cv 2 --n-jobs 1 --split-strategy group
python scripts/evaluate_demo.py
```

The pytest suite avoids webcam, OpenCV, and MediaPipe execution. It tests demo dataset loading, grouped training, model serialization, metadata, and metrics writing.

## Real-Time Inference

`tlse.realtime` applies a confidence threshold and temporal smoothing:

```bash
python -m tlse.realtime --model models/model_depth47.p --confidence-threshold 0.75 --smoothing-window 8
```

The confidence value comes from `predict_proba`, which is not guaranteed to be calibrated for Random Forests. For experiments that depend on probability quality, train with:

```bash
python -m tlse.train --dataset data/processed/landmarks.pickle --calibrate-probabilities --calibration-method sigmoid
```

## Demo Visuals

For the portfolio README, add a real capture or GIF under `docs/media/` showing:

- webcam frame
- MediaPipe hand landmarks
- predicted class
- confidence score

Recommended final embed:

```markdown
![TLSE webcam demo](docs/media/tlse-demo.gif)
```

## Limitations

- Recognizes isolated static poses; it does not translate full Spanish Sign Language.
- Accuracy depends heavily on the signer, camera, lighting, distance, and class balance.
- MediaPipe can fail with occlusion, motion blur, partial hands, or unusual hand orientations.
- The `None` class is a practical UI state, not a linguistic sign.
- Dynamic signs need temporal modeling and video sequences; the current Random Forest uses single-frame features.
- Future feature variants should test MediaPipe handedness and 3D world landmarks, not only the current normalized 47-feature vector.

## Legacy Map

| Legacy path | New target |
| --- | --- |
| `Recopilar Fotos/2DataSet.py` | `src/tlse/capture.py` |
| `Procesamiento del Dataset/CrearDataSet.py` | `src/tlse/preprocess.py` |
| `Entrenar Modelo/EntrenamientoModelo5.py` | `src/tlse/train.py` |
| `Versiones 1/V1.4/PM_V1.4.9.py` | `src/tlse/realtime.py` |
| `Data2/` | `data/raw/` |
| `Pickles/model_depth47.p` | `models/model_depth47.p` |
