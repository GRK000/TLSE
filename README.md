# TLSE

TLSE is a small Python/ML project for recognizing isolated static hand signs or
gesture-like alphabet poses from webcam frames. It uses MediaPipe hand landmarks
as features and a scikit-learn Random Forest as the maintained baseline model.

It is not a full Spanish Sign Language translator. It does not model grammar,
sentence context, signer variation, or dynamic signs over time.

## Project Status

This repository is a maintainable MVP/demo. The package layout, CLI, tests, and
CI are intended to support reproducible local experiments without requiring a
webcam in automated checks. Real model quality still needs a reproducible
held-out or grouped benchmark on real captures.

## Requirements

- Python 3.9, 3.10, or 3.11
- A webcam only for capture or real-time inference
- OpenCV and MediaPipe for image preprocessing and webcam workflows

Python 3.12 is intentionally not declared as supported because MediaPipe wheels
can lag behind the newest Python releases.

## Installation

Recommended development setup:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`pyproject.toml` is the canonical packaging and dependency source. The
`requirements.txt` file is kept only as a convenience for simple local/demo
installs:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quickstart

Run the synthetic demo workflow in about a minute:

```bash
python -m pip install -e ".[dev]"
python -m tlse demo
python -m tlse evaluate --dataset demo/sample_data/landmarks_demo.pickle --model models/demo_model.p --report-output reports/demo_eval_metrics.json
pytest
```

The demo dataset is synthetic and deliberately separable. A perfect demo score
only proves the package, CLI, training, serialization, and metrics paths work; it
is not a sign-language benchmark.

## CLI Examples

```bash
python -m tlse --help
python -m tlse demo --help
python -m tlse evaluate --help
python -m tlse benchmark --help
```

Generate and train on the synthetic demo dataset:

```bash
python -m tlse demo --dataset demo/sample_data/landmarks_demo.pickle --model-output models/demo_model.p
```

Evaluate a trained model on a processed landmark dataset:

```bash
python -m tlse evaluate --dataset data/processed/landmarks.pickle --model models/model_depth47.p --report-output reports/eval_metrics.json
```

Compare baseline classifiers:

```bash
python -m tlse benchmark --dataset data/processed/landmarks.pickle --split-strategy group --report-output reports/model_benchmark.json
```

The CLI raises clear file errors when the requested dataset or model artifact is
missing.

Legacy-compatible entry points still exist:

```bash
tlse-capture
tlse-preprocess
tlse-train
tlse-realtime
tlse-evaluate
tlse-benchmark
```

## Repository Structure

| Path | Purpose |
| --- | --- |
| `src/tlse/` | Maintained importable package. |
| `src/tlse/cli.py` | Unified argparse CLI used by `python -m tlse`. |
| `src/tlse/demo.py` | Synthetic dataset generator for smoke tests. |
| `src/tlse/train.py` | Dataset loading, splitting, Random Forest training, metrics, artifacts. |
| `src/tlse/evaluate.py` | Model evaluation helpers. |
| `src/tlse/benchmark.py` | Baseline classifier benchmark. |
| `scripts/` | Compatibility wrappers for older workflows. |
| `tests/` | Fast package, CLI, dataset, and demo training tests. |
| `demo/sample_data/` | Committed synthetic demo fixture. |
| `Versiones 1/`, `Versiones 2/`, `Entrenar Modelo/`, `Procesamiento del Dataset/`, `Recopilar Fotos/` | Historical experiment scripts. |

New work should target `src/tlse/`. The historical folders are kept for project
history and should not be treated as the maintained API.

## Data Format

Processed datasets are Python pickle files containing a dictionary:

```python
{
    "data": numpy.ndarray,        # shape: (num_samples, 47)
    "labels": numpy.ndarray,      # shape: (num_samples,)
    "class_names": list[str],     # optional but recommended
    "groups": numpy.ndarray,      # optional session/person/batch ids
}
```

The maintained feature vector is `depth47-v1`: normalized 2D MediaPipe hand
landmarks plus five fingertip depth features.

For webcam captures, prefer grouped splits by session, person, or capture batch:

```bash
python -m tlse.train --dataset data/processed/landmarks.pickle --split-strategy group
```

Random splits can leak near-duplicate consecutive frames between train and test.

## Pickle Security

TLSE currently uses `.pickle`/`.p` artifacts for compatibility with the existing
dataset and model workflows. Only load pickle files you created locally or
received from a trusted source. Python pickle can execute arbitrary code during
loading.

A future dataset format should move to a safer interchange format such as
`.npz`, `.csv`, `.json`, or Parquet. That migration is intentionally not done
here to avoid breaking the current ML workflow.

## Tests And CI

Run the local test suite:

```bash
pytest
```

CI installs the package in editable mode, checks CLI help, compiles Python files,
and runs tests on Python 3.10 and 3.11. It avoids webcam, GPU, credentials, and
local-only datasets.

## Evaluation And Benchmarking

Demo smoke test:

```bash
python -m tlse demo
```

Real processed dataset evaluation:

```bash
python -m tlse evaluate --dataset data/processed/landmarks.pickle --model models/model_depth47.p
```

Classifier benchmark:

```bash
python -m tlse benchmark --dataset data/processed/landmarks.pickle --split-strategy group
```

Current reproducible demo result:

| Dataset | Classes | Samples | Accuracy | Macro F1 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `demo/sample_data/landmarks_demo.pickle` | 4 | 160 | 1.00 | 1.00 | Synthetic smoke test only |

Real-world benchmark status:

| Dataset/model | Result |
| --- | --- |
| Held-out/grouped real captures | Pending reproducible benchmark |

## Real Dataset Workflow

Capture images:

```bash
python -m tlse.capture --output-dir data/raw --num-classes 20 --images-per-class 100
```

Preprocess images into landmark features:

```bash
python -m tlse.preprocess --data-dir data/raw --output data/processed/landmarks.pickle
```

Train:

```bash
python -m tlse.train --dataset data/processed/landmarks.pickle --model-output models/model_depth47.p --report-output reports/metrics.json
```

Run webcam inference:

```bash
python -m tlse.realtime --model models/model_depth47.p --confidence-threshold 0.75 --smoothing-window 8
```

## Known Limitations

- Recognizes isolated static poses, not full Spanish Sign Language.
- Accuracy depends on signer, camera, lighting, distance, framing, and class balance.
- MediaPipe can fail with occlusion, motion blur, partial hands, or unusual hand orientations.
- The `None` class is a practical UI state, not a linguistic sign.
- Dynamic signs need temporal modeling over video sequences.
- Random Forest probabilities from `predict_proba` are not guaranteed to be calibrated.

## Next Steps

- Publish a reproducible grouped benchmark on real captures.
- Add session/person/batch group metadata during preprocessing.
- Consider a safer processed dataset format such as `.npz` or Parquet.
- Add a short real webcam GIF or image under `docs/media/` for portfolio use.
- Evaluate whether dynamic signs need sequence models rather than single-frame features.
