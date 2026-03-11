# TLSE – Sign-language translator research stack

This project packages the end-to-end journey of a Spanish sign-language recognizer: controlled data capture, landmark preprocessing, classifier training, and a real-time interface that streams character-by-character translations to the screen (and optionally to a speaker). The repository is deliberately versioned so you can walk a recruiter through how each iteration improved accuracy, latency, and user feedback.

---

## Why this matters

- **Human-centric accessibility proof of concept** – Transforms webcam input into character predictions while logging the confidence, supporting both visual and audible cues (via `pyttsx3`).
- **Reproducible ML workflow** – Every script is tracked: from capturing datasets, through MediaPipe landmark extraction, to Random Forest training with hyperparameter tuning.
- **Story-ready version history** – The `Versiones 1` folder documents the incremental engineering decisions that make a compelling interview narrative.

## Repository layout at a glance

| Area | What it holds |
| --- | --- |
| `Recopilar Fotos/2DataSet.py` | Webcam tool for capturing letter samples (20 folders × 100 frames). |
| `Procesamiento del Dataset/CrearDataSet.py` | MediaPipe Hands parsing + normalization that writes `data.pickle`. |
| `Entrenar Modelo/EntrenamientoModelo5.py` | Latest training pipeline: 47-feature normalization, stratified split, RandomizedSearchCV, and `Pickles/model_depth47.p`. |
| `Versiones 1/V1.4/PM_V1.4.9.py` | Current real-time recognizer that loads the pickle and overlays prediction + confidence. |
| `Pickles/` | Stored models (`model_depth47.p`, legacy `model.p`, and future retrained versions). |

The root also exposes experiments (`Entrenar Modelo/EntrenamientoModelo[2-4].py`), dataset archives, and notebooks in future releases.

## Version highlights for storytelling

- **V1.1 – Baseline recognition demo**: single-frame inference, bounding box overlay, and fps counter. Showcases initial feasibility with a static `model.p`.
- **V1.2 – Voice + sentence builder**: adds `pyttsx3`, an “s” hotkey to append recognized characters to a running phrase, and smarter handling of `None` gestures for spacing.
- **V1.3 – Edit controls & debouncing**: includes “b” for backspace, guards against duplicate predictions, and keeps the sentence output tidy for longer expressions.
- **V1.4 – Feature-engineered classifier**: introduces `Data2`, 47-feature normalization (translation + rotation + scaling + depth), and training via `RandomizedSearchCV` over Random Forests. `PM_V1.4.9.py` demonstrates the final recognizer UI.
- **V1.5 – Async controls & retraining paths**: runs keyboard listeners in parallel threads, records data for future finetuning, and saves `modelo_reentrenado.pkl` after live sessions.

## Getting started (recruiter-friendly demo path)

1. **Install dependencies** (Python 3.9+): `pip install opencv-python mediapipe numpy scikit-learn pyttsx3 tqdm pynput`.
2. **Collect labeled samples**: run `python "Recopilar Fotos/2DataSet.py"` and follow the prompts (`q` to start each letter, `q` again to stop the capture session).
3. **Generate landmark dataset**: `python "Procesamiento del Dataset/CrearDataSet.py"` to produce `data.pickle` from the captured folders.
4. **Train the model**: `python "Entrenar Modelo/EntrenamientoModelo5.py"` builds the Random Forest, prints classification reports, and saves `Pickles/model_depth47.p`.
5. **Launch the translator**: `python "Versiones 1/V1.4/PM_V1.4.9.py"` to open a live window that visualizes predictions and confidence on each frame. Press `q` or close the window to exit.

## Technical takeaways

- **Normalized geometries** – Most recent scripts translate the hand to the wrist origin, rotate to a stable base, scale uniformly, and append depth cues per fingertip.
- **Structured experimentation** – Each version folder documents what was added; recruiters can step through the history to see how UI, controls, and model reliability improved.
- **Production-ready model training** – `EntrenamientoModelo5.py` runs `RandomizedSearchCV` with sensible hyperparameter bounds and reports a confusion matrix, highlighting the candidate’s ability to benchmark ML models.

## Notes

- Large pickles (`*.p`) and dataset folders are excluded by `.gitignore`.
- The real-time script assumes the latest pickle lives at `Pickles/model_depth47.p`; regenerate it after any data change.
- The capture + training loop is reproducible in any environment with a webcam and Python 3.9+; you can use it to demo rapid prototyping skills in interviews.

Let me know if you want a shorter “elevator pitch” version of this README or a version in Spanish for collaborators.
