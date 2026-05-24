# TLSE Studio Product Vision

TLSE Studio is a local product experience for real-time static hand sign recognition, dataset review, model comparison, and guided practice.

It is positioned as an accessible computer-vision studio, not as a full Spanish Sign Language translator. The current scope is isolated static hand poses represented by MediaPipe-style landmarks and evaluated with transparent confidence and dataset quality signals.

## Target Users

- Students and makers exploring sign recognition with computer vision.
- Educators who need a local tool to demonstrate static hand-pose recognition.
- Developers improving the TLSE model, dataset, and evaluation workflow.
- Accessibility-focused teams evaluating early-stage assistive interfaces.

## Use Cases

- Run local live recognition with visible confidence and demo fallback.
- Inspect dataset balance before training or reporting model quality.
- Compare model artifacts and metrics across experiments.
- Practice selected static signs with feedback that avoids false certainty.
- Produce credible project documentation for reviews, demos, and portfolios.

## Non Goals

- Full LSE translation.
- Grammar, syntax, discourse, or linguistic context modeling.
- Dynamic signs requiring temporal video understanding.
- Cloud-hosted inference or remote camera streaming.
- Medical, legal, or production accessibility guarantees.

## Product Principles

- Always make uncertainty visible.
- Use demo mode honestly when model, camera, or dataset access is missing.
- Prefer local-first processing and clear privacy language.
- Treat dataset quality as part of the product, not an afterthought.
- Keep CLI and Python workflows stable while adding the studio layer.

## Roadmap

1. Local FastAPI runtime with demo-safe responses.
2. TLSE Studio web UI with dashboard, live recognition, practice, dataset, and model views.
3. Real camera bridge and browser capture pipeline.
4. Safer dataset format and richer session/person metadata.
5. Grouped evaluation reports and model promotion workflow.
6. Optional Tauri desktop packaging once the runtime/UI contract stabilizes.
