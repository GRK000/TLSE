# Architecture

TLSE Studio is layered around the existing Python package. The CLI remains the maintained ML workflow, while the new API and web app provide a product interface on top.

```text
apps/web React UI
  | REST: health, metrics, dataset, models, predict
  | WebSocket: live prediction events
FastAPI runtime: src/tlse/api
  | services isolate filesystem, metrics, model registry, demo mode
Existing TLSE package: src/tlse
  | capture, preprocess, train, evaluate, realtime, benchmark, demo
Local artifacts
  | models/*.p, reports/*.json, demo/sample_data/*.pickle
```

## Backend

The backend lives in `src/tlse/api` and exposes a small local runtime:

- `GET /health`
- `GET /api/models`
- `GET /api/models/current`
- `GET /api/metrics/summary`
- `GET /api/dataset/summary`
- `POST /api/predict`
- `GET /api/app/config`
- `WS /ws/live`

Model and dataset loading are isolated in services so endpoint handlers do not depend on artifact details. Pickle artifacts are only loaded from local project paths and are documented as trusted-local files.

## Frontend

The web app lives in `apps/web` and uses React, TypeScript, Vite, Tailwind CSS, TanStack Query, Recharts, and lucide-react. It is organized by product area:

- `components/layout`
- `components/dashboard`
- `components/live`
- `components/dataset`
- `components/model-lab`
- `components/practice`
- `components/settings`
- `routes`
- `hooks`
- `lib`
- `types`

## Data Flow

Dashboard views read summaries through REST. Live recognition listens to `/ws/live`; when the socket is not available, the UI switches to explicit demo mode using mock data from `src/lib/mock-data.ts`.

Prediction requests can also be sent to `/api/predict` with a flattened landmark vector. The API returns top predictions, confidence, accepted state, runtime mode, and latency.

## Demo Mode

Demo mode is a first-class state. It is used when no camera/model is available or when the web socket cannot connect. Demo predictions are labeled and surfaced in the UI so users do not confuse simulated results with real inference.

## Future Tauri

The architecture keeps a browser-compatible UI and a local API boundary. Tauri can later package the UI shell and manage the FastAPI sidecar or a native Rust/Python bridge without rewriting product screens.
