# Future Tauri Packaging

Tauri is a good future fit because TLSE Studio already separates the web UI from the local runtime API.

## Proposed Shape

```text
Tauri shell
  | serves built apps/web assets
  | starts/stops local Python API sidecar
FastAPI sidecar
  | exposes localhost REST and WebSocket endpoints
  | loads local models, reports, and datasets
```

## Packaging Steps

1. Build the web app with `npm run build`.
2. Add a Tauri project under `apps/desktop` or `src-tauri`.
3. Configure Tauri to serve `apps/web/dist`.
4. Bundle or discover the Python runtime.
5. Launch `uvicorn tlse.api.main:app` as a sidecar on a local port.
6. Store user settings in the Tauri app data directory.

## Open Decisions

- Whether to bundle Python or require a managed local install.
- How model artifacts and datasets are selected outside the repository.
- How camera permissions are surfaced consistently across platforms.
- Whether inference should eventually move into a native plugin.
