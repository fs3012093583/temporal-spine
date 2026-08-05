# Project summaries

## Current integration milestone

The standalone OpenHands adapter keeps the existing Temporal-Spine core independent from the optional SDK. Historical user requests remain in the JSON User-Spine, while assistant results and execution records remain append-only and can be loaded through `load_components`. The offline runner produces inspectable `latest_raw_context.json`, per-step raw-context files, event logs, model-message logs, and a run summary. The live path is intentionally opt-in and reads model credentials only from an environment variable.

The adapter also records provider-ready chat requests before dispatch, including normalized messages and function-tool schemas, while redacting credential-shaped request fields.
