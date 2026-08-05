# Development log

## 2026-08-05 — OpenHands SDK adapter

- Added an independent OpenHands integration in `src/temporal_spine/openhands_adapter.py`.
- Added a safe offline test mode that exercises compression, progressive component loading, and raw-context logging without an SDK or API call.
- Added the live extension points: an OpenHands condenser, a native `load_components` tool, an event callback, and JSON/JSONL diagnostics.
- Added provider-ready request capture (`latest_model_request.json` plus per-call files) with redaction of API credentials and authorization-like fields.
- Verified the offline path with the repository test suite: 29 tests passed.
- Inspected the current OpenHands SDK 1.40.0 and tools 1.36.1 wheel APIs in a temporary environment. The full live runtime was not installed because its optional telemetry/MCP dependency graph requires a larger environment; live API execution remains a follow-up validation step.
