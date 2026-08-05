# Temporal-Spine Core

Temporal-Spine is a training-free post-compression memory runtime that gives a large language model
selective access to older conversation history.

It keeps historical **user requests** visible in chronological order as a stable `User-Spine`.
Assistant results, execution records, and exact originals of unusually long user requests are kept
as source-faithful components. The model loads only the components needed for the current task.

> This repository implements text-context orchestration for OpenAI-compatible APIs. It does not
> claim zero-copy KV-page attachment, GPU KV-cache compression, or latency acceleration.

## Core invariants

1. **Query-independent User-Spine** — historical user requests are organized before the current
   query is known. Long requests use a deterministic head-tail preview with a lossless recovery
   path.
2. **Model-guided component access** — the model selects legal `user:T`, `result:T`, and `trace:T`
   components through native tool calls.
3. **Monotonic completeness** — a component is loaded at most once; later rounds only append new
   information.
4. **Source-faithful recovery** — loaded content is copied from immutable historical records rather
   than regenerated from a summary.
5. **Stable request envelope** — system prompt, memory tools, model settings, and thinking settings
   remain identical across loading rounds. Runtime changes are append-only events.
6. **Deterministic enforcement** — the runtime validates IDs, removes ambiguity about provenance,
   orders each batch by historical turn, and rejects invalid or repeated loading.
7. **Optional complete fallback** — when enabled and the underlying model can hold the full history,
   all still-unloaded components can be appended as a safety endpoint.

## Protocol

```text
fixed system prompt
+ chronological User-Spine
+ component catalog
+ current request
        |
        v
LLM -- load_components([...]) --> runtime appends original components
        |
        +-- optionally load another non-repeated batch
        |
        `-- ordinary assistant response --> answer
```

The model calls `load_components` only when compressed historical evidence is needed. If the visible
User-Spine and already-loaded evidence are sufficient, it responds normally. Temporal-Spine does not
own an agent's planning, business tools, or termination protocol.

## Installation

```bash
git clone https://github.com/fs3012093583/temporal-spine.git
cd temporal-spine
python -m pip install -e .
```

The core package has no third-party runtime dependencies.

## Offline example

```bash
python examples/offline_demo.py
```

The example uses a scripted model and makes no network request.

## OpenAI-compatible API example

```python
import os

from temporal_spine import (
    HistoricalTurn,
    OpenAICompatibleClient,
    RunConfig,
    TemporalSpine,
)

client = OpenAICompatibleClient(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/chat/completions",
    api_key_env="DEEPSEEK_API_KEY",
    max_tokens=512,
)

history = [
    HistoricalTurn(0, "I prefer vegetarian meals.", "I will remember that."),
    HistoricalTurn(1, "Suggest a dinner.", "Try a lentil curry."),
]

result = TemporalSpine(
    client,
    RunConfig(max_load_rounds=1, full_fallback_enabled=True),
).run(history, "What dinner did you previously suggest?")

print(result.answer)
print(result.loaded_action_ids)
```

Set the API key only through the environment:

```bash
export DEEPSEEK_API_KEY="..."
python examples/openai_compatible_demo.py
```

Never commit `.env` or private conversation logs.

## One-Shot and Progressive loading

The same runtime supports one-shot and progressive access:

```python
# Default one-shot selection followed by one answer call.
RunConfig(max_load_rounds=1)

# Progressive selection with up to three loading rounds.
RunConfig(max_load_rounds=3)

# Extremely long histories where complete fallback is not feasible.
RunConfig(max_load_rounds=3, full_fallback_enabled=False)
```

`max_load_rounds` bounds memory-loading calls, not ordinary agent reasoning. The model may answer
immediately when the visible information is sufficient. At the bound, optional complete fallback
loads every remaining component before one ordinary answer call.

## LangGraph integration

Install the optional integration dependency:

```bash
python -m pip install -e '.[agent]'
```

`LangGraphTemporalSpine` uses a graph edge to continue after `load_components` and ends naturally
when the model returns an ordinary assistant response. Temporal-Spine remains only the compressed
history backend; LangGraph remains responsible for the wider agent workflow.

## OpenHands SDK integration

The OpenHands integration is a separate coding-agent adapter. It adds a Temporal-Spine condenser,
the native `load_components` tool, an append-only event archive, and raw model-context logging. The
OpenHands SDK currently requires Python 3.12 or newer.

Install it in an environment dedicated to the coding agent:

```bash
python -m pip install -e '.[openhands]'
```

Run the deterministic test mode first. It makes no API call and prints the context before and after
loading a historical assistant result:

```bash
python examples/openhands_temporal_spine_demo.py --test-mode
```

The log directory contains:

- `latest_raw_context.json` — the latest model-visible message list;
- `raw_context_*.json` — every recorded context snapshot;
- `latest_model_request.json` — the latest provider-ready messages, function-tool schemas, and non-secret request parameters;
- `model_request_*.json` — one provider-ready request per model call;
- `events.jsonl` — OpenHands event records;
- `llm_messages.jsonl` — SDK event-to-LLM-message conversions;
- `summary.json` — the test result and loaded component IDs.

Run a real local OpenHands coding conversation only after the offline test passes:

```bash
export DEEPSEEK_API_KEY="..."
python examples/openhands_temporal_spine_demo.py \
  --live \
  --api-key-env DEEPSEEK_API_KEY \
  --workspace . \
  --log-dir .openhands-temporal-spine/live-run
```

The adapter does not write the API key. The live workspace is the directory passed by
`--workspace`; use a disposable checkout when allowing file edits or terminal commands.

## Local web demo

Install the UI dependencies and start the Gradio page:

```bash
python -m pip install -e '.[ui]'
export DEEPSEEK_API_KEY="..."
python examples/gradio_app.py
```

Then open `http://127.0.0.1:7860`. The page is a continuous chat interface. Before compression it
sends the ordinary full conversation; pressing **Compress History** moves every completed hot turn
into the User-Spine and source-faithful component store. Conversation continues with recent turns in
a Hot Window, and the button can be used again to archive those turns. **Run 10-turn Demo** sends ten
predefined user messages through the real Agent one by one, streams each completed turn into the
current conversation, and deliberately leaves compression under manual control. The page shows current memory
state, loaded component IDs, model-call/token audit, runtime trace, and the complete JSON request body
for every model call. Authorization headers and API keys are excluded from the display and are never
written by the demo. Submitting a message sends the visible request data to the configured endpoint.

## Audit output

`RunResult` exposes:

- final answer;
- runtime-recorded loaded component IDs;
- whether complete fallback was used;
- per-call fixed-request hash and append-only check;
- provider token usage when returned by the API;
- a deterministic event trace.

The fixed-request hash audits logical stability. It is not evidence that an API provider actually
served a prefix-cache hit. Provider-reported cached tokens must be recorded separately.

## Testing

```bash
python -m pip install -e '.[dev]'
pytest -q
```

The tests cover natural completion, monotonic loading, chronological batch ordering, lossless
long-request recovery, complete fallback, stable request hashes, append-only messages, and duplicate
rejection.

## Scope

This release intentionally excludes benchmark datasets, model answers, judge outputs, API keys,
private experiment logs, paper build artifacts, and third-party baseline code. Dataset adapters and
full reproduction scripts will be released separately after their licenses and evaluation manifests
are finalized.

## License

Apache License 2.0. See [LICENSE](LICENSE).
