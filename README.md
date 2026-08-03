# Temporal-Spine Core

Temporal-Spine is a training-free protocol for giving a large language model selective access to
long conversation history.

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
5. **Stable request envelope** — system prompt, tools, model settings, and thinking settings remain
   identical across control and answer calls. Phase changes are append-only runtime events.
6. **Deterministic enforcement** — the runtime validates IDs, removes ambiguity about provenance,
   orders each batch by historical turn, and rejects premature final submission.
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
        `-- commit_final_result
                    |
                    v
          append final_result_authorized
                    |
                    v
          LLM -- submit_final_result(answer)
```

The complete message sequence is append-only. The final answer call sees the same system prompt,
tool schema, model, temperature, token limit, and thinking configuration as every control call.

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
    RunConfig(max_control_rounds=1, full_fallback_enabled=True),
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

## One-Shot and Progressive modes

The same runtime supports both paper variants:

```python
# Default one-shot selection followed by one answer call.
RunConfig(max_control_rounds=1)

# Progressive selection with up to three control decisions.
RunConfig(max_control_rounds=3)

# Extremely long histories where complete fallback is not feasible.
RunConfig(max_control_rounds=3, full_fallback_enabled=False)
```

`max_control_rounds` is a bound, not a requirement to use every round. The model may commit as soon
as the visible information is sufficient.

## Audit output

`RunResult` exposes:

- final answer;
- runtime-recorded loaded component IDs;
- focused user turns;
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

The tests cover monotonic loading, chronological batch ordering, lossless long-request recovery,
complete fallback, stable request hashes, append-only messages, duplicate rejection, and phase
authorization.

## Scope

This release intentionally excludes benchmark datasets, model answers, judge outputs, API keys,
private experiment logs, paper build artifacts, and third-party baseline code. Dataset adapters and
full reproduction scripts will be released separately after their licenses and evaluation manifests
are finalized.

## License

Apache License 2.0. See [LICENSE](LICENSE).
