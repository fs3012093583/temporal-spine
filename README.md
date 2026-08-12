# Temporal-Spine Core

Temporal-Spine Core is the provider-agnostic reference implementation of the paper's current
**bounded One-Shot MemAddr** algorithm. It gives a large language model selective, source-faithful
access to long conversation history without training a retriever or generating history summaries.

Historical user requests remain visible in chronological order as a stable `User-Spine`.
Assistant results, observable execution records, and exact originals of truncated long user
requests are hidden components addressed as `result:T`, `trace:T`, and `user:T`. For each current
request, the model either answers directly or loads one bounded batch of components and then
answers.

> This package implements text-context orchestration for OpenAI-compatible APIs. It does not claim
> zero-copy KV-page attachment, GPU KV-cache compression, or provider-side prefix-cache hits.

## Current protocol

The `memaddr-access-state-v1-one-shot-max5-keep-first-no-fallback` protocol used by this release has
the following fixed behavior:

1. **User-Spine is always visible.** Short user requests are retained verbatim. A long request uses
   a deterministic head-tail preview, while its complete original remains loadable.
2. **Only two tools exist.** The stable tool array contains `load_components` and
   `submit_final_result`; there is no focus action, commit action, `remaining` action, or Full
   History fallback.
3. **Access is adaptive but one-shot.** The first model call may submit immediately or request one
   component batch. Once a batch is returned, history access is closed and the second call must
   submit the result. Normal execution therefore uses one or two model calls; a third call is
   reserved only for recovery from one invalid protocol action.
4. **At most five components are loaded.** If a non-conforming provider returns more than five IDs,
   the runtime keeps the first five requested IDs, records the dropped IDs, and reports the
   truncation in the tool result. Retained components are presented in historical order.
5. **Loaded exchanges are source explicit.** For an assistant result or execution record, the tool
   result presents its already-visible paired user request first and then the unmodified hidden
   component.
6. **State is explicit and append-only.** Every request sees the latest `history_access_state`,
   including model-call count, access-call count, remaining budget, batch limit, and authorized
   tools. System prompt, tool schema, model settings, and thinking settings remain byte-stable.
7. **The runtime is authoritative.** It validates IDs and arguments, records actual provenance,
   checks append-only messages, and marks a natural-language answer captured without
   `submit_final_result` as protocol non-compliance.

```text
stable System + tools + model configuration
                 |
       User-Spine + current request
        + history_access_state(calls_used=0)
                 |
                 v
           first model call
              /       \
 submit_final_result   load_components([up to 5 IDs])
        |                         |
     answer            runtime appends paired requests,
                       original components, and state
                                  |
                                  v
                         second model call
                                  |
                         submit_final_result
```

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

The offline example uses a scripted model and makes no network request.

## OpenAI-compatible API example

```python
from temporal_spine import HistoricalTurn, OpenAICompatibleClient, TemporalSpine

client = OpenAICompatibleClient(
    model="your-model",
    base_url="https://provider.example/v1/chat/completions",
    api_key_env="PROVIDER_API_KEY",
    temperature=0.0,
    max_tokens=512,
    thinking={"type": "disabled"},
)

history = [
    HistoricalTurn(0, "I prefer vegetarian meals.", "I will remember that."),
    HistoricalTurn(1, "Give me a dinner idea.", "A lentil curry would be a good option."),
]

result = TemporalSpine(client).run(
    history,
    "What dinner did you previously suggest?",
    current_date="2026-08-12",
)

print(result.answer)
print(result.loaded_action_ids)
print(result.protocol_followed)
```

Set the key only through the environment. Never commit API keys, `.env` files, private histories,
or raw provider logs.

## Runtime output

`RunResult` contains:

- `answer`: the final user-facing result;
- `loaded_action_ids`: the retained components actually loaded, in presentation order;
- `dropped_action_ids`: overflow IDs that were not loaded;
- `invalid_action_count`: rejected protocol actions before completion;
- `protocol_followed`: whether the model used `submit_final_result` as required;
- `submission_mode`: native submission or audited runtime capture;
- `calls`: per-call stable-envelope hash, append-only status, message count, and provider usage;
- `trace`: component provenance, overflow notice, access state, and final-submission event.

The fixed-request hash and append-only check audit **logical cache eligibility**. They are not
evidence of provider cache hits. Use provider-reported cached-input fields for empirical cache-hit
measurement.

## Testing

```bash
python -m pip install -e '.[dev]'
pytest -q
```

The test suite covers direct submission, one-shot loading, five-component keep-first overflow,
historical presentation order, paired source layout, long-request recovery, stable request
envelopes, append-only messages, closed-access enforcement, malformed IDs, empty answers, and
audited natural-language recovery.

## Scope

This release intentionally excludes benchmark datasets, model answers, judge outputs, API keys,
private experiment logs, paper build artifacts, and third-party baseline code. It is the compact
algorithm implementation, not the complete experimental repository.

## License

Apache License 2.0. See [LICENSE](LICENSE).
