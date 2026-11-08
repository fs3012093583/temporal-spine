# MemAddr

**User-Spine-guided direct addressing for long-term dialogue memory**

[Paper](https://doi.org/10.1016/j.knosys.2026.117296) · [中文快速部署](README.zh-CN.md) · [Security](SECURITY.md)

MemAddr keeps historical user messages visible in chronological order (the **User-Spine**).
Original assistant replies stay in a component store under stable addresses such as `result:12`.
The same language model that answers the question selects which addresses to load. The runtime
checks those addresses and returns the original text; it does not rank relevance.

No generated memory summaries, embedding model, vector database, external relevance retriever,
or method-specific training is required. This is a small Python library and CLI, not an inference
server. When using a hosted model, the method runs on a CPU machine without a GPU.

## Quick start: no API key needed

Requirements: **Python 3.10+**, Git, and a terminal. The core has **no third-party runtime dependencies**.

```bash
git clone https://github.com/fs3012093583/temporal-spine.git
cd temporal-spine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
memaddr demo
```

On Windows, use `py -m venv .venv`, then `.venv\Scripts\Activate.ps1` in PowerShell.
The demo is scripted, makes **no network requests**, and demonstrates loading `result:1` followed
by final submission. It is not a model-quality evaluation. Expected fields include:

```json
{
  "answer": "I previously suggested lentil curry for dinner.",
  "loaded_action_ids": ["result:1"],
  "access_calls": 1,
  "model_calls": 2,
  "protocol_followed": true
}
```

`python -m temporal_spine demo` is equivalent if the `memaddr` executable is not on your PATH.

## Run with a real model

Use an OpenAI-compatible **Chat Completions API with native function/tool calling**. Ordinary
text generation alone is insufficient for address selection. Choose an available model and its
documented endpoint; no provider or model subscription is included in this repository.

Set these variables in your terminal. Replace the placeholders with your provider's values:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export MEMADDR_BASE_URL="https://your-provider.example/v1"
export MEMADDR_MODEL="YOUR_TOOL_CAPABLE_MODEL"

memaddr run \
  --history examples/history.json \
  --question "What dinner did you previously suggest?" \
  --current-date 2026-01-12
```

**This command sends the visible history and selected originals to your provider and may incur
API charges.** The API key is read from the environment, never from the history file. For another
key variable, use `--api-key-env PROVIDER_API_KEY` (the variable name, not the secret itself).
Both an API root ending in `/v1` and a full `/chat/completions` URL are accepted.

For a local OpenAI-compatible server, use its model name and `http://localhost:PORT/v1`.
If the server ignores authentication, set `OPENAI_API_KEY=local`. Remote APIs must use HTTPS.

### Your own history

Replace the synthetic example with a UTF-8 JSON file of the same form:

```json
{
  "turns": [
    {
      "turn_id": 12,
      "timestamp": "2026-01-11",
      "user_request": "Give me a dinner idea.",
      "assistant_result": "Try lentil curry."
    }
  ]
}
```

`turn_id` (unique, non-negative integer) and `user_request` are required. `assistant_result` and
`timestamp` are optional. Use source dates when available. Supply the **current question separately**;
do not include its reference answer or any future messages in the history. The runtime orders
turns by `turn_id`; assign IDs in chronological order and never renumber existing turns.

After a completed interaction, your application can append a new turn with the next ID and its
original user request and assistant answer. Each `run` builds the visible spine from the history
you supply; it does not modify your JSON file or silently persist private conversations.
An optional `execution_records` array can retain observable tool outputs under `trace:T` addresses.
It is a library extension; private model reasoning is not required.

### Access and output limits

| Setting | Default | Meaning |
| --- | --- | --- |
| `--access-rounds` | `1` | Valid history-access decisions, `K`. Use `2` for progressive access, or `0` for User-Spine only. |
| `--batch-size` | `5` | Maximum addresses processed per access (`1`–`5`). |
| `--max-model-calls` | `3` | Total API calls, including final submission and any format repair. |
| `--max-tokens` | `8192` | Output limit **per API call**, including reasoning when the provider counts it. |
| `--timeout` | `120` | Timeout in seconds for each API request. |

For two access decisions followed by an answer:

```bash
memaddr run --history examples/history.json \
  --question "What dinner did you previously suggest?" --access-rounds 2
```

The model can submit early; these limits do not force extra calls. A format repair shares the
total call budget and can reduce the remaining opportunity to load history. No automatic HTTP
retries or full-history fallback are performed. Increase output limits for models that need more
reasoning, subject to the provider's maximum and your cost budget.

Provider-specific options:

- Use `--token-parameter max_completion_tokens` if that is your API's output-limit field.
- Use `--temperature omit` if your model does not accept temperature.
- Use `--thinking-json '{"type":"disabled"}'` only if supported by your provider. Thinking
  controls are not standardized across APIs; there is no universal setting.
- Use `--strict-tools` to reject ordinary-text final answers. By default, the runtime can capture
  such answers but reports `protocol_followed: false` and the submission mode explicitly.

## Python integration

The import name remains `temporal_spine` for compatibility; the package distribution is
`temporal-spine-core` (version 0.3.0). Install it from this repository, not an assumed PyPI release.

```python
import os
from temporal_spine import (
    HistoricalTurn, OpenAICompatibleClient, RunConfig, TemporalSpine,
)

client = OpenAICompatibleClient(
    model=os.environ["MEMADDR_MODEL"],
    base_url=os.environ["MEMADDR_BASE_URL"],
    api_key_env="OPENAI_API_KEY",
    max_tokens=8192,
)
history = [
    HistoricalTurn(12, "Give me a dinner idea.", "Try lentil curry.", timestamp="2026-01-11"),
]
memory = TemporalSpine(client, RunConfig(max_access_calls=2))
result = memory.run(history, "What did you suggest?", current_date="2026-01-12")
print(result.answer)
print(result.loaded_action_ids)
print(result.protocol_followed)
```

For a non-compatible provider, implement the small `ChatModel` interface in
[`model.py`](src/temporal_spine/model.py): expose `fixed_parameters` and return a `ModelResponse`
from `complete(messages, tools)`. The address store and runtime are provider-independent.

## Method behavior

1. **Build the User-Spine without the current query.** Requests of at most 512 deterministic
   spans remain intact; longer ones expose their first and last 256 spans, with exact originals
   recoverable as `user:T`. Each Chinese character, alphanumeric sequence, or punctuation mark
   contributes a span. These are **not provider billing tokens**. `SpinePolicy` retains its legacy
   `*_tokens` argument names, but they configure these span limits.
2. **Select addresses, not search queries.** The fixed tools are `load_components` and
   `submit_final_result`. Hidden bodies do not appear in the initial context.
3. **Restore deterministically.** Keep the first `b` requested IDs, validate the retained list,
   omit already loaded components, and return each new batch in historical order. Assistant
   records are paired with the already-visible user request. Content is restored verbatim.
4. **Bound access.** Default `K=1`; `K=2` permits a second decision after reading the first batch.
   A valid decision with no new records still uses one access. Final submission is not an access.
5. **Keep the trajectory append-only.** Earlier messages and batches are not reordered. System
   prompt, tools, and model parameters stay fixed; runtime state controls phase permissions.

`RunResult` includes the answer, loaded/dropped IDs, protocol status, per-call usage and request
hashes, and a detailed trace. The trace contains recovered source text: keep it private.
CLI output omits full restored records. Cache-related usage fields are retained when supplied
by the provider; stable prefixes alone do **not** guarantee actual provider cache hits.

This release implements the core bounded-address mechanism, not all benchmark-specific evaluation
workflows. The visible User-Spine grows with history: the whole request must still fit the model's
context window. The package does not compress GPU KV pages or supply a production multi-tenant
database. Your application must handle access authorization, storage, retention and context limits.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `memaddr: command not found` | Activate `.venv`, or use `python -m temporal_spine`. |
| Missing API key | Export the variable named by `--api-key-env` in the same terminal. |
| HTTP 401 / 403 | Credential and model permissions. |
| HTTP 400 / 404 | Endpoint, model ID, native tools, token-limit field and Thinking settings. |
| HTTP 429 / connection failure | Provider limits or network; no retries are made automatically. |
| Incomplete output (`length`) | Raise `--max-tokens` within the provider limit; partial answers are rejected. |
| Context too long | Shorten the supplied history or use a larger-context model; hidden access does not bound spine growth. |
| `protocol_followed: false` | Inspect invalid actions or natural-language capture; do not treat this as strict tool compliance. |

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

Tests and examples use synthetic conversations only. Tests mock the HTTP transport, exercise the
CLI end to end, and make no paid API calls. Do not commit keys, private histories or runtime traces.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation and license

Shun Fan and Chaolang Hu. *MemAddr: User-spine-guided direct addressing for long-term dialogue
memory*. Knowledge-Based Systems (2026). [DOI: 10.1016/j.knosys.2026.117296](https://doi.org/10.1016/j.knosys.2026.117296).

```bibtex
@article{fan2026memaddr,
  title = {MemAddr: User-spine-guided direct addressing for long-term dialogue memory},
  author = {Fan, Shun and Hu, Chaolang},
  journal = {Knowledge-Based Systems},
  year = {2026},
  doi = {10.1016/j.knosys.2026.117296}
}
```

Apache License 2.0; see [LICENSE](LICENSE). This method-only repository excludes benchmark datasets,
experimental configurations and results, model answers, judge logs, paper build artifacts, and
third-party baseline implementations.
