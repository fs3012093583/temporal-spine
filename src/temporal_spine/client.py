from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
import urllib.parse
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .model import ModelResponse, ToolCall


class ProviderError(RuntimeError):
    """A provider request failed or returned an unusable response."""


@dataclass(repr=False)
class OpenAICompatibleClient:
    """Non-streaming client accepting an API root or full chat/completions URL.

    No automatic retries: callers control additional paid requests.
    """

    model: str
    base_url: str
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float | None = 0.0
    max_tokens: int = 8192
    timeout: float = 120
    thinking: Mapping[str, Any] | None = None
    token_parameter: str = "max_tokens"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be supplied explicitly")
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("token_parameter must be max_tokens or max_completion_tokens")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, query, or fragment")
        if not parsed.hostname or parsed.scheme not in {"http", "https"}:
            raise ValueError("base_url must be an HTTP(S) API root or endpoint")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("use HTTPS for remote APIs; HTTP is allowed only on localhost")
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url.endswith("/chat/completions"):
            self.base_url += "/chat/completions"
        self.api_key = self.api_key or os.getenv(self.api_key_env)
        if not self.api_key:
            raise ValueError(f"API key is missing; set {self.api_key_env}")

    @property
    def fixed_parameters(self) -> Mapping[str, Any]:
        parameters: dict[str, Any] = {
            "model": self.model,
            self.token_parameter: self.max_tokens,
        }
        if self.temperature is not None:
            parameters["temperature"] = self.temperature
        if self.thinking is not None:
            parameters["thinking"] = dict(self.thinking)
        return parameters

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ModelResponse:
        body: dict[str, Any] = {
            **self.fixed_parameters,
            "messages": list(messages),
            "tools": list(tools),
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Provider bodies may contain private prompts; do not echo them.
            raise ProviderError(
                f"provider returned HTTP {exc.code}; check endpoint, model, credentials and tool support"
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"provider connection failed ({type(exc).__name__}); no automatic retry") from None
        except (ValueError, UnicodeError):
            raise ProviderError("provider returned invalid JSON") from None
        try:
            choice = payload["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            if not isinstance(content, str):
                raise ValueError("expected text content")
            calls: list[ToolCall] = []
            for raw_call in message.get("tool_calls") or []:
                function = raw_call["function"]
                raw_arguments = function.get("arguments") or "{}"
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be an object")
                if not isinstance(raw_call.get("id"), str) or not raw_call["id"]:
                    raise ValueError("tool call id is missing")
                calls.append(ToolCall(raw_call["id"], str(function["name"]), arguments))
            if len({call.call_id for call in calls}) != len(calls):
                raise ValueError("duplicate tool call id")
            raw_usage = payload.get("usage") or {}
            usage = {
                key: int(value) for key, value in raw_usage.items()
                if type(value) in (int, float)
            }
            for detail_key, prefix in (("prompt_tokens_details", "prompt"), ("completion_tokens_details", "completion")):
                for key, value in (raw_usage.get(detail_key) or {}).items():
                    if type(value) in (int, float):
                        usage[f"{prefix}_{key}"] = int(value)
            reasoning = message.get("reasoning_content")
            if reasoning is not None and not isinstance(reasoning, str):
                raise ValueError("reasoning_content must be text")
            return ModelResponse(
                content=content.strip(), tool_calls=tuple(calls), usage=usage,
                finish_reason=choice.get("finish_reason"), reasoning_content=reasoning,
            )
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            raise ProviderError("provider returned malformed choices or native tool calls") from None
