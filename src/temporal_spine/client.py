from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .model import ModelResponse, ToolCall


@dataclass
class OpenAICompatibleClient:
    """Small dependency-free client for OpenAI-compatible chat-completion APIs."""

    model: str
    base_url: str
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: int = 120
    thinking: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv(self.api_key_env)
        if not self.api_key:
            raise ValueError(f"API key is missing; set {self.api_key_env}")

    @property
    def fixed_parameters(self) -> Mapping[str, Any]:
        parameters: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
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
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        message = payload["choices"][0]["message"]
        calls: list[ToolCall] = []
        for raw_call in message.get("tool_calls") or []:
            function = raw_call["function"]
            raw_arguments = function.get("arguments") or "{}"
            arguments = (
                json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            )
            calls.append(
                ToolCall(
                    call_id=str(raw_call.get("id") or f"call-{len(calls)}"),
                    name=str(function["name"]),
                    arguments=arguments,
                )
            )
        usage = {
            key: int(value)
            for key, value in (payload.get("usage") or {}).items()
            if isinstance(value, (int, float))
        }
        return ModelResponse(
            content=(message.get("content") or "").strip(),
            tool_calls=tuple(calls),
            usage=usage,
        )
