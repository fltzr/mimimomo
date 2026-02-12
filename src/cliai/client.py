"""
OpenAI SDK-based chat client for CLIAI.

Uses the official OpenAI Python SDK which handles SSE streaming, retries
with exponential backoff, timeouts, and error handling for any
OpenAI-compatible endpoint.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Generator, Optional

import json
import openai

from config import Config
from network import validate_endpoint, BlockedHostError


@dataclass
class ChatChunk:
    """A single chunk from the streaming response."""

    delta_content: str = ""
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[dict] = None
    tool_calls: Optional[list] = None  # List of {id, name, arguments}


class APIError(Exception):
    """Raised for non-retryable API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)


class ChatClient:
    """OpenAI-compatible chat completions client with streaming support."""

    def __init__(self, config: Config):
        self.config = config

        # Debug log path (cwd-relative)
        self._debug_log = "debug_requests.json"

        # Custom httpx client with request logging hook
        import httpx

        def _log_raw_request(request: httpx.Request):
            """Capture the full raw HTTP request to debug_requests.json."""
            try:
                body_bytes = request.content
                try:
                    body = json.loads(body_bytes) if body_bytes else None
                except (json.JSONDecodeError, UnicodeDecodeError):
                    body = body_bytes.decode("utf-8", errors="replace") if body_bytes else None

                entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "method": request.method,
                    "url": str(request.url),
                    "headers": dict(request.headers),
                    "body": body,
                }

                try:
                    with open(self._debug_log, "r") as f:
                        data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    data = []

                data.append(entry)

                with open(self._debug_log, "w") as f:
                    json.dump(data, f, indent=2, default=str)
            except Exception:
                pass  # Never break the chat for logging

        http_client = httpx.Client(
            event_hooks={"request": [_log_raw_request]},
        )

        self._client = openai.OpenAI(
            base_url=config.endpoint.rstrip("/"),
            api_key=config.api_key or "not-needed",
            timeout=openai.Timeout(
                120.0,  # default for all
                connect=10.0,
            ),
            max_retries=3,
            http_client=http_client,
        )

        # Network security
        self._allowed_hosts = config.allowed_hosts
        self._enforce_allowlist = config.enforce_allowlist

    def _validate_network(self) -> Optional[ChatChunk]:
        """Validate endpoint against the allowlist. Returns error chunk if blocked."""
        try:
            url = self.config.endpoint.rstrip("/") + "/chat/completions"
            validate_endpoint(url, self._allowed_hosts, self._enforce_allowlist)
            return None
        except BlockedHostError as e:
            return ChatChunk(error=str(e))

    def stream_chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Generator[ChatChunk, None, None]:
        """
        Send a chat completion request and yield streaming chunks.

        Yields:
            ChatChunk objects with delta content, finish reasons, errors, or usage.
        """
        # Network security gate
        blocked = self._validate_network()
        if blocked:
            yield blocked
            return

        kwargs = self._build_kwargs(
            messages, model, temperature, max_tokens,
            tools=tools, response_format=response_format,
        )

        pending_tool_calls = []  # Accumulate streamed tool call fragments

        try:
            with self._client.chat.completions.create(
                **kwargs, stream=True,
                stream_options={"include_usage": True},
            ) as stream:
                for chunk in stream:
                    # Usage info (final chunk)
                    if chunk.usage:
                        yield ChatChunk(usage={
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens,
                        })

                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    delta_content = choice.delta.content or ""
                    finish_reason = choice.finish_reason

                    # Accumulate tool calls from streaming deltas
                    if choice.delta.tool_calls:
                        for tc in choice.delta.tool_calls:
                            while len(pending_tool_calls) <= tc.index:
                                pending_tool_calls.append({
                                    "id": "", "name": "", "arguments": ""
                                })
                            entry = pending_tool_calls[tc.index]
                            if tc.id:
                                entry["id"] = tc.id
                            if tc.function and tc.function.name:
                                entry["name"] = tc.function.name
                            if tc.function and tc.function.arguments:
                                entry["arguments"] += tc.function.arguments

                    if finish_reason == "tool_calls":
                        yield ChatChunk(
                            finish_reason="tool_calls",
                            tool_calls=pending_tool_calls,
                        )
                        return

                    if delta_content or finish_reason:
                        yield ChatChunk(
                            delta_content=delta_content,
                            finish_reason=finish_reason,
                        )

                    if finish_reason == "stop":
                        return

        except openai.APIConnectionError as e:
            yield ChatChunk(error=(
                f"Connection failed: {e}\n"
                "Is the server running? Check your --endpoint setting."
            ))
        except openai.RateLimitError as e:
            yield ChatChunk(error=f"Rate limited: {e}")
        except openai.APIStatusError as e:
            yield ChatChunk(error=_format_api_error(e))
        except openai.APITimeoutError:
            yield ChatChunk(error=(
                "Request timed out.\n"
                "The model may be loading or the request is too large."
            ))
        except KeyboardInterrupt:
            yield ChatChunk(finish_reason="interrupted")

    def send_chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> ChatChunk:
        """
        Send a non-streaming chat completion request.

        Returns:
            A single ChatChunk with the full response.
        """
        # Network security gate
        blocked = self._validate_network()
        if blocked:
            return blocked

        kwargs = self._build_kwargs(
            messages, model, temperature, max_tokens,
            tools=tools, response_format=response_format,
        )

        try:
            response = self._client.chat.completions.create(**kwargs, stream=False)

            if not response.choices:
                return ChatChunk(error="Empty response: no choices returned.")

            choice = response.choices[0]
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            # Handle tool calls
            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                tool_calls = []
                for tc in choice.message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    })
                return ChatChunk(
                    finish_reason="tool_calls",
                    tool_calls=tool_calls,
                    usage=usage,
                )

            return ChatChunk(
                delta_content=choice.message.content or "",
                finish_reason=choice.finish_reason or "stop",
                usage=usage,
            )

        except openai.APIConnectionError as e:
            return ChatChunk(error=f"Connection failed: {e}")
        except openai.RateLimitError as e:
            return ChatChunk(error=f"Rate limited: {e}")
        except openai.APIStatusError as e:
            return ChatChunk(error=_format_api_error(e))
        except openai.APITimeoutError:
            return ChatChunk(error="Request timed out.")

    def _build_kwargs(
        self,
        messages: list[dict],
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        tools: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> dict:
        """Build keyword arguments for the SDK call."""
        kwargs: dict = {
            "model": model or self.config.model,
            "messages": messages,
        }

        temp = temperature if temperature is not None else self.config.temperature
        if temp is not None:
            kwargs["temperature"] = temp

        tokens = max_tokens or self.config.max_tokens
        if tokens is not None:
            kwargs["max_tokens"] = tokens

        if tools:
            kwargs["tools"] = tools

        if response_format:
            kwargs["response_format"] = response_format

        return kwargs



def _format_api_error(e: openai.APIStatusError) -> str:
    """Format an API status error into a human-readable message."""
    status_hints = {
        401: "Authentication failed — check your API key.",
        403: "Access forbidden — your API key may not have permission.",
        404: "Endpoint not found — check your --endpoint URL and model name.",
        422: "Invalid request — the server rejected the payload.",
        429: "Rate limited — too many requests.",
    }

    hint = status_hints.get(e.status_code, "")
    msg = f"[{e.status_code}] {e.message}"
    if hint:
        msg += f"\n💡 {hint}"
    return msg
