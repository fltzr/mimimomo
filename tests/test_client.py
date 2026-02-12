"""Tests for client.py — OpenAI SDK-based chat client."""

import sys
from pathlib import Path
from unittest import mock

import pytest

from client import ChatClient, ChatChunk, _format_api_error
from config import Config


class TestBuildKwargs:
    """Test request keyword argument construction."""

    def setup_method(self):
        self.client = ChatClient(Config(
            endpoint="http://test:8080/v1",
            model="gpt-4",
            temperature=0.5,
        ))

    def test_default_kwargs(self):
        messages = [{"role": "user", "content": "Hi"}]
        kwargs = self.client._build_kwargs(messages, None, None, None)
        assert kwargs["model"] == "gpt-4"
        assert kwargs["messages"] == messages
        assert kwargs["temperature"] == 0.5

    def test_override_model(self):
        kwargs = self.client._build_kwargs([], "gpt-3.5-turbo", None, None)
        assert kwargs["model"] == "gpt-3.5-turbo"

    def test_override_temperature(self):
        kwargs = self.client._build_kwargs([], None, 0.9, None)
        assert kwargs["temperature"] == 0.9

    def test_max_tokens_in_kwargs(self):
        client = ChatClient(Config(max_tokens=512))
        kwargs = client._build_kwargs([], None, None, None)
        assert kwargs["max_tokens"] == 512

    def test_max_tokens_override(self):
        kwargs = self.client._build_kwargs([], None, None, 256)
        assert kwargs["max_tokens"] == 256

    def test_no_max_tokens_when_none(self):
        kwargs = self.client._build_kwargs([], None, None, None)
        assert "max_tokens" not in kwargs


class TestErrorFormatting:
    """Test API error message formatting."""

    def test_401_error(self):
        err = mock.MagicMock()
        err.status_code = 401
        err.message = "Invalid API key"
        msg = _format_api_error(err)
        assert "401" in msg
        assert "Invalid API key" in msg
        assert "Authentication failed" in msg

    def test_404_error(self):
        err = mock.MagicMock()
        err.status_code = 404
        err.message = "Not found"
        msg = _format_api_error(err)
        assert "404" in msg
        assert "Endpoint not found" in msg

    def test_unknown_status(self):
        err = mock.MagicMock()
        err.status_code = 418
        err.message = "I'm a teapot"
        msg = _format_api_error(err)
        assert "418" in msg
        assert "I'm a teapot" in msg
        # No hint for unknown status
        assert "💡" not in msg


class TestClientInit:
    """Test client initialization."""

    def test_client_creates_openai_instance(self):
        client = ChatClient(Config(endpoint="http://test:8080/v1", api_key="sk-test"))
        assert client._client is not None
        assert client._client.api_key == "sk-test"

    def test_client_no_api_key(self):
        """When no API key, it should use a placeholder."""
        client = ChatClient(Config(api_key=""))
        assert client._client.api_key == "not-needed"

    def test_client_stores_security_config(self):
        client = ChatClient(Config(
            allowed_hosts=["api.openai.com"],
            enforce_allowlist=True,
        ))
        assert client._allowed_hosts == ["api.openai.com"]
        assert client._enforce_allowlist is True


class TestNetworkSecurity:
    """Test host allowlist enforcement in ChatClient."""

    def test_stream_chat_blocked_host(self):
        """stream_chat should yield an error chunk when host is blocked."""
        config = Config(
            endpoint="https://evil.com/v1",
            allowed_hosts=["api.openai.com"],
            enforce_allowlist=True,
        )
        client = ChatClient(config)
        messages = [{"role": "user", "content": "hi"}]

        chunks = list(client.stream_chat(messages))
        assert len(chunks) == 1
        assert chunks[0].error is not None
        assert "evil.com" in chunks[0].error

    def test_send_chat_blocked_host(self):
        """send_chat should return an error chunk when host is blocked."""
        config = Config(
            endpoint="https://evil.com/v1",
            allowed_hosts=["api.openai.com"],
            enforce_allowlist=True,
        )
        client = ChatClient(config)
        messages = [{"role": "user", "content": "hi"}]

        result = client.send_chat(messages)
        assert result.error is not None
        assert "evil.com" in result.error

    def test_no_enforcement_allows_any_host(self):
        """When enforce_allowlist=False, no blocking occurs."""
        config = Config(
            endpoint="https://anything.com/v1",
            allowed_hosts=[],
            enforce_allowlist=False,
        )
        client = ChatClient(config)
        assert client._enforce_allowlist is False
        # Validation should pass
        assert client._validate_network() is None

    def test_allowed_host_passes_validation(self):
        """When host is in allowlist, validation passes."""
        config = Config(
            endpoint="https://api.openai.com/v1",
            allowed_hosts=["api.openai.com"],
            enforce_allowlist=True,
        )
        client = ChatClient(config)
        assert client._validate_network() is None

    def test_validate_network_returns_error_chunk(self):
        """_validate_network returns a ChatChunk with error when blocked."""
        config = Config(
            endpoint="https://evil.com/v1",
            allowed_hosts=["api.openai.com"],
            enforce_allowlist=True,
        )
        client = ChatClient(config)
        result = client._validate_network()
        assert result is not None
        assert result.error is not None
        assert "evil.com" in result.error


class TestChatChunk:
    """Test ChatChunk dataclass."""

    def test_default_values(self):
        chunk = ChatChunk()
        assert chunk.delta_content == ""
        assert chunk.finish_reason is None
        assert chunk.error is None
        assert chunk.usage is None

    def test_with_content(self):
        chunk = ChatChunk(delta_content="Hello", finish_reason="stop")
        assert chunk.delta_content == "Hello"
        assert chunk.finish_reason == "stop"

    def test_with_error(self):
        chunk = ChatChunk(error="Something went wrong")
        assert chunk.error == "Something went wrong"

    def test_with_usage(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        chunk = ChatChunk(usage=usage)
        assert chunk.usage["total_tokens"] == 30
