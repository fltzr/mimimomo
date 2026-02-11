"""Tests for client.py — API client SSE parsing and error handling."""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client import ChatClient, ChatChunk, _extract_error_message, _calculate_backoff
from config import Config


class TestSSEParsing:
    """Test Server-Sent Events line parsing."""

    def setup_method(self):
        self.client = ChatClient(Config(endpoint="http://test:8080/v1"))

    def test_parse_content_chunk(self):
        data = {
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]
        }
        line = f"data: {json.dumps(data)}"
        chunk = self.client._parse_sse_line(line)
        assert chunk is not None
        assert chunk.delta_content == "Hello"
        assert chunk.finish_reason is None

    def test_parse_done_signal(self):
        chunk = self.client._parse_sse_line("data: [DONE]")
        assert chunk is not None
        assert chunk.finish_reason == "stop"

    def test_parse_finish_reason(self):
        data = {
            "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]
        }
        line = f"data: {json.dumps(data)}"
        chunk = self.client._parse_sse_line(line)
        assert chunk is not None
        assert chunk.finish_reason == "stop"

    def test_parse_error_response(self):
        data = {"error": {"message": "Model not found"}}
        line = f"data: {json.dumps(data)}"
        chunk = self.client._parse_sse_line(line)
        assert chunk is not None
        assert chunk.error == "Model not found"

    def test_parse_usage_chunk(self):
        data = {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
        line = f"data: {json.dumps(data)}"
        chunk = self.client._parse_sse_line(line)
        assert chunk is not None
        assert chunk.usage["total_tokens"] == 30

    def test_parse_empty_line(self):
        assert self.client._parse_sse_line("") is None

    def test_parse_comment_line(self):
        assert self.client._parse_sse_line(": keep-alive") is None

    def test_parse_invalid_json(self):
        assert self.client._parse_sse_line("data: {invalid}") is None

    def test_parse_non_data_line(self):
        assert self.client._parse_sse_line("event: message") is None


class TestPayloadBuilding:
    """Test request payload construction."""

    def test_default_payload(self):
        client = ChatClient(Config(model="gpt-4", temperature=0.5))
        messages = [{"role": "user", "content": "Hi"}]
        payload = client._build_payload(messages)
        assert payload["model"] == "gpt-4"
        assert payload["messages"] == messages
        assert payload["stream"] is True
        assert payload["temperature"] == 0.5

    def test_override_model(self):
        client = ChatClient(Config(model="gpt-4"))
        payload = client._build_payload([], model="gpt-3.5-turbo")
        assert payload["model"] == "gpt-3.5-turbo"

    def test_non_stream_payload(self):
        client = ChatClient(Config())
        payload = client._build_payload([], stream=False)
        assert payload["stream"] is False
        assert "stream_options" not in payload

    def test_max_tokens_in_payload(self):
        client = ChatClient(Config(max_tokens=512))
        payload = client._build_payload([])
        assert payload["max_tokens"] == 512

    def test_stream_options_included(self):
        client = ChatClient(Config())
        payload = client._build_payload([], stream=True)
        assert payload["stream_options"] == {"include_usage": True}


class TestErrorExtraction:
    """Test error message parsing from HTTP responses."""

    def test_json_error_body(self):
        body = json.dumps({"error": {"message": "Invalid API key"}})
        msg = _extract_error_message(body, 401)
        assert "Invalid API key" in msg
        assert "401" in msg

    def test_plain_text_error(self):
        msg = _extract_error_message("Bad Request", 400)
        assert "Bad Request" in msg

    def test_status_hint_included(self):
        msg = _extract_error_message("", 429)
        assert "Rate limited" in msg

    def test_unknown_status(self):
        msg = _extract_error_message("", 418)
        assert "418" in msg


class TestBackoff:
    """Test exponential backoff calculation."""

    def test_first_attempt(self):
        delay = _calculate_backoff(0)
        assert delay == 1.0

    def test_second_attempt(self):
        delay = _calculate_backoff(1)
        assert delay == 2.0

    def test_third_attempt(self):
        delay = _calculate_backoff(2)
        assert delay == 4.0

    def test_retry_after_header(self):
        mock_response = mock.MagicMock()
        mock_response.headers = {"retry-after": "5"}
        delay = _calculate_backoff(0, response=mock_response)
        assert delay == 5.0


class TestClientHeaders:
    """Test that client sets correct headers."""

    def test_headers_with_api_key(self):
        client = ChatClient(Config(api_key="sk-test123"))
        assert client._headers["Authorization"] == "Bearer sk-test123"
        assert client._headers["Content-Type"] == "application/json"

    def test_headers_without_api_key(self):
        client = ChatClient(Config(api_key=""))
        assert "Authorization" not in client._headers
