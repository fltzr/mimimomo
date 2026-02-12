"""Tests for session.py — chat session and per-exchange logging."""

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

# Use a temp directory for exchanges during tests
_test_data_dir = Path(tempfile.mkdtemp()) / "data"
_test_exchanges_dir = _test_data_dir / "exchanges"
_test_exchanges_dir.mkdir(parents=True, exist_ok=True)

import config
config.DATA_DIR = _test_data_dir

from session import ChatSession, save_exchange  # noqa: E402
import session as session_module
session_module.EXCHANGES_DIR = _test_exchanges_dir


class TestSessionMessages:
    """Test message management."""

    def test_add_user_message(self):
        s = ChatSession()
        s.add_user("Hello")
        assert s.message_count == 1
        msgs = s.get_messages()
        assert msgs[0] == {"role": "user", "content": "Hello"}

    def test_add_assistant_message(self):
        s = ChatSession()
        s.add_assistant("Hi there!")
        msgs = s.get_messages()
        assert msgs[0] == {"role": "assistant", "content": "Hi there!"}

    def test_system_prompt_prepended(self):
        s = ChatSession(system_prompt="You are helpful.")
        s.add_user("Hello")
        msgs = s.get_messages()
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "You are helpful."}
        assert msgs[1] == {"role": "user", "content": "Hello"}

    def test_no_system_prompt(self):
        s = ChatSession()
        s.add_user("Hello")
        msgs = s.get_messages()
        assert len(msgs) == 1

    def test_clear_messages(self):
        s = ChatSession(system_prompt="system")
        s.add_user("msg1")
        s.add_assistant("resp1")
        s.clear()
        assert s.message_count == 0
        assert s.is_empty
        # System prompt should still be there
        msgs = s.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_get_last_user_message(self):
        s = ChatSession()
        s.add_user("first")
        s.add_assistant("resp")
        s.add_user("second")
        assert s.get_last_user_message() == "second"

    def test_get_last_user_message_empty(self):
        s = ChatSession()
        assert s.get_last_user_message() is None


class TestRetry:
    """Test pop_last_exchange for retry."""

    def test_pop_last_exchange(self):
        s = ChatSession()
        s.add_user("Hello")
        s.add_assistant("Hi!")
        s.add_user("How are you?")
        s.add_assistant("Good!")

        msg = s.pop_last_exchange()
        assert msg == "How are you?"
        assert s.message_count == 2

    def test_pop_no_messages(self):
        s = ChatSession()
        msg = s.pop_last_exchange()
        assert msg is None

    def test_pop_only_user_no_assistant(self):
        s = ChatSession()
        s.add_user("Hello")
        msg = s.pop_last_exchange()
        assert msg == "Hello"
        assert s.message_count == 0


class TestContextTrimming:
    """Test context window management."""

    def test_trim_to_last_n(self):
        s = ChatSession()
        for i in range(10):
            s.add_user(f"msg{i}")
            s.add_assistant(f"resp{i}")

        s.trim_to_last_n(4)
        assert s.message_count == 4

    def test_trim_no_op_when_under(self):
        s = ChatSession()
        s.add_user("only one")
        s.trim_to_last_n(10)
        assert s.message_count == 1


class TestSaveExchange:
    """Test per-exchange file saving."""

    def test_save_exchange_creates_file(self):
        with mock.patch.object(session_module, "EXCHANGES_DIR", _test_exchanges_dir):
            path = save_exchange(
                prompt="What is Python?",
                response="Python is a programming language.",
                model="gpt-4",
                endpoint="https://api.openai.com/v1",
            )
        assert path.exists()
        assert path.suffix == ".json"

        data = json.loads(path.read_text())
        assert data["prompt"] == "What is Python?"
        assert data["response"] == "Python is a programming language."
        assert data["model"] == "gpt-4"
        assert data["endpoint"] == "https://api.openai.com/v1"
        assert "timestamp" in data

    def test_save_exchange_unique_filenames(self):
        import time
        paths = []
        with mock.patch.object(session_module, "EXCHANGES_DIR", _test_exchanges_dir):
            for i in range(3):
                p = save_exchange(f"prompt {i}", f"response {i}", "model", "http://x")
                paths.append(p)
                time.sleep(0.001)  # Ensure unique microsecond timestamps

        # All paths should be unique
        assert len(set(paths)) == 3


