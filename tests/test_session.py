"""Tests for session.py — chat session management and persistence."""

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Use a temp directory for sessions during tests
_test_sessions_dir = Path(tempfile.mkdtemp()) / "sessions"
_test_sessions_dir.mkdir(parents=True, exist_ok=True)


with mock.patch("config.SESSIONS_DIR", _test_sessions_dir):
    # Need to also patch in the session module
    import config
    original_sessions_dir = config.SESSIONS_DIR
    config.SESSIONS_DIR = _test_sessions_dir

    from session import ChatSession, _sanitize_filename


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


class TestPersistence:
    """Test session save/load."""

    def test_save_and_load(self):
        s = ChatSession(system_prompt="Be concise.")
        s.add_user("Hello")
        s.add_assistant("Hi!")

        path = s.save("test_roundtrip")
        assert path.exists()

        loaded = ChatSession.load("test_roundtrip")
        assert loaded.message_count == 2
        assert loaded.system_prompt == "Be concise."
        assert loaded.session_name == "test_roundtrip"

    def test_save_auto_name(self):
        s = ChatSession()
        s.add_user("test")
        path = s.save()
        assert path.exists()
        assert path.suffix == ".json"

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            ChatSession.load("does_not_exist_xyz")

    def test_list_sessions(self):
        # Create a session to ensure something is listed
        s = ChatSession()
        s.add_user("list test")
        s.save("list_test_session")

        sessions = ChatSession.list_sessions()
        assert len(sessions) > 0
        names = [s["name"] for s in sessions]
        assert "list_test_session" in names

    def test_delete_session(self):
        s = ChatSession()
        s.add_user("delete me")
        s.save("delete_test")

        assert ChatSession.delete_session("delete_test") is True
        assert ChatSession.delete_session("delete_test") is False


class TestSanitizeFilename:
    """Test filename sanitization."""

    def test_normal_name(self):
        assert _sanitize_filename("my_session") == "my_session"

    def test_spaces_replaced(self):
        assert _sanitize_filename("my session") == "my_session"

    def test_special_chars(self):
        result = _sanitize_filename("test/file:name")
        assert "/" not in result
        assert ":" not in result

    def test_empty_string(self):
        assert _sanitize_filename("") == "unnamed"
