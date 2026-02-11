"""Tests for commands.py — slash command parsing and handling."""

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commands import parse_command, CommandHandler, COMMAND_HELP


class TestParseCommand:
    """Test slash command parsing."""

    def test_parse_simple_command(self):
        result = parse_command("/help")
        assert result == ("/help", "")

    def test_parse_command_with_args(self):
        result = parse_command("/model gpt-4o")
        assert result == ("/model", "gpt-4o")

    def test_parse_command_with_multi_word_args(self):
        result = parse_command("/system You are a helpful assistant.")
        assert result == ("/system", "You are a helpful assistant.")

    def test_non_command_returns_none(self):
        assert parse_command("Hello world") is None

    def test_empty_string_returns_none(self):
        assert parse_command("") is None

    def test_command_case_insensitive(self):
        result = parse_command("/HELP")
        assert result == ("/help", "")

    def test_command_with_leading_spaces(self):
        result = parse_command("  /clear  ")
        assert result == ("/clear", "")


class TestCommandHandler:
    """Test command dispatch and effects."""

    def _make_handler(self):
        session = mock.MagicMock()
        ui = mock.MagicMock()
        config = mock.MagicMock()
        config.model = "test-model"
        client = mock.MagicMock()
        client.config = config
        handler = CommandHandler(session, ui, config, client)
        return handler, session, ui, config

    def test_help_command(self):
        handler, session, ui, _ = self._make_handler()
        handled = handler.handle("/help", "")
        assert handled is True
        ui.show_help.assert_called_once()

    def test_clear_command(self):
        handler, session, ui, _ = self._make_handler()
        handler.handle("/clear", "")
        session.clear.assert_called_once()
        ui.show_success.assert_called_once()

    def test_exit_command(self):
        handler, _, _, _ = self._make_handler()
        handler.handle("/exit", "")
        assert handler.should_exit is True

    def test_quit_command(self):
        handler, _, _, _ = self._make_handler()
        handler.handle("/quit", "")
        assert handler.should_exit is True

    def test_model_switch(self):
        handler, session, ui, config = self._make_handler()
        handler.handle("/model", "gpt-4o")
        assert config.model == "gpt-4o"
        ui.show_success.assert_called_once()

    def test_model_show_current(self):
        handler, session, ui, config = self._make_handler()
        handler.handle("/model", "")
        ui.show_info_msg.assert_called_once()

    def test_system_prompt_set(self):
        handler, session, ui, _ = self._make_handler()
        handler.handle("/system", "You are a pirate.")
        session.__setattr__("system_prompt", "You are a pirate.")

    def test_retry_command(self):
        handler, session, ui, _ = self._make_handler()
        session.pop_last_exchange.return_value = "Hello"
        handler.handle("/retry", "")
        msg = handler.retry_message
        assert msg == "Hello"

    def test_retry_no_messages(self):
        handler, session, ui, _ = self._make_handler()
        session.pop_last_exchange.return_value = None
        handler.handle("/retry", "")
        ui.show_error.assert_called_once()

    def test_info_command(self):
        handler, _, ui, config = self._make_handler()
        handler.handle("/info", "")
        ui.show_info.assert_called_once_with(config)

    def test_unknown_command(self):
        handler, _, ui, _ = self._make_handler()
        handled = handler.handle("/foobar", "")
        assert handled is False
        ui.show_error.assert_called_once()


class TestCommandHelp:
    """Test that all commands have help text."""

    def test_all_commands_documented(self):
        expected = ["/help", "/clear", "/model", "/system", "/retry", "/info", "/exit"]
        for cmd in expected:
            matching = [k for k in COMMAND_HELP if k.startswith(cmd)]
            assert len(matching) > 0, f"Missing help for {cmd}"

    def test_removed_commands_not_present(self):
        """Session persistence commands should not exist."""
        for cmd in ["/save", "/load", "/sessions", "/delete"]:
            matching = [k for k in COMMAND_HELP if k.startswith(cmd)]
            assert len(matching) == 0, f"{cmd} should have been removed"
