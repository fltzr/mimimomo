"""
Slash commands for CLIAI.

Parses and dispatches /commands within the chat loop.
"""

from typing import Optional


# Command registry: command_name -> description
COMMAND_HELP = {
    "/help": "Show this help message",
    "/clear": "Clear conversation history",
    "/save [name]": "Save session to file",
    "/load <name>": "Load session from file",
    "/sessions": "List saved sessions",
    "/delete <name>": "Delete a saved session",
    "/model [name]": "Show or switch the current model",
    "/system [prompt]": "Show or set the system prompt",
    "/retry": "Re-send the last user message",
    "/info": "Show current configuration",
    "/exit": "Quit the chat",
}


def parse_command(user_input: str) -> Optional[tuple[str, str]]:
    """
    Parse a slash command from user input.

    Returns:
        (command, args) tuple, or None if not a command.
    """
    stripped = user_input.strip()
    if not stripped.startswith("/"):
        return None

    parts = stripped.split(None, 1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    return (command, args)


class CommandHandler:
    """
    Handles slash command dispatch.

    This class ties together session, ui, config, and client
    to execute commands that affect chat state.
    """

    def __init__(self, session, ui, config, client):
        self.session = session
        self.ui = ui
        self.config = config
        self.client = client
        self._should_exit = False
        self._retry_message: Optional[str] = None

    @property
    def should_exit(self) -> bool:
        return self._should_exit

    @property
    def retry_message(self) -> Optional[str]:
        """If set, the chat loop should re-send this message."""
        msg = self._retry_message
        self._retry_message = None
        return msg

    def handle(self, command: str, args: str) -> bool:
        """
        Execute a slash command.

        Returns:
            True if the command was handled, False if unknown.
        """
        dispatch = {
            "/help": self._cmd_help,
            "/clear": self._cmd_clear,
            "/save": self._cmd_save,
            "/load": self._cmd_load,
            "/sessions": self._cmd_sessions,
            "/delete": self._cmd_delete,
            "/model": self._cmd_model,
            "/system": self._cmd_system,
            "/retry": self._cmd_retry,
            "/info": self._cmd_info,
            "/exit": self._cmd_exit,
            "/quit": self._cmd_exit,
        }

        handler = dispatch.get(command)
        if handler:
            handler(args)
            return True

        self.ui.show_error(f"Unknown command: {command}. Type /help for available commands.")
        return False

    # ── Command Implementations ─────────────────────────────────

    def _cmd_help(self, args: str):
        self.ui.show_help(COMMAND_HELP)

    def _cmd_clear(self, args: str):
        self.session.clear()
        self.ui.show_success("Conversation cleared.")

    def _cmd_save(self, args: str):
        name = args.strip() or None
        try:
            filepath = self.session.save(name)
            self.ui.show_success(f"Session saved: {filepath}")
        except Exception as e:
            self.ui.show_error(f"Failed to save session: {e}")

    def _cmd_load(self, args: str):
        name = args.strip()
        if not name:
            self.ui.show_error("Usage: /load <session_name>")
            return

        try:
            from session import ChatSession
            loaded = ChatSession.load(name)
            # Transfer state to current session
            self.session._messages = loaded._messages
            self.session._system_prompt = loaded._system_prompt
            self.session._session_name = loaded._session_name
            self.session._created_at = loaded._created_at
            self.ui.show_success(
                f"Loaded session '{loaded.session_name}' "
                f"({loaded.message_count} messages)"
            )
        except FileNotFoundError:
            self.ui.show_error(f"Session not found: {name}")
        except Exception as e:
            self.ui.show_error(f"Failed to load session: {e}")

    def _cmd_sessions(self, args: str):
        from session import ChatSession
        sessions = ChatSession.list_sessions()
        self.ui.show_sessions_list(sessions)

    def _cmd_delete(self, args: str):
        name = args.strip()
        if not name:
            self.ui.show_error("Usage: /delete <session_name>")
            return

        from session import ChatSession
        if ChatSession.delete_session(name):
            self.ui.show_success(f"Deleted session: {name}")
        else:
            self.ui.show_error(f"Session not found: {name}")

    def _cmd_model(self, args: str):
        name = args.strip()
        if not name:
            self.ui.show_info_msg(f"Current model: {self.config.model}")
            return

        old_model = self.config.model
        self.config.model = name
        # Update client config too
        self.client.config.model = name
        self.ui.show_success(f"Model switched: {old_model} → {name}")

    def _cmd_system(self, args: str):
        prompt = args.strip()
        if not prompt:
            current = self.session.system_prompt or "(none)"
            self.ui.show_info_msg(f"System prompt: {current}")
            return

        self.session.system_prompt = prompt
        self.ui.show_success("System prompt updated.")

    def _cmd_retry(self, args: str):
        last_msg = self.session.pop_last_exchange()
        if last_msg:
            self._retry_message = last_msg
            self.ui.show_info_msg("Retrying last message…")
        else:
            self.ui.show_error("No previous message to retry.")

    def _cmd_info(self, args: str):
        self.ui.show_info(self.config)

    def _cmd_exit(self, args: str):
        self._should_exit = True
