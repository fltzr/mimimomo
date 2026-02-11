"""
Chat session manager for CLIAI.

Manages conversation history, system prompts, persistence,
and context window management.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import SESSIONS_DIR


class ChatSession:
    """Manages a single chat conversation's state and history."""

    def __init__(self, system_prompt: str = ""):
        self._messages: list[dict] = []
        self._system_prompt: str = system_prompt
        self._session_name: Optional[str] = None
        self._created_at: str = datetime.now(timezone.utc).isoformat()
        self._modified_at: str = self._created_at

    # ── Properties ──────────────────────────────────────────────

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str):
        self._system_prompt = value

    @property
    def session_name(self) -> Optional[str]:
        return self._session_name

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def is_empty(self) -> bool:
        return len(self._messages) == 0

    # ── Message Management ──────────────────────────────────────

    def add_user(self, content: str):
        """Add a user message to history."""
        self._messages.append({"role": "user", "content": content})
        self._touch()

    def add_assistant(self, content: str):
        """Add an assistant message to history."""
        self._messages.append({"role": "assistant", "content": content})
        self._touch()

    def get_messages(self) -> list[dict]:
        """
        Get the full message list for API submission.
        Prepends system prompt if set.
        """
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(self._messages)
        return messages

    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message content, or None."""
        for msg in reversed(self._messages):
            if msg["role"] == "user":
                return msg["content"]
        return None

    def pop_last_exchange(self) -> Optional[str]:
        """
        Remove the last user+assistant exchange.
        Returns the user message content for retry, or None.
        """
        # Find and remove last assistant message
        for i in range(len(self._messages) - 1, -1, -1):
            if self._messages[i]["role"] == "assistant":
                self._messages.pop(i)
                break

        # Find and remove last user message
        user_content = None
        for i in range(len(self._messages) - 1, -1, -1):
            if self._messages[i]["role"] == "user":
                user_content = self._messages.pop(i)["content"]
                break

        self._touch()
        return user_content

    def clear(self):
        """Clear all messages (keeps system prompt)."""
        self._messages.clear()
        self._touch()

    def trim_to_last_n(self, n: int):
        """Keep only the last n messages (for context window management)."""
        if len(self._messages) > n:
            self._messages = self._messages[-n:]
            self._touch()

    # ── Persistence ─────────────────────────────────────────────

    def save(self, name: Optional[str] = None) -> Path:
        """
        Save session to a JSON file.

        Args:
            name: Session name. If not provided, auto-generates from timestamp.

        Returns:
            Path to the saved session file.
        """
        if name:
            self._session_name = name
        elif not self._session_name:
            self._session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")

        safe_name = _sanitize_filename(self._session_name)
        filepath = SESSIONS_DIR / f"{safe_name}.json"

        data = {
            "name": self._session_name,
            "created_at": self._created_at,
            "modified_at": self._modified_at,
            "system_prompt": self._system_prompt,
            "messages": self._messages,
        }

        filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return filepath

    @classmethod
    def load(cls, name: str) -> "ChatSession":
        """
        Load a session from a JSON file.

        Args:
            name: Session name (with or without .json extension).

        Returns:
            ChatSession instance with restored state.

        Raises:
            FileNotFoundError: If the session file doesn't exist.
        """
        safe_name = _sanitize_filename(name)
        filepath = SESSIONS_DIR / f"{safe_name}.json"

        if not filepath.exists():
            # Try exact name with .json
            filepath = SESSIONS_DIR / name
            if not filepath.exists():
                filepath = SESSIONS_DIR / f"{name}.json"
                if not filepath.exists():
                    raise FileNotFoundError(f"Session not found: {name}")

        data = json.loads(filepath.read_text())

        session = cls(system_prompt=data.get("system_prompt", ""))
        session._messages = data.get("messages", [])
        session._session_name = data.get("name", name)
        session._created_at = data.get("created_at", "")
        session._modified_at = data.get("modified_at", "")

        return session

    @staticmethod
    def list_sessions() -> list[dict]:
        """
        List all saved sessions.

        Returns:
            List of dicts with 'name', 'file', 'created_at', 'message_count'.
        """
        sessions = []
        for filepath in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
            try:
                data = json.loads(filepath.read_text())
                sessions.append(
                    {
                        "name": data.get("name", filepath.stem),
                        "file": filepath.name,
                        "created_at": data.get("created_at", ""),
                        "message_count": len(data.get("messages", [])),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                sessions.append(
                    {
                        "name": filepath.stem,
                        "file": filepath.name,
                        "created_at": "",
                        "message_count": 0,
                    }
                )
        return sessions

    @staticmethod
    def delete_session(name: str) -> bool:
        """Delete a saved session. Returns True if deleted."""
        safe_name = _sanitize_filename(name)
        filepath = SESSIONS_DIR / f"{safe_name}.json"

        if not filepath.exists():
            filepath = SESSIONS_DIR / f"{name}.json"

        if filepath.exists():
            filepath.unlink()
            return True
        return False

    # ── Internal ────────────────────────────────────────────────

    def _touch(self):
        """Update modified timestamp."""
        self._modified_at = datetime.now(timezone.utc).isoformat()


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    # Replace whitespace and special chars
    sanitized = re.sub(r"[^\w\-.]", "_", name)
    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip("_.")
    return sanitized or "unnamed"
