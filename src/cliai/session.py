"""
Chat session manager for CLIAI.

Manages in-memory conversation history and per-exchange logging.
Each prompt+response pair is saved as an individual JSON file.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import DATA_DIR


EXCHANGES_DIR = DATA_DIR / "exchanges"
EXCHANGES_DIR.mkdir(parents=True, exist_ok=True)


class ChatSession:
    """Manages a single chat conversation's in-memory state."""

    def __init__(self, system_prompt: str = ""):
        self._messages: list[dict] = []
        self._system_prompt: str = system_prompt
        self._system_hint: str = ""  # Ephemeral per-turn system hint

    # ── Properties ──────────────────────────────────────────────

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str):
        self._system_prompt = value

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

    def add_assistant(self, content: str):
        """Add an assistant message to history."""
        self._messages.append({"role": "assistant", "content": content})

    def add_system_hint(self, hint: str):
        """Set an ephemeral system hint (cleared after each turn)."""
        self._system_hint = hint

    def clear_system_hint(self):
        """Remove the ephemeral system hint."""
        self._system_hint = ""

    def get_messages(self) -> list[dict]:
        """
        Get the full message list for API submission.
        Prepends system prompt and any ephemeral hint.
        """
        messages = []
        prompt_parts = []
        if self._system_prompt:
            prompt_parts.append(self._system_prompt)
        if self._system_hint:
            prompt_parts.append(self._system_hint)
        if prompt_parts:
            messages.append({"role": "system", "content": "\n\n".join(prompt_parts)})
        messages.extend(self._messages)
        return messages

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

        return user_content

    def clear(self):
        """Clear all messages (keeps system prompt)."""
        self._messages.clear()

    def trim_to_last_n(self, n: int):
        """Keep only the last n messages (for context window management)."""
        if len(self._messages) > n:
            self._messages = self._messages[-n:]


def save_exchange(
    prompt: str,
    response: str,
    model: str,
    endpoint: str,
) -> Path:
    """
    Save a single prompt+response exchange as an individual JSON file.

    Args:
        prompt: The user's prompt text.
        response: The assistant's response text.
        model: Model name used.
        endpoint: API endpoint used.

    Returns:
        Path to the saved exchange file.
    """
    timestamp = datetime.now(timezone.utc)
    filename = timestamp.strftime("%Y%m%d_%H%M%S_%f") + ".json"
    filepath = EXCHANGES_DIR / filename

    data = {
        "timestamp": timestamp.isoformat(),
        "model": model,
        "endpoint": endpoint,
        "prompt": prompt,
        "response": response,
    }

    filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return filepath
