"""
Rich TUI layer for CLIAI.

Handles all terminal rendering: welcome banner, input prompts,
streaming markdown output, spinners, and status displays.
"""

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML
from typing import Optional, Generator

from config import Config, DATA_DIR

# Persistent input history file
HISTORY_FILE = DATA_DIR / "input_history"

console = Console()


class ChatUI:
    """Rich terminal interface for the chat session."""

    def __init__(self, config: Config):
        self.config = config
        self._session = PromptSession(
            history=FileHistory(str(HISTORY_FILE))
        )

    # ── Banner & Info ───────────────────────────────────────────

    def show_welcome(self):
        """Display the welcome banner with config info."""
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_column("Key", style="dim")
        info_table.add_column("Value", style="bold")
        info_table.add_row("Endpoint", self.config.display_endpoint())
        info_table.add_row("Model", self.config.model)
        info_table.add_row("Profile", self.config.profile_name)
        if self.config.api_key:
            info_table.add_row("API Key", self.config.display_key())
        info_table.add_row("Streaming", "✓" if self.config.stream else "✗")

        panel = Panel(
            info_table,
            title="[bold bright_cyan]⚡ CLIAI[/bold bright_cyan]",
            subtitle="[dim]Type /help for commands • /exit to quit[/dim]",
            border_style="bright_cyan",
            padding=(1, 2),
        )
        console.print()
        console.print(panel)
        console.print()

    def show_info(self, config: Config):
        """Display current configuration info."""
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_column("Key", style="dim")
        info_table.add_column("Value", style="bold")
        for key, val in config.to_dict().items():
            if key == "profile_name":
                continue
            info_table.add_row(key, str(val))

        panel = Panel(
            info_table,
            title=f"[bold]Profile: {config.profile_name}[/bold]",
            border_style="bright_cyan",
        )
        console.print(panel)

    # ── Input ───────────────────────────────────────────────────

    def get_input(self) -> Optional[str]:
        """
        Get multi-line user input.
        End a line with \\ to continue on next line.
        Returns None on EOF.
        """
        try:
            user_input = ""
            while True:
                if not user_input:
                    prompt_text = HTML(
                        '<style fg="ansibrightcyan" bg="" bold="true">You › </style>'
                    )
                else:
                    prompt_text = HTML(
                        '<style fg="ansigray">... </style>'
                    )

                line = self._session.prompt(prompt_text)

                if line.strip().endswith("\\"):
                    user_input += line.rstrip()[:-1] + "\n"
                else:
                    user_input += line
                    break

            return user_input
        except KeyboardInterrupt:
            return ""  # Empty string = skip this input
        except EOFError:
            return None  # None = exit

    # ── Output ──────────────────────────────────────────────────

    def stream_response(self, chunks: Generator) -> tuple[str, Optional[dict]]:
        """
        Render streaming chunks as live-updating markdown.

        Args:
            chunks: Generator yielding ChatChunk objects.

        Returns:
            Tuple of (full_response_text, usage_dict_or_None).
        """
        from client import ChatChunk

        full_response = ""
        usage = None
        interrupted = False

        console.print()
        console.print(
            Text("AI ›", style="bold bright_magenta"), end=" "
        )

        try:
            # Show spinner until first content arrives
            first_chunk = True
            with console.status(
                "[dim]Thinking…[/dim]", spinner="dots", spinner_style="bright_magenta"
            ) as status:
                for chunk in chunks:
                    if chunk.error:
                        status.stop()
                        self.show_error(chunk.error)
                        return "", None

                    if chunk.usage:
                        usage = chunk.usage

                    if chunk.delta_content:
                        if first_chunk:
                            status.stop()
                            first_chunk = False
                        break

                    if chunk.finish_reason:
                        status.stop()
                        break
                else:
                    # Generator was empty
                    return "", usage

            # Now stream remaining content with live markdown display
            if chunk.delta_content:
                full_response = chunk.delta_content

            with Live(
                Markdown(full_response),
                console=console,
                refresh_per_second=12,
                vertical_overflow="visible",
            ) as live:
                try:
                    for chunk in chunks:
                        if chunk.error:
                            live.stop()
                            self.show_error(chunk.error)
                            return full_response, usage

                        if chunk.usage:
                            usage = chunk.usage

                        if chunk.delta_content:
                            full_response += chunk.delta_content
                            live.update(Markdown(full_response))

                        if chunk.finish_reason == "interrupted":
                            interrupted = True
                            break

                        if chunk.finish_reason:
                            break
                except KeyboardInterrupt:
                    interrupted = True

        except KeyboardInterrupt:
            interrupted = True

        if interrupted:
            console.print("\n[bold yellow]⚠ Response stopped.[/bold yellow]")

        console.print()

        # Show usage if available
        if usage:
            self._show_usage(usage)

        return full_response, usage

    def show_non_stream_response(self, content: str, usage: Optional[dict] = None):
        """Display a non-streamed response."""
        console.print()
        console.print(Text("AI ›", style="bold bright_magenta"))
        console.print(Markdown(content))
        console.print()

        if usage:
            self._show_usage(usage)

    def _show_usage(self, usage: dict):
        """Display token usage info."""
        prompt_tokens = usage.get("prompt_tokens", "?")
        completion_tokens = usage.get("completion_tokens", "?")
        total_tokens = usage.get("total_tokens", "?")
        console.print(
            f"  [dim]tokens: {prompt_tokens} prompt + "
            f"{completion_tokens} completion = {total_tokens} total[/dim]"
        )

    # ── Status Messages ─────────────────────────────────────────

    def show_error(self, message: str):
        """Display an error message."""
        console.print(f"\n[bold red]✗ Error:[/bold red] {message}")

    def show_warning(self, message: str):
        """Display a warning message."""
        console.print(f"[bold yellow]⚠ {message}[/bold yellow]")

    def show_success(self, message: str):
        """Display a success message."""
        console.print(f"[bold green]✓ {message}[/bold green]")

    def show_info_msg(self, message: str):
        """Display an info message."""
        console.print(f"[dim]{message}[/dim]")

    def show_goodbye(self):
        """Display goodbye message."""
        console.print("\n[bold bright_cyan]👋 Goodbye![/bold bright_cyan]\n")

    def show_sessions_list(self, sessions: list[dict]):
        """Display a table of saved sessions."""
        if not sessions:
            console.print("[dim]No saved sessions.[/dim]")
            return

        table = Table(title="Saved Sessions", border_style="bright_cyan")
        table.add_column("Name", style="bold")
        table.add_column("Messages", justify="right")
        table.add_column("Created", style="dim")
        table.add_column("File", style="dim")

        for s in sessions:
            created = s.get("created_at", "")[:19].replace("T", " ")
            table.add_row(
                s["name"],
                str(s["message_count"]),
                created,
                s["file"],
            )

        console.print(table)

    def show_help(self, commands: dict[str, str]):
        """Display the help table."""
        table = Table(
            title="Available Commands",
            border_style="bright_cyan",
            show_lines=False,
        )
        table.add_column("Command", style="bold bright_cyan")
        table.add_column("Description")

        for cmd, desc in commands.items():
            table.add_row(cmd, desc)

        console.print(table)
