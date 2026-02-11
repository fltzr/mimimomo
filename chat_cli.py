#!/usr/bin/env python3
"""
CLIAI — A robust CLI AI chat platform.

Works seamlessly with any OpenAI-compatible endpoint and API key.
Supports streaming, session persistence, slash commands, and named profiles.
"""

import sys
from pathlib import Path
from typing import Optional

import typer

# Ensure project root is on sys.path for imports
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config, load_config, create_default_config, list_profiles, CONFIG_FILE
from client import ChatClient
from session import ChatSession
from ui import ChatUI
from commands import CommandHandler, parse_command

app = typer.Typer(
    name="cliai",
    help="⚡ CLIAI — CLI AI chat for any OpenAI-compatible endpoint.",
    add_completion=False,
    no_args_is_help=False,
)


# ── Main Chat Command ──────────────────────────────────────────


@app.command()
def chat(
    endpoint: Optional[str] = typer.Option(
        None, "--endpoint", "-e",
        help="API endpoint URL (e.g. https://api.openai.com/v1)",
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k",
        help="API key for authentication",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model name to use",
    ),
    profile: Optional[str] = typer.Option(
        None, "--profile", "-p",
        help="Named profile from config file",
    ),
    system: Optional[str] = typer.Option(
        None, "--system", "-s",
        help="System prompt to prepend",
    ),
    temperature: Optional[float] = typer.Option(
        None, "--temperature", "-t",
        help="Sampling temperature (0.0-2.0)",
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens",
        help="Maximum tokens in the response",
    ),
    no_stream: bool = typer.Option(
        False, "--no-stream",
        help="Disable streaming (wait for full response)",
    ),
    load_session: Optional[str] = typer.Option(
        None, "--load", "-l",
        help="Load a saved session by name",
    ),
):
    """Start an interactive chat session."""

    # Create default config on first run
    create_default_config()

    # Build CLI overrides dict (only non-None values)
    cli_overrides = {}
    if endpoint is not None:
        cli_overrides["endpoint"] = endpoint
    if api_key is not None:
        cli_overrides["api_key"] = api_key
    if model is not None:
        cli_overrides["model"] = model
    if temperature is not None:
        cli_overrides["temperature"] = temperature
    if max_tokens is not None:
        cli_overrides["max_tokens"] = max_tokens
    if no_stream:
        cli_overrides["stream"] = False
    if system is not None:
        cli_overrides["system_prompt"] = system

    # Load config
    config = load_config(profile=profile, cli_overrides=cli_overrides)

    # Initialize components
    client = ChatClient(config)
    session = ChatSession(system_prompt=config.system_prompt)
    ui = ChatUI(config)
    cmd_handler = CommandHandler(session, ui, config, client)

    # Optionally load a previous session
    if load_session:
        try:
            loaded = ChatSession.load(load_session)
            session._messages = loaded._messages
            session._system_prompt = loaded._system_prompt
            session._session_name = loaded._session_name
            ui.show_success(
                f"Loaded session '{loaded.session_name}' "
                f"({loaded.message_count} messages)"
            )
        except FileNotFoundError:
            ui.show_error(f"Session not found: {load_session}")
            raise typer.Exit(1)
        except Exception as e:
            ui.show_error(f"Failed to load session: {e}")
            raise typer.Exit(1)

    # Show welcome
    ui.show_welcome()

    # ── Chat Loop ───────────────────────────────────────────

    try:
        while True:
            # Check for retry from command handler
            retry_msg = cmd_handler.retry_message
            if retry_msg:
                user_input = retry_msg
            else:
                user_input = ui.get_input()

            # EOF → exit
            if user_input is None:
                break

            # Empty input → skip
            if not user_input.strip():
                continue

            # Check for slash commands
            parsed = parse_command(user_input)
            if parsed:
                cmd, args = parsed
                cmd_handler.handle(cmd, args)

                if cmd_handler.should_exit:
                    break

                # Check if a retry was queued
                retry_msg = cmd_handler.retry_message
                if retry_msg:
                    user_input = retry_msg
                else:
                    continue

            # Add user message to session
            session.add_user(user_input)

            # Send to API
            if config.stream:
                chunks = client.stream_chat(session.get_messages())
                response_text, usage = ui.stream_response(chunks)
            else:
                result = client.send_chat(session.get_messages())
                if result.error:
                    ui.show_error(result.error)
                    # Remove the user message since we failed
                    session._messages.pop()
                    continue
                response_text = result.delta_content
                ui.show_non_stream_response(response_text, result.usage)

            # Add assistant response to history
            if response_text:
                session.add_assistant(response_text)

    except KeyboardInterrupt:
        pass
    finally:
        # Auto-save if there's content
        if not session.is_empty:
            try:
                path = session.save()
                ui.show_info_msg(f"Session auto-saved: {path}")
            except Exception:
                pass

        ui.show_goodbye()


# ── Config Management Command ──────────────────────────────────


@app.command("config")
def show_config(
    profile: Optional[str] = typer.Option(
        None, "--profile", "-p",
        help="Show a specific profile",
    ),
    init: bool = typer.Option(
        False, "--init",
        help="Create default config file",
    ),
):
    """Show or initialize configuration."""
    if init:
        path = create_default_config()
        typer.echo(f"Config file: {path}")
        return

    config = load_config(profile=profile)
    ui = ChatUI(config)
    ui.show_info(config)

    # Show file location
    typer.echo(f"\nConfig file: {CONFIG_FILE}")

    # List all profiles
    profiles = list_profiles()
    if profiles:
        typer.echo(f"Available profiles: {', '.join(profiles.keys())}")


# ── Sessions Command ───────────────────────────────────────────


@app.command("sessions")
def show_sessions():
    """List saved chat sessions."""
    from ui import ChatUI
    config = Config()
    ui = ChatUI(config)
    sessions = ChatSession.list_sessions()
    ui.show_sessions_list(sessions)


# ── Entry Point ────────────────────────────────────────────────

if __name__ == "__main__":
    app()
