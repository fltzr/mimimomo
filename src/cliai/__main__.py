"""Entry point for `python -m cliai`."""
import sys
from pathlib import Path

# Ensure the package directory is on sys.path so bare imports work
_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from chat_cli import app

if __name__ == "__main__":
    app()
