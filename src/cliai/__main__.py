"""Entry point for `python -m cliai` and `cliai` console script."""
import sys
from pathlib import Path

# Ensure the package directory is on sys.path so bare imports work
# (e.g. `from config import ...` instead of `from cliai.config import ...`)
_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)


def main():
    """Console script entry point."""
    from chat_cli import app
    app()


if __name__ == "__main__":
    main()
