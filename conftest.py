"""Root conftest — adds src/cliai to sys.path for all tests."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src" / "cliai"))
sys.path.insert(0, str(root / "src"))
