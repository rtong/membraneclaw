"""Opt-in prototype CLI, separate from ae.py and the current experiment matrix."""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_evaluate.skill_runtime.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
