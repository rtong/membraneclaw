from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_evaluate.cli import main  # noqa: E402
from auto_evaluate.selection import prepare_selected_argv  # noqa: E402


if __name__ == "__main__":
    try:
        selected_argv = prepare_selected_argv(sys.argv[1:], ROOT)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(main(selected_argv))
