"""Source-checkout entrypoint; installed users can run the membraneclaw command."""
from membraneclaw_runtime.cli import main, render

if __name__ == "__main__":
    raise SystemExit(main())
