from __future__ import annotations

import json
import sys
import time
from typing import Any

from .codex_automation import (
    _RESULT_PREFIX,
    build_repair_prompt,
    parse_json_object,
    validate_task_output,
)


def _thread_id(thread: Any, result: Any) -> str | None:
    value = getattr(thread, "id", None) or getattr(result, "thread_id", None)
    return str(value) if value is not None else None


def run(request: dict[str, Any]) -> dict[str, Any]:
    try:
        from openai_codex import Codex, Sandbox
    except ImportError as exc:
        raise RuntimeError(
            "The active Miniforge environment does not contain openai-codex. "
            "Run: python -m pip install -e ."
        ) from exc

    stage = request["stage"]
    task = request["task"]
    prompt = request["prompt"]
    retries = int(request.get("retries", 2))
    model = request["model"]
    started = time.perf_counter()
    last_errors: list[str] = []
    with Codex() as codex:
        sandbox = getattr(Sandbox, "read_only", Sandbox.workspace_write)
        thread = codex.thread_start(model=model, sandbox=sandbox)
        result = None
        for attempt in range(1, retries + 2):
            result = thread.run(prompt)
            response = getattr(result, "final_response", None)
            if not isinstance(response, str) or not response.strip():
                last_errors = ["Codex returned an empty final_response"]
            else:
                try:
                    output = parse_json_object(response)
                    last_errors = validate_task_output(stage, task, output)
                except ValueError as exc:
                    last_errors = [str(exc)]
                if not last_errors:
                    return {
                        "ok": True,
                        "output": output,
                        "thread_id": _thread_id(thread, result),
                        "attempts": attempt,
                        "latency_ms": round((time.perf_counter() - started) * 1000),
                    }
            if attempt <= retries:
                prompt = build_repair_prompt(stage, task, last_errors)
    raise RuntimeError("Codex output remained invalid after repair: " + "; ".join(last_errors))


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        result = run(request)
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(_RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
