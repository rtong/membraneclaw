from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .benchmark import import_all
from .io_utils import read_json, utc_now, write_json


def _option_value(argv: list[str], name: str) -> str | None:
    for index, value in enumerate(argv):
        if value == name:
            if index + 1 >= len(argv):
                raise ValueError(f"{name} requires a value")
            return argv[index + 1]
        prefix = name + "="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def _extract_cases(argv: list[str]) -> tuple[list[str], list[str]]:
    cleaned: list[str] = []
    cases: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == "--case":
            if index + 1 >= len(argv):
                raise ValueError("--case requires a case ID")
            cases.append(argv[index + 1].strip())
            index += 2
            continue
        if value.startswith("--case="):
            cases.append(value.split("=", 1)[1].strip())
            index += 1
            continue
        cleaned.append(value)
        index += 1
    if any(not case_id for case_id in cases):
        raise ValueError("--case cannot be empty")
    return cleaned, list(dict.fromkeys(cases))


def _resolve_set(root: Path, set_name: str | None) -> tuple[str, dict[str, Any]]:
    registry = read_json(root / "configs" / "benchmark_sets.json")
    selected = set_name or registry.get("active_set")
    entry = (registry.get("sets") or {}).get(selected)
    if not selected or not isinstance(entry, dict):
        available = sorted((registry.get("sets") or {}).keys())
        raise ValueError(f"Unknown benchmark set {selected!r}; available: {available}")
    return selected, entry


def prepare_selected_argv(argv: list[str], root: Path) -> list[str]:
    """Materialize repeatable run/auto --case selections inside the run directory."""
    if not argv or argv[0] not in {"run", "auto", "router-eval"}:
        return list(argv)

    cleaned, requested = _extract_cases(list(argv))
    if not requested:
        return cleaned
    if _option_value(cleaned, "--benchmarks-dir"):
        raise ValueError("--case cannot be combined with --benchmarks-dir")

    run_id = _option_value(cleaned, "--run-id")
    if not run_id:
        run_id = datetime.now().astimezone().strftime("run-%Y%m%d-%H%M%S")
        cleaned.extend(["--run-id", run_id])

    set_name, entry = _resolve_set(root, _option_value(cleaned, "--benchmark-set"))
    normalized_raw = entry.get("normalized_dir")
    if not normalized_raw:
        raise ValueError(f"Benchmark set {set_name!r} is missing normalized_dir")
    normalized_dir = (root / normalized_raw).resolve()
    source_raw = entry.get("source_config")
    if source_raw:
        import_all((root / source_raw).resolve(), normalized_dir, root)
    elif not normalized_dir.exists():
        raise FileNotFoundError(f"Normalized benchmark directory not found: {normalized_dir}")

    index = read_json(normalized_dir / "index.json")
    by_id = {row["case_id"]: row for row in index.get("cases", [])}
    missing = [case_id for case_id in requested if case_id not in by_id]
    if missing:
        raise ValueError(
            f"Unknown benchmark case ID(s): {missing}. Available cases: {sorted(by_id)}"
        )

    runs_root = (root / "runs").resolve()
    selection_dir = (runs_root / run_id / "benchmarks").resolve()
    if not selection_dir.is_relative_to(runs_root):
        raise ValueError(f"Invalid run ID outside runs directory: {run_id}")
    selection_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    for case_id in requested:
        row = by_id[case_id]
        source = normalized_dir / row["file"]
        if not source.exists():
            raise FileNotFoundError(f"Normalized benchmark file not found: {source}")
        shutil.copy2(source, selection_dir / row["file"])
        selected_rows.append(row)
    write_json(
        selection_dir / "index.json",
        {
            "schema_version": index.get("schema_version", "1.0"),
            "generated_at": utc_now(),
            "selection": {
                "benchmark_set": set_name,
                "case_ids": requested,
                "source_normalized_dir": str(normalized_dir),
            },
            "cases": selected_rows,
        },
    )
    print(f"[benchmarks] Selected {len(requested)} case(s): {', '.join(requested)}")
    return cleaned + ["--benchmarks-dir", str(selection_dir)]
