from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .benchmark import import_all
from .io_utils import read_json, sha256_file, utc_now, write_json


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


def _extract_case_file(argv: list[str]) -> tuple[list[str], str | None]:
    cleaned: list[str] = []
    case_file: str | None = None
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == "--case-file":
            if index + 1 >= len(argv):
                raise ValueError("--case-file requires a JSON path")
            if case_file is not None:
                raise ValueError("--case-file may be specified only once")
            case_file = argv[index + 1].strip()
            index += 2
            continue
        if value.startswith("--case-file="):
            if case_file is not None:
                raise ValueError("--case-file may be specified only once")
            case_file = value.split("=", 1)[1].strip()
            index += 1
            continue
        cleaned.append(value)
        index += 1
    if case_file == "":
        raise ValueError("--case-file cannot be empty")
    return cleaned, case_file


def _case_ids_from_file(root: Path, raw_path: str) -> tuple[list[str], Path, dict[str, Any] | None]:
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Case selection file not found: {path}")
    payload = read_json(path)
    case_ids = payload.get("case_ids") if isinstance(payload, dict) else payload
    if not isinstance(case_ids, list) or not case_ids:
        raise ValueError(f"Case selection file must contain a non-empty case_ids list: {path}")
    normalized = [str(case_id).strip() for case_id in case_ids]
    if any(not case_id for case_id in normalized):
        raise ValueError(f"Case selection file contains an empty case ID: {path}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Case selection file contains duplicate case IDs: {path}")
    return normalized, path, payload if isinstance(payload, dict) else None


def _source_benchmarks_from_selection(
    root: Path,
    payload: dict[str, Any] | None,
) -> Path | None:
    if not payload:
        return None
    raw = payload.get("source_benchmarks_dir")
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("source_benchmarks_dir in case selection file must be a non-empty path")
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not (path / "index.json").exists():
        raise FileNotFoundError(f"Frozen source benchmark index not found: {path / 'index.json'}")
    return path


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
    cleaned, case_file_raw = _extract_case_file(cleaned)
    case_file: Path | None = None
    case_file_payload: dict[str, Any] | None = None
    if requested and case_file_raw:
        raise ValueError("--case and --case-file cannot be combined")
    if case_file_raw:
        requested, case_file, case_file_payload = _case_ids_from_file(root, case_file_raw)
    if not requested:
        return cleaned
    if _option_value(cleaned, "--benchmarks-dir"):
        raise ValueError("--case cannot be combined with --benchmarks-dir")

    run_id = _option_value(cleaned, "--run-id")
    if not run_id:
        run_id = datetime.now().astimezone().strftime("run-%Y%m%d-%H%M%S")
        cleaned.extend(["--run-id", run_id])

    set_name, entry = _resolve_set(root, _option_value(cleaned, "--benchmark-set"))
    normalized_dir = _source_benchmarks_from_selection(root, case_file_payload)
    if normalized_dir is None:
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
    existing_index_path = selection_dir / "index.json"
    if existing_index_path.exists():
        existing = read_json(existing_index_path)
        existing_ids = ((existing.get("selection") or {}).get("case_ids") or [])
        if existing_ids != requested:
            raise ValueError(
                f"Run {run_id!r} already has a different benchmark selection; "
                "use a new run ID instead of overwriting it"
            )
        print(f"[benchmarks] Reusing frozen {len(requested)}-case selection for run '{run_id}'")
        return cleaned + ["--benchmarks-dir", str(selection_dir)]

    selection_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    for case_id in requested:
        row = by_id[case_id]
        source = normalized_dir / row["file"]
        if not source.exists():
            raise FileNotFoundError(f"Normalized benchmark file not found: {source}")
        shutil.copy2(source, selection_dir / row["file"])
        selected_rows.append(row)
    selection_metadata = {
        "benchmark_set": set_name,
        "case_ids": requested,
        "source_normalized_dir": str(normalized_dir),
    }
    if case_file is not None:
        selection_metadata.update(
            {
                "case_file": str(case_file),
                "case_file_sha256": sha256_file(case_file),
            }
        )
    write_json(
        selection_dir / "index.json",
        {
            "schema_version": index.get("schema_version", "1.0"),
            "generated_at": utc_now(),
            "selection": selection_metadata,
            "cases": selected_rows,
        },
    )
    print(f"[benchmarks] Selected {len(requested)} case(s): {', '.join(requested)}")
    return cleaned + ["--benchmarks-dir", str(selection_dir)]
