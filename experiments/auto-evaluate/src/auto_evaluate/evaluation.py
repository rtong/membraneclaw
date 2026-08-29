from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json


def profiles_path(root: Path) -> Path:
    return root / "configs" / "evaluation_profiles.json"


def load_profiles(root: Path) -> dict[str, Any]:
    path = profiles_path(root)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation profile registry not found: {path}")
    payload = read_json(path)
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("evaluation_profiles.json must define at least one profile")
    return payload


def resolve_profile(root: Path, benchmark_entry: dict[str, Any] | None, explicit_profile: str | None = None) -> tuple[str, dict[str, Any]]:
    registry = load_profiles(root)
    profile_id = explicit_profile or (benchmark_entry or {}).get("evaluation_profile") or registry.get("default_profile")
    profile = (registry.get("profiles") or {}).get(profile_id)
    if not isinstance(profile, dict):
        available = ", ".join(sorted(registry.get("profiles") or {}))
        raise ValueError(f"Unknown evaluation profile '{profile_id}'. Available profiles: {available}")
    if not isinstance(profile.get("system_ids"), list) or not profile["system_ids"]:
        raise ValueError(f"Evaluation profile '{profile_id}' has no system_ids")
    if not isinstance(profile.get("teachers"), list):
        raise ValueError(f"Evaluation profile '{profile_id}' must define teachers as a list")
    return str(profile_id), profile


def write_profile_snapshot(run_dir: Path, profile_id: str, profile: dict[str, Any], benchmark_set: str | None) -> Path:
    output = run_dir / "evaluation_profile.json"
    write_json(output, {"schema_version": "1.0", "profile_id": profile_id, "benchmark_set": benchmark_set, **profile})
    return output


def load_run_profile(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "evaluation_profile.json"
    return read_json(path) if path.exists() else None
