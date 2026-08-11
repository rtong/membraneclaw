from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .benchmark import iter_benchmarks
from .io_utils import read_json, sha256_tree, stable_hash, utc_now, write_json
from .openwebui import OpenWebUIClient, OpenWebUIError


def load_systems(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    systems = config.get("systems") or []
    if [item.get("id") for item in systems] != ["baseline", "environment", "environment-skill"]:
        raise ValueError(
            "systems.json must define baseline, environment, environment-skill in that order"
        )
    for system in systems:
        env_name = system.get("model_env")
        model_id = os.environ.get(env_name or "", "").strip()
        if not model_id:
            raise ValueError(f"Missing model ID environment variable: {env_name}")
        system["model_id"] = model_id
        rag_version_env = system.get("rag_version_env")
        if system.get("rag_enabled"):
            system["rag_version"] = os.environ.get(rag_version_env or "", "unversioned").strip() or "unversioned"
        skill_version = system.get("skill_version")
        if skill_version:
            skill_id, version = skill_version.split("@", 1)
            skill_dir = config_path.parents[1] / "skills" / skill_id / f"v{version}"
            if not skill_dir.exists():
                raise ValueError(f"Configured Skill directory does not exist: {skill_dir}")
            system["skill_artifact_sha256"] = sha256_tree(skill_dir)
    return config


def make_client(generation: dict[str, Any]) -> OpenWebUIClient:
    base_url = os.environ.get("OPENWEBUI_BASE_URL", "").strip()
    api_key = os.environ.get("OPENWEBUI_API_KEY", "").strip()
    if not base_url:
        raise ValueError("OPENWEBUI_BASE_URL is missing; fill it in .env")
    if not api_key:
        raise ValueError("OPENWEBUI_API_KEY is missing; fill it in .env")
    return OpenWebUIClient(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=int(generation.get("timeout_seconds", 600)),
    )


def _record_path(run_dir: Path, case_id: str, system_id: str) -> Path:
    return run_dir / "responses" / f"{case_id}__{system_id}.json"


def execute_run(
    *,
    benchmarks_dir: Path,
    systems_path: Path,
    run_dir: Path,
    force: bool = False,
) -> dict[str, int]:
    config = load_systems(systems_path)
    generation = config["generation"]
    client = make_client(generation)
    run_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = list(iter_benchmarks(benchmarks_dir))

    manifest = {
        "schema_version": "1.0",
        "run_id": run_dir.name,
        "created_at": utc_now(),
        "benchmark_cases": [item["case_id"] for item in benchmarks],
        "systems": config["systems"],
        "generation": generation,
        "shared_system_prompt": config["shared_system_prompt"],
        "openwebui_base_url": os.environ.get("OPENWEBUI_BASE_URL", "").rstrip("/"),
    }
    write_json(run_dir / "manifest.json", manifest)

    counts = {"success": 0, "error": 0, "skipped": 0}
    total = len(benchmarks) * len(config["systems"])
    position = 0
    for benchmark in benchmarks:
        for system in config["systems"]:
            position += 1
            label = f"[{position}/{total}] {benchmark['case_id']} / {system['id']}"
            system_generation = {
                **generation,
                **(system.get("generation_overrides") or {}),
            }
            messages = [
                {"role": "system", "content": config["shared_system_prompt"]},
                {"role": "user", "content": benchmark["question_prompt"]},
            ]
            request_fingerprint = stable_hash(
                {
                    "case_sha256": benchmark["source"]["sha256"],
                    "system": system,
                    "generation": system_generation,
                    "messages": messages,
                }
            )
            path = _record_path(run_dir, benchmark["case_id"], system["id"])
            if path.exists() and not force:
                previous = read_json(path)
                if previous.get("status") == "success" and previous.get("request_hash") == request_fingerprint:
                    counts["skipped"] += 1
                    print(f"{label}: skipped (matching successful response)", flush=True)
                    continue

            print(f"{label}: running", flush=True)

            base_record = {
                "schema_version": "1.0",
                "run_id": run_dir.name,
                "case_id": benchmark["case_id"],
                "system_id": system["id"],
                "display_name": system["display_name"],
                "model_id": system["model_id"],
                "tools_enabled": system.get("tools_enabled", False),
                "skill_version": system.get("skill_version"),
                "rag_enabled": system.get("rag_enabled", False),
                "request_hash": request_fingerprint,
                "started_at": utc_now(),
            }
            last_error = None
            max_retries = int(system_generation.get("max_retries", 2))
            for attempt in range(max_retries + 1):
                try:
                    chat_method = client.chat_stream if system_generation.get("stream") else client.chat
                    result = chat_method(
                        model=system["model_id"], messages=messages, generation=system_generation
                    )
                    write_json(
                        path,
                        {
                            **base_record,
                            "status": "success",
                            "attempts": attempt + 1,
                            "completed_at": utc_now(),
                            "latency_ms": result.latency_ms,
                            "response_text": result.content,
                            "usage": result.raw.get("usage"),
                            "generation": system_generation,
                            "raw_response": result.raw,
                        },
                    )
                    counts["success"] += 1
                    print(
                        f"{label}: success in {result.latency_ms / 1000:.1f}s "
                        f"(attempt {attempt + 1})",
                        flush=True,
                    )
                    break
                except OpenWebUIError as exc:
                    last_error = str(exc)
                    if attempt < max_retries:
                        print(
                            f"{label}: attempt {attempt + 1} failed; retrying: {last_error}",
                            flush=True,
                        )
                        time.sleep(min(2 ** attempt, 8))
            else:
                write_json(
                    path,
                    {
                        **base_record,
                        "status": "error",
                        "attempts": max_retries + 1,
                        "completed_at": utc_now(),
                        "error": last_error,
                    },
                )
                counts["error"] += 1
                print(f"{label}: error after {max_retries + 1} attempts: {last_error}", flush=True)
    return counts
