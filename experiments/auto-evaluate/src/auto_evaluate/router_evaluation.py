from __future__ import annotations

import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .benchmark import iter_benchmarks
from .io_utils import read_json, sha256_tree, stable_hash, utc_now, write_json
from .openwebui import OpenWebUIError
from .routing_metrics import binary_route_metrics, paired_exact_mcnemar
from .runner import classify_execution_error, make_client, parse_rag_route


def load_router_config(
    config_path: Path,
    selected_variant_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Load router variants and materialize their prompts and model IDs."""
    config = read_json(config_path)
    variants = config.get("variants") or []
    ids = [row.get("id") for row in variants]
    if not variants or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("router_evaluation.json must define variants with unique IDs")
    if selected_variant_ids:
        missing = sorted(set(selected_variant_ids) - set(ids))
        if missing:
            raise ValueError(f"Unknown router variant IDs: {missing}")
        by_id = {row["id"]: row for row in variants}
        variants = [by_id[variant_id] for variant_id in selected_variant_ids]

    project_root = config_path.parents[1]
    for variant in variants:
        model_env = variant.get("model_env")
        model_id = variant.get("model_id") or os.environ.get(model_env or "", "").strip()
        if not model_id:
            raise ValueError(
                f"{variant['id']}: missing router model ID environment variable: {model_env}"
            )
        variant["model_id"] = model_id
        skill_version = variant.get("skill_version")
        if skill_version:
            if "@" not in skill_version:
                raise ValueError(f"{variant['id']}: invalid skill_version {skill_version!r}")
            skill_id, version = skill_version.split("@", 1)
            skill_dir = project_root / "skills" / skill_id / f"v{version}"
            skill_path = skill_dir / "SKILL.md"
            if not skill_path.exists():
                raise FileNotFoundError(f"Router Skill not found: {skill_path}")
            variant["prompt"] = skill_path.read_text(encoding="utf-8")
            variant["skill_artifact_sha256"] = sha256_tree(skill_dir)
        prompt = variant.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{variant['id']}: prompt or skill_version is required")
        variant["prompt_sha256"] = stable_hash(prompt)
    config["variants"] = variants
    return config


def _expected_route(benchmark: dict[str, Any], default: str | None) -> str | None:
    expected = (benchmark.get("benchmark_view") or {}).get("expected_route", default)
    if expected not in {None, "use_rag", "skip_rag"}:
        raise ValueError(
            f"{benchmark['case_id']}: expected route must be use_rag or skip_rag"
        )
    return expected


def summarize_router_evaluation(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "router_manifest.json")
    records = [
        read_json(path)
        for path in sorted((run_dir / "router_responses").glob("*.json"))
    ]
    by_variant: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_variant[str(record["variant_id"])].append(record)

    summaries: dict[str, Any] = {}
    for variant in manifest["variants"]:
        variant_id = variant["id"]
        rows = by_variant.get(variant_id, [])
        successful = [row for row in rows if row.get("status") == "success"]
        labeled = [row for row in successful if row.get("expected_action")]
        correct = [
            row for row in labeled if row.get("action") == row.get("expected_action")
        ]
        confidences = [float(row["confidence"]) for row in successful]
        attempt_latencies = [
            float(row["latency_ms"]) for row in rows if row.get("latency_ms")
        ]
        success_latencies = [
            float(row["latency_ms"]) for row in successful if row.get("latency_ms")
        ]
        actions = Counter(row.get("action") for row in successful if row.get("action"))
        errors = Counter(row.get("error_type") or "unknown" for row in rows if row.get("status") != "success")
        classification = binary_route_metrics(
            successful,
            expected_key="expected_action",
            predicted_key="action",
        )
        summaries[variant_id] = {
            "display_name": variant.get("display_name", variant_id),
            "skill_version": variant.get("skill_version"),
            "n_expected": len(manifest["benchmark_cases"]),
            "n_success": len(successful),
            "valid_response_rate": (
                len(successful) / len(manifest["benchmark_cases"])
                if manifest["benchmark_cases"]
                else None
            ),
            "n_labeled": len(labeled),
            "n_correct": len(correct),
            "routing_accuracy": len(correct) / len(labeled) if labeled else None,
            "routing_classification": classification,
            "use_rag_rate": actions["use_rag"] / len(successful) if successful else None,
            "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
            "mean_success_latency_ms": (
                sum(success_latencies) / len(success_latencies)
                if success_latencies
                else None
            ),
            "mean_attempt_latency_ms": (
                sum(attempt_latencies) / len(attempt_latencies)
                if attempt_latencies
                else None
            ),
            "action_counts": dict(actions),
            "error_counts": dict(errors),
        }

    comparison = None
    zero_shot = summaries.get("zero-shot")
    skill = summaries.get("router-skill")
    if zero_shot and skill:
        accuracy_gain = None
        if zero_shot["routing_accuracy"] is not None and skill["routing_accuracy"] is not None:
            accuracy_gain = skill["routing_accuracy"] - zero_shot["routing_accuracy"]
        comparison = {
            "baseline_variant": "zero-shot",
            "candidate_variant": "router-skill",
            "routing_accuracy_gain": accuracy_gain,
            "valid_response_rate_gain": (
                skill["valid_response_rate"] - zero_shot["valid_response_rate"]
            ),
            "paired_mcnemar": paired_exact_mcnemar(
                by_variant.get("zero-shot", []),
                by_variant.get("router-skill", []),
            ),
        }

    return {
        "schema_version": "1.0",
        "run_id": run_dir.name,
        "benchmark_set": manifest.get("benchmark_set"),
        "default_expected_route": manifest.get("default_expected_route"),
        "variants": summaries,
        "skill_effect": comparison,
    }


def execute_router_evaluation(
    *,
    benchmarks_dir: Path,
    config_path: Path,
    run_dir: Path,
    benchmark_set: str | None,
    default_expected_route: str | None,
    selected_variant_ids: list[str] | None = None,
    selected_case_ids: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run short routing calls only; no solver, WaterTAP tool, or RAG call follows."""
    if default_expected_route not in {None, "use_rag", "skip_rag"}:
        raise ValueError("default_expected_route must be use_rag or skip_rag")
    config = load_router_config(config_path, selected_variant_ids)
    generation = {
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 256,
        "max_retries": 0,
        **(config.get("generation") or {}),
    }
    client = make_client(generation)
    benchmarks = list(iter_benchmarks(benchmarks_dir))
    if selected_case_ids:
        by_id = {row["case_id"]: row for row in benchmarks}
        missing = [case_id for case_id in selected_case_ids if case_id not in by_id]
        if missing:
            raise ValueError(f"Router pilot references unknown benchmark cases: {missing}")
        benchmarks = [by_id[case_id] for case_id in selected_case_ids]
    if not benchmarks:
        raise ValueError("Router evaluation selected no benchmark cases")

    run_dir.mkdir(parents=True, exist_ok=True)
    response_dir = run_dir / "router_responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    manifest_variants = [
        {key: value for key, value in variant.items() if key != "prompt"}
        for variant in config["variants"]
    ]
    write_json(
        run_dir / "router_manifest.json",
        {
            "schema_version": "1.0",
            "run_id": run_dir.name,
            "created_at": utc_now(),
            "benchmark_set": benchmark_set,
            "benchmark_cases": [row["case_id"] for row in benchmarks],
            "default_expected_route": default_expected_route,
            "generation": generation,
            "variants": manifest_variants,
        },
    )

    counts = {"success": 0, "error": 0, "skipped": 0}
    total = len(benchmarks) * len(config["variants"])
    position = 0
    for benchmark in benchmarks:
        expected = _expected_route(benchmark, default_expected_route)
        for variant in config["variants"]:
            position += 1
            label = f"[{position}/{total}] {benchmark['case_id']} / {variant['id']}"
            messages = [
                {"role": "system", "content": variant["prompt"]},
                {"role": "user", "content": benchmark["question_prompt"]},
            ]
            fingerprint = stable_hash(
                {
                    "case_sha256": benchmark["source"]["sha256"],
                    "variant_id": variant["id"],
                    "model_id": variant["model_id"],
                    "prompt_sha256": variant["prompt_sha256"],
                    "generation": generation,
                    "messages": messages,
                    "expected_action": expected,
                }
            )
            path = response_dir / f"{benchmark['case_id']}__{variant['id']}.json"
            if path.exists() and not force:
                previous = read_json(path)
                if previous.get("status") == "success" and previous.get("request_hash") == fingerprint:
                    counts["skipped"] += 1
                    print(f"{label}: skipped (matching successful route)", flush=True)
                    continue

            print(f"{label}: running", flush=True)
            started = time.perf_counter()
            raw_content = ""
            try:
                chat_method = client.chat_stream if generation.get("stream") else client.chat
                result = chat_method(
                    model=variant["model_id"],
                    messages=messages,
                    generation=generation,
                )
                raw_content = result.content
                route = parse_rag_route(result.content)
                write_json(
                    path,
                    {
                        "schema_version": "1.0",
                        "run_id": run_dir.name,
                        "case_id": benchmark["case_id"],
                        "variant_id": variant["id"],
                        "display_name": variant.get("display_name", variant["id"]),
                        "model_id": variant["model_id"],
                        "skill_version": variant.get("skill_version"),
                        "status": "success",
                        "expected_action": expected,
                        "routing_correct": route["action"] == expected if expected else None,
                        "request_hash": fingerprint,
                        "completed_at": utc_now(),
                        "latency_ms": result.latency_ms,
                        "raw_response": result.raw,
                        "raw_content": result.content,
                        **route,
                    },
                )
                counts["success"] += 1
                verdict = "correct" if route["action"] == expected else "wrong"
                print(f"{label}: {route['action']} ({verdict})", flush=True)
            except (OpenWebUIError, ValueError) as exc:
                latency_ms = round((time.perf_counter() - started) * 1000)
                error_type = classify_execution_error(str(exc))
                write_json(
                    path,
                    {
                        "schema_version": "1.0",
                        "run_id": run_dir.name,
                        "case_id": benchmark["case_id"],
                        "variant_id": variant["id"],
                        "display_name": variant.get("display_name", variant["id"]),
                        "model_id": variant["model_id"],
                        "skill_version": variant.get("skill_version"),
                        "status": "error",
                        "expected_action": expected,
                        "request_hash": fingerprint,
                        "completed_at": utc_now(),
                        "latency_ms": latency_ms,
                        "error": str(exc),
                        "error_type": error_type,
                        "raw_content": raw_content,
                    },
                )
                counts["error"] += 1
                print(f"{label}: error [{error_type}]", flush=True)

    summary = summarize_router_evaluation(run_dir)
    write_json(run_dir / "router_summary.json", summary)
    return {**counts, "summary": summary}
