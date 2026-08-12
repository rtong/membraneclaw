from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .benchmark import import_all, iter_benchmarks, validate_benchmark
from .codex_automation import run_codex_tasks
from .io_utils import load_dotenv, read_json, read_jsonl, write_jsonl
from .judge import prepare_judge_tasks, prepare_teacher_tasks, validate_ratings
from .report import build_report
from .runner import execute_run, load_systems, make_client
from .skill_gate import DEFAULT_CONFIG, evaluate_skill_gate

def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_run_id() -> str:
    return datetime.now().astimezone().strftime("run-%Y%m%d-%H%M%S")


def _benchmark_registry_path(root: Path) -> Path:
    return root / "configs" / "benchmark_sets.json"


def _load_benchmark_registry(root: Path) -> dict[str, Any] | None:
    path = _benchmark_registry_path(root)
    if not path.exists():
        return None
    return read_json(path)


def _resolve_benchmark_set(
    root: Path,
    set_name: str | None = None,
) -> tuple[str, dict[str, Any]]:
    registry = _load_benchmark_registry(root)
    if not registry:
        raise ValueError(
            f"Benchmark set registry not found: {_benchmark_registry_path(root)}"
        )
    selected = set_name or registry.get("active_set")
    if not selected:
        raise ValueError("benchmark_sets.json must define active_set")
    sets = registry.get("sets")
    if not isinstance(sets, dict) or not sets:
        raise ValueError("benchmark_sets.json must define at least one benchmark set")
    entry = sets.get(selected)
    if not isinstance(entry, dict):
        available = ", ".join(sorted(sets))
        raise ValueError(
            f"Unknown benchmark set '{selected}'. Available sets: {available}"
        )
    return selected, entry


def _prepare_benchmarks(
    args,
    *,
    sync: bool,
) -> tuple[Path, str | None]:
    root = _root()
    explicit_dir = getattr(args, "benchmarks_dir", None)
    if explicit_dir:
        return (root / explicit_dir).resolve(), None

    selected, entry = _resolve_benchmark_set(
        root,
        getattr(args, "benchmark_set", None),
    )
    normalized_raw = entry.get("normalized_dir")
    if not normalized_raw:
        raise ValueError(f"Benchmark set '{selected}' is missing normalized_dir")
    normalized_dir = (root / normalized_raw).resolve()
    source_config_raw = entry.get("source_config")
    if sync and source_config_raw:
        source_config = (root / source_config_raw).resolve()
        print(
            f"[benchmarks] Sync set '{selected}' from {source_config} -> {normalized_dir}"
        )
        import_all(source_config, normalized_dir, root)
    elif sync and not normalized_dir.exists():
        raise FileNotFoundError(
            f"Benchmark set '{selected}' points to missing normalized dir: {normalized_dir}"
        )
    return normalized_dir, selected


def _benchmarks_arg_value(benchmarks_dir: Path) -> str:
    return str(benchmarks_dir)


def _paths(args) -> tuple[Path, Path, Path]:
    root = _root()
    benchmarks = root / (getattr(args, "benchmarks_dir", None) or "benchmarks/normalized")
    systems = root / (getattr(args, "systems", None) or "configs/systems.json")
    run_id = getattr(args, "run_id", None) or _default_run_id()
    run_dir = root / "runs" / run_id
    return benchmarks, systems, run_dir


def _run_auto_step(label: str, func, args) -> None:
    print(f"[auto] {label}...")
    code = int(func(args))
    if code:
        raise RuntimeError(f"{label} failed with exit code {code}")


def command_import(args) -> int:
    root = _root()
    config = args.config
    output = args.output
    selected = None
    if not config and not output:
        selected, entry = _resolve_benchmark_set(root, getattr(args, "benchmark_set", None))
        config = entry.get("source_config")
        output = entry.get("normalized_dir")
        if not config or not output:
            raise ValueError(
                f"Benchmark set '{selected}' cannot be imported because it does not define both source_config and normalized_dir"
            )
    elif bool(config) != bool(output):
        raise ValueError(
            "import-benchmarks requires both --config and --output, or neither"
        )
    config_path = (root / config).resolve()
    output_dir = (root / output).resolve()
    if selected:
        print(f"[benchmarks] Import set '{selected}'")
    benchmarks = import_all(config_path, output_dir, root)
    for item in benchmarks:
        print(
            f"imported {item['case_id']}: {len(item['rubric']['steps'])} steps, "
            f"{item['rubric']['total_points']:.0f} points"
        )
    print(f"wrote {output_dir / 'index.json'}")
    return 0


def command_validate_benchmarks(args) -> int:
    benchmarks, selected = _prepare_benchmarks(args, sync=True)
    if selected:
        print(f"[benchmarks] Validate set '{selected}'")
    count = 0
    for item in iter_benchmarks(benchmarks):
        validate_benchmark(item)
        count += 1
        print(f"ok {item['case_id']} ({len(item['rubric']['steps'])} steps)")
    print(f"validated {count} benchmark(s)")
    return 0


def command_probe(args) -> int:
    root = _root()
    load_dotenv(root / ".env")
    _, systems_path, _ = _paths(args)
    config = load_systems(systems_path)
    client = make_client(config["generation"])
    payload = client.list_models()
    data = payload.get("data") if isinstance(payload, dict) else None
    models = data if isinstance(data, list) else payload.get("models", []) if isinstance(payload, dict) else []
    ids = {
        row.get("id") or row.get("model") or row.get("name")
        for row in models
        if isinstance(row, dict)
    }
    expected = {row["model_id"] for row in config["systems"]}
    output = {
        "available_model_ids": sorted(x for x in ids if x),
        "expected": sorted(expected),
    }
    binding_errors = []
    binding_warnings = []
    if args.details:
        expected_rows = []
        details_by_id = {}
        for row in models:
            if not isinstance(row, dict):
                continue
            row_id = row.get("id") or row.get("model") or row.get("name")
            if row_id not in expected:
                continue
            info = row.get("info") if isinstance(row.get("info"), dict) else {}
            meta = info.get("meta") if isinstance(info.get("meta"), dict) else {}
            raw_knowledge = meta.get("knowledge") if isinstance(meta.get("knowledge"), list) else []
            knowledge = [
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "type": item.get("type"),
                }
                for item in raw_knowledge
                if isinstance(item, dict)
            ]
            detail = {
                    "id": row_id,
                    "name": row.get("name"),
                    "owned_by": row.get("owned_by"),
                    "base_model_id": (
                        row.get("base_model_id")
                        or info.get("base_model_id")
                        or meta.get("base_model_id")
                    ),
                    "connection_type": row.get("connection_type"),
                    "capabilities": meta.get("capabilities"),
                    "knowledge": knowledge,
                    "skill_ids": meta.get("skillIds"),
                    "info_keys": sorted(info),
                    "meta_keys": sorted(meta),
                }
            expected_rows.append(detail)
            details_by_id[row_id] = detail
        output["expected_model_details"] = expected_rows
        if set(details_by_id) == expected:
            baseline = details_by_id["baseline"]
            environment = details_by_id["environment"]
            environment_skill = details_by_id["environment-skill"]
            baseline_knowledge = {item["id"] for item in baseline["knowledge"] if item.get("id")}
            environment_knowledge = {
                item["id"] for item in environment["knowledge"] if item.get("id")
            }
            environment_skill_knowledge = {
                item["id"] for item in environment_skill["knowledge"] if item.get("id")
            }
            if baseline_knowledge:
                binding_errors.append("Baseline must not have Knowledge/RAG attached")
            if not environment_knowledge:
                binding_errors.append("Environment must have at least one Knowledge collection")
            if environment_knowledge != environment_skill_knowledge:
                binding_errors.append(
                    "Environment and Environment-Skill must use identical Knowledge IDs"
                )
            baseline_skills = set(baseline.get("skill_ids") or [])
            environment_skills = set(environment.get("skill_ids") or [])
            environment_skill_skills = set(environment_skill.get("skill_ids") or [])
            if baseline_skills:
                binding_errors.append("Baseline must not have Skills attached")
            if environment_skills:
                binding_errors.append("Environment must not have Skills attached")
            if "swro-watertap" not in environment_skill_skills:
                binding_errors.append("Environment-Skill must attach swro-watertap")
            noisy = {"vision", "web_search", "image_generation", "code_interpreter", "terminal"}
            for system_id, detail in details_by_id.items():
                capabilities = detail.get("capabilities") or {}
                enabled = sorted(name for name in noisy if capabilities.get(name))
                if enabled:
                    binding_warnings.append(
                        f"{system_id} has unrelated capabilities enabled: {', '.join(enabled)}"
                    )
        output["binding_errors"] = binding_errors
        output["binding_warnings"] = binding_warnings
    print(json.dumps(output, ensure_ascii=False, indent=2))
    missing = expected - ids if ids else set()
    if missing:
        print(f"missing expected model IDs: {sorted(missing)}", file=sys.stderr)
        return 2
    if binding_errors:
        print("remote preset bindings do not match the experiment design", file=sys.stderr)
        return 3
    return 0


def command_probe_chat(args) -> int:
    root = _root()
    load_dotenv(root / ".env")
    _, systems_path, _ = _paths(args)
    config = load_systems(systems_path)
    systems = {row["id"]: row for row in config["systems"]}
    if not args.model and args.system not in systems:
        raise ValueError(f"Unknown system ID: {args.system}")
    generation = dict(config["generation"])
    generation["max_tokens"] = args.max_tokens
    generation["timeout_seconds"] = args.timeout
    client = make_client(generation)
    selected = systems.get(args.system, {})
    model_id = args.model or selected["model_id"]
    if args.case:
        benchmarks_dir, benchmark_set = _prepare_benchmarks(args, sync=True)
        if benchmark_set:
            print(f"[benchmarks] Probe case from set '{benchmark_set}'")
        benchmarks = {
            row["case_id"]: row for row in iter_benchmarks(benchmarks_dir)
        }
        if args.case not in benchmarks:
            raise ValueError(f"Unknown benchmark case ID: {args.case}")
        messages = [
            {"role": "system", "content": config["shared_system_prompt"]},
            {"role": "user", "content": benchmarks[args.case]["question_prompt"]},
        ]
    else:
        messages = [{"role": "user", "content": "Reply with exactly: OK"}]
    chat_method = client.chat_stream if args.stream else client.chat
    result = chat_method(model=model_id, messages=messages, generation=generation)
    print(
        json.dumps(
            {
                "system_id": selected.get("id") if not args.model else None,
                "model_id": model_id,
                "case_id": args.case,
                "stream": args.stream,
                "latency_ms": result.latency_ms,
                "response": result.content,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_run(args) -> int:
    root = _root()
    load_dotenv(root / ".env")
    benchmarks, selected = _prepare_benchmarks(args, sync=True)
    if selected:
        print(f"[benchmarks] Run set '{selected}'")
    systems = (root / getattr(args, "systems", "configs/systems.json")).resolve()
    run_id = getattr(args, "run_id", None) or _default_run_id()
    run_dir = (root / "runs" / run_id).resolve()
    counts = execute_run(
        benchmarks_dir=benchmarks,
        systems_path=systems,
        run_dir=run_dir,
        force=args.force,
    )
    print(json.dumps({"run_dir": str(run_dir), **counts}, ensure_ascii=False, indent=2))
    return 1 if counts["error"] else 0


def command_validate_ratings(args) -> int:
    _, _, run_dir = _paths(args)
    errors = validate_ratings(run_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("ratings are valid")
    return 0


def command_report(args) -> int:
    _, _, run_dir = _paths(args)
    output = Path(args.output).resolve() if args.output else None
    result = build_report(run_dir, output)
    print(f"wrote {result}")
    return 0


def command_skill_gate(args) -> int:
    root = _root()
    _, _, run_dir = _paths(args)
    config_path = root / args.config
    config = read_json(config_path) if config_path.exists() else DEFAULT_CONFIG
    result = evaluate_skill_gate(run_dir, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("passed") else 1


def command_auto(args) -> int:
    benchmarks, selected = _prepare_benchmarks(args, sync=True)
    benchmark_dir_arg = _benchmarks_arg_value(benchmarks)
    if selected:
        print(f"[benchmarks] Auto set '{selected}'")
    run_id = getattr(args, "run_id", None) or _default_run_id()
    root = _root()
    run_dir = (root / "runs" / run_id).resolve()
    auto_run_id = run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[auto] Run ID: {auto_run_id}")

    validate_args = argparse.Namespace(
        benchmarks_dir=benchmark_dir_arg,
        benchmark_set=None,
    )
    _run_auto_step("validate benchmarks", command_validate_benchmarks, validate_args)

    probe_args = argparse.Namespace(systems=args.systems, details=True)
    _run_auto_step("probe remote presets", command_probe, probe_args)

    run_args = argparse.Namespace(
        run_id=auto_run_id,
        benchmarks_dir=benchmark_dir_arg,
        benchmark_set=None,
        systems=args.systems,
        force=args.force,
    )
    _run_auto_step("run evaluated systems", command_run, run_args)

    print("[auto] Generate blind teacher answers with fresh Codex tasks...")
    teacher_batch = prepare_teacher_tasks(benchmarks, run_dir)
    teacher_tasks = read_jsonl(teacher_batch)
    teacher_outputs = run_codex_tasks(
        stage="teacher",
        tasks=teacher_tasks,
        run_dir=run_dir,
        project_root=root,
        model=args.codex_model,
        concurrency=args.codex_concurrency,
        retries=args.codex_retries,
        timeout_seconds=args.codex_timeout,
        force=args.force_codex,
    )
    write_jsonl(run_dir / "teacher_responses.jsonl", teacher_outputs)

    print("[auto] Score anonymous candidates with one fresh Codex task per response...")
    judge_batch = prepare_judge_tasks(benchmarks, run_dir, seed=args.seed)
    judge_tasks = read_jsonl(judge_batch)
    ratings = run_codex_tasks(
        stage="judge",
        tasks=judge_tasks,
        run_dir=run_dir,
        project_root=root,
        model=args.codex_model,
        concurrency=args.codex_concurrency,
        retries=args.codex_retries,
        timeout_seconds=args.codex_timeout,
        force=args.force_codex,
    )
    write_jsonl(run_dir / "ratings.jsonl", ratings)

    ratings_args = argparse.Namespace(run_id=auto_run_id)
    _run_auto_step("validate ratings", command_validate_ratings, ratings_args)

    report_args = argparse.Namespace(run_id=auto_run_id, output=args.output)
    _run_auto_step("build report", command_report, report_args)

    gate_args = argparse.Namespace(run_id=auto_run_id, config=args.config)
    print("[auto] evaluate skill gate...")
    gate_code = int(command_skill_gate(gate_args))
    report_path = Path(args.output).resolve() if args.output else run_dir / "report.html"
    print(
        json.dumps(
            {
                "status": "complete",
                "run_dir": str(run_dir),
                "report": str(report_path),
                "skill_gate_passed": gate_code == 0,
                "codex_model": args.codex_model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return gate_code if args.fail_on_gate else 0


def command_list_benchmark_sets(args) -> int:
    root = _root()
    registry = _load_benchmark_registry(root)
    if not registry:
        print("benchmark set registry not found")
        return 1
    active = registry.get("active_set")
    rows = []
    for name, entry in sorted((registry.get("sets") or {}).items()):
        rows.append(
            {
                "name": name,
                "active": name == active,
                "description": entry.get("description", ""),
                "source_config": entry.get("source_config"),
                "normalized_dir": entry.get("normalized_dir"),
            }
        )
    print(json.dumps({"active_set": active, "sets": rows}, ensure_ascii=False, indent=2))
    return 0


def _add_benchmark_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmarks-dir", default=None)
    parser.add_argument("--benchmark-set", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SWRO step-level auto evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sets = sub.add_parser("list-benchmark-sets", help="show configured benchmark sets and active set")
    p_sets.set_defaults(func=command_list_benchmark_sets)

    p_import = sub.add_parser("import-benchmarks", help="convert source Excel files to normalized JSON")
    p_import.add_argument("--config", default=None)
    p_import.add_argument("--output", default=None)
    p_import.add_argument("--benchmark-set", default=None)
    p_import.set_defaults(func=command_import)

    p_validate = sub.add_parser("validate-benchmarks", help="validate normalized benchmark JSON")
    _add_benchmark_selection_args(p_validate)
    p_validate.set_defaults(func=command_validate_benchmarks)

    p_probe = sub.add_parser("probe", help="list remote OpenWebUI models and check preset IDs")
    p_probe.add_argument("--systems", default="configs/systems.json")
    p_probe.add_argument("--details", action="store_true")
    p_probe.set_defaults(func=command_probe)

    p_probe_chat = sub.add_parser(
        "probe-chat", help="send a minimal chat request to one OpenWebUI preset"
    )
    p_probe_chat.add_argument("--systems", default="configs/systems.json")
    p_probe_chat.add_argument("--system", default="agent")
    p_probe_chat.add_argument("--model", default=None)
    p_probe_chat.add_argument("--case", default=None)
    p_probe_chat.add_argument("--max-tokens", type=int, default=32)
    p_probe_chat.add_argument("--timeout", type=int, default=60)
    p_probe_chat.add_argument("--stream", action="store_true")
    _add_benchmark_selection_args(p_probe_chat)
    p_probe_chat.set_defaults(func=command_probe_chat)

    p_run = sub.add_parser("run", help="run every benchmark against the three OpenWebUI presets")
    p_run.add_argument("--run-id", default=None)
    _add_benchmark_selection_args(p_run)
    p_run.add_argument("--systems", default="configs/systems.json")
    p_run.add_argument("--force", action="store_true")
    p_run.set_defaults(func=command_run)

    p_auto = sub.add_parser(
        "auto",
        help="run systems, Codex teacher/judge tasks, validation and HTML reporting",
    )
    p_auto.add_argument("--run-id", default=None)
    _add_benchmark_selection_args(p_auto)
    p_auto.add_argument("--systems", default="configs/systems.json")
    p_auto.add_argument("--force", action="store_true", help="rerun successful OpenWebUI responses")
    p_auto.add_argument("--force-codex", action="store_true", help="rerun successful Codex tasks")
    p_auto.add_argument("--seed", type=int, default=20260806)
    p_auto.add_argument("--output", default=None)
    p_auto.add_argument("--config", default="configs/skill_promotion.json")
    p_auto.add_argument("--codex-model", default="gpt-5.6-terra")
    p_auto.add_argument("--codex-concurrency", type=int, default=1)
    p_auto.add_argument("--codex-retries", type=int, default=2)
    p_auto.add_argument("--codex-timeout", type=int, default=3600)
    p_auto.add_argument("--fail-on-gate", action="store_true")
    p_auto.set_defaults(func=command_auto)

    p_ratings = sub.add_parser("validate-ratings", help="check score bounds, sums and failure codes")
    p_ratings.add_argument("--run-id", required=True)
    p_ratings.set_defaults(func=command_validate_ratings)

    p_report = sub.add_parser("report", help="generate a self-contained HTML report")
    p_report.add_argument("--run-id", required=True)
    p_report.add_argument("--output", default=None)
    p_report.set_defaults(func=command_report)

    p_gate = sub.add_parser("skill-gate", help="compare Environment-Skill with Environment")
    p_gate.add_argument("--run-id", required=True)
    p_gate.add_argument("--config", default="configs/skill_promotion.json")
    p_gate.set_defaults(func=command_skill_gate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
