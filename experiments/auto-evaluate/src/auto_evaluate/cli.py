from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .benchmark import import_all, iter_benchmarks, validate_benchmark
from .codex_automation import run_codex_tasks, validate_stage_environment
from .evaluation import resolve_profile, write_profile_snapshot
from .failure_analysis import build_failure_analysis, render_text_analysis
from .figures import PAPER_FIGURE_IDS, export_all_figures, export_figure
from .io_utils import load_dotenv, read_json, read_jsonl, utc_now, write_json, write_jsonl
from .judge import prepare_judge_tasks, prepare_teacher_tasks, validate_ratings
from .report import build_report
from .reward_analysis import build_reward_analysis, build_router_update_plan
from .router_evaluation import execute_router_evaluation
from .runner import execute_run, load_systems, make_client, summarize_run_completeness

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
    if entry.get("enabled") is False:
        raise ValueError(
            f"Benchmark set '{selected}' is reserved but not enabled. Add its source config "
            "and normalized data, then set enabled to true in configs/benchmark_sets.json."
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


def _ensure_run_benchmark_snapshot(benchmarks_dir: Path, run_dir: Path) -> Path:
    target = run_dir / "benchmarks"
    if benchmarks_dir.resolve() == target.resolve():
        return target
    if (target / "index.json").exists():
        return target
    if not (benchmarks_dir / "index.json").exists():
        return benchmarks_dir
    target.mkdir(parents=True, exist_ok=True)
    for source in benchmarks_dir.glob("*.json"):
        shutil.copy2(source, target / source.name)
    return target


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


def _probe_binding_expectations(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve physical preset expectations behind direct and virtual systems."""
    expectations: dict[str, dict[str, Any]] = {}

    def add(model_id: str, alias: str, *, rag_enabled: bool, skill_version: str | None) -> None:
        expected_skill = (skill_version or "").split("@", 1)[0] or None
        existing = expectations.get(model_id)
        signature = (rag_enabled, expected_skill)
        if existing and (existing["rag_enabled"], existing["expected_skill"]) != signature:
            raise ValueError(
                f"Physical preset {model_id!r} has conflicting experiment bindings: "
                f"{existing['aliases']} versus {alias}"
            )
        if existing:
            existing["aliases"].append(alias)
            return
        expectations[model_id] = {
            "model_id": model_id,
            "aliases": [alias],
            "rag_enabled": rag_enabled,
            "expected_skill": expected_skill,
        }

    for system in config.get("systems", []):
        adaptive = system.get("adaptive_rag")
        if adaptive:
            add(
                adaptive["router_model_id"],
                f"{system['id']}:router",
                rag_enabled=False,
                skill_version=None,
            )
            add(
                adaptive["no_rag_model_id"],
                f"{system['id']}:skip_rag",
                rag_enabled=False,
                skill_version=system.get("skill_version"),
            )
            add(
                adaptive["rag_model_id"],
                f"{system['id']}:use_rag",
                rag_enabled=True,
                skill_version=system.get("skill_version"),
            )
        else:
            add(
                system["model_id"],
                system["id"],
                rag_enabled=bool(system.get("rag_enabled", False)),
                skill_version=system.get("skill_version"),
            )
    return expectations


def command_probe(args) -> int:
    root = _root()
    load_dotenv(root / ".env")
    _, systems_path, _ = _paths(args)
    selected_ids = getattr(args, "selected_system_ids", None)
    if selected_ids is None and getattr(args, "benchmark_set", None):
        _, entry = _resolve_benchmark_set(root, args.benchmark_set)
        _, profile = resolve_profile(root, entry, getattr(args, "evaluation_profile", None))
        selected_ids = profile["system_ids"]
    config = load_systems(systems_path, selected_ids)
    client = make_client(config["generation"])
    payload = client.list_models()
    data = payload.get("data") if isinstance(payload, dict) else None
    models = data if isinstance(data, list) else payload.get("models", []) if isinstance(payload, dict) else []
    ids = {row.get("id") or row.get("model") or row.get("name") for row in models if isinstance(row, dict)}
    binding_expectations = _probe_binding_expectations(config)
    expected = set(binding_expectations)
    output = {"available_model_ids": sorted(x for x in ids if x), "expected": sorted(expected)}
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
            detail = {
                "id": row_id,
                "name": row.get("name"),
                "base_model_id": row.get("base_model_id") or info.get("base_model_id") or meta.get("base_model_id"),
                "capabilities": meta.get("capabilities") or {},
                "knowledge": [{"id": item.get("id"), "name": item.get("name"), "type": item.get("type")} for item in raw_knowledge if isinstance(item, dict)],
                "skill_ids": meta.get("skillIds") or [],
            }
            expected_rows.append(detail)
            details_by_id[row_id] = detail
        output["expected_model_details"] = expected_rows
        if set(details_by_id) == expected:
            rag_knowledge_sets = []
            for model_id, detail in details_by_id.items():
                expectation = binding_expectations[model_id]
                aliases = ", ".join(expectation["aliases"])
                knowledge = {item["id"] for item in detail["knowledge"] if item.get("id")}
                skills = set(detail.get("skill_ids") or [])
                if expectation["rag_enabled"]:
                    if not knowledge:
                        binding_errors.append(f"{aliases} must have Knowledge/RAG attached")
                    rag_knowledge_sets.append((aliases, knowledge))
                elif knowledge:
                    binding_errors.append(f"{aliases} must not have Knowledge/RAG attached")
                expected_skill = expectation["expected_skill"]
                if expected_skill and expected_skill not in skills:
                    binding_errors.append(f"{aliases} must attach {expected_skill}")
                if not expected_skill and skills:
                    binding_errors.append(f"{aliases} must not have Skills attached")
                noisy = {"vision", "web_search", "image_generation", "code_interpreter", "terminal"}
                enabled = sorted(name for name in noisy if detail["capabilities"].get(name))
                if enabled:
                    binding_warnings.append(f"{aliases} has unrelated capabilities enabled: {', '.join(enabled)}")
            if len(rag_knowledge_sets) > 1:
                first_name, first_ids = rag_knowledge_sets[0]
                for name, knowledge in rag_knowledge_sets[1:]:
                    if knowledge != first_ids:
                        binding_errors.append(f"{first_name} and {name} must use identical Knowledge IDs")
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
    if args.max_tokens is None:
        generation.pop("max_tokens", None)
    else:
        generation["max_tokens"] = args.max_tokens
    if args.no_thinking:
        generation["enable_thinking"] = False
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
                "max_tokens": generation.get("max_tokens"),
                "enable_thinking": generation.get("enable_thinking"),
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
    set_name = selected or getattr(args, "benchmark_set", None)
    entry = _resolve_benchmark_set(root, set_name)[1] if set_name and _load_benchmark_registry(root) else None
    profile_id, profile = resolve_profile(root, entry, getattr(args, "evaluation_profile", None))
    if selected:
        print(f"[benchmarks] Run set '{selected}' with profile '{profile_id}'")
    systems = (root / getattr(args, "systems", "configs/systems.json")).resolve()
    run_id = getattr(args, "run_id", None) or _default_run_id()
    run_dir = (root / "runs" / run_id).resolve()
    write_profile_snapshot(run_dir, profile_id, profile, set_name)
    counts = execute_run(
        benchmarks_dir=benchmarks,
        systems_path=systems,
        run_dir=run_dir,
        force=args.force,
        selected_system_ids=profile["system_ids"],
        evaluation_profile=profile_id,
        system_concurrency=getattr(args, "system_concurrency", 2),
    )
    print(json.dumps({"run_dir": str(run_dir), "evaluation_profile": profile_id, **counts}, ensure_ascii=False, indent=2))
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


def command_router_eval(args) -> int:
    root = _root()
    load_dotenv(root / ".env")
    benchmarks, selected = _prepare_benchmarks(args, sync=True)
    set_name = selected or getattr(args, "benchmark_set", None)
    entry = (
        _resolve_benchmark_set(root, set_name)[1]
        if set_name and _load_benchmark_registry(root)
        else None
    )
    profile_id, profile = resolve_profile(
        root, entry, getattr(args, "evaluation_profile", None)
    )
    route_spec = profile.get("adaptive_rag_analysis") or {}
    default_expected_route = route_spec.get("default_expected_route")
    run_id = getattr(args, "run_id", None) or _default_run_id()
    run_dir = (root / "runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = _ensure_run_benchmark_snapshot(benchmarks, run_dir)
    config_path = (root / args.config).resolve()
    config = read_json(config_path)
    selected_cases = config.get("pilot_case_ids") if args.pilot else None
    print(
        f"[router-eval] Run ID: {run_id}; set: {set_name}; "
        f"profile: {profile_id}; pilot: {bool(args.pilot)}"
    )
    result = execute_router_evaluation(
        benchmarks_dir=benchmarks,
        config_path=config_path,
        run_dir=run_dir,
        benchmark_set=set_name,
        default_expected_route=default_expected_route,
        selected_variant_ids=args.variant,
        selected_case_ids=selected_cases,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "router_summary": str(run_dir / "router_summary.json"),
                **result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if result["error"] else 0


def command_auto(args) -> int:
    benchmarks, selected = _prepare_benchmarks(args, sync=True)
    benchmark_dir_arg = _benchmarks_arg_value(benchmarks)
    root = _root()
    set_name = selected or getattr(args, "benchmark_set", None)
    entry = _resolve_benchmark_set(root, set_name)[1] if set_name and _load_benchmark_registry(root) else None
    profile_id, profile = resolve_profile(root, entry, getattr(args, "evaluation_profile", None))
    if selected:
        print(f"[benchmarks] Auto set '{selected}' with profile '{profile_id}'")
    run_id = getattr(args, "run_id", None) or _default_run_id()
    load_dotenv(root / ".env")
    run_dir = (root / "runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = _ensure_run_benchmark_snapshot(benchmarks, run_dir)
    benchmark_dir_arg = _benchmarks_arg_value(benchmarks)
    write_profile_snapshot(run_dir, profile_id, profile, set_name)
    auto_run_id = run_dir.name
    stage = getattr(args, "stage", "all")
    print(f"[auto] Run ID: {auto_run_id}; stage: {stage}; profile: {profile_id}")

    timing = {"run_id": auto_run_id, "stage": stage, "started_at": utc_now()}
    timing_started = time.perf_counter()

    def _finish_timing(status: str) -> None:
        timing["status"] = status
        timing["completed_at"] = utc_now()
        timing["duration_seconds"] = round(time.perf_counter() - timing_started, 1)
        timing_path = run_dir / "timing.jsonl"
        rows = read_jsonl(timing_path)
        rows.append(timing)
        write_jsonl(timing_path, rows)
        print(f"[auto] {status} in {timing['duration_seconds']}s (see runs/{auto_run_id}/timing.jsonl)")

    try:
        validate_args = argparse.Namespace(benchmarks_dir=benchmark_dir_arg, benchmark_set=None)
        _run_auto_step("validate benchmarks", command_validate_benchmarks, validate_args)

        legacy_concurrency = getattr(args, "codex_concurrency", None)
        teacher_general_concurrency = legacy_concurrency or getattr(args, "teacher_general_concurrency", 2)
        teacher_tools_concurrency = legacy_concurrency or getattr(args, "teacher_tools_concurrency", 1)
        judge_concurrency = legacy_concurrency or getattr(args, "judge_concurrency", 4)
        system_concurrency = getattr(args, "system_concurrency", 2)

        if stage in {"all", "systems"}:
            probe_args = argparse.Namespace(
                systems=args.systems,
                details=True,
                selected_system_ids=profile["system_ids"],
                benchmark_set=None,
                evaluation_profile=profile_id,
            )
            _run_auto_step("probe remote presets", command_probe, probe_args)
            run_args = argparse.Namespace(
                run_id=auto_run_id,
                benchmarks_dir=benchmark_dir_arg,
                benchmark_set=set_name,
                evaluation_profile=profile_id,
                systems=args.systems,
                force=args.force,
                system_concurrency=system_concurrency,
            )
            print(f"[auto] Run evaluated OpenWebUI systems (concurrency={system_concurrency}; independent conversations)...")
            system_run_code = int(command_run(run_args))
            if system_run_code:
                print("[auto] One or more system requests failed; preserving them and continuing.")

        completeness = None
        if getattr(args, "require_complete_systems", False) and stage in {"all", "systems", "teachers", "judges"}:
            completeness = summarize_run_completeness(
                benchmarks_dir=benchmarks,
                run_dir=run_dir,
                system_ids=profile["system_ids"],
            )
            if completeness["incomplete"]:
                examples = "; ".join(
                    f"{row['case_id']}/{row['system_id']}={row['status']}"
                    + (f"[{row['error_type']}]" if row.get("error_type") else "")
                    for row in completeness["items"][:8]
                )
                raise RuntimeError(
                    "OpenWebUI system stage is incomplete: "
                    f"{completeness['success']}/{completeness['expected']} successful; "
                    f"examples: {examples}. Re-run the same run ID to reuse successful cache entries."
                )
            print(
                f"[auto] OpenWebUI completeness check passed: "
                f"{completeness['success']}/{completeness['expected']} "
                f"(native={completeness['native_success']}, "
                f"recovered={completeness['recovered_success']}, "
                f"policy_replay={completeness.get('policy_replay_success', 0)})"
            )

        if stage in {"all", "teachers"}:
            teacher_outputs = []
            for teacher in profile["teachers"]:
                if teacher.get("system_id") == "gpt-5.6-teacher":
                    batch = prepare_teacher_tasks(benchmarks, run_dir)
                else:
                    batch = prepare_teacher_tasks(benchmarks, run_dir, teacher)
                tasks = read_jsonl(batch)
                validate_stage_environment("teacher", tasks)
                concurrency = teacher_tools_concurrency if teacher.get("tools_enabled") else teacher_general_concurrency
                print(f"[auto] Generate {teacher['display_name']} answers (concurrency={concurrency})...")
                outputs = run_codex_tasks(
                    stage="teacher", tasks=tasks, run_dir=run_dir, project_root=root,
                    model=args.codex_model, concurrency=concurrency, retries=args.codex_retries,
                    timeout_seconds=args.codex_timeout, force=args.force_codex,
                )
                if teacher.get("system_id") != "gpt-5.6-teacher":
                    display_by_task = {task["task_id"]: task.get("display_name") for task in tasks}
                    for output in outputs:
                        output["display_name"] = display_by_task.get(output.get("task_id")) or teacher["display_name"]
                teacher_outputs.extend(outputs)
            write_jsonl(run_dir / "teacher_responses.jsonl", teacher_outputs)

        if stage in {"all", "judges"}:
            if profile["teachers"] and not (run_dir / "teacher_responses.jsonl").exists():
                raise FileNotFoundError("teacher_responses.jsonl is missing; run --stage teachers first")
            print(f"[auto] Score anonymous candidates (concurrency={judge_concurrency}; Judge tools forbidden)...")
            judge_batch = prepare_judge_tasks(benchmarks, run_dir, seed=args.seed)
            judge_tasks = read_jsonl(judge_batch)
            validate_stage_environment("judge", judge_tasks)
            ratings = run_codex_tasks(
                stage="judge", tasks=judge_tasks, run_dir=run_dir, project_root=root,
                model=args.codex_model, concurrency=judge_concurrency, retries=args.codex_retries,
                timeout_seconds=args.codex_timeout, force=args.force_codex,
            )
            write_jsonl(run_dir / "ratings.jsonl", ratings)

        if stage in {"all", "judges", "report"}:
            ratings_args = argparse.Namespace(run_id=auto_run_id)
            _run_auto_step("validate ratings", command_validate_ratings, ratings_args)
            reward_args = argparse.Namespace(
                run_id=auto_run_id,
                output=None,
                router_plan_output=None,
            )
            _run_auto_step("build reward analysis", command_reward_analysis, reward_args)
        if stage in {"all", "report"}:
            report_args = argparse.Namespace(run_id=auto_run_id, output=args.output)
            _run_auto_step("build report", command_report, report_args)

        report_path = Path(args.output).resolve() if args.output else run_dir / "report.html"
        print(json.dumps({
            "status": "complete", "completed_stage": stage, "run_dir": str(run_dir),
            "report": str(report_path) if report_path.exists() else None,
            "evaluation_profile": profile_id,
            "codex_model": args.codex_model,
            "system_concurrency": system_concurrency,
            "system_completeness": completeness,
        }, ensure_ascii=False, indent=2))
        _finish_timing("complete")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError):
        _finish_timing("error")
        raise


def command_failure_analysis(args) -> int:
    root = _root()
    run_dir = (root / "runs" / args.run_id).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    analysis = build_failure_analysis(run_dir)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "ok", "output": str(output), "run_id": run_dir.name}, ensure_ascii=False))
        return 0
    print(render_text_analysis(analysis))
    return 0


def command_plot(args) -> int:
    _, _, run_dir = _paths(args)
    if args.figure == "all":
        if args.output:
            raise ValueError("--output can be used only when exporting one figure")
        outputs = export_all_figures(run_dir)
        print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))
        return 0
    output = Path(args.output).resolve() if args.output else None
    result = export_figure(run_dir, args.figure, output)
    print(f"wrote {result}")
    return 0


def command_reward_analysis(args) -> int:
    root = _root()
    run_dir = (root / "runs" / args.run_id).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    analysis = build_reward_analysis(run_dir)
    update_plan = build_router_update_plan(run_dir, analysis)
    analysis_path = Path(args.output).resolve() if args.output else run_dir / "reward_analysis.json"
    update_path = (
        Path(args.router_plan_output).resolve()
        if args.router_plan_output
        else run_dir / "router_update_plan.json"
    )
    write_json(analysis_path, analysis)
    write_json(update_path, update_plan)
    print(
        json.dumps(
            {
                "status": "ok",
                "run_id": run_dir.name,
                "reward_analysis": str(analysis_path),
                "router_update_plan": str(update_path),
                "adaptive_rag": analysis.get("adaptive_rag", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0



def command_cleanup_codex_threads(args) -> int:
    """Archive Codex teacher/judge threads recorded for a run (recoverable).

    Results are already persisted under runs/<run-id>, so archiving the source
    conversations is safe cleanup: they leave the active UI list and can still
    be restored from the app's archive until manually deleted there.
    """
    root = _root()
    run_dir = (root / "runs" / args.run_id).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    records_dir = run_dir / "codex" / "records"
    thread_rows: list[tuple[str, str, str]] = []
    if records_dir.exists():
        for path in sorted(records_dir.glob("*/*.json")):
            record = read_json(path)
            thread_id = record.get("thread_id")
            if thread_id:
                thread_rows.append(
                    (
                        str(record.get("stage", "?")),
                        str(record.get("task_id", "?")),
                        str(thread_id),
                    )
                )
    seen: set[str] = set()
    unique: list[tuple[str, str, str]] = []
    for stage, task_id, thread_id in thread_rows:
        if thread_id not in seen:
            seen.add(thread_id)
            unique.append((stage, task_id, thread_id))
    if not unique:
        print(f"[cleanup-codex-threads] no thread IDs recorded for run {args.run_id}")
        return 0
    if args.dry_run:
        for stage, task_id, thread_id in unique:
            print(f"[cleanup-codex-threads] would archive {thread_id} ({stage} :: {task_id})")
        print(f"[cleanup-codex-threads] dry-run: {len(unique)} thread(s)")
        return 0
    try:
        from openai_codex import Codex
    except ImportError as exc:
        raise RuntimeError(
            "openai_codex is not installed in the active environment; run: python -m pip install -e ."
        ) from exc
    errors: list[str] = []
    archived = 0
    with Codex() as codex:
        for stage, task_id, thread_id in unique:
            try:
                codex.thread_archive(thread_id)
                archived += 1
                print(f"[cleanup-codex-threads] archived {thread_id} ({stage} :: {task_id})")
            except Exception as exc:
                errors.append(f"{thread_id} ({stage} :: {task_id}): {exc}")
    print(f"[cleanup-codex-threads] archived {archived}/{len(unique)}")
    for error in errors[:10]:
        print(f"[cleanup-codex-threads] error: {error}")
    return 0 if not errors else 1


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
                "view_config": entry.get("view_config"),
                "view_id": entry.get("view_id"),
                "normalized_dir": entry.get("normalized_dir"),
                "evaluation_profile": entry.get("evaluation_profile"),
                "enabled": entry.get("enabled", True),
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
    p_probe.add_argument("--benchmark-set", default=None)
    p_probe.add_argument("--evaluation-profile", default=None)
    p_probe.set_defaults(func=command_probe)

    p_probe_chat = sub.add_parser(
        "probe-chat", help="send a minimal chat request to one OpenWebUI preset"
    )
    p_probe_chat.add_argument("--systems", default="configs/systems.json")
    p_probe_chat.add_argument("--system", default="agent")
    p_probe_chat.add_argument("--model", default=None)
    p_probe_chat.add_argument("--case", default=None)
    p_probe_chat.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="client-side output cap; omit to use the server/model limit",
    )
    p_probe_chat.add_argument(
        "--no-thinking",
        action="store_true",
        help="request chat-template thinking mode to be disabled",
    )
    p_probe_chat.add_argument("--timeout", type=int, default=60)
    p_probe_chat.add_argument("--stream", action="store_true")
    _add_benchmark_selection_args(p_probe_chat)
    p_probe_chat.set_defaults(func=command_probe_chat)

    p_run = sub.add_parser("run", help="run every benchmark against the selected OpenWebUI presets")
    p_run.add_argument("--run-id", default=None)
    _add_benchmark_selection_args(p_run)
    p_run.add_argument("--systems", default="configs/systems.json")
    p_run.add_argument("--evaluation-profile", default=None)
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--system-concurrency", type=int, default=2)
    p_run.set_defaults(func=command_run)

    p_router = sub.add_parser(
        "router-eval",
        help="compare zero-shot and Skill-guided RAG routing without running the solver",
    )
    p_router.add_argument("--run-id", default=None)
    _add_benchmark_selection_args(p_router)
    p_router.add_argument("--evaluation-profile", default=None)
    p_router.add_argument("--config", default="configs/router_evaluation.json")
    p_router.add_argument(
        "--variant",
        action="append",
        default=None,
        help="router variant ID; repeat to select multiple variants",
    )
    p_router.add_argument(
        "--pilot",
        action="store_true",
        help="run the fixed 12-case D1-D6 routing pilot from the router config",
    )
    p_router.add_argument("--force", action="store_true")
    p_router.set_defaults(func=command_router_eval)

    p_auto = sub.add_parser(
        "auto",
        help="run resumable systems, dual-teacher, judge, and report stages",
    )
    p_auto.add_argument("--run-id", default=None)
    _add_benchmark_selection_args(p_auto)
    p_auto.add_argument("--systems", default="configs/systems.json")
    p_auto.add_argument("--evaluation-profile", default=None)
    p_auto.add_argument("--stage", choices=("all", "systems", "teachers", "judges", "report"), default="all")
    p_auto.add_argument("--force", action="store_true", help="rerun successful OpenWebUI responses")
    p_auto.add_argument("--force-codex", action="store_true", help="rerun successful Codex tasks")
    p_auto.add_argument("--seed", type=int, default=20260806)
    p_auto.add_argument("--output", default=None)
    p_auto.add_argument("--codex-model", default="gpt-5.6-sol")
    p_auto.add_argument("--codex-concurrency", type=int, default=None, help="legacy override for all Codex stages")
    p_auto.add_argument("--system-concurrency", type=int, default=2, help="parallel independent OpenWebUI requests (1-8)")
    p_auto.add_argument("--teacher-general-concurrency", type=int, default=2)
    p_auto.add_argument("--teacher-tools-concurrency", type=int, default=1)
    p_auto.add_argument("--judge-concurrency", type=int, default=4, help="parallel independent Judge tasks (default: 4)")
    p_auto.add_argument("--codex-retries", type=int, default=2)
    p_auto.add_argument("--codex-timeout", type=int, default=3600)
    p_auto.add_argument("--require-complete-systems", action="store_true", help="stop before Teacher/Judge when any OpenWebUI response is missing or failed")
    p_auto.set_defaults(func=command_auto)

    p_ratings = sub.add_parser("validate-ratings", help="check score bounds, sums and failure codes")
    p_ratings.add_argument("--run-id", required=True)
    p_ratings.set_defaults(func=command_validate_ratings)

    p_analysis = sub.add_parser(
        "failure-analysis",
        help="aggregate scores, failure codes and tool/RAG utilization of a run",
    )
    p_analysis.add_argument("--run-id", required=True)
    p_analysis.add_argument(
        "--output",
        default=None,
        help="write the full analysis as JSON instead of printing the text view",
    )
    p_analysis.set_defaults(func=command_failure_analysis)

    p_reward = sub.add_parser(
        "reward-analysis",
        help="build paired rubric rewards, adaptive-RAG regret and a Router update plan",
    )
    p_reward.add_argument("--run-id", required=True)
    p_reward.add_argument("--output", default=None)
    p_reward.add_argument("--router-plan-output", default=None)
    p_reward.set_defaults(func=command_reward_analysis)

    p_report = sub.add_parser("report", help="generate a self-contained HTML report")
    p_report.add_argument("--run-id", required=True)
    p_report.add_argument("--output", default=None)
    p_report.set_defaults(func=command_report)

    p_plot = sub.add_parser(
        "plot",
        help="export one paper-ready SVG figure from an evaluated run",
    )
    p_plot.add_argument("--run-id", required=True)
    p_plot.add_argument(
        "--figure",
        required=True,
        choices=(*PAPER_FIGURE_IDS, "all"),
    )
    p_plot.add_argument("--output", default=None)
    p_plot.set_defaults(func=command_plot)

    p_cleanup = sub.add_parser(
        "cleanup-codex-threads",
        help="archive Codex teacher/judge threads recorded for a run (keeps them recoverable)",
    )
    p_cleanup.add_argument("--run-id", required=True)
    p_cleanup.add_argument("--dry-run", action="store_true")
    p_cleanup.set_defaults(func=command_cleanup_codex_threads)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
