from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from .benchmark import import_all, iter_benchmarks, validate_benchmark
from .io_utils import load_dotenv, read_json
from .judge import prepare_judge_tasks, prepare_teacher_tasks, validate_ratings
from .report import build_report
from .runner import execute_run, load_systems, make_client
from .skill_gate import DEFAULT_CONFIG, evaluate_skill_gate

TEACHER_WEB_PROMPT = (
    "Please process the uploaded teacher_batch.jsonl and return only valid "
    "teacher_responses.jsonl content."
)

JUDGE_WEB_PROMPT = (
    "Please process the uploaded judge_batch.jsonl and return only valid ratings.jsonl content."
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_run_id() -> str:
    return datetime.now().astimezone().strftime("run-%Y%m%d-%H%M%S")


def _paths(args) -> tuple[Path, Path, Path]:
    root = _root()
    benchmarks = root / getattr(args, "benchmarks_dir", "benchmarks/normalized")
    systems = root / getattr(args, "systems", "configs/systems.json")
    run_id = getattr(args, "run_id", None) or _default_run_id()
    run_dir = root / "runs" / run_id
    return benchmarks, systems, run_dir


def _run_cycle_step(label: str, func, args) -> None:
    print(f"[cycle] {label}...")
    code = int(func(args))
    if code:
        raise RuntimeError(f"{label} failed with exit code {code}")


def _wait_for_file(path: Path, poll_seconds: float) -> None:
    if path.exists():
        print(f"[cycle] Found {path.name}, continuing.")
        return
    print(f"[cycle] Waiting for {path.name}: {path}")
    while not path.exists():
        time.sleep(poll_seconds)
    print(f"[cycle] Detected {path.name}, continuing.")


def _print_manual_stage(
    *,
    stage: str,
    upload_path: Path,
    save_path: Path,
    instructions_path: Path,
    prompt: str,
) -> None:
    print(f"[cycle] {stage} requires one manual web step.")
    print(f"[cycle] Upload: {upload_path}")
    print(f"[cycle] Save returned JSONL to: {save_path}")
    print(f"[cycle] Instructions: {instructions_path}")
    print(f"[cycle] Web input prompt: {prompt}")


def command_import(args) -> int:
    root = _root()
    config = root / args.config
    output = root / args.output
    benchmarks = import_all(config, output, root)
    for item in benchmarks:
        print(
            f"imported {item['case_id']}: {len(item['rubric']['steps'])} steps, "
            f"{item['rubric']['total_points']:.0f} points"
        )
    print(f"wrote {output / 'index.json'}")
    return 0


def command_validate_benchmarks(args) -> int:
    benchmarks, _, _ = _paths(args)
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
            agent = details_by_id["agent"]
            rag = details_by_id["agent-rag"]
            full = details_by_id["agent-rag-skill"]
            agent_knowledge = {item["id"] for item in agent["knowledge"] if item.get("id")}
            rag_knowledge = {item["id"] for item in rag["knowledge"] if item.get("id")}
            full_knowledge = {item["id"] for item in full["knowledge"] if item.get("id")}
            if agent_knowledge:
                binding_errors.append("Agent must not have Knowledge/RAG attached")
            if not rag_knowledge:
                binding_errors.append("Agent-RAG must have at least one Knowledge collection")
            if rag_knowledge != full_knowledge:
                binding_errors.append("Agent-RAG and Agent-RAG-Skill must use identical Knowledge IDs")
            agent_skills = set(agent.get("skill_ids") or [])
            rag_skills = set(rag.get("skill_ids") or [])
            full_skills = set(full.get("skill_ids") or [])
            if agent_skills:
                binding_errors.append("Agent must not have Skills attached")
            if rag_skills:
                binding_errors.append("Agent-RAG must not have Skills attached")
            if "swro-watertap" not in full_skills:
                binding_errors.append("Agent-RAG-Skill must attach swro-watertap")
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
        benchmarks = {
            row["case_id"]: row for row in iter_benchmarks(root / "benchmarks" / "normalized")
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
    benchmarks, systems, run_dir = _paths(args)
    counts = execute_run(
        benchmarks_dir=benchmarks,
        systems_path=systems,
        run_dir=run_dir,
        force=args.force,
    )
    print(json.dumps({"run_dir": str(run_dir), **counts}, ensure_ascii=False, indent=2))
    return 1 if counts["error"] else 0


def command_prepare_teacher(args) -> int:
    benchmarks, _, run_dir = _paths(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    output = prepare_teacher_tasks(benchmarks, run_dir)
    print(f"wrote {output}")
    return 0


def command_prepare_judge(args) -> int:
    benchmarks, _, run_dir = _paths(args)
    output = prepare_judge_tasks(benchmarks, run_dir, seed=args.seed)
    print(f"wrote {output}")
    return 0


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


def command_cycle(args) -> int:
    benchmarks, _, run_dir = _paths(args)
    cycle_run_id = run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cycle] Run ID: {cycle_run_id}")

    validate_args = argparse.Namespace(benchmarks_dir=args.benchmarks_dir)
    _run_cycle_step("validate benchmarks", command_validate_benchmarks, validate_args)

    probe_args = argparse.Namespace(systems=args.systems, details=True)
    _run_cycle_step("probe remote presets", command_probe, probe_args)

    run_args = argparse.Namespace(
        run_id=cycle_run_id,
        benchmarks_dir=args.benchmarks_dir,
        systems=args.systems,
        force=args.force,
    )
    _run_cycle_step("run evaluated systems", command_run, run_args)

    teacher_response_path = run_dir / "teacher_responses.jsonl"
    teacher_batch_path = run_dir / "teacher_batch.jsonl"
    teacher_instructions_path = run_dir / "TEACHER_INSTRUCTIONS.md"
    if not teacher_response_path.exists():
        print("[cycle] Preparing teacher batch...")
        prepare_teacher_tasks(benchmarks, run_dir)
        _print_manual_stage(
            stage="Teacher stage",
            upload_path=teacher_batch_path,
            save_path=teacher_response_path,
            instructions_path=teacher_instructions_path,
            prompt=TEACHER_WEB_PROMPT,
        )
        _wait_for_file(teacher_response_path, args.poll_seconds)
    else:
        print(f"[cycle] Reusing existing teacher responses: {teacher_response_path}")

    ratings_path = run_dir / "ratings.jsonl"
    judge_batch_path = run_dir / "judge_batch.jsonl"
    judge_instructions_path = run_dir / "JUDGE_INSTRUCTIONS.md"
    if not ratings_path.exists():
        print("[cycle] Preparing judge batch...")
        prepare_judge_tasks(benchmarks, run_dir, seed=args.seed)
        _print_manual_stage(
            stage="Judge stage",
            upload_path=judge_batch_path,
            save_path=ratings_path,
            instructions_path=judge_instructions_path,
            prompt=JUDGE_WEB_PROMPT,
        )
        _wait_for_file(ratings_path, args.poll_seconds)
    else:
        print(f"[cycle] Reusing existing ratings: {ratings_path}")

    ratings_args = argparse.Namespace(run_id=cycle_run_id)
    _run_cycle_step("validate ratings", command_validate_ratings, ratings_args)

    report_args = argparse.Namespace(run_id=cycle_run_id, output=args.output)
    _run_cycle_step("build report", command_report, report_args)

    gate_args = argparse.Namespace(run_id=cycle_run_id, config=args.config)
    print("[cycle] evaluate skill gate...")
    gate_code = int(command_skill_gate(gate_args))
    print(f"[cycle] Complete. Run directory: {run_dir}")
    return gate_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SWRO step-level auto evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import-benchmarks", help="convert source Excel files to normalized JSON")
    p_import.add_argument("--config", default="configs/benchmarks.json")
    p_import.add_argument("--output", default="benchmarks/normalized")
    p_import.set_defaults(func=command_import)

    p_validate = sub.add_parser("validate-benchmarks", help="validate normalized benchmark JSON")
    p_validate.add_argument("--benchmarks-dir", default="benchmarks/normalized")
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
    p_probe_chat.set_defaults(func=command_probe_chat)

    p_run = sub.add_parser("run", help="run every benchmark against the three OpenWebUI presets")
    p_run.add_argument("--run-id", default=None)
    p_run.add_argument("--benchmarks-dir", default="benchmarks/normalized")
    p_run.add_argument("--systems", default="configs/systems.json")
    p_run.add_argument("--force", action="store_true")
    p_run.set_defaults(func=command_run)

    p_cycle = sub.add_parser(
        "cycle",
        help="run the local pipeline, pause for teacher/judge web steps, then finish automatically",
    )
    p_cycle.add_argument("--run-id", default=None)
    p_cycle.add_argument("--benchmarks-dir", default="benchmarks/normalized")
    p_cycle.add_argument("--systems", default="configs/systems.json")
    p_cycle.add_argument("--force", action="store_true")
    p_cycle.add_argument("--seed", type=int, default=20260806)
    p_cycle.add_argument("--poll-seconds", type=float, default=2.0)
    p_cycle.add_argument("--output", default=None)
    p_cycle.add_argument("--config", default="configs/skill_promotion.json")
    p_cycle.set_defaults(func=command_cycle)

    p_teacher = sub.add_parser("prepare-teacher", help="write blind GPT-5.6 teacher tasks")
    p_teacher.add_argument("--run-id", required=True)
    p_teacher.add_argument("--benchmarks-dir", default="benchmarks/normalized")
    p_teacher.set_defaults(func=command_prepare_teacher)

    p_judge = sub.add_parser("prepare-judge", help="write anonymized GPT-5.6 judge tasks")
    p_judge.add_argument("--run-id", required=True)
    p_judge.add_argument("--benchmarks-dir", default="benchmarks/normalized")
    p_judge.add_argument("--seed", type=int, default=20260806)
    p_judge.set_defaults(func=command_prepare_judge)

    p_ratings = sub.add_parser("validate-ratings", help="check score bounds, sums and failure codes")
    p_ratings.add_argument("--run-id", required=True)
    p_ratings.set_defaults(func=command_validate_ratings)

    p_report = sub.add_parser("report", help="generate a self-contained HTML report")
    p_report.add_argument("--run-id", required=True)
    p_report.add_argument("--output", default=None)
    p_report.set_defaults(func=command_report)

    p_gate = sub.add_parser("skill-gate", help="compare Agent-RAG-Skill with Agent-RAG")
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
