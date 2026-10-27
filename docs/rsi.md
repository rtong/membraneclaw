# Offline RSI controller

This implementation runs a **synthetic offline orchestration experiment**. It does not call a model, train model weights, run WaterTAP or demonstrate improved real-agent performance. The recorded scores are computed by a deterministic fixture Evaluator from actual fixture Solver outputs and private expected values.

## Install and run

Python 3.11+ on a single POSIX host is required. Install from the repository root:

```sh
python -m pip install .
python -m membraneclaw_runtime.rsi.cli demo --output ./var/rsi/demo-001
```

Choose a new output directory. The command refuses to overwrite an existing run. Packaged fixture data is included, so `demo` also works outside the source checkout without credentials or model/tool extras.

Expected demonstration behavior:

1. Run baseline development and validation tasks.
2. Add a synthetic unit-conversion instruction; accept its validation improvement.
3. Start round two from that accepted version.
4. Add an unconditional rounding instruction. It helps a development case but regresses a validation case, so reject it and retain the first accepted version.
5. Write `report.json` and `report.md` with `synthetic_only=true`, selected asset/version identities, per-round decisions, and the synthetic-unit ledger. Actual API dollars/tokens are `null`.

The default fixture executes 26 admitted component operations across two rounds. This count is a deterministic fixture expectation, not a recommended research budget. Its deliberately simple instruction parser is a test double, not a language model. Group IDs in the toy dataset are synthetic; they are not evidence of a leakage-free split of real engineering problem families.

## Configure a run

Export an editable copy of the packaged synthetic fixtures to a new directory:

```sh
python -m membraneclaw_runtime.rsi.cli init-example --output ./var/rsi/example-001
python -m membraneclaw_runtime.rsi.cli run \
  --config ./var/rsi/example-001/offline_config.json --output ./var/rsi/custom-001
```

`init-example` only copies files; it does not start a run, invoke components, or require credentials. It works after package installation, including outside the source checkout. It refuses any existing output path, including an empty directory or symlink. If a copy fails partway through, choose a new directory or inspect and remove the incomplete export before retrying.

The exported directory contains `offline_config.json`, `assets/prompt.txt`, `assets/skills/unit-checks.md`, `tasks_public.json`, and `evaluation_private.json`. Edit these files before running the `run` command above. The private file contains synthetic expected answers; keep real evaluation references private when replacing the toy data. Exported directories and files use private POSIX permissions. Packaged data under `membraneclaw_runtime/rsi/data/` is the single source for both `demo` and exports; there is no separate repository-level `examples/rsi/` copy.

The configuration pins a seed schedule, exact asset allowlist, round limit, decision policy, and operation/unit/wall-time bounds. Text input paths must remain inside the configuration directory. Prompt/skill assets must not be executable or symlinked. Allowed mutations are `prompt.txt` and explicitly listed flat `skills/*.md` files only.

All configured input bytes and relevant controller/tracing source bytes are captured in the root registration. Source files can subsequently move or change without rewriting those snapshots. Resume uses the stored inputs and rejects a changed running implementation. To change configuration, components or datasets, start a new run.

`asset_digest` identifies asset content; `version_id` includes lineage/proposal provenance. The shared deterministic `compile_agent_v1` passes prompt text and sorted skill text to the Solver. Candidates do not execute arbitrary Python/shell code. Workspaces are disposable materializations, verified against immutable assets. Separate directories are **not an OS sandbox**; only trusted fixture components execute in this version.

## Inspect, verify and rebuild reports

```sh
python -m membraneclaw_runtime.rsi.cli inspect --run-dir ./var/rsi/demo-001
python -m membraneclaw_runtime.rsi.cli verify --run-dir ./var/rsi/demo-001
python -m membraneclaw_runtime.rsi.cli report --run-dir ./var/rsi/demo-001
```

These commands do not dispatch models or tools. Verification checks trace hashes/payloads, segment lineage, asset/version identity, operation/result identity, raw evidence and ledger consistency. Reports exclude private expected answers and raw candidate assets/outputs. Raw traces contain private evaluation references and must remain private.

Reports are derived files. Deleting them does not change a decision; `report` reconstructs them. `semantic_digest` compares semantic decisions/assets, excluding runtime IDs and timestamps. `record_digest` identifies exact evidence and will differ across independent executions. Neither digest proves engineering correctness.

## Recover from interruption

```sh
python -m membraneclaw_runtime.rsi.cli resume --run-dir ./var/rsi/demo-001
```

A run lock prevents two local processes from owning the same run. Completed operations are reused; a prepared operation with no start record can dispatch once using its existing reservation. Terminal episodes remain immutable. A continuation segment registers its predecessor/head and bootstrap state in immutable genesis; no mutable locator file decides which version won.

A started operation with no durable outcome is unknown. Normal resume stops rather than automatically repeating it. For the built-in pure deterministic fixtures only, explicit recomputation is available:

```sh
python -m membraneclaw_runtime.rsi.cli resume \
  --run-dir ./var/rsi/demo-001 --recompute-offline
```

Recomputation records a new admitted attempt and retains the old uncertain reservation; it does not erase unknown usage. This flag does not authorize retrying live models or external tool actions. Live adapters are not enabled by this CLI. There is no operator override that silently turns unknown work into a successful result.

Wall time includes interruptions, with a persisted elapsed floor and a monotonic clock during active execution. Resume does not reset the budget. Over-reported usage is retained, never clamped. A budget-stop run cannot be extended by changing its saved configuration; start a new run.

An invalid returned result is stored as raw evidence before semantic rejection. Available valid usage is still charged. Credential-bearing raw bytes are rejected before encoding/persistence. An exception stores its type rather than potentially sensitive exception text. A crash after component return but before evidence capture remains unknown: this implementation cannot guarantee persistence of bytes across an instantaneous process kill.

A completed run remains completed on resume. Rejecting a candidate retains the incumbent; this MVP does not provide a command to rewrite historical promotion decisions or arbitrarily roll back a completed run.

## Final test evaluation

Search never sends test tasks to Solver/Proposer components. Evaluate the frozen selected version in a separate run:

```sh
python -m membraneclaw_runtime.rsi.cli final-evaluate \
  --run-dir ./var/rsi/demo-001 --output ./var/rsi/final-001 \
  --operations 2 --units 2 --wall-seconds 120
```

This budget fits the single packaged test task and seed (one solve plus one evaluation). Real datasets need their own measured budget. The parent search must be completed. Its trace is not modified. The evaluation registers the exact selected version and parent head, uses no Proposer, and cannot select another candidate. Its report includes its own ledger and the linked search-plus-this-evaluation ledger. Multiple evaluation directories do not implement a shared global cap, and repeatedly inspecting final-test results can compromise a research protocol.

## Integration and limits

Component protocols live in `contracts.py`; the controller owns all commits, promotion and budget transitions. Proposer requests contain allowed assets and public development tasks/outputs/scores, never validation/test reference answers. Configured callbacks are trusted code; in-process projection tests are not an adversarial memory-isolation boundary.

The native runtime, real WaterTAP environment, statistical promotion thresholds, real benchmark partition audit, learned meta-policy, multi-host execution, arbitrary tool-wrapper/code mutations, and dashboard are follow-up work. Only synthetic fixture adapters are accepted by this CLI. Before opening code execution, enforce a separate process/container boundary; before claiming improvement, freeze and run an independent real evaluation protocol.

The initial implementation checkpoints the full controller state on each transition. Storage and read/verification work grow with history; this is intended for small serial offline experiments, not high-volume distributed search. Run on a local filesystem with the existing FileTraceStore POSIX locking guarantees; do not run concurrent writers on copied/shared network directories.

Run the tests:

```sh
python -m unittest discover -s tests/rsi -v
```

Process-death tests include prepare/start/return/capture/completion/promotion boundaries and successor registration. Passing them verifies the tested local recovery protocol, not power-loss durability of every storage device or live API behavior. See [implementation plan](rsi-offline-implementation-plan.md) for scope, review corrections and live-readiness gates.
