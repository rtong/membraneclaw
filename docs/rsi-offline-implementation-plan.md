# Offline-first RSI implementation plan

Status: implementation contract. Offline A-D are implemented with synthetic-only validation; live adapters (E) and real-agent effectiveness remain unverified.

Implementation base: upstream `ec69f89` (the traced runtime merged in PR #43; source submission `635890c1b883c957022adbf202c061374ba952c4`). Recheck upstream changes before implementation. This plan covers agent prompt/skill improvement, not model-weight training.

## 1. Outcome and boundaries

Deliver one local command that runs a baseline, proposes candidate changes, evaluates them, accepts or rejects them, and starts a second round from the selected version. It must also recover from process interruption without silently repeating external work.

Before model APIs are available, use explicitly labeled scripted Solver and Proposer adapters and an executable deterministic Evaluator. The fixtures test orchestration, not WaterTAP engineering accuracy or model capability. Never plot fixture scores as experimental evidence.

The first implementation supports declarative changes to a dedicated agent prompt and plain-text skill files. Arbitrary Python/shell patches, tool-wrapper edits, model-weight updates, learned meta-policies, distributed scheduling, and a web dashboard are deferred. These introduce independent execution and evaluation risks and are not necessary to prove the first loop.

A single candidate is generated per round. Execution is serial. Rejected candidates leave the incumbent unchanged. Accepted candidates become the next round's parent. Use configurable multi-round operation, with a two-round demonstration as the initial acceptance case.

## 2. What exists and what must be added

| Existing component | Reuse | Missing integration |
|---|---|---|
| `agent/factory.py::runtime_session` | Future native Solver execution | Turn an immutable agent version into its prompt/skills |
| `membraneclaw_runtime/gateway.py` | Model budgets, native state, durable model/tool capture | Link child execution episodes to RSI operations |
| `tracing/schema.py` | Canonical JSON, hashing, episode/event envelopes | Versioned RSI documents and domain validation |
| `tracing/file_store.py`, `recorder.py` | Durable commits, snapshots, local execution ownership | Controller checkpoints, operation ledger, version graph |
| `tracing/validation.py`, `export.py` | Trace integrity validation and explicit private export | RSI-specific consistency checks and reports |
| `experiments/auto-evaluate` | Existing benchmark import, scoring conventions and reports | A separately tested adapter; not a direct model correctness oracle |
| `mcp_watertap` and MCP adapters | Future physical simulation evidence | Live environment smoke and answer-to-constraint verification |

Keep the core in `membraneclaw_runtime/rsi/`. Do not make importing it depend on the evaluation package, HTTP, model credentials, Codex login, or WaterTAP. Add no new third-party dependency for offline orchestration.

## 3. Module and artifact layout

Planned files:

```
membraneclaw_runtime/rsi/
  __init__.py
  contracts.py       # validated immutable documents and component protocols
  datasets.py        # public task projection, split manifests, group audit
  versions.py        # content-addressed prompt/skill versions and candidate edits
  workspace.py       # materialize only permitted assets; verify before/after
  policy.py          # explicit acceptance rules and decision explanations
  budget.py          # reserve, settle, and retain uncertain charges
  journal.py         # controller events, checkpoints, recovery and ownership
  controller.py      # serial state machine; no provider-specific behavior
  report.py          # reports derived from committed evidence
  fixtures.py        # scripted proposer/solver and deterministic evaluator
  cli.py             # init-example, demo, run, resume, inspect, report, verify
  data/              # packaged synthetic fixtures; init-example exports editable copies
    offline_config.json
    assets/prompt.txt
    assets/skills/unit-checks.md
    tasks_public.json
    evaluation_private.json
tests/rsi/
  test_contracts.py
  test_versions.py
  test_datasets.py
  test_budget.py
  test_controller.py
  test_recovery.py
  test_report_cli.py
docs/rsi.md           # English installation, operation and limitation guide
```

Run artifacts live under an ignored `var/rsi/` directory, outside source assets. Canonical events/checkpoints and referenced payloads are authoritative. Materialized workspaces, `state.json`, and report files are disposable projections; their absence must not change a committed decision.

No database, GitHub archive, or network connection is required to run the offline controller. Existing private export/archive facilities can be used separately after recording.

## 4. Configuration and immutable inputs

`RunSpec` records schema version, mode, controller/component versions, source commit, initial agent digest, dataset/split digests, policy, budgets, seed schedule, and maximum rounds. Hash the canonical document. Exclude credentials; record secret references only when a future adapter needs them.

Pin all referenced asset/task/evaluator bytes at initialization. An uncommitted or modified working tree must either be rejected or explicitly captured as a content-addressed source snapshot. A Git commit alone is insufficient provenance for dirty files.

Resume requires the original configuration and component identities. Reject changed dataset bytes, missing snapshots, unknown schemas, or incompatible adapter versions. A revised configuration starts a new run linked to the previous run.

Initial offline configuration requirements:

- `mode=synthetic_offline`, enforced by explicit adapter capability checks.
- `max_rounds=2` for the demonstration; not a research stopping rule.
- One candidate per round; no concurrency.
- Explicit prompt/skill path allowlist; no wildcard access to the repository.
- Exact seed lists and task IDs per split, not an unspecified random split.
- Demo acceptance rule: positive validation gain, no per-case score regression, and complete valid execution. This is a deterministic test policy, not a statistically justified live-experiment threshold.
- Positive operation-count and wall-time limits. The fixture files specify their simulated unit charges; actual model dollars/tokens are absent, not invented.

## 5. Documents and component contracts

Implement frozen dataclasses plus strict serialization/validation. Reject extra fields when they could change execution semantics, non-finite scores/costs, duplicate identifiers, and unresolved references.

| Document | Required information |
|---|---|
| `AgentVersion` | Parent digest, sorted path/content-digest manifest, source candidate, immutable asset references |
| `Candidate` | Parent version digest, exact file edits, before/after digests, rationale, development evidence references, proposer identity |
| `TaskInput` | Public task ID and text, public constraints/tool permissions; no reference answer or private rubric |
| `SolverResult` | Task/version/seed/operation identity, execution status, answer/artifact references, child trace reference, observed usage |
| `CaseEvaluation` | Same task/version/seed identity, bounded score and scale, constraint checks, execution/evidence completeness, evaluator identity |
| `EvaluationBatch` | Exact ordered case roster, repetition schedule, result references, missing/invalid cases and aggregate calculation |
| `Decision` | Parent/candidate and evaluation digests, accepted/rejected/inconclusive, rule-by-rule reasons, selected incumbent |
| `OperationRecord` | Stable operation ID, input digest, prepared/started/completed/failed/unknown state, reservation and outcome references |
| `ControllerCheckpoint` | Run/segment IDs, phase, round, incumbent, candidate, completed operation references, ledger and remaining work |

Interfaces, with concrete request/result types defined in `contracts.py`:

```python
class Solver(Protocol):
    async def solve(self, request: SolveRequest) -> SolverResult: ...

class Proposer(Protocol):
    async def propose(self, request: ProposalRequest) -> Candidate: ...

class Evaluator(Protocol):
    async def evaluate(self, request: EvaluationRequest) -> CaseEvaluation: ...
```

The controller owns durable transitions. Components return data; they cannot update the incumbent or decide promotion. A component may return a child trace reference, but the controller verifies its operation/version/task identity before accepting it.

`ProposalRequest` contains the incumbent's allowed assets and development evidence only. Validation feedback returned to the Proposer is limited to an explicitly configured aggregate decision summary. Never supply hidden test inputs/answers or validation reference answers. Any permitted feedback is recorded as part of the adaptive search protocol.

The Solver receives public task data only. The trusted Evaluator loads its private references separately. An Evaluator must distinguish an incorrect answer from a failed/incomplete execution; missing results are not silently dropped from the denominator.

## 6. Versions, workspaces and trust boundary

Candidate application is deterministic: verify the parent manifest and before-digests, validate every edit, compute the new manifest, and only then register a new immutable version. Reject empty/no-op candidates and duplicate previously seen versions with a recorded reason.

For the MVP, allow only UTF-8 prompt/skill text assets. Disallow absolute paths, `..`, alternate separators, symlinks, executable mode changes, Git metadata, evaluator/configuration changes and writes outside the selected asset root. Add explicit path-collision tests, including case-insensitive filesystem behavior.

Materialize each evaluation workspace from its version manifest, never by applying a patch onto a shared mutable checkout. Preserve all baseline and accepted versions. Rollback means retaining/reselecting a version digest; it does not run `git reset` against the user's repository.

Path validation and separate directories are not an OS sandbox. The offline adapters execute trusted code only, with no generated code, arbitrary command execution, dynamic candidate imports or credentials. Before allowing code/tool-wrapper mutations or real tools with filesystem/network access, add a separate process/container boundary with enforced mounts, credential separation, process/network limits and an independently tested evaluator boundary.

## 7. Dataset split and evaluator preparation

Create a manifest containing task digest, source workbook/version, task-family/group ID, split, and grouping rationale. Keep all known variants of the same underlying problem in one split. Check exact duplicates automatically; flag uncertain semantic relationships for human review rather than claiming automatic leakage detection.

Existing D1-D6 normalized tasks are inputs to this audit, not proof that a clean RSI split already exists. Do not use the synthetic D7 mocks as a substitute for missing real D7 data.

For the offline demonstration, generate separate tiny development and validation fixtures. The fixture Solver produces outputs from controlled scenario rules; the Evaluator computes correctness from those outputs and private fixture answers. Avoid a fixture that simply returns a score based on a candidate's name. Include a candidate that improves development behavior but fails validation, so the acceptance path is actually exercised.

The final test split is excluded from the search command, adapter requests, workspaces and report details during optimization. Use a separate explicit final-evaluation command after selecting a frozen version, recording both its version and evaluation protocol. Repeated validation feedback is adaptive selection and can overfit; a validation improvement is not a generalization claim.

Optional parallel work: run fixed WaterTAP input cases without any model. Capture successful results, invalid-input behavior, solver failures and tool/environment versions. Record observations from actual executions only. A skipped/unavailable simulator remains unverified.

## 8. Controller state machine and commit order

Phases:

1. `initialized`: pin configuration, versions, split roster and component identities.
2. `baseline_development`: collect incumbent development results for diagnosis.
3. `baseline_validation`: collect the comparison batch under the frozen protocol.
4. `proposing`: request one candidate using permitted development evidence.
5. `candidate_checked`: validate edits and register immutable candidate assets.
6. `candidate_evaluating`: evaluate incumbent/candidate on the same validation roster and seed schedule.
7. `deciding`: check completeness and apply acceptance policy.
8. `round_committed`: atomically record the Decision and selected incumbent.
9. `next_round` or `finished`: collect development evidence for the selected incumbent and continue, or stop with an explicit reason.

Task/seed ordering is fixed in the RunSpec. Initial baseline results can be reused in fixture mode only when all operation inputs and adapter determinism claims match exactly. Live repeated comparisons need a separately frozen repetition/order policy; do not assume seeds make remote APIs deterministic.

For every component invocation:

- Validate input, calculate maximum permitted reservation, and atomically persist `operation.prepared` plus the updated ledger/checkpoint.
- Persist `operation.started` before dispatch. Failure to commit prevents dispatch.
- Invoke the component under execution ownership and remaining wall-time budget.
- Validate identity, result shape and usage. Persist the result reference, outcome, budget settlement and next checkpoint together.
- Advance only from a committed valid result. Outcome persistence failure stops new work; it never triggers a repeated external invocation.

Persist `candidate.accepted`/`candidate.rejected` and the selected incumbent in the same commit. Materialize a convenient pointer/report only afterward. A crash between those steps must reconstruct the same incumbent from committed evidence.

Candidate evaluation failures produce an inconclusive decision, not an artificial score gain or a promotion. Continue or stop according to an explicit configured failure policy; the initial policy stops for operator inspection.

## 9. Journal, locking and recovery

Use a controller trace episode per execution segment, with a shared RSI run identifier in metadata. The controller holds its execution guard across component waits and cleanup. Child Solver model/tool episodes use their own existing gateway ownership; they do not write controller state directly.

Only one active controller segment may own an RSI run. Add a run-level nonblocking local lock and a durable segment locator so two processes cannot both create continuation segments. Lock ordering is run lock, then segment execution guard, then normal store commits. Never wait for the run lock while holding a child episode lock.

Each transition commits its payload and ControllerCheckpoint through the existing TraceStore. Do not create a competing JSONL journal or depend on an uncommitted cache file. Stable operation/commit identities permit storage reconciliation after a lost commit acknowledgement. Recover state from verified committed checkpoints, not in-memory pending requests.

Respect existing terminal-state rules: a recovered/finished `interrupted`, `failed`, `cancelled` or `completed` episode cannot accept new commits. Resumption after an interrupted terminal segment creates a new linked controller segment under the run lock, using the last verified checkpoint. Do not weaken TraceStore terminal semantics to implement RSI resume.

Recovery cases:

| Last durable evidence | Required action |
|---|---|
| Prepared, never started | Dispatch once after restoring its existing reservation |
| Completed, outcome and checkpoint committed | Reuse; do not call the component again |
| Started, no authoritative outcome | Mark unresolved/unknown; do not automatically retry a potentially external action |
| Remote outcome independently retrievable by operation ID | Reconcile through an adapter capability, recording the receipt and original operation identity |
| Offline fixture explicitly declares pure deterministic behavior | Allow explicit safe recomputation with a recorded recovery event; do not generalize this permission to live adapters |
| Promotion committed, cache/workspace missing | Rebuild projections from the selected version digest |
| Corrupt/missing referenced data | Fail verification and stop; never guess the incumbent |

Cancellation stops further dispatch, drains active work as required by its adapter, captures any available late outcome, and records a stopped/unknown state. Timeouts do not imply external cancellation. Operator reconciliation is a local recorded command, not an unrecorded edit to checkpoint JSON.

## 10. Budgets and promotion policy

Track separate dimensions: component invocations, tasks, tool calls, model tokens when available, wall time, and money only when a trustworthy price/accounting source is configured. Never label synthetic units as actual dollars. Missing token/cost observations remain unknown. Define wall time as elapsed time from the recorded run start, including interruptions, and check it again on resume; use a monotonic clock during each live segment. A pause never silently resets the time budget.

Reserve before dispatch using an enforceable per-operation bound. Unknown outcomes retain their reservation. Observed usage above a reservation is recorded in full and stops further dispatch; never clamp it to make the budget appear respected. An external endpoint can overrun a local estimate, so distinguish an admission budget from a hard provider-enforced spending cap. Reject live configurations claiming a strict financial cap when no defensible bound/enforcement exists.

The experiment's total ledger includes baseline, diagnosis/proposal, candidate evaluation, validation, failures, retries explicitly authorized by adapter semantics, and final test evaluation if requested. Reuse records provenance and zero new invocations rather than erasing original cost.

Decisions require matching evaluation protocols and complete rosters. Record mean score change, per-case differences, hard constraint regressions, execution completeness and known/unknown cost dimensions. Keep policy weights/thresholds frozen for the run. An unavailable metric makes a rule requiring it inconclusive.

The fixture policy's exact arithmetic and expected decisions are test expectations only. Select live thresholds, repetitions and hard constraints after baseline measurements and before candidate selection. Preserve the initial unmodified baseline as well as the current incumbent in all reports.

## 11. Report and planned CLI

The English report contains mode/provenance banners, pinned inputs, version lineage, per-round candidate edits, accepted/rejected/inconclusive reasons, paired case results, baseline-versus-incumbent changes, cumulative ledger, unknown outcomes and trace links. No score smoothing or missing-case omission.

Start with JSON plus Markdown; defer an interactive dashboard until the underlying records are stable. Escape candidate text and do not include private answers or credentials in default reports. Private evidence remains available through explicit authorized artifact access.

Implemented offline commands (see `docs/rsi.md` for operational semantics):

```sh
python -m membraneclaw_runtime.rsi.cli demo --output ./var/rsi/demo-001
python -m membraneclaw_runtime.rsi.cli init-example --output ./var/rsi/example-001
python -m membraneclaw_runtime.rsi.cli run --config ./var/rsi/example-001/offline_config.json --output ./var/rsi/run-001
python -m membraneclaw_runtime.rsi.cli inspect --run-dir ./var/rsi/run-001
python -m membraneclaw_runtime.rsi.cli resume --run-dir ./var/rsi/run-001
python -m membraneclaw_runtime.rsi.cli verify --run-dir ./var/rsi/run-001
python -m membraneclaw_runtime.rsi.cli report --run-dir ./var/rsi/run-001
```

Creating an existing run must fail rather than overwrite it. Resume discovers its immutable configuration; it does not accept silent overrides. Verification/reporting perform no model/tool actions. A completed run does not become active merely because `resume` was called.

## 12. Implementation PR sequence

All implementation PRs ultimately target `rtong/membraneclaw`. Use fork branches for development. Submit each dependency after its prerequisite is merged, or explicitly agree an upstream stacking arrangement; do not present fork-internal PRs as upstream submissions.

| PR | Deliverables | Acceptance evidence |
|---|---|---|
| A: contracts, splits and immutable versions | Documents, protocols, input snapshots, group manifest, declarative edit validation and workspace materialization | Serialization/hash stability; forbidden paths; duplicate/no-op candidates; variant-group leakage; public/private projections |
| B: durable journal and budgets | Segment/run ownership, operation ledger, reserve/settle rules, checkpoints and reconciliation | Process-lock exclusion; lost-ack storage recovery; terminal-segment continuation; unknown reservations retained |
| C: controller and offline adapters | Serial state machine, paired evaluation, decision policy and two-round fixtures | Accept/reject/inconclusive branches; selected parent in round two; cancellation prevents further dispatch |
| D: CLI, reports and operational guide | Planned commands, JSON/Markdown reports, process-crash tests, CI wiring and English README | One-command demo; repeated resume; report rebuild; clean-environment install; complete fault matrix |
| E: live adapters after API readiness | Native Solver/Proposer, trusted evaluation integration and real MCP smoke | Small genuine end-to-end run with actual usage/trace evidence; no synthetic-effectiveness claim |

For the first live Solver adapter, disable unrestricted built-in execution/file tools (`include_builtin_tools=False`) and expose only explicitly approved simulation tools. Keep provider/MCP credentials in the trusted executor, outside candidate assets and proposer-visible data. This restriction does not replace the OS isolation required before enabling generated code.

A-D can be implemented without model access. WaterTAP local smoke is an independent optional task and does not block A-D. E requires real credentials/environment and its own review. Each PR must be independently reviewable; do not combine model-weight training or second-level policy learning into this scope.

## 13. Required fault and behavioral tests

- Candidate improves development but regresses validation: reject and preserve incumbent.
- Candidate genuinely improves deterministic fixture validation: accept; round two uses its exact asset digest.
- Equal score, invalid output, missing case, duplicate case, changed score scale, NaN and mismatched version: never promote accidentally.
- Candidate attempts to change rubric, configuration, hidden data, path traversal, symlink or executable asset: reject before invocation.
- Missing outcome/unknown usage: record explicitly; do not silently omit from aggregate or release uncertain reservations.
- Crash after prepare, after start, after component return but before commit, after result commit, and after decision commit but before projection update.
- Fail storage before dispatch: zero component calls. Lose acknowledgement after commit: one result/decision, no duplicate work.
- Resume an interrupted terminal segment: linked continuation segment; original terminal trace remains immutable.
- Two controller processes race to start/resume: one owner, no duplicated segment/operation.
- Cancel during a component call, worker cleanup or state commit: no next operation; retain available late outcomes.
- Budget exactly fits, cannot fit, is over-reported, or has unresolved charge: explicit and correct dispatch/stop behavior.
- Dirty source, changed fixture bytes, changed config, missing asset or corrupt event chain: fail verification/resume.
- Rebuild reports/workspaces after deleting disposable projections: identical decisions and version lineage.
- Inject public/private canaries: private reference answers are absent from Solver/Proposer requests, candidate workspaces and default reports.
- Run with missing model credentials and blocked network: offline mode still works; model factories are never invoked.

Use process termination tests for durability and concurrent-process tests for locks. Exceptions alone do not reproduce process death. Deterministic fixture runs must match semantic results, version/decision digests and invocation counts; timestamps and execution IDs need not be identical.

## 14. Exit gates and next steps

Gate 1, structural readiness: contracts and immutable versions pass validation; no hidden-data exposure in component projections.

Gate 2, offline loop readiness: two-round demonstration plus a rejection scenario, all fault cases, explicit budget stops, and recovery verification pass without any network or credentials.

Gate 3, live smoke readiness: operator provides model route/credentials, accessible simulation environment, and an approved experimental budget. Confirm one real model call and one real simulator call with preserved observations before running a search.

Gate 4, experimental readiness: freeze live baseline, evaluator, data grouping, repetition schedule, acceptance policy and final-test protocol. Run genuine candidates and publish failures/costs alongside successes. No effectiveness conclusion is available before this gate.

Only after collecting enough comparable real improvement records should the second-level meta-policy be considered. Compare it against simple fixed and random direction-selection baselines under equal total budgets. The offline controller is infrastructure for that study, not evidence that learned recursive improvement already works.

## 15. Review corrections and implementation mapping

The plan review identified five contract gaps. These requirements supersede any less specific wording above:

1. **Recovery registration:** a successor's deterministic ID is derived from its terminal predecessor's commit hash. Its immutable genesis contains the exact predecessor/head, bootstrap checkpoint and checkpoint digest. Under a run lock, recovery discovers this registration without a mutable head pointer, repairs a crash before genesis using the known predecessor, and adopts an already registered but unstarted successor. An unregistered/corrupt root fails closed. Process-death tests cover every registration boundary.
2. **Invalid outcome evidence:** capture a returned JSON/dataclass representation as an immutable raw-byte envelope before interpreting result fields. Then record semantic validation and usage settlement separately. NaN/extra fields/identity mismatches remain inspectable. Credential policy checks raw bytes before any base64 encoding; rejected credentials are not persisted. Exceptions preserve sanitized exception type and an uncertain reservation. A crash before capture leaves an explicitly unknown operation.
3. **Distinct digests:** `asset_digest` identifies compiled asset content; `version_id` identifies assets plus lineage/proposal provenance; `record_digest` identifies exact records including execution IDs and timestamps; `semantic_digest` compares outcome/asset/decision projections without runtime identities. Reverting to previously seen assets is rejected even if lineage would produce a new version ID.
4. **Compilation contract:** `compile_agent_v1` maps `prompt.txt` and lexically ordered skill text to an immutable `CompiledAgent`. The fixture Solver consumes that object; the future live Solver must use the same compiler. Tests change skill text and verify the actual Solver input/output changes.
5. **Final evaluation:** `final-evaluate` creates an independent linked run from a frozen completed search. It cannot append to or promote within the terminal parent. It requires a separate explicit budget, records the selected version/parent head, and reports both its own ledger and search-plus-this-evaluation totals. Multiple independent final-evaluation directories do not share a global spending cap.

Implementation delivery adjustment: A-D are implemented together in one upstream offline-controller PR because recovery, evidence validation and controller acceptance tests exercise the same end-to-end protocol. The module boundaries above remain independently reviewable. E remains deferred until credentials and the real execution environment are available.

Pinned source bytes and input snapshots are authoritative. Changes to original external dataset/configuration files do not silently alter an existing run; modified pinned records fail integrity validation. Resume refuses changed controller/tracing implementation bytes. Schema/behavior changes require a new run or the original implementation, not an implicit migration.
