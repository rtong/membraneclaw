# D7 mock short Router probe

`d7_mock_router_probe` is a Router-only diagnostic set. It contains three short matched pairs:
pressure policy, membrane-cleaning criteria, and recovery policy. In each pair, the engineering
decision is held constant; only the availability of the governing external thresholds changes.

- The missing-threshold member is R2 / `use_rag`.
- The supplied-threshold member is R0 / `skip_rag`.

This design tests whether a routing failure comes from the Router policy itself or from the relevant
knowledge gap being obscured by a long benchmark prompt. It is synthetic development data and is
not paper evidence.

## Commands

Initialize and validate the six cases:

```bat
python ae.py validate-benchmarks --benchmark-set d7_mock_router_probe
```

Evaluate both Router variants:

```bat
python ae.py router-eval --benchmark-set d7_mock_router_probe --run-id d7-mock-short-probe-v1
```

Do not run `auto`, systems, Teacher, or Judge on this set. The short prompts replace the workbook
questions only for routing diagnosis; the original workbook answers and scoring rubrics no longer
describe those replacement prompts.

## Interpretation

- Both variants identify R2 here but failed on the long mixed mock: long-prompt salience is the main issue.
- Only Router Skill identifies R2: the Skill adds useful routing behavior under a clean signal.
- Both variants still always skip RAG: the Router/Skill policy is biased toward `skip_rag` and needs revision.

## Natural follow-up

The first probe explicitly says that thresholds are not provided. After it reaches the ceiling, run
the natural short set, which removes those direct missing-knowledge cues while keeping the same
three engineering decisions and matched R0 controls:

```bat
python ae.py validate-benchmarks --benchmark-set d7_mock_router_natural
python ae.py router-eval --benchmark-set d7_mock_router_natural --run-id d7-mock-natural-probe-v1
```

This follow-up is also Router-only and must not enter systems, Teacher, or Judge stages.
