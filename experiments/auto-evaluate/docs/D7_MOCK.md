# D7 mock pipeline set

`d7_mock` is a balanced six-case engineering-pipeline rehearsal set. It references existing source
workbooks without copying or editing the Excel files. Three cases remain self-contained R0 controls;
three receive versioned, corpus-grounded R2 overlays only in normalized data. Its run outputs are
independent from both D1-D6 and the future real D7.

## Included coverage

| Mock case | Original domain | Coverage |
| --- | --- | --- |
| `D7-mock-01-feasibility` | D1 + pressure-policy overlay | R2 / `use_rag` |
| `D7-mock-02-constraint-conflict` | D2 + manufacturer cleaning-trigger overlay | R2 / `use_rag` |
| `D7-mock-03-salinity-intrusion` | D3 + site recovery-policy overlay | R2 / `use_rag` |
| `D7-mock-04-capex-sensitivity` | unchanged D4 | R0 / `skip_rag` |
| `D7-mock-05-recovery-limit` | unchanged D5 | R0 / `skip_rag` |
| `D7-mock-06-multisimulator` | unchanged D6 | R0 / `skip_rag` |

The R2 overlays use pressure and recovery bands from `Safety file.xlsx` and cleaning criteria from
page 135, section 6.3 of `RO-operational Manual.pdf`. The hidden labels and evidence are never added
to the Router question. The set can validate both route classes, import, OpenWebUI execution,
Teacher/Judge orchestration, reward analysis, and reporting. It remains synthetic development data
and must not enter paper result tables.

## Commands

Initialize and validate the independent normalized set:

```bat
python ae.py validate-benchmarks --benchmark-set d7_mock
```

Run the short Router-only rehearsal on all six cases:

```bat
python ae.py router-eval --benchmark-set d7_mock --run-id d7-mock-mixed-router-v1
```

Run a two-case end-to-end smoke test before attempting all six:

```bat
python ae.py auto --benchmark-set d7_mock --run-id d7-mock-e2e-smoke --stage systems --case D7-mock-01-feasibility --case D7-mock-05-recovery-limit
```

If both cases complete, run all six system conditions:

```bat
python ae.py auto --benchmark-set d7_mock --run-id d7-mock-e2e-v1 --stage systems
```

Only continue to Teacher/Judge after reviewing system completion. Use the same run ID:

```bat
python ae.py auto --benchmark-set d7_mock --run-id d7-mock-e2e-v1 --stage teachers
python ae.py auto --benchmark-set d7_mock --run-id d7-mock-e2e-v1 --stage judges
python ae.py auto --benchmark-set d7_mock --run-id d7-mock-e2e-v1 --stage report
```
