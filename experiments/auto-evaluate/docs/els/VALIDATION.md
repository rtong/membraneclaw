# Offline validation

Validated on 2026-10-19 in a separate checkout of upstream commit
`38084160d1642bb26c516d49369fdcaa0173cd9d`, using Python 3.12.14.

From `experiments/auto-evaluate`:

| Command | Result |
|---|---|
| `python examples/els/offline_demo.py` | Two synthetic observations; derived increment 100 USD/pump; both source records retained |
| `python -m unittest discover -s tests -p 'test_els*.py' -v` | 26 tests passed; network connections blocked in ELS tests |
| `python -m unittest discover -s tests` | 105 tests passed, including the 26 ELS tests |

Existing repository tests require their existing development dependencies.
The ELS-only tests and demo use the Python standard library.

These are synthetic/unit checks, not new model, WaterTAP or Judge experiments.
No benchmark score, performance uplift or full D1–D7 integration is established
by these checks. The production experiment runner and its original frozen
artifacts are outside this PR.
