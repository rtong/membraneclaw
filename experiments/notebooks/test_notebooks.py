"""Execution and consistency tests for the notebook series.

The notebooks are deliberately standalone: each one redefines the MDP in its own
first code cells so that it can be opened on its own in Colab. That duplication
is a feature, and it is also the risk — every copy of the same constants can
drift apart silently while every notebook still runs green. Nothing else in the
series checks that they still agree.

These tests do two things:

* execute every notebook end to end, so a cell that has quietly stopped
  working is caught;
* compare the MDP definition, the discount factor, and the exact ``v*`` answer
  key across every notebook that defines them.

The answer key is the load-bearing one. Notebook 02 computes ``v*`` exactly, and
03 onwards all grade their sampled estimates against it. If one notebook's
copy of the MDP drifts, they are silently grading against different problems.

Requirements: ``numpy``, ``matplotlib``, ``nbformat``, ``nbclient``, ``ipykernel``,
``pytest``. The repository ``.venv`` has all of them:

    cd experiments/notebooks && ../../.venv/bin/python -m pytest

The whole suite executes every notebook once and takes about a minute.
"""

from __future__ import annotations

import json
import pathlib

import nbformat
import pytest
from nbclient import NotebookClient

NB_DIR = pathlib.Path(__file__).parent
NOTEBOOKS = sorted(p.name for p in NB_DIR.glob("[0-9][0-9]_*.ipynb"))

# Constants every notebook that models the MDP should agree on. A notebook is
# only checked for the names it actually defines: 01 has no GAMMA because it has
# not introduced discounting yet, and that is not drift.
SHARED_SCALARS = (
    "N_STATES",
    "N_ACTIONS",
    "GAMMA",
    "COST_CHECK",
    "COST_REPEAT_CHECK",
    "REWARD_SIM_SUCCESS",
    "REWARD_SIM_FAILURE",
    "REWARD_SUBMIT_SUCCESS",
    "REWARD_SUBMIT_FAILURE",
)
SHARED_DICTS = ("SIM_SUCCESS_PROB", "SUBMIT_SUCCESS_PROB")
SHARED_ARRAYS = ("P", "R")

# Each notebook's name for the exact optimal value function from notebook 02.
ANSWER_KEY_NAMES = {
    "02_bellman_value_iteration.ipynb": "v_star",
    "03_monte_carlo.ipynb": "V_TRUE",
    "04_reinforce_baseline.ipynb": "V_STAR",
    "05_ppo_clipping.ipynb": "V_STAR",
    "06_td_sarsa_qlearning.ipynb": "V_STAR",
    "07_model_based_dyna.ipynb": "V_STAR",
}

# Hand-computed for gamma=0.95, and stated in notebook 02's prose. Pinned here so
# that a change to the MDP has to be a deliberate edit to this file rather than a
# silent re-baselining of every notebook at once.
EXPECTED_V_STAR = [6.245, 7.1, 7.1, 8.0, 0.0, 0.0]

TOL = 1e-9

_PROBE = """
import json as _json
import numpy as _np

def _plain(_v):
    if isinstance(_v, _np.ndarray):
        return _v.tolist()
    if isinstance(_v, dict):
        return {int(_k): float(_x) for _k, _x in _v.items()}
    if isinstance(_v, (set, frozenset)):
        return sorted(int(_x) for _x in _v)
    if isinstance(_v, (list, tuple)):
        return [int(_x) for _x in _v]
    return float(_v)

_names = %(names)r
_out = {}
for _n in _names:
    if _n in globals():
        try:
            _out[_n] = _plain(globals()[_n])
        except Exception as _e:          # a name reused for something unrelated
            _out[_n] = {"__unreadable__": str(_e)}
print("PROBE_START")
print(_json.dumps(_out))
print("PROBE_END")
"""


def _execute(name):
    """Run one notebook, then read the values of interest out of its kernel.

    Returns (executed_notebook, namespace). Raises CellExecutionError if any
    cell fails, which is what makes this a test of the notebook itself.
    """
    nb = nbformat.read(NB_DIR / name, as_version=4)
    wanted = (
        list(SHARED_SCALARS)
        + list(SHARED_DICTS)
        + list(SHARED_ARRAYS)
        + ["TERMINAL_STATES", "NONTERMINAL", "v_reckless"]
        + ["Q_STAR", "Q_EPS", "Q_s", "Q_q", "Q_hat"]
        + sorted(set(ANSWER_KEY_NAMES.values()))
    )
    probe = nbformat.v4.new_code_cell(_PROBE % {"names": wanted})

    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(NB_DIR)}},
    )
    with client.setup_kernel():
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                client.execute_cell(cell, index)
        nb.cells.append(probe)
        client.execute_cell(probe, len(nb.cells) - 1)
    nb.cells.pop()

    text = "".join(
        o.get("text", "") for o in probe.outputs if o.output_type == "stream"
    )
    payload = text.split("PROBE_START")[1].split("PROBE_END")[0]
    return nb, json.loads(payload)


@pytest.fixture(scope="session")
def executed():
    """Every notebook, run once for the whole session."""
    return {name: _execute(name) for name in NOTEBOOKS}


def test_the_series_is_every_numbered_notebook():
    assert NOTEBOOKS == [
        "01_finite_mdp.ipynb",
        "02_bellman_value_iteration.ipynb",
        "03_monte_carlo.ipynb",
        "04_reinforce_baseline.ipynb",
        "05_ppo_clipping.ipynb",
        "06_td_sarsa_qlearning.ipynb",
        "07_model_based_dyna.ipynb",
    ]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_runs_without_error(executed, name):
    nb, _ = executed[name]
    errors = [
        (i, o.get("ename"), o.get("evalue"))
        for i, cell in enumerate(nb.cells)
        if cell.cell_type == "code"
        for o in cell.get("outputs", [])
        if o.output_type == "error"
    ]
    assert errors == []


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_defines_the_mdp(executed, name):
    """Every notebook is self-contained, so every one of them builds the MDP."""
    _, ns = executed[name]
    for key in ("N_STATES", "N_ACTIONS", "COST_CHECK", "SIM_SUCCESS_PROB"):
        assert key in ns, f"{name} does not define {key}"
    assert ns["N_STATES"] == 6
    assert ns["N_ACTIONS"] == 4


def _agree(a, b):
    """Compare two probed values, allowing for float noise in derived tables.

    P and R are built by arithmetic rather than written down, so two notebooks
    that agree perfectly still differ in the last bit or two. Real drift is many
    orders of magnitude larger than TOL, so nothing meaningful slips through.
    """
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_agree(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_agree(a[k], b[k]) for k in a)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= TOL
    return a == b


@pytest.mark.parametrize("key", SHARED_SCALARS + SHARED_DICTS + SHARED_ARRAYS)
def test_the_copies_of_the_mdp_agree(executed, key):
    """The real point: one inline definition per notebook, one problem."""
    seen = {name: ns[key] for name, (_, ns) in executed.items() if key in ns}
    assert seen, f"no notebook defines {key}"

    reference_name, reference = next(iter(seen.items()))
    for name, value in seen.items():
        assert _agree(value, reference), (
            f"{key} differs between {reference_name} and {name}: "
            f"{reference!r} vs {value!r}"
        )


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_transition_probabilities_are_distributions(executed, name):
    _, ns = executed[name]
    if "P" not in ns:
        pytest.skip(f"{name} does not build the dense P table")
    for s, rows in enumerate(ns["P"]):
        for a, row in enumerate(rows):
            assert abs(sum(row) - 1.0) < 1e-12, f"P[{s},{a}] sums to {sum(row)}"


@pytest.mark.parametrize("name", sorted(ANSWER_KEY_NAMES))
def test_the_answer_key_is_the_same_everywhere(executed, name):
    """02 solves the MDP exactly; 03, 04 and 05 are all graded against it."""
    _, ns = executed[name]
    key = ANSWER_KEY_NAMES[name]
    assert key in ns, f"{name} no longer defines {key}"
    v = ns[key]
    assert len(v) == len(EXPECTED_V_STAR)
    for state, (got, want) in enumerate(zip(v, EXPECTED_V_STAR)):
        assert abs(got - want) < 1e-6, (
            f"{name}: {key}[{state}] is {got}, expected {want}. If the MDP or "
            f"gamma changed on purpose, update EXPECTED_V_STAR."
        )


def test_terminal_states_are_worth_nothing(executed):
    """SUCCESS and FAILURE are absorbing, so v* is exactly 0 there."""
    for name, key in ANSWER_KEY_NAMES.items():
        _, ns = executed[name]
        assert ns[key][4] == 0.0 and ns[key][5] == 0.0, name


def test_action_values_are_consistent_with_state_values(executed):
    """Notebook 06 grades control against q*, so q* had better match v*."""
    _, ns = executed["06_td_sarsa_qlearning.ipynb"]
    q_star, v_star = ns["Q_STAR"], ns["V_STAR"]
    for s in ns["NONTERMINAL"]:
        assert abs(max(q_star[s]) - v_star[s]) < 1e-9, (
            f"max_a q*({s}, a) should equal v*({s})"
        )


def test_sarsa_and_q_learning_converge_to_different_targets(executed):
    """Notebook 06's central claim, pinned.

    SARSA is on-policy and converges to q for the eps-greedy policy; Q-learning
    is off-policy and converges to q*. Both saw the same behaviour policy, so if
    this ever collapses to one answer the notebook is teaching something false.
    """
    _, ns = executed["06_td_sarsa_qlearning.ipynb"]
    states = ns["NONTERMINAL"]

    def max_gap(a, b):
        return max(abs(a[s][i] - b[s][i]) for s in states for i in range(4))

    sarsa_to_eps = max_gap(ns["Q_s"], ns["Q_EPS"])
    sarsa_to_star = max_gap(ns["Q_s"], ns["Q_STAR"])
    qlearn_to_star = max_gap(ns["Q_q"], ns["Q_STAR"])
    qlearn_to_eps = max_gap(ns["Q_q"], ns["Q_EPS"])

    assert sarsa_to_eps < sarsa_to_star, (
        f"SARSA should sit nearer q_eps ({sarsa_to_eps:.3f}) "
        f"than q* ({sarsa_to_star:.3f})"
    )
    assert qlearn_to_star < qlearn_to_eps, (
        f"Q-learning should sit nearer q* ({qlearn_to_star:.3f}) "
        f"than q_eps ({qlearn_to_eps:.3f})"
    )


def test_exploration_breaks_the_tie_at_the_start_state(executed):
    """q* is indifferent between the two checks; q_eps is not."""
    _, ns = executed["06_td_sarsa_qlearning.ipynb"]
    salinity, fouling = 0, 1  # Action.CHECK_SALINITY, Action.CHECK_FOULING
    no_info = 0

    assert abs(ns["Q_STAR"][no_info][salinity] - ns["Q_STAR"][no_info][fouling]) < 1e-9
    assert ns["Q_EPS"][no_info][salinity] > ns["Q_EPS"][no_info][fouling] + 0.1


def test_planning_on_expert_data_is_blind_off_the_expert_path(executed):
    """Notebook 07's failure demo, pinned.

    Two thousand episodes of the optimal policy never enter FOULING_CHECKED, so
    every action there is estimated from nothing and the four values come back
    identical. If this ever stops being true the demonstration is gone.
    """
    _, ns = executed["07_model_based_dyna.ipynb"]
    fouling = 2  # State.FOULING_CHECKED
    row = ns["Q_hat"][fouling]
    assert max(row) - min(row) < 1e-9, (
        f"expected no information about FOULING_CHECKED, got {row}"
    )
    # ...while the true action values there span a wide range.
    true_row = ns["Q_STAR"][fouling]
    assert max(true_row) - min(true_row) > 10.0


def test_gathering_evidence_beats_acting_blind(executed):
    """The lesson the whole series is built to demonstrate."""
    _, ns = executed["02_bellman_value_iteration.ipynb"]
    v_star = ns["v_star"]
    assert v_star[0] > 0 > ns["v_reckless"][0], (
        "the careful policy should be worth more than acting blind"
    )
    assert v_star[3] > v_star[1] > v_star[0], (
        "more evidence in hand should never be worth less"
    )
