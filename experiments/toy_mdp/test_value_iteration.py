"""Tests for the exact solvers: convergence, optimality, and policy extraction."""
from __future__ import annotations

import numpy as np
import pytest

from tiny_mdp import N_STATES, Action, State, is_terminal, transition_tables
from value_iteration import (
    NON_TERMINAL,
    _gamma,
    action_values,
    greedy_policy,
    main,
    policy_value,
    value_iteration,
)

# Worked out by hand from the reward and probability tables; see README.
EXPECTED_V_STAR = {
    State.NO_INFO: 7.0,
    State.SALINITY_CHECKED: 7.5,
    State.FOULING_CHECKED: 7.5,
    State.BOTH_CHECKED: 8.0,
    State.SUCCESS: 0.0,
    State.FAILURE: 0.0,
}


@pytest.fixture
def solved():
    V, _ = value_iteration()
    return V


def test_value_iteration_matches_the_hand_computed_optimum(solved):
    for state, expected in EXPECTED_V_STAR.items():
        assert solved[state] == pytest.approx(expected)


def test_terminal_states_are_worth_nothing(solved):
    assert solved[State.SUCCESS] == 0.0
    assert solved[State.FAILURE] == 0.0


def test_v_star_satisfies_the_bellman_optimality_equation(solved):
    # The defining property: V* is a fixed point of the optimality operator.
    np.testing.assert_allclose(action_values(solved).max(axis=1), solved, atol=1e-9)


def test_optimal_policy_gathers_evidence_then_validates(solved):
    policy = greedy_policy(solved)
    assert policy[State.NO_INFO] in (Action.CHECK_SALINITY, Action.CHECK_FOULING)
    assert policy[State.SALINITY_CHECKED] == Action.CHECK_FOULING
    assert policy[State.FOULING_CHECKED] == Action.CHECK_SALINITY
    assert policy[State.BOTH_CHECKED] == Action.RUN_SIMULATION


def test_submitting_blind_is_never_optimal(solved):
    policy = greedy_policy(solved)
    for state in NON_TERMINAL:
        assert policy[state] != Action.SUBMIT_DIRECTLY


def test_both_checks_are_tied_at_the_start(solved):
    # Order does not matter when no evidence is held yet -- so the optimal policy
    # is not unique, and argmax's tie-break picks one of two equally good actions.
    Q = action_values(solved)
    assert Q[State.NO_INFO, Action.CHECK_SALINITY] == pytest.approx(
        Q[State.NO_INFO, Action.CHECK_FOULING]
    )


def test_greedy_policy_achieves_the_optimal_value(solved):
    # Evaluating the extracted policy must reproduce V* exactly, which is the
    # consistency check tying value iteration to policy extraction.
    np.testing.assert_allclose(policy_value(greedy_policy(solved)), solved, atol=1e-9)


@pytest.mark.parametrize("action", list(Action))
def test_no_fixed_policy_beats_the_optimum(solved, action):
    flat = np.full(N_STATES, action)
    if action in (Action.CHECK_SALINITY, Action.CHECK_FOULING):
        pytest.skip("never terminates; covered by test_improper_policy_is_rejected")
    assert (policy_value(flat) <= solved + 1e-9).all()


@pytest.mark.parametrize(
    ("action", "expected"),
    [(Action.SUBMIT_DIRECTLY, -9.0), (Action.RUN_SIMULATION, -8.0)],
)
def test_acting_without_evidence_is_worse_than_gathering_it(action, expected):
    value = policy_value(np.full(N_STATES, action))[State.NO_INFO]
    assert value == pytest.approx(expected)
    assert value < EXPECTED_V_STAR[State.NO_INFO]


@pytest.mark.parametrize("action", [Action.CHECK_SALINITY, Action.CHECK_FOULING])
def test_improper_policy_is_rejected_rather_than_silently_wrong(action):
    # A policy that only ever re-checks loops forever at -1.0 a step: its
    # undiscounted value is -inf, and the linear system is singular.
    with pytest.raises(ValueError, match="never reaches a terminal state"):
        policy_value(np.full(N_STATES, action))


@pytest.mark.parametrize("action", [Action.CHECK_SALINITY, Action.CHECK_FOULING])
def test_discounting_makes_the_looping_policy_evaluable(action):
    # With gamma < 1 the same policy has a finite, very negative value: the
    # geometric sum of -1.0 per step, i.e. -1/(1-gamma).
    value = policy_value(np.full(N_STATES, action), gamma=0.9)
    assert value[State.BOTH_CHECKED] == pytest.approx(-10.0)


def test_discounting_only_bites_where_reward_is_delayed(solved):
    V_discounted, _ = value_iteration(gamma=0.9)

    # From BOTH_CHECKED the optimal action terminates immediately, so its whole
    # value is immediate reward and discounting cannot touch it.
    assert V_discounted[State.BOTH_CHECKED] == pytest.approx(8.0)

    # From further back the payoff is two checks away, so discounting shrinks it.
    assert V_discounted[State.NO_INFO] < solved[State.NO_INFO]
    assert V_discounted[State.SALINITY_CHECKED] < solved[State.SALINITY_CHECKED]


def test_value_iteration_converges_quickly_on_this_mdp(solved):
    # Four sweeps: one per layer of the evidence lattice, plus one to detect the
    # fixed point. Worth pinning -- it is why exact methods are cheap here.
    _, sweeps = value_iteration()
    assert sweeps == 4


def test_action_values_agree_with_the_transition_tables(solved):
    P, R = transition_tables()
    Q = action_values(solved)
    for s in State:
        for a in Action:
            expected = R[s, a] + sum(P[s, a, s2] * solved[s2] for s2 in State)
            assert Q[s, a] == pytest.approx(expected)


def test_non_terminal_index_is_correct():
    assert [State(s) for s in NON_TERMINAL] == [s for s in State if not is_terminal(s)]


@pytest.mark.parametrize("text", ["0", "0.5", "1", "1.0"])
def test_gamma_argument_accepts_the_valid_range(text):
    assert 0.0 <= _gamma(text) <= 1.0


@pytest.mark.parametrize("text", ["-0.1", "1.5", "2"])
def test_gamma_argument_rejects_values_outside_the_range(text):
    # Above 1.0 the Bellman operator is not a contraction and value iteration
    # never converges, so this must fail at the boundary rather than time out.
    import argparse

    with pytest.raises(argparse.ArgumentTypeError, match=r"must be in \[0, 1\]"):
        _gamma(text)


def test_cli_defaults_to_undiscounted(capsys):
    main([])
    assert "gamma=1.0" in capsys.readouterr().out


def test_cli_honours_the_gamma_flag(capsys):
    main(["--gamma", "0.5"])
    out = capsys.readouterr().out
    assert "gamma=0.5" in out
    # V*(NO_INFO) drops from 7.00 to 1.25 once payoffs two steps out are halved.
    assert "V*=  1.25" in out


def test_myopic_agent_stops_gathering_evidence():
    # At gamma=0 only immediate reward counts, so from SALINITY_CHECKED the
    # agent simulates (0.00 now) rather than pay -0.5 for the second check.
    V, _ = value_iteration(gamma=0.0)
    policy = greedy_policy(V, gamma=0.0)
    assert policy[State.SALINITY_CHECKED] == Action.RUN_SIMULATION
