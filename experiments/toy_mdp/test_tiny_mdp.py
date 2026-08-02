"""Tests pinning down the MDP definition itself: probabilities, rewards, dynamics."""
from __future__ import annotations

import numpy as np
import pytest

from tiny_mdp import (
    COST_CHECK,
    COST_REPEAT_CHECK,
    N_ACTIONS,
    N_STATES,
    REWARD_SIM_FAILURE,
    REWARD_SIM_SUCCESS,
    REWARD_SUBMIT_FAILURE,
    REWARD_SUBMIT_SUCCESS,
    SIM_SUCCESS_PROB,
    SUBMIT_SUCCESS_PROB,
    Action,
    State,
    is_terminal,
    step,
    transition_tables,
    transitions,
)

NON_TERMINAL = [
    State.NO_INFO,
    State.SALINITY_CHECKED,
    State.FOULING_CHECKED,
    State.BOTH_CHECKED,
]


@pytest.mark.parametrize("state", list(State))
@pytest.mark.parametrize("action", list(Action))
def test_probabilities_form_a_distribution(state, action):
    probs = [t.prob for t in transitions(state, action)]
    assert all(p >= 0.0 for p in probs)
    assert sum(probs) == pytest.approx(1.0)


@pytest.mark.parametrize("state", [State.SUCCESS, State.FAILURE])
@pytest.mark.parametrize("action", list(Action))
def test_terminal_states_absorb_with_no_reward(state, action):
    assert transitions(state, action) == ((1.0, state, 0.0),)
    assert is_terminal(state)


@pytest.mark.parametrize(
    ("state", "action", "expected"),
    [
        (State.NO_INFO, Action.CHECK_SALINITY, State.SALINITY_CHECKED),
        (State.NO_INFO, Action.CHECK_FOULING, State.FOULING_CHECKED),
        (State.SALINITY_CHECKED, Action.CHECK_FOULING, State.BOTH_CHECKED),
        (State.FOULING_CHECKED, Action.CHECK_SALINITY, State.BOTH_CHECKED),
    ],
)
def test_checks_accumulate_evidence_at_the_standard_cost(state, action, expected):
    assert transitions(state, action) == ((1.0, expected, COST_CHECK),)


@pytest.mark.parametrize(
    ("state", "action"),
    [
        (State.SALINITY_CHECKED, Action.CHECK_SALINITY),
        (State.FOULING_CHECKED, Action.CHECK_FOULING),
        (State.BOTH_CHECKED, Action.CHECK_SALINITY),
        (State.BOTH_CHECKED, Action.CHECK_FOULING),
    ],
)
def test_repeat_checks_stay_put_and_cost_more(state, action):
    assert transitions(state, action) == ((1.0, state, COST_REPEAT_CHECK),)


@pytest.mark.parametrize("state", NON_TERMINAL)
def test_simulation_outcomes(state):
    p = SIM_SUCCESS_PROB[state]
    assert transitions(state, Action.RUN_SIMULATION) == (
        (p, State.SUCCESS, REWARD_SIM_SUCCESS),
        (1.0 - p, State.FAILURE, REWARD_SIM_FAILURE),
    )


@pytest.mark.parametrize("state", NON_TERMINAL)
def test_direct_submission_outcomes(state):
    p = SUBMIT_SUCCESS_PROB[state]
    assert transitions(state, Action.SUBMIT_DIRECTLY) == (
        (p, State.SUCCESS, REWARD_SUBMIT_SUCCESS),
        (1.0 - p, State.FAILURE, REWARD_SUBMIT_FAILURE),
    )


def test_simulation_reward_carries_the_implicit_cost_of_simulating():
    # Simulating and succeeding is worth one unit less than submitting and
    # succeeding; the gap is the cost of running the simulator.
    assert REWARD_SUBMIT_SUCCESS - REWARD_SIM_SUCCESS == pytest.approx(1.0)
    assert REWARD_SIM_FAILURE - REWARD_SUBMIT_FAILURE == pytest.approx(-1.0)


@pytest.mark.parametrize("state", NON_TERMINAL)
def test_validating_beats_submitting_blind_from_every_state(state):
    assert SIM_SUCCESS_PROB[state] > SUBMIT_SUCCESS_PROB[state]


@pytest.mark.parametrize("probs", [SIM_SUCCESS_PROB, SUBMIT_SUCCESS_PROB])
def test_more_evidence_never_hurts(probs):
    # NO_INFO < each single check < BOTH_CHECKED, for both kinds of proposal.
    for partial in (State.SALINITY_CHECKED, State.FOULING_CHECKED):
        assert probs[State.NO_INFO] < probs[partial] < probs[State.BOTH_CHECKED]


def test_transition_tables_match_the_sparse_definition():
    P, R = transition_tables()
    assert P.shape == (N_STATES, N_ACTIONS, N_STATES)
    assert R.shape == (N_STATES, N_ACTIONS)
    np.testing.assert_allclose(P.sum(axis=2), 1.0)

    for s in State:
        for a in Action:
            expected_r = sum(t.prob * t.reward for t in transitions(s, a))
            assert R[s, a] == pytest.approx(expected_r)
            for t in transitions(s, a):
                assert P[s, a, t.next_state] == pytest.approx(t.prob)


def test_step_is_deterministic_for_checks_and_flags_termination():
    rng = np.random.default_rng(0)
    next_state, reward, done = step(State.NO_INFO, Action.CHECK_SALINITY, rng)
    assert (next_state, reward, done) == (State.SALINITY_CHECKED, COST_CHECK, False)


def test_step_samples_terminal_outcomes_at_the_stated_rate():
    rng = np.random.default_rng(0)
    n = 20_000
    successes = 0
    for _ in range(n):
        next_state, reward, done = step(State.BOTH_CHECKED, Action.RUN_SIMULATION, rng)
        assert done
        assert next_state in (State.SUCCESS, State.FAILURE)
        assert reward == (
            REWARD_SIM_SUCCESS if next_state is State.SUCCESS else REWARD_SIM_FAILURE
        )
        successes += next_state is State.SUCCESS

    p = SIM_SUCCESS_PROB[State.BOTH_CHECKED]
    stderr = (p * (1 - p) / n) ** 0.5
    assert abs(successes / n - p) < 5 * stderr
