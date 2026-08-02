"""Tests for Monte Carlo evaluation, checked against the exact V_pi."""
from __future__ import annotations

import numpy as np
import pytest

from monte_carlo import NON_TERMINAL, generate_episode, mc_evaluate
from tiny_mdp import N_STATES, Action, State, is_terminal
from value_iteration import greedy_policy, policy_value, value_iteration

N_EPISODES = 20_000


@pytest.fixture
def optimal_policy():
    V, _ = value_iteration()
    return greedy_policy(V)


@pytest.fixture
def exact(optimal_policy):
    return policy_value(optimal_policy)


def test_episode_ends_in_a_terminal_state(optimal_policy):
    rng = np.random.default_rng(0)
    episode = generate_episode(optimal_policy, rng)
    # Only non-terminal states are acted in; the terminal state ends the list.
    assert all(not is_terminal(s) for s, _, _ in episode)
    # Check salinity, check fouling, simulate -- three steps, every time.
    assert [a for _, a, _ in episode] == [
        Action.CHECK_SALINITY,
        Action.CHECK_FOULING,
        Action.RUN_SIMULATION,
    ]


def test_estimate_lands_within_a_few_standard_errors(optimal_policy, exact):
    rng = np.random.default_rng(0)
    V, counts, stderr = mc_evaluate(optimal_policy, N_EPISODES, rng)
    for s in NON_TERMINAL:
        if counts[s] == 0:
            continue
        assert abs(V[s] - exact[s]) < 5 * stderr[s]


def test_unvisited_states_report_no_data_rather_than_zero(optimal_policy):
    # Started from NO_INFO the optimal policy checks salinity first, so it never
    # sets foot in FOULING_CHECKED. NaN says "no samples"; 0.0 would be a lie.
    rng = np.random.default_rng(0)
    V, counts, _ = mc_evaluate(optimal_policy, 200, rng)
    assert counts[State.FOULING_CHECKED] == 0
    assert np.isnan(V[State.FOULING_CHECKED])


def test_exploring_starts_reach_every_state(optimal_policy, exact):
    rng = np.random.default_rng(0)
    V, counts, stderr = mc_evaluate(optimal_policy, N_EPISODES, rng, start=None)
    assert (counts[NON_TERMINAL] > 0).all()
    for s in NON_TERMINAL:
        assert abs(V[s] - exact[s]) < 5 * stderr[s]


def test_fixed_start_visits_every_reached_state_once_per_episode(optimal_policy):
    rng = np.random.default_rng(0)
    _, counts, _ = mc_evaluate(optimal_policy, 500, rng)
    reached = [State.NO_INFO, State.SALINITY_CHECKED, State.BOTH_CHECKED]
    assert [counts[s] for s in reached] == [500, 500, 500]


def test_first_visit_and_every_visit_agree_when_no_state_repeats(optimal_policy):
    # This policy never revisits a state, so the two variants see exactly the
    # same returns. The distinction only matters once a policy loops.
    first, _, _ = mc_evaluate(
        optimal_policy, 300, np.random.default_rng(1), first_visit=True
    )
    every, _, _ = mc_evaluate(
        optimal_policy, 300, np.random.default_rng(1), first_visit=False
    )
    np.testing.assert_array_equal(first, every)


def test_gamma_zero_gives_the_immediate_reward_exactly(optimal_policy):
    # With gamma=0 the return is just the first reward. From NO_INFO that is a
    # deterministic -0.5, so this must come back exact, not merely close.
    rng = np.random.default_rng(0)
    V, _, _ = mc_evaluate(optimal_policy, 50, rng, gamma=0.0)
    assert V[State.NO_INFO] == pytest.approx(-0.5)


def test_discounting_is_applied_to_later_rewards(optimal_policy):
    rng = np.random.default_rng(0)
    V, _, _ = mc_evaluate(optimal_policy, N_EPISODES, rng, gamma=0.5)
    exact_discounted = policy_value(optimal_policy, gamma=0.5)
    assert V[State.NO_INFO] == pytest.approx(exact_discounted[State.NO_INFO], abs=0.1)


def test_same_seed_gives_the_same_estimate(optimal_policy):
    a, _, _ = mc_evaluate(optimal_policy, 200, np.random.default_rng(7))
    b, _, _ = mc_evaluate(optimal_policy, 200, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_error_shrinks_with_more_episodes(optimal_policy, exact):
    errors = []
    for n in (500, 8_000, 64_000):
        rng = np.random.default_rng(3)
        V, _, _ = mc_evaluate(optimal_policy, n, rng)
        errors.append(abs(V[State.NO_INFO] - exact[State.NO_INFO]))
    assert errors[-1] < errors[0]


def test_standard_error_falls_like_one_over_sqrt_n(optimal_policy):
    _, _, small = mc_evaluate(optimal_policy, 2_000, np.random.default_rng(0))
    _, _, large = mc_evaluate(optimal_policy, 32_000, np.random.default_rng(0))
    # 16x the episodes should cut the standard error by roughly 4x.
    assert small[State.NO_INFO] / large[State.NO_INFO] == pytest.approx(4.0, rel=0.15)


@pytest.mark.parametrize("action", [Action.CHECK_SALINITY, Action.CHECK_FOULING])
def test_improper_policy_is_reported_rather_than_truncated(action):
    # Truncating a non-terminating episode would quietly bias every return, so
    # generate_episode refuses instead -- the same stance policy_value takes.
    policy = np.full(N_STATES, action)
    with pytest.raises(RuntimeError, match="did not terminate"):
        generate_episode(policy, np.random.default_rng(0), max_steps=50)
