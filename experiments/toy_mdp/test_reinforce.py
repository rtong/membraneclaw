"""Tests for REINFORCE: the policy it learns and the gradient estimator itself."""
from __future__ import annotations

import numpy as np
import pytest

from reinforce import (
    NON_TERMINAL,
    generate_episode,
    gradient_samples,
    greedy_from,
    reinforce,
    softmax,
)
from tiny_mdp import N_ACTIONS, N_STATES, Action, State, is_terminal
from value_iteration import (
    action_values,
    stochastic_policy_value,
    value_iteration,
)


@pytest.fixture(scope="module")
def trained():
    theta, history = reinforce(20_000, np.random.default_rng(0))
    return theta, history


def test_softmax_rows_are_distributions():
    theta = np.random.default_rng(0).normal(size=(N_STATES, N_ACTIONS)) * 5
    probs = softmax(theta)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert (probs > 0).all()


def test_zero_logits_give_a_uniform_policy():
    np.testing.assert_allclose(softmax(np.zeros((N_STATES, N_ACTIONS))), 0.25)


def test_softmax_survives_extreme_logits():
    # Without the max-shift this overflows to inf/inf = nan.
    probs = softmax(np.array([[1000.0, 999.0, -1000.0, 0.0]]))
    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1.0)


def test_episodes_only_contain_states_that_can_be_acted_in():
    rng = np.random.default_rng(0)
    episode = generate_episode(np.zeros((N_STATES, N_ACTIONS)), rng)
    assert all(not is_terminal(s) for s, _, _ in episode)


def test_reinforce_learns_an_optimal_action_everywhere_it_goes(trained):
    theta, _ = trained
    learned = greedy_from(theta)
    V_star, _ = value_iteration()
    Q_star = action_values(V_star)
    # Compare on optimality, not on identity with a particular argmax: at
    # NO_INFO both checks are optimal, so either one is a correct answer.
    for s in NON_TERMINAL:
        assert Q_star[s, learned[s]] == pytest.approx(V_star[s])


def test_learned_policy_is_worth_nearly_the_optimum(trained):
    theta, _ = trained
    V_star, _ = value_iteration()
    value = stochastic_policy_value(softmax(theta))[State.NO_INFO]
    assert value == pytest.approx(V_star[State.NO_INFO], abs=0.05)


def test_learning_actually_improves_the_return(trained):
    _, history = trained
    early = np.mean(history[:1000])
    late = np.mean(history[-1000:])
    assert late > early + 1.0


def test_training_is_reproducible():
    a, _ = reinforce(300, np.random.default_rng(5))
    b, _ = reinforce(300, np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)


def test_terminal_rows_are_never_updated(trained):
    theta, _ = trained
    # Terminal states are never acted in, so their logits must stay untouched.
    np.testing.assert_array_equal(theta[State.SUCCESS], np.zeros(N_ACTIONS))
    np.testing.assert_array_equal(theta[State.FAILURE], np.zeros(N_ACTIONS))


@pytest.fixture(scope="module")
def gradients():
    theta = np.zeros((N_STATES, N_ACTIONS))
    V_pi = stochastic_policy_value(softmax(theta))
    plain = gradient_samples(theta, 4_000, np.random.default_rng(1))
    centred = gradient_samples(theta, 4_000, np.random.default_rng(1), baseline=V_pi)
    return plain, centred


def test_baseline_reduces_gradient_variance(gradients):
    plain, centred = gradients
    assert centred.var(axis=0).sum() < plain.var(axis=0).sum()


def test_baseline_leaves_the_gradient_unbiased(gradients):
    plain, centred = gradients
    # Both estimate the same true gradient, so their means must agree to within
    # the sampling error of the difference.
    n = plain.shape[0]
    difference = plain - centred
    stderr = difference.std(axis=0) / np.sqrt(n)
    off_by = np.abs(difference.mean(axis=0))
    assert (off_by <= 4 * stderr + 1e-9).all()


def test_gradient_variance_shrinks_as_the_policy_commits():
    # A near-deterministic policy takes the same actions every episode, so the
    # only noise left is the terminal coin flip.
    rng = np.random.default_rng(0)
    uniform = gradient_samples(np.zeros((N_STATES, N_ACTIONS)), 2_000, rng)
    trained_theta, _ = reinforce(2_000, np.random.default_rng(0))
    committed = gradient_samples(trained_theta, 2_000, np.random.default_rng(2))
    assert committed.var(axis=0).sum() < uniform.var(axis=0).sum()


def test_baseline_run_also_reaches_an_optimal_policy():
    theta, _ = reinforce(20_000, np.random.default_rng(0), baseline=True)
    learned = greedy_from(theta)
    V_star, _ = value_iteration()
    Q_star = action_values(V_star)
    for s in NON_TERMINAL:
        assert Q_star[s, learned[s]] == pytest.approx(V_star[s])
