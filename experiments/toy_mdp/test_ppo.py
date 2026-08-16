"""Tests for PPO-clip: the ratio's gradient, the clip's effect, and convergence.

Reproduced with real runs, not asserted from theory: an unclipped multi-epoch
update on stale batches genuinely diverges on this MDP (see
`test_clipping_prevents_the_collapse_unclipped_training_suffers`), and the
clipped version genuinely does not (`test_clipped_training_reaches_near_optimal`).
"""
from __future__ import annotations

import numpy as np
import pytest

from ppo import (
    STUCK_EPISODE_RETURN,
    clip_fraction,
    collect_batch,
    greedy_from,
    ppo,
    ppo_epoch,
)
from reinforce import NON_TERMINAL, softmax
from tiny_mdp import N_ACTIONS, N_STATES, Action, State
from value_iteration import action_values, stochastic_policy_value, value_iteration


@pytest.fixture(scope="module")
def V_star_and_optimal():
    V_star, _ = value_iteration()
    return V_star, action_values(V_star)


def test_ratio_is_one_when_theta_has_not_moved():
    theta = np.random.default_rng(0).normal(size=(N_STATES, N_ACTIONS))
    batch, _, n_stuck = collect_batch(theta, 20, np.random.default_rng(1))
    assert n_stuck == 0
    probs = softmax(theta)
    ratio = probs[batch["state"], batch["action"]] / batch["old_prob"]
    np.testing.assert_allclose(ratio, 1.0)
    assert clip_fraction(theta, batch, eps=0.2) == 0.0


def test_ppo_epoch_moves_probability_toward_a_positive_advantage_action():
    # A single state, single sample, clearly positive advantage: the taken
    # action's probability must rise.
    theta = np.zeros((N_STATES, N_ACTIONS))
    batch = {
        "state": np.array([State.BOTH_CHECKED]),
        "action": np.array([Action.RUN_SIMULATION]),
        "return": np.array([9.0]),
        "old_prob": np.array([0.25]),
        "discount": np.array([1.0]),
    }
    baseline = np.zeros(N_STATES)
    new_theta = ppo_epoch(theta, batch, baseline, alpha=0.1, eps=0.2)
    before = softmax(theta)[State.BOTH_CHECKED, Action.RUN_SIMULATION]
    after = softmax(new_theta)[State.BOTH_CHECKED, Action.RUN_SIMULATION]
    assert after > before


def test_clip_zeroes_the_gradient_once_the_ratio_saturates():
    # Ratio already at 3x old_prob with eps=0.2 (cap at 1.2x): the clipped
    # term is picked and the ratio is saturated, so the update must vanish.
    theta = np.zeros((N_STATES, N_ACTIONS))
    theta[State.BOTH_CHECKED, Action.RUN_SIMULATION] = 5.0  # pi(a|s) far above old_prob
    batch = {
        "state": np.array([State.BOTH_CHECKED]),
        "action": np.array([Action.RUN_SIMULATION]),
        "return": np.array([9.0]),  # positive advantage: clip caps the upside
        "old_prob": np.array([softmax(theta)[State.BOTH_CHECKED, Action.RUN_SIMULATION] / 3]),
        "discount": np.array([1.0]),
    }
    baseline = np.zeros(N_STATES)
    new_theta = ppo_epoch(theta, batch, baseline, alpha=0.1, eps=0.2)
    np.testing.assert_allclose(new_theta, theta)


def test_clip_does_not_block_a_good_action_still_below_the_window():
    # Positive advantage, but the current probability sits below old_prob by
    # enough that ratio=0.5 is already under 1-eps=0.8. The clip only stops
    # *further* movement once the ratio has crossed into the window from
    # below -- it must not block getting there in the first place, so the
    # unclipped term should still be the one selected and the gradient nonzero.
    theta = np.zeros((N_STATES, N_ACTIONS))  # pi(a|s) = 0.25 for every action
    batch = {
        "state": np.array([State.BOTH_CHECKED]),
        "action": np.array([Action.RUN_SIMULATION]),
        "return": np.array([9.0]),
        "old_prob": np.array([0.5]),  # ratio = 0.25 / 0.5 = 0.5, below 1-eps
        "discount": np.array([1.0]),
    }
    baseline = np.zeros(N_STATES)
    new_theta = ppo_epoch(theta, batch, baseline, alpha=0.1, eps=0.2)
    assert not np.allclose(new_theta, theta)
    before = softmax(theta)[State.BOTH_CHECKED, Action.RUN_SIMULATION]
    after = softmax(new_theta)[State.BOTH_CHECKED, Action.RUN_SIMULATION]
    assert after > before


def test_collect_batch_reports_stuck_episodes_without_crashing():
    # An improper policy that only ever re-checks: episodes never terminate.
    theta = np.zeros((N_STATES, N_ACTIONS))
    theta[:, Action.CHECK_SALINITY] = 50.0  # near-deterministic re-check loop
    batch, episode_returns, n_stuck = collect_batch(
        theta, 5, np.random.default_rng(0), max_steps=50
    )
    assert n_stuck == 5
    assert episode_returns == [STUCK_EPISODE_RETURN] * 5
    assert batch["state"].size == 0


def test_empty_batch_leaves_theta_unchanged():
    theta = np.random.default_rng(0).normal(size=(N_STATES, N_ACTIONS))
    batch = {k: np.array([]) for k in ("state", "action", "return", "old_prob", "discount")}
    batch["state"] = batch["state"].astype(int)
    batch["action"] = batch["action"].astype(int)
    new_theta = ppo_epoch(theta, batch, np.zeros(N_STATES), alpha=0.5, eps=0.2)
    np.testing.assert_array_equal(new_theta, theta)


def test_clipped_training_reaches_near_optimal(V_star_and_optimal):
    V_star, Q_star = V_star_and_optimal
    theta, history, _, stuck = ppo(80, 16, np.random.default_rng(0), alpha=0.5, eps=0.2, n_epochs=4)
    assert sum(stuck) == 0
    value = stochastic_policy_value(softmax(theta))[State.NO_INFO]
    assert value == pytest.approx(V_star[State.NO_INFO], abs=0.5)


def test_clipping_prevents_the_collapse_unclipped_training_suffers(V_star_and_optimal):
    # Same aggressive schedule (multiple epochs on a stale batch), only the
    # clip differs. This is measured, not asserted from the PPO paper: 20
    # seeds put clipped at 6.84 +/- 0.65 and unclipped at 0.30 +/- 5.88 with
    # over a thousand stuck episodes (see README). Five seeds is enough to
    # make the direction of that gap a reliable regression check.
    V_star, _ = V_star_and_optimal
    clipped_values, unclipped_values, unclipped_stuck = [], [], 0

    for seed in range(5):
        theta, _, _, stuck = ppo(80, 16, np.random.default_rng(seed), alpha=0.5, eps=0.2, n_epochs=4)
        clipped_values.append(stochastic_policy_value(softmax(theta))[State.NO_INFO])

        theta, _, _, stuck = ppo(80, 16, np.random.default_rng(seed), alpha=0.5, eps=None, n_epochs=4)
        unclipped_stuck += sum(stuck)
        try:
            v = stochastic_policy_value(softmax(theta))[State.NO_INFO]
        except ValueError:
            v = float("-inf")
        unclipped_values.append(v)

    assert np.mean(clipped_values) > np.mean(unclipped_values) + 2.0
    assert unclipped_stuck > 0


def test_greedy_from_matches_theta_argmax():
    theta = np.random.default_rng(0).normal(size=(N_STATES, N_ACTIONS))
    np.testing.assert_array_equal(greedy_from(theta), theta.argmax(axis=1))


def test_training_is_reproducible():
    a, hist_a, _, _ = ppo(20, 8, np.random.default_rng(3), alpha=0.5, eps=0.2, n_epochs=4)
    b, hist_b, _, _ = ppo(20, 8, np.random.default_rng(3), alpha=0.5, eps=0.2, n_epochs=4)
    np.testing.assert_array_equal(a, b)
    assert hist_a == hist_b
