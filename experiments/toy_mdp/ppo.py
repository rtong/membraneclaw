"""PPO-clip: the probability ratio, and what clipping actually does to its gradient.

REINFORCE takes one gradient step per episode and throws the episode away. PPO
instead collects a batch with the current policy, then takes several gradient
steps on that *same* batch -- which means by the second step the policy being
updated is no longer the policy that generated the data. The probability ratio

    r(s, a) = pi_theta(a|s) / pi_theta_old(a|s)

is the importance-sampling correction for that mismatch, and clipping it is what
stops those extra steps from running away on stale data.

The surrogate objective is L = min(r * A, clip(r, 1-eps, 1+eps) * A). Its gradient
is not the gradient of either term everywhere -- it is the gradient of whichever
term the min actually selects, and that term is 0 whenever clipping is what
selected it:

    dL/dr = A                        if r*A <= clip(r)*A   (unclipped term wins)
          = A  if r is not saturated  (clipped term wins, but clip(r) == r there)
          = 0  if r is saturated      (clipped term wins because it flattened out)

Combined with dr/dtheta[s,:] = r * (e_a - pi(.|s)), that is the entire update --
no autodiff, same hand-derived style as reinforce.py.

Run `python3 ppo.py` for a normal run, or `python3 ppo.py --compare-clipping` to
see what the same aggressive schedule (many epochs, large alpha) does with and
without the clip.
"""
from __future__ import annotations

import argparse

import numpy as np

from reinforce import NON_TERMINAL, generate_episode, softmax
from tiny_mdp import N_ACTIONS, N_STATES, Action, State


STUCK_EPISODE_RETURN = -1000.0
"""Sentinel logged for an episode that never terminated (see collect_batch)."""


def collect_batch(
    theta_old: np.ndarray,
    n_episodes: int,
    rng: np.random.Generator,
    gamma: float = 1.0,
    start: State = State.NO_INFO,
    max_steps: int = 200,
) -> tuple[dict[str, np.ndarray], list[float], int]:
    """Roll out `n_episodes` under theta_old and flatten every step into arrays.

    A large enough alpha can push the softmax into re-checking forever -- an
    improper policy in the value_iteration.policy_value sense. Such an episode
    contributes no samples to the batch (there is no well-defined return to
    train on) but is logged via a sentinel return and a stuck count, since a
    training run that silently drops these would hide exactly the failure mode
    this file exists to demonstrate.

    Returns (batch, episode_returns, n_stuck).
    """
    probs_old = softmax(theta_old)
    states, actions, returns, old_probs, discounts = [], [], [], [], []
    episode_returns = []
    n_stuck = 0

    for _ in range(n_episodes):
        try:
            episode = generate_episode(theta_old, rng, start=start, max_steps=max_steps)
        except RuntimeError:
            n_stuck += 1
            episode_returns.append(STUCK_EPISODE_RETURN)
            continue

        G = 0.0
        step_returns = [0.0] * len(episode)
        for t in reversed(range(len(episode))):
            G = episode[t][2] + gamma * G
            step_returns[t] = G
        episode_returns.append(step_returns[0] if episode else 0.0)

        for t, (state, action, _) in enumerate(episode):
            states.append(state)
            actions.append(action)
            returns.append(step_returns[t])
            old_probs.append(probs_old[state, action])
            discounts.append(gamma ** t)

    batch = {
        "state": np.array(states, dtype=int),
        "action": np.array(actions, dtype=int),
        "return": np.array(returns, dtype=float),
        "old_prob": np.array(old_probs, dtype=float),
        "discount": np.array(discounts, dtype=float),
    }
    return batch, episode_returns, n_stuck


def clip_fraction(theta: np.ndarray, batch: dict[str, np.ndarray], eps: float) -> float:
    """Share of batch samples whose ratio currently falls outside [1-eps, 1+eps]."""
    if batch["state"].size == 0:
        return float("nan")
    probs = softmax(theta)
    ratio = probs[batch["state"], batch["action"]] / batch["old_prob"]
    return float(np.mean((ratio < 1 - eps) | (ratio > 1 + eps)))


def ppo_epoch(
    theta: np.ndarray,
    batch: dict[str, np.ndarray],
    baseline: np.ndarray,
    alpha: float,
    eps: float,
) -> np.ndarray:
    """One gradient-ascent step of the clipped surrogate objective on `batch`.

    `eps=None` disables clipping entirely (plain importance-weighted REINFORCE
    on stale data), which is the "what clipping prevents" side of the comparison.
    """
    probs = softmax(theta)
    pi_a = probs[batch["state"], batch["action"]]
    ratio = pi_a / batch["old_prob"]
    advantage = batch["return"] - baseline[batch["state"]]

    if eps is None:
        dL_dr = advantage
    else:
        clipped_ratio = np.clip(ratio, 1 - eps, 1 + eps)
        unclipped_term = ratio * advantage
        clipped_term = clipped_ratio * advantage
        use_unclipped = unclipped_term <= clipped_term
        saturated = clipped_ratio != ratio
        dL_dr = np.where(use_unclipped, advantage, np.where(saturated, 0.0, advantage))

    # d(ratio)/d theta[s, :] = ratio * (e_a - pi(.|s)); accumulate per state row.
    e_a_minus_pi = -probs[batch["state"]]
    e_a_minus_pi[np.arange(len(batch["state"])), batch["action"]] += 1.0
    contrib = (batch["discount"] * dL_dr * ratio)[:, None] * e_a_minus_pi

    grad = np.zeros_like(theta)
    np.add.at(grad, batch["state"], contrib)
    return theta + alpha * grad


def ppo(
    n_iterations: int,
    episodes_per_iter: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
    eps: float | None = 0.2,
    n_epochs: int = 4,
    gamma: float = 1.0,
    baseline: bool = False,
    start: State = State.NO_INFO,
) -> tuple[np.ndarray, list[float], list[float], list[int]]:
    """Train a softmax policy by PPO-clip.

    Returns (theta, per-iteration mean return, per-iteration clip fraction
    measured after that iteration's epochs, per-iteration stuck-episode count).
    """
    theta = np.zeros((N_STATES, N_ACTIONS))
    baseline_sum = np.zeros(N_STATES)
    baseline_count = np.zeros(N_STATES, dtype=int)
    history: list[float] = []
    clip_fractions: list[float] = []
    stuck_counts: list[int] = []

    for _ in range(n_iterations):
        theta_old = theta.copy()
        batch, episode_returns, n_stuck = collect_batch(
            theta_old, episodes_per_iter, rng, gamma, start
        )
        history.append(float(np.mean(episode_returns)))
        stuck_counts.append(n_stuck)

        baseline_vec = np.zeros(N_STATES)
        if baseline:
            has_data = baseline_count > 0
            baseline_vec[has_data] = baseline_sum[has_data] / baseline_count[has_data]
            for s, g in zip(batch["state"], batch["return"]):
                baseline_sum[s] += g
                baseline_count[s] += 1

        for _epoch in range(n_epochs):
            theta = ppo_epoch(theta, batch, baseline_vec, alpha / episodes_per_iter, eps)

        # Measured *after* every epoch has run on this batch, against the same
        # old_prob the ratio was defined against -- this is how far the clip
        # actually let the policy drift on stale data before the next batch.
        clip_fractions.append(
            clip_fraction(theta, batch, eps) if eps is not None else 0.0
        )

    return theta, history, clip_fractions, stuck_counts


def greedy_from(theta: np.ndarray) -> np.ndarray:
    return theta.argmax(axis=1)


def _safe_value(probs: np.ndarray, gamma: float, stochastic_policy_value) -> float:
    """`stochastic_policy_value`, reporting -inf instead of raising.

    A collapsed policy that has drifted almost, but not quite, to a one-hot
    re-check loop leaves the linear system ill-conditioned rather than exactly
    singular, so this can also legitimately return a huge-magnitude finite
    number -- that is not a bug, it is what "nearly never terminates" looks
    like in closed form.
    """
    try:
        return float(stochastic_policy_value(probs, gamma)[State.NO_INFO])
    except ValueError:
        return float("-inf")


def main(argv: list[str] | None = None) -> None:
    from value_iteration import (
        action_values,
        greedy_policy,
        stochastic_policy_value,
        value_iteration,
    )

    parser = argparse.ArgumentParser(
        description="Learn the toy MDP's policy by PPO-clip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--episodes-per-iter", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument(
        "--compare-clipping", action="store_true",
        help="run the same aggressive schedule with and without the clip",
    )
    args = parser.parse_args(argv)

    V_star, _ = value_iteration(args.gamma)
    optimal = greedy_policy(V_star, args.gamma)
    Q_star = action_values(V_star, args.gamma)

    if args.compare_clipping:
        print(f"same schedule, {args.iterations} iterations x {args.epochs} epochs, "
              f"alpha={args.alpha}, {args.episodes_per_iter} episodes/iter:\n")
        for label, eps in (("clipped, eps=0.2", 0.2), ("unclipped", None)):
            theta, history, _, stuck = ppo(
                args.iterations, args.episodes_per_iter, np.random.default_rng(args.seed),
                alpha=args.alpha, eps=eps, n_epochs=args.epochs, gamma=args.gamma,
                baseline=args.baseline,
            )
            probs = softmax(theta)
            value = _safe_value(probs, args.gamma, stochastic_policy_value)
            best = float(np.mean(history[-10:]))
            worst = float(np.min(history))
            total_stuck = sum(stuck)
            print(f"{label:<18}  final V(NO_INFO)={value:7.3f}  "
                  f"last-10-iter mean return={best:7.3f}  worst iter mean={worst:7.3f}  "
                  f"stuck episodes={total_stuck}")
        return

    theta, history, clip_fractions, stuck_counts = ppo(
        args.iterations, args.episodes_per_iter, np.random.default_rng(args.seed),
        alpha=args.alpha, eps=args.eps, n_epochs=args.epochs, gamma=args.gamma,
        baseline=args.baseline,
    )
    learned = greedy_from(theta)
    probs = softmax(theta)

    print(f"PPO-clip, {args.iterations} iterations x {args.epochs} epochs, "
          f"alpha={args.alpha}, eps={args.eps}\n")

    print("learned policy:")
    print(f"  {'state':<18} {'action':<17} {'pi(a|s)':>8}   {'Q* gap':>7}")
    for s in NON_TERMINAL:
        gap = V_star[s] - Q_star[s, learned[s]]
        mark = " " if np.isclose(gap, 0.0) else "x"
        print(f"{mark} {State(s).name:<18} {Action(learned[s]).name:<17} "
              f"{probs[s, learned[s]]:>8.3f}   {gap:>7.3f}")

    value = _safe_value(probs, args.gamma, stochastic_policy_value)
    best = V_star[State.NO_INFO]
    print(f"\nvalue of the softmax policy as it stands   {value:>7.3f}")
    print(f"value of the optimal policy                {best:>7.3f}")

    window = max(1, args.iterations // 10)
    print(f"\nmean return per {window}-iteration block, and clip fraction "
          "measured after that block's epochs:")
    for i in range(0, args.iterations, window):
        block = history[i:i + window]
        cf = np.nanmean(clip_fractions[i:i + window])
        stuck = sum(stuck_counts[i:i + window])
        stuck_note = f"   stuck={stuck}" if stuck else ""
        print(f"  iters {i:>5}-{i + window:<5} return={np.mean(block):>7.3f}   "
              f"clipped={cf:>5.1%}{stuck_note}")


if __name__ == "__main__":
    main()
