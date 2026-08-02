"""REINFORCE: learn a policy directly by following the gradient of return.

Monte Carlo evaluation answered "how good is this policy?". REINFORCE answers
"which policy?" without ever computing a value function: it samples episodes,
then pushes the log-probability of each action up or down in proportion to the
return that followed it.

The policy is tabular softmax over logits theta[s, a]. At this size the score
function is worth writing by hand rather than handing to autograd, because it is
the entire idea in one line:

    grad log pi(a | s) = e_a - pi(. | s)

so the update nudges the taken action's logit up by (1 - pi(a|s)) and every other
action's down by pi(a'|s), all scaled by the return. Nothing here needs torch.

Run `python3 reinforce.py` to watch a uniform policy turn into the optimal one,
scored against the exact V* that value_iteration computes in closed form.
"""
from __future__ import annotations

import argparse

import numpy as np

from tiny_mdp import N_ACTIONS, N_STATES, Action, State, is_terminal, step

NON_TERMINAL = np.array([s for s in State if not is_terminal(s)])


def softmax(theta: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shifted by the row max for numerical stability."""
    shifted = theta - theta.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def generate_episode(
    theta: np.ndarray,
    rng: np.random.Generator,
    start: State = State.NO_INFO,
    max_steps: int = 1000,
) -> list[tuple[State, Action, float]]:
    """Roll out one episode by sampling from the softmax policy."""
    probs = softmax(theta)
    episode: list[tuple[State, Action, float]] = []
    state = State(start)

    for _ in range(max_steps):
        if is_terminal(state):
            return episode
        action = Action(rng.choice(N_ACTIONS, p=probs[state]))
        next_state, reward, _ = step(state, action, rng)
        episode.append((state, action, reward))
        state = next_state

    raise RuntimeError(f"episode did not terminate in {max_steps} steps")


def reinforce(
    n_episodes: int,
    rng: np.random.Generator,
    alpha: float = 0.01,
    gamma: float = 1.0,
    baseline: bool = False,
    start: State = State.NO_INFO,
) -> tuple[np.ndarray, list[float]]:
    """Train a softmax policy by REINFORCE. Returns (theta, per-episode returns).

    With `baseline=True` the return is centred by a running per-state mean before
    it scales the gradient. Any baseline that does not depend on the action leaves
    the gradient unbiased, and this one measurably cuts its variance -- but by
    only 10-15% here (see `gradient_samples`), because most of the variance in
    this MDP comes from how the terminal coin lands, not from which state the
    agent is in. A state-value baseline cannot cancel that.

    It is also not a free win at a fixed `alpha`: centring shrinks the returns
    that scale the update, so the logits sharpen more slowly than they do when a
    uniformly positive return keeps pushing them apart.
    """
    theta = np.zeros((N_STATES, N_ACTIONS))
    probs = softmax(theta)
    baseline_sum = np.zeros(N_STATES)
    baseline_count = np.zeros(N_STATES, dtype=int)
    history: list[float] = []

    for _ in range(n_episodes):
        episode = generate_episode(theta, rng, start=start)

        returns = np.empty(len(episode))
        G = 0.0
        for t in reversed(range(len(episode))):
            G = episode[t][2] + gamma * G
            returns[t] = G
        history.append(returns[0] if len(episode) else 0.0)

        for t, (state, action, _) in enumerate(episode):
            advantage = returns[t]
            if baseline:
                if baseline_count[state] > 0:
                    advantage -= baseline_sum[state] / baseline_count[state]
                baseline_sum[state] += returns[t]
                baseline_count[state] += 1

            # grad log pi(a|s) = e_a - pi(.|s), scaled by gamma^t and the return.
            grad = -probs[state].copy()
            grad[action] += 1.0
            theta[state] += alpha * (gamma ** t) * advantage * grad

        probs = softmax(theta)

    return theta, history


def gradient_samples(
    theta: np.ndarray,
    n_episodes: int,
    rng: np.random.Generator,
    gamma: float = 1.0,
    baseline: np.ndarray | None = None,
    start: State = State.NO_INFO,
) -> np.ndarray:
    """Per-episode gradient estimates at a fixed theta, shape (n, S, A).

    Holding theta still is what makes the baseline's effect measurable: the
    estimates all target the same true gradient, so their spread is pure
    estimator variance rather than a mixture of learning and noise.
    """
    probs = softmax(theta)
    b = np.zeros(N_STATES) if baseline is None else baseline
    samples = np.zeros((n_episodes, N_STATES, N_ACTIONS))

    for i in range(n_episodes):
        episode = generate_episode(theta, rng, start=start)
        G = 0.0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + gamma * G
            grad = -probs[state].copy()
            grad[action] += 1.0
            samples[i, state] += (gamma ** t) * (G - b[state]) * grad

    return samples


def greedy_from(theta: np.ndarray) -> np.ndarray:
    """The deterministic policy the softmax logits currently favour."""
    return theta.argmax(axis=1)


def main(argv: list[str] | None = None) -> None:
    from value_iteration import (
        action_values,
        greedy_policy,
        policy_value,
        stochastic_policy_value,
        value_iteration,
    )

    parser = argparse.ArgumentParser(
        description="Learn the toy MDP's policy by REINFORCE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline", action="store_true",
                        help="centre returns by a running per-state mean")
    args = parser.parse_args(argv)

    V_star, _ = value_iteration(args.gamma)
    optimal = greedy_policy(V_star, args.gamma)

    theta, history = reinforce(
        args.episodes,
        np.random.default_rng(args.seed),
        alpha=args.alpha,
        gamma=args.gamma,
        baseline=args.baseline,
    )
    learned = greedy_from(theta)
    probs = softmax(theta)

    tag = "with baseline" if args.baseline else "plain"
    print(f"REINFORCE ({tag}), {args.episodes} episodes, alpha={args.alpha}\n")

    # Compare on optimality, not on identity with greedy_policy's tie-break:
    # at NO_INFO both checks are optimal, and picking the other one is not a miss.
    Q_star = action_values(V_star, args.gamma)

    print("learned policy:")
    print(f"  {'state':<18} {'action':<17} {'pi(a|s)':>8}   {'Q* gap':>7}")
    for s in NON_TERMINAL:
        gap = V_star[s] - Q_star[s, learned[s]]
        mark = " " if np.isclose(gap, 0.0) else "x"
        print(f"{mark} {State(s).name:<18} {Action(learned[s]).name:<17} "
              f"{probs[s, learned[s]]:>8.3f}   {gap:>7.3f}")

    actual = stochastic_policy_value(probs, args.gamma)[State.NO_INFO]
    greedy = policy_value(learned, args.gamma)[State.NO_INFO]
    best = policy_value(optimal, args.gamma)[State.NO_INFO]
    print(f"\nvalue of the softmax policy as it stands   {actual:>7.3f}")
    print(f"value of its greedy extraction             {greedy:>7.3f}")
    print(f"value of the optimal policy                {best:>7.3f}")

    window = max(1, args.episodes // 20)
    print(f"\nmean return per {window}-episode block:")
    blocks = np.array(history[: len(history) // window * window]).reshape(-1, window)
    for i, mean in enumerate(blocks.mean(axis=1)):
        if i % 4 == 0 or i == len(blocks) - 1:
            print(f"  episodes {i * window:>6}-{(i + 1) * window:<6} {mean:>7.3f}")


if __name__ == "__main__":
    main()
