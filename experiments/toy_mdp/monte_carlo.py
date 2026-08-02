"""Monte Carlo policy evaluation: estimate V_pi by averaging sampled returns.

This is the first model-free method in the project. It never reads the transition
tables -- it only rolls out episodes through `tiny_mdp.step` and averages what
comes back. The environment obviously still knows its own dynamics; the point is
that the estimator does not, so nothing here would change if `step` were a real
plant instead of a lookup table.

Run `python3 monte_carlo.py` to watch the estimate converge on the exact V_pi
that `value_iteration.policy_value` computes in closed form.

Two properties of this MDP are worth keeping in mind while reading the output:

* A deterministic policy visits only a thin slice of the state space. The optimal
  policy started from NO_INFO never reaches FOULING_CHECKED at all, so MC returns
  NaN there -- an honest "no data", not a zero.
* No sensible policy revisits a state, so first-visit and every-visit MC give
  bit-identical answers here. The distinction only bites once a policy loops.
"""
from __future__ import annotations

import argparse

import numpy as np

from tiny_mdp import N_STATES, Action, State, is_terminal, step

NON_TERMINAL = np.array([s for s in State if not is_terminal(s)])


def generate_episode(
    policy: np.ndarray,
    rng: np.random.Generator,
    start: State = State.NO_INFO,
    max_steps: int = 1000,
) -> list[tuple[State, Action, float]]:
    """Roll out one episode as a list of (state, action, reward) triples."""
    episode: list[tuple[State, Action, float]] = []
    state = State(start)

    for _ in range(max_steps):
        if is_terminal(state):
            return episode
        action = Action(policy[state])
        next_state, reward, _ = step(state, action, rng)
        episode.append((state, action, reward))
        state = next_state

    raise RuntimeError(
        f"episode did not terminate in {max_steps} steps; the policy is probably "
        "improper (it loops on re-checks forever). Raise max_steps if the policy "
        "really is proper but slow."
    )


def mc_evaluate(
    policy: np.ndarray,
    n_episodes: int,
    rng: np.random.Generator,
    gamma: float = 1.0,
    first_visit: bool = True,
    start: State | None = State.NO_INFO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate V_pi by averaging returns. Returns (V, counts, standard errors).

    `start=None` uses exploring starts -- a uniformly random non-terminal state
    each episode -- which is how you get coverage of states the policy would
    never reach on its own. States with no samples come back as NaN.
    """
    total = np.zeros(N_STATES)
    total_sq = np.zeros(N_STATES)
    counts = np.zeros(N_STATES, dtype=int)

    for _ in range(n_episodes):
        first_state = State(rng.choice(NON_TERMINAL)) if start is None else start
        episode = generate_episode(policy, rng, start=first_state)

        first_index: dict[State, int] = {}
        for t, (state, _, _) in enumerate(episode):
            first_index.setdefault(state, t)

        # Accumulate returns backwards: G_t = r_t + gamma * G_{t+1}.
        G = 0.0
        for t in reversed(range(len(episode))):
            state, _, reward = episode[t]
            G = reward + gamma * G
            if first_visit and first_index[state] != t:
                continue
            total[state] += G
            total_sq[state] += G * G
            counts[state] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        V = np.where(counts > 0, total / counts, np.nan)
        variance = np.where(counts > 1, total_sq / counts - V * V, np.nan)
        stderr = np.sqrt(np.maximum(variance, 0.0) / np.maximum(counts, 1))
        stderr = np.where(counts > 1, stderr, np.nan)

    return V, counts, stderr


def main(argv: list[str] | None = None) -> None:
    from value_iteration import greedy_policy, policy_value, value_iteration

    parser = argparse.ArgumentParser(
        description="Monte Carlo policy evaluation on the toy MDP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exploring-starts",
        action="store_true",
        help="start each episode in a random non-terminal state, so that states "
             "the policy would never reach on its own still get sampled",
    )
    args = parser.parse_args(argv)

    V_star, _ = value_iteration(args.gamma)
    policy = greedy_policy(V_star, args.gamma)
    exact = policy_value(policy, args.gamma)
    start = None if args.exploring_starts else State.NO_INFO

    print(f"policy under evaluation (optimal at gamma={args.gamma}):")
    for s in NON_TERMINAL:
        print(f"  {State(s).name:<18} {Action(policy[s]).name}")

    print(f"\nconvergence towards the exact V_pi ({args.episodes} episodes max):")
    print(f"  {'episodes':>9}  {'V(NO_INFO)':>11}  {'abs error':>10}  {'std err':>9}")

    ladder = []
    n = 500
    while n < args.episodes:
        ladder.append(n)
        n *= 4
    ladder.append(args.episodes)

    for n in ladder:
        rng = np.random.default_rng(args.seed)
        V, _, stderr = mc_evaluate(policy, n, rng, args.gamma, start=start)
        estimate = V[State.NO_INFO]
        error = abs(estimate - exact[State.NO_INFO])
        print(f"  {n:>9}  {estimate:>11.4f}  {error:>10.4f}  {stderr[State.NO_INFO]:>9.4f}")

    rng = np.random.default_rng(args.seed)
    V, counts, stderr = mc_evaluate(policy, args.episodes, rng, args.gamma, start=start)

    print(f"\nper-state estimates after {args.episodes} episodes:")
    print(f"  {'state':<18} {'MC':>9} {'exact':>9} {'error':>8} {'visits':>8}")
    for s in NON_TERMINAL:
        state = State(s)
        if counts[s] == 0:
            print(f"  {state.name:<18} {'--':>9} {exact[s]:>9.3f} {'--':>8} {0:>8}")
            continue
        error = abs(V[s] - exact[s])
        print(f"  {state.name:<18} {V[s]:>9.3f} {exact[s]:>9.3f} "
              f"{error:>8.3f} {counts[s]:>8}")

    unvisited = [State(s).name for s in NON_TERMINAL if counts[s] == 0]
    if unvisited:
        print(f"\nnever visited: {', '.join(unvisited)}")
        print("  the policy simply does not go there; --exploring-starts fixes it")


if __name__ == "__main__":
    main()
