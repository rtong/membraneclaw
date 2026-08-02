"""Exact solutions for the toy MDP: value iteration, policy extraction, evaluation.

These are the model-based methods — they read the transition tables directly
instead of sampling, so the answers they produce are exact up to the convergence
tolerance. They are the ground truth the sampling methods (Monte Carlo,
REINFORCE, PPO) will later be checked against.

Run `python3 value_iteration.py` to print the optimal values and policy, or
`python3 value_iteration.py --gamma 0.5` to see how impatience changes them.

Discounting defaults to gamma=1.0. The MDP is an episodic shortest-path problem:
every terminal state absorbs with zero reward, so undiscounted returns stay
finite and are directly readable as "expected reward for solving the ticket".
The one catch is that a policy which only ever re-checks known evidence never
terminates; under gamma=1.0 its value is -inf, and `policy_value` says so
explicitly rather than returning a meaningless number.
"""
from __future__ import annotations

import argparse

import numpy as np

from tiny_mdp import (
    N_ACTIONS,
    N_STATES,
    Action,
    State,
    is_terminal,
    transition_tables,
)

NON_TERMINAL = np.array([s for s in State if not is_terminal(s)])


def action_values(V: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """One Bellman backup: Q[s, a] = R[s, a] + gamma * sum_s' P[s, a, s'] V[s']."""
    P, R = transition_tables()
    return R + gamma * (P @ V)


def value_iteration(
    gamma: float = 1.0, tol: float = 1e-12, max_iterations: int = 10_000
) -> tuple[np.ndarray, int]:
    """Iterate the Bellman optimality operator to a fixed point.

    Returns the optimal state values and the number of sweeps taken. Terminal
    states need no special handling: their rows are a zero-reward self-loop, so
    V stays pinned at 0 once it starts there.
    """
    V = np.zeros(N_STATES)
    for sweep in range(1, max_iterations + 1):
        V_next = action_values(V, gamma).max(axis=1)
        delta = np.abs(V_next - V).max()
        V = V_next
        if delta < tol:
            return V, sweep
    raise RuntimeError(f"value iteration did not converge in {max_iterations} sweeps")


def greedy_policy(V: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Extract a deterministic policy that acts greedily with respect to V.

    Ties are broken by lowest action index, so this returns one optimal policy,
    not the only one -- see the tie at NO_INFO between the two checks.
    """
    return action_values(V, gamma).argmax(axis=1)


def policy_value(policy: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Exact value of a deterministic policy, by solving the linear system.

    V_pi = R_pi + gamma * P_pi V_pi, restricted to non-terminal states, gives
    (I - gamma P_pi) V_pi = R_pi -- a direct solve rather than an iteration.
    """
    P, R = transition_tables()
    rows = NON_TERMINAL
    P_pi = P[rows, policy[rows]][:, rows]
    R_pi = R[rows, policy[rows]]

    A = np.eye(len(rows)) - gamma * P_pi
    if np.linalg.matrix_rank(A) < len(rows):
        raise ValueError(
            "policy never reaches a terminal state, so its undiscounted value is "
            "-inf; pass gamma < 1 to evaluate it anyway"
        )

    V = np.zeros(N_STATES)
    V[rows] = np.linalg.solve(A, R_pi)
    return V


def stochastic_policy_value(probs: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Exact value of a stochastic policy given as action probabilities pi(a|s).

    Same linear solve as `policy_value`, but the per-state transition and reward
    are averaged over the policy's action distribution first. This is what a
    softmax policy is actually worth -- taking its argmax and evaluating that
    instead flatters a policy that has not yet committed.
    """
    P, R = transition_tables()
    rows = NON_TERMINAL
    P_pi = np.einsum("sa,sat->st", probs, P)[rows][:, rows]
    R_pi = (probs * R).sum(axis=1)[rows]

    A = np.eye(len(rows)) - gamma * P_pi
    if np.linalg.matrix_rank(A) < len(rows):
        raise ValueError(
            "policy never reaches a terminal state, so its undiscounted value is "
            "-inf; pass gamma < 1 to evaluate it anyway"
        )

    V = np.zeros(N_STATES)
    V[rows] = np.linalg.solve(A, R_pi)
    return V


def _format_table(values: np.ndarray, row_labels, col_labels) -> str:
    width = max(len(c) for c in col_labels) + 3
    lines = [f"{'':<18}" + "".join(f"{c:>{width}}" for c in col_labels)]
    for label, row in zip(row_labels, values):
        lines.append(f"{label:<18}" + "".join(f"{v:>{width}.2f}" for v in row))
    return "\n".join(lines)


def _gamma(text: str) -> float:
    """argparse type: a discount factor must lie in [0, 1].

    Above 1.0 the Bellman operator stops being a contraction and value iteration
    runs away, so reject it here rather than after 10,000 fruitless sweeps.
    """
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in [0, 1], got {value}")
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Solve the toy membrane-diagnosis MDP exactly.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gamma",
        type=_gamma,
        default=1.0,
        help="discount factor; 1.0 leaves returns undiscounted, lower values "
             "make the agent impatient about payoffs that are several steps away",
    )
    gamma = parser.parse_args(argv).gamma

    V, sweeps = value_iteration(gamma)
    Q = action_values(V, gamma)
    policy = greedy_policy(V, gamma)

    print(f"value iteration converged in {sweeps} sweeps (gamma={gamma})\n")

    print("optimal action values Q*[s, a]:")
    print(_format_table(Q, [s.name for s in State], [a.name for a in Action]))

    print("\noptimal values and policy:")
    for s in State:
        if is_terminal(s):
            print(f"  {s.name:<18} V*={V[s]:>6.2f}   (terminal)")
            continue
        ties = [Action(a).name for a in range(N_ACTIONS) if np.isclose(Q[s, a], V[s])]
        note = f"  (tied with {', '.join(t for t in ties[1:])})" if len(ties) > 1 else ""
        print(f"  {s.name:<18} V*={V[s]:>6.2f}   {Action(policy[s]).name}{note}")

    print("\nvalue from NO_INFO under fixed policies:")
    baselines = {
        "submit blind immediately": Action.SUBMIT_DIRECTLY,
        "simulate immediately": Action.RUN_SIMULATION,
    }
    for name, action in baselines.items():
        flat = np.full(N_STATES, action)
        print(f"  {name:<28} {policy_value(flat, gamma)[State.NO_INFO]:>6.2f}")

    check_then_sim = np.full(N_STATES, Action.RUN_SIMULATION)
    check_then_sim[State.NO_INFO] = Action.CHECK_SALINITY
    print(f"  {'check salinity, simulate':<28} "
          f"{policy_value(check_then_sim, gamma)[State.NO_INFO]:>6.2f}")
    # Deliberately unlabelled: which policy is optimal depends on gamma, so
    # naming it here would go stale the moment someone passes --gamma.
    print(f"  {'optimal (see above)':<28} "
          f"{policy_value(policy, gamma)[State.NO_INFO]:>6.2f}")


if __name__ == "__main__":
    main()
