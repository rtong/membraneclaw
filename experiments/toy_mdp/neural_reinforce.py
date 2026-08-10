"""Neural REINFORCE for the membrane-diagnosis toy MDP.

The tabular implementation in ``reinforce.py`` writes the softmax gradient out
by hand.  This version represents the policy with a tiny PyTorch MLP and lets
autograd perform exactly the same update.  The two state features say whether
salinity and fouling have been checked:

    [salinity_known, fouling_known] -> Linear(2, H) -> tanh -> Linear(H, 4)

Rollouts use only ``tiny_mdp.step``.  The transition table and value iteration
are imported by the CLI after training solely to report an exact score, never to
choose an action or form a training target.

Run the configuration discussed in the accompanying experiment with:

    python3 neural_reinforce.py --episodes 10000 --gamma 0.95 \
        --hidden-size 16 --lr 0.01 --seed 0 --device cpu

Pass ``--baseline`` to train a second MLP as a state-value baseline.  Its output
is detached before forming the policy loss, so it reduces variance without
letting the policy exploit or backpropagate through the baseline.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from tiny_mdp import N_ACTIONS, N_STATES, Action, State, is_terminal, step


NON_TERMINAL = np.array([s for s in State if not is_terminal(s)])

# Terminal rows are never encoded during a rollout.  Keeping rows for them makes
# it convenient to obtain an (N_STATES, N_ACTIONS) policy matrix for the exact
# evaluator; their values do not affect that evaluation because they absorb.
STATE_FEATURES = np.array(
    [
        [0.0, 0.0],  # NO_INFO
        [1.0, 0.0],  # SALINITY_CHECKED
        [0.0, 1.0],  # FOULING_CHECKED
        [1.0, 1.0],  # BOTH_CHECKED
        [0.0, 0.0],  # SUCCESS (terminal, never acted in)
        [0.0, 0.0],  # FAILURE (terminal, never acted in)
    ],
    dtype=np.float32,
)


class PolicyNetwork(nn.Module):
    """A two-layer categorical policy over the four discrete actions."""

    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        output = nn.Linear(hidden_size, N_ACTIONS)
        # Start from an exactly uniform policy, as the tabular implementation
        # does with zero logits.  The hidden layer is still randomly initialised
        # and begins learning as soon as the output layer has moved off zero.
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            output,
        )

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        return self.net(state_features)


class ValueNetwork(nn.Module):
    """Optional learned state-value baseline with the same hidden width."""

    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        return self.net(state_features).squeeze(-1)


@dataclass
class Episode:
    """One sampled trajectory and the differentiable log probabilities in it."""

    states: list[State]
    actions: list[Action]
    rewards: list[float]
    log_probs: list[torch.Tensor]


@dataclass
class TrainingResult:
    """Artifacts returned by :func:`train`."""

    policy: PolicyNetwork
    returns: list[float]
    value_network: ValueNetwork | None = None


def _network_device(network: nn.Module) -> torch.device:
    return next(network.parameters()).device


def encode_state(state: State, device: torch.device | str = "cpu") -> torch.Tensor:
    """Encode an MDP state as ``[salinity_known, fouling_known]``."""
    return torch.as_tensor(STATE_FEATURES[int(State(state))], device=device)


def generate_episode(
    policy: PolicyNetwork,
    rng: np.random.Generator,
    start: State = State.NO_INFO,
    max_steps: int = 1000,
) -> Episode:
    """Sample an episode from ``policy`` while retaining action log-prob graphs."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    states: list[State] = []
    actions: list[Action] = []
    rewards: list[float] = []
    log_probs: list[torch.Tensor] = []
    state = State(start)
    device = _network_device(policy)

    for _ in range(max_steps):
        if is_terminal(state):
            return Episode(states, actions, rewards, log_probs)

        logits = policy(encode_state(state, device))
        distribution = Categorical(logits=logits)
        action_tensor = distribution.sample()
        action = Action(action_tensor.item())
        next_state, reward, _ = step(state, action, rng)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(distribution.log_prob(action_tensor))
        state = next_state

    raise RuntimeError(f"episode did not terminate in {max_steps} steps")


def discounted_returns(
    rewards: list[float], gamma: float, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Compute every reward-to-go, ``G_t = r_t + gamma * G_(t+1)``."""
    returns = np.empty(len(rewards), dtype=np.float32)
    reward_to_go = 0.0
    for t in reversed(range(len(rewards))):
        reward_to_go = rewards[t] + gamma * reward_to_go
        returns[t] = reward_to_go
    return torch.as_tensor(returns, device=device)


def _resolve_device(name: str | torch.device) -> torch.device:
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def train(
    n_episodes: int = 10_000,
    *,
    gamma: float = 0.95,
    hidden_size: int = 16,
    lr: float = 0.01,
    seed: int = 0,
    device: str | torch.device = "cpu",
    baseline: bool = False,
    max_steps: int = 1000,
) -> TrainingResult:
    """Train a neural policy with one Monte Carlo REINFORCE update per episode.

    The policy objective is the loss form of the discounted policy-gradient
    estimator::

        loss = -sum_t gamma**t * (G_t - b(s_t)) * log pi(a_t | s_t)

    ``b`` is zero by default.  With ``baseline=True`` it is a learned value MLP
    fitted to the same Monte Carlo returns using mean-squared error.
    """
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if lr <= 0.0:
        raise ValueError("lr must be positive")

    resolved_device = _resolve_device(device)
    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    policy = PolicyNetwork(hidden_size).to(resolved_device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    value_network = ValueNetwork(hidden_size).to(resolved_device) if baseline else None
    value_optimizer = (
        torch.optim.Adam(value_network.parameters(), lr=lr)
        if value_network is not None
        else None
    )

    history: list[float] = []
    for _ in range(n_episodes):
        episode = generate_episode(policy, rng, max_steps=max_steps)
        returns = discounted_returns(episode.rewards, gamma, resolved_device)
        history.append(float(returns[0].item()) if len(returns) else 0.0)

        log_probs = torch.stack(episode.log_probs)
        discounts = gamma ** torch.arange(len(episode.rewards), device=resolved_device)

        if value_network is None:
            advantages = returns
        else:
            features = torch.stack(
                [encode_state(state, resolved_device) for state in episode.states]
            )
            values = value_network(features)
            advantages = returns - values.detach()

            assert value_optimizer is not None
            value_loss = nn.functional.mse_loss(values, returns)
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

        policy_loss = -(discounts * advantages * log_probs).sum()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

    return TrainingResult(policy, history, value_network)


def policy_probabilities(policy: PolicyNetwork) -> np.ndarray:
    """Return ``pi(a|s)`` for every state as a NumPy matrix."""
    device = _network_device(policy)
    features = torch.as_tensor(STATE_FEATURES, device=device)
    with torch.no_grad():
        probs = torch.softmax(policy(features), dim=-1)
    return probs.cpu().numpy()


def greedy_from(policy: PolicyNetwork) -> np.ndarray:
    """Extract the deterministic action favoured by the neural policy."""
    return policy_probabilities(policy).argmax(axis=1)


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def _gamma(text: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in [0, 1], got {value}")
    return value


def main(argv: list[str] | None = None) -> None:
    from value_iteration import (
        action_values,
        greedy_policy,
        policy_value,
        stochastic_policy_value,
        value_iteration,
    )

    parser = argparse.ArgumentParser(
        description="Learn the toy MDP's policy with a small PyTorch MLP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=_positive_int, default=10_000)
    parser.add_argument("--gamma", type=_gamma, default=0.95)
    parser.add_argument("--hidden-size", type=_positive_int, default=16)
    parser.add_argument("--lr", type=_positive_float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="fit a learned state-value MLP and subtract it from returns",
    )
    parser.add_argument(
        "--report-every",
        type=_positive_int,
        default=500,
        help="number of episodes in each reported mean-return block",
    )
    args = parser.parse_args(argv)

    result = train(
        args.episodes,
        gamma=args.gamma,
        hidden_size=args.hidden_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        baseline=args.baseline,
    )

    probabilities = policy_probabilities(result.policy)
    learned = probabilities.argmax(axis=1)
    V_star, _ = value_iteration(args.gamma)
    Q_star = action_values(V_star, args.gamma)
    optimal = greedy_policy(V_star, args.gamma)

    tag = "with learned baseline" if args.baseline else "plain"
    print(
        f"Neural REINFORCE ({tag}), {args.episodes} episodes, "
        f"gamma={args.gamma}, lr={args.lr}, device={args.device}"
    )
    print(f"network: 2 -> {args.hidden_size} -> {N_ACTIONS}\n")

    action_names = [action.name for action in Action]
    print("final action probabilities:")
    print(f"  {'state':<18} " + " ".join(f"{name:>17}" for name in action_names))
    for s in NON_TERMINAL:
        values = " ".join(f"{probabilities[s, a]:>17.3f}" for a in range(N_ACTIONS))
        print(f"  {State(s).name:<18} {values}")

    print("\nlearned greedy policy:")
    print(f"  {'state':<18} {'action':<17} {'pi(a|s)':>8}   {'Q* gap':>7}")
    for s in NON_TERMINAL:
        gap = V_star[s] - Q_star[s, learned[s]]
        mark = " " if np.isclose(gap, 0.0) else "x"
        print(
            f"{mark} {State(s).name:<18} {Action(learned[s]).name:<17} "
            f"{probabilities[s, learned[s]]:>8.3f}   {gap:>7.3f}"
        )

    softmax_value = stochastic_policy_value(probabilities, args.gamma)[State.NO_INFO]
    greedy_value = policy_value(learned, args.gamma)[State.NO_INFO]
    optimal_value = policy_value(optimal, args.gamma)[State.NO_INFO]
    print(f"\nvalue of the neural softmax policy          {softmax_value:>7.3f}")
    print(f"value of its greedy extraction             {greedy_value:>7.3f}")
    print(f"value of the optimal policy                {optimal_value:>7.3f}")

    print(f"\nmean return per {args.report_every}-episode block:")
    for start in range(0, len(result.returns), args.report_every):
        stop = min(start + args.report_every, len(result.returns))
        mean = np.mean(result.returns[start:stop])
        print(f"  episodes {start:>6}-{stop:<6} {mean:>7.3f}")


if __name__ == "__main__":
    main()
