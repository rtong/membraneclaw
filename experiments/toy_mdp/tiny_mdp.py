"""A 6-state membrane-diagnosis MDP, small enough to solve exactly.

The scenario: RO permeate quality is off spec. The agent should gather evidence
(feed salinity, fouling indicators) before proposing a fix, and it can either
validate the fix in a simulator or submit it directly. Gathering evidence costs a
little; submitting an unsupported fix costs a lot.

Every probability and reward here is a teaching parameter chosen by hand. None of
it comes from WaterTAP or from real RO data, and this model is not a plant
simulator — it exists to make Bellman backups, value iteration, and policy
gradients concrete on a state space you can print in full.
"""
from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import numpy as np


class State(IntEnum):
    NO_INFO = 0           # nothing checked yet
    SALINITY_CHECKED = 1  # feed salinity known
    FOULING_CHECKED = 2   # fouling indicators known
    BOTH_CHECKED = 3      # both kinds of evidence in hand
    SUCCESS = 4           # problem solved (terminal)
    FAILURE = 5           # wrong or unsafe fix submitted (terminal)


class Action(IntEnum):
    CHECK_SALINITY = 0
    CHECK_FOULING = 1
    RUN_SIMULATION = 2
    SUBMIT_DIRECTLY = 3


N_STATES = len(State)
N_ACTIONS = len(Action)

TERMINAL_STATES = frozenset({State.SUCCESS, State.FAILURE})

# --- rewards ---------------------------------------------------------------
COST_CHECK = -0.5          # first look at a piece of evidence
COST_REPEAT_CHECK = -1.0   # re-checking something already known: pure waste
REWARD_SIM_SUCCESS = 9.0   # +10 outcome, minus the -1 implicit cost of simulating
REWARD_SIM_FAILURE = -11.0
REWARD_SUBMIT_SUCCESS = 10.0
REWARD_SUBMIT_FAILURE = -10.0

# --- success probabilities, keyed by how much evidence the agent holds ------
SIM_SUCCESS_PROB: dict[State, float] = {
    State.NO_INFO: 0.15,
    State.SALINITY_CHECKED: 0.55,
    State.FOULING_CHECKED: 0.45,
    State.BOTH_CHECKED: 0.95,
}
SUBMIT_SUCCESS_PROB: dict[State, float] = {
    State.NO_INFO: 0.05,
    State.SALINITY_CHECKED: 0.35,
    State.FOULING_CHECKED: 0.25,
    State.BOTH_CHECKED: 0.75,
}

# Evidence held in each non-terminal state, used to work out where a check lands.
_EVIDENCE: dict[State, frozenset[Action]] = {
    State.NO_INFO: frozenset(),
    State.SALINITY_CHECKED: frozenset({Action.CHECK_SALINITY}),
    State.FOULING_CHECKED: frozenset({Action.CHECK_FOULING}),
    State.BOTH_CHECKED: frozenset({Action.CHECK_SALINITY, Action.CHECK_FOULING}),
}
_STATE_BY_EVIDENCE = {evidence: state for state, evidence in _EVIDENCE.items()}


class Transition(NamedTuple):
    """One outcome branch of a (state, action) pair."""

    prob: float
    next_state: State
    reward: float


def is_terminal(state: State) -> bool:
    return State(state) in TERMINAL_STATES


def transitions(state: State, action: Action) -> tuple[Transition, ...]:
    """Every outcome of taking `action` in `state`, with probabilities summing to 1.

    Terminal states absorb: any action loops back with zero reward, so episodes
    that have ended contribute nothing further to the return.
    """
    state, action = State(state), Action(action)

    if is_terminal(state):
        return (Transition(1.0, state, 0.0),)

    if action in (Action.CHECK_SALINITY, Action.CHECK_FOULING):
        already_known = action in _EVIDENCE[state]
        next_state = (
            state if already_known
            else _STATE_BY_EVIDENCE[_EVIDENCE[state] | {action}]
        )
        reward = COST_REPEAT_CHECK if already_known else COST_CHECK
        return (Transition(1.0, next_state, reward),)

    if action is Action.RUN_SIMULATION:
        p = SIM_SUCCESS_PROB[state]
        return (
            Transition(p, State.SUCCESS, REWARD_SIM_SUCCESS),
            Transition(1.0 - p, State.FAILURE, REWARD_SIM_FAILURE),
        )

    p = SUBMIT_SUCCESS_PROB[state]
    return (
        Transition(p, State.SUCCESS, REWARD_SUBMIT_SUCCESS),
        Transition(1.0 - p, State.FAILURE, REWARD_SUBMIT_FAILURE),
    )


def transition_tables() -> tuple[np.ndarray, np.ndarray]:
    """Dense tables for exact methods: P[s, a, s'] and expected R[s, a]."""
    P = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    R = np.zeros((N_STATES, N_ACTIONS))
    for s in State:
        for a in Action:
            for prob, next_state, reward in transitions(s, a):
                P[s, a, next_state] += prob
                R[s, a] += prob * reward
    return P, R


def step(
    state: State, action: Action, rng: np.random.Generator
) -> tuple[State, float, bool]:
    """Sample one environment step: (next_state, reward, done).

    This is the sampling interface the model-free methods use — Monte Carlo
    evaluation, REINFORCE, PPO — none of which get to look at `transitions`.
    """
    outcomes = transitions(state, action)
    probs = [t.prob for t in outcomes]
    chosen = outcomes[rng.choice(len(outcomes), p=probs)]
    return chosen.next_state, chosen.reward, is_terminal(chosen.next_state)
