# toy_mdp

## Purpose

This toy project is designed to build practical understanding of:

* finite Markov Decision Processes;
* state, action, transition, reward, and terminal-state definitions;
* Bellman expectation and optimality equations;
* value iteration;
* policy extraction;
* Monte Carlo policy evaluation;
* REINFORCE;
* variance reduction with a baseline or critic;
* basic PPO probability-ratio and clipping behavior.

The toy MDP is not intended to simulate a real reverse-osmosis plant.

## The MDP

RO permeate quality is off spec. The agent should check feed salinity and fouling
indicators before proposing a fix, and validate that fix in a simulator rather
than submitting it blind.

```mermaid
stateDiagram-v2
    direction LR

    [*] --> NO_INFO

    NO_INFO --> SALINITY_CHECKED: check salinity, -0.5
    NO_INFO --> FOULING_CHECKED: check fouling, -0.5
    SALINITY_CHECKED --> BOTH_CHECKED: check fouling, -0.5
    FOULING_CHECKED --> BOTH_CHECKED: check salinity, -0.5

    SALINITY_CHECKED --> SALINITY_CHECKED: re-check, -1.0
    FOULING_CHECKED --> FOULING_CHECKED: re-check, -1.0
    BOTH_CHECKED --> BOTH_CHECKED: re-check, -1.0

    BOTH_CHECKED --> SUCCESS: simulate, p=0.95, +9
    BOTH_CHECKED --> FAILURE: simulate, p=0.05, -11

    SUCCESS --> [*]
    FAILURE --> [*]
```

`NO_INFO` has no self-loop: from there both checks are new, so there is nothing to
waste a step on yet.

To keep the diagram readable it draws the terminal edges only from `BOTH_CHECKED`.
Every evidence state has the same two exits — `RUN_SIMULATION` and
`SUBMIT_DIRECTLY`, each splitting into `SUCCESS` and `FAILURE` — which is 16 edges
if drawn in full. The success probabilities table below is that omitted half of
the graph.

### States

| State | Meaning |
| --- | --- |
| `NO_INFO` | nothing checked yet |
| `SALINITY_CHECKED` | feed salinity known |
| `FOULING_CHECKED` | fouling indicators known |
| `BOTH_CHECKED` | both kinds of evidence in hand |
| `SUCCESS` | problem solved (terminal) |
| `FAILURE` | wrong or unsafe fix submitted (terminal) |

### Actions

| Action | Meaning |
| --- | --- |
| `CHECK_SALINITY` | check feed salinity |
| `CHECK_FOULING` | check fouling indicators |
| `RUN_SIMULATION` | simulate the proposed fix on the evidence so far |
| `SUBMIT_DIRECTLY` | submit the fix without validating it |

### Rewards

| Event | Reward |
| --- | --- |
| check a new piece of evidence | -0.5 |
| re-check something already known | -1.0 |
| simulation succeeds | +9 |
| simulation fails | -11 |
| direct submission succeeds | +10 |
| direct submission fails | -10 |

Running the simulator carries an implicit cost of -1, so success there is worth
+9 rather than +10.

### Success probabilities

Success depends on how much evidence backs the proposal:

| Evidence held | Simulation succeeds | Direct submission succeeds |
| --- | --- | --- |
| none | 15% | 5% |
| salinity only | 55% | 35% |
| fouling only | 45% | 25% |
| both | 95% | 75% |

These are hand-picked teaching parameters, not WaterTAP output or real RO data.

## Solving it exactly

`value_iteration.py` iterates the Bellman optimality operator to its fixed point,
extracts the greedy policy, and evaluates any deterministic policy by solving
(I - gamma P_pi) V_pi = R_pi directly.

```
python3 value_iteration.py
```

With gamma=1.0 it converges in 4 sweeps — one per layer of the evidence lattice,
plus one to notice it is done:

| State | V* | Optimal action |
| --- | --- | --- |
| `NO_INFO` | 7.0 | either check (tied) |
| `SALINITY_CHECKED` | 7.5 | `CHECK_FOULING` |
| `FOULING_CHECKED` | 7.5 | `CHECK_SALINITY` |
| `BOTH_CHECKED` | 8.0 | `RUN_SIMULATION` |
| `SUCCESS` / `FAILURE` | 0.0 | terminal |

So the optimal policy is *gather both pieces of evidence, then validate* — worth
7.0 from a cold start, against -8.0 for simulating blind and -9.0 for submitting
blind. `SUBMIT_DIRECTLY` is never optimal in any state: validating dominates it
everywhere, because the -1 implicit cost of simulating buys more than 1 unit of
expected outcome at every evidence level.

Two details worth noticing, both pinned by tests:

* The start state has **two** optimal actions — the checks commute, so the
  optimal policy is not unique and `argmax` merely picks one.
* Discounting does not touch V*(`BOTH_CHECKED`), because from there the optimal
  action terminates immediately and its whole value is immediate reward. It only
  bites where the payoff is still several steps away.

## Layout

* `tiny_mdp.py` — the MDP: `transitions()` for exact methods, `step()` for sampling.
* `value_iteration.py` — value iteration, policy extraction, exact policy evaluation.
* `test_tiny_mdp.py`, `test_value_iteration.py` — tests for both.

Run the tests from this directory:

```
python3 -m pytest
```

This experiment installs its own dependencies (see `requirements.txt`) and does
not use the repository's `.venv`, which carries the much heavier WaterTAP stack.
