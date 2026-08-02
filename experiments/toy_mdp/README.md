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

## Layout

* `tiny_mdp.py` — the MDP: `transitions()` for exact methods, `step()` for sampling.
* `test_tiny_mdp.py` — tests pinning down the dynamics, rewards, and tables.

Run the tests from this directory:

```
python -m pytest
```
