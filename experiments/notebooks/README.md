# RL from scratch — seven notebooks

A progressive, self-contained introduction to reinforcement learning built on a
single small problem. Each notebook adds one layer, and each runs on its own.

## Requirements

`numpy` and a Jupyter kernel. That is all.

```
pip install numpy jupyterlab
jupyter lab
```

`matplotlib` is optional. Notebook 01 draws a proper heatmap if it is installed
and falls back to a text rendering if it is not; nothing else needs it.

The committed notebooks have their outputs saved, so they read as written on
GitHub without running anything.

## The notebooks

| # | File | Covers | Colab |
| --- | --- | --- | --- |
| 01 | `01_finite_mdp.ipynb` | finite MDPs; states, actions, transitions, rewards, terminal states; MDP diagram and transition matrices | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bayanasar/membraneclaw/blob/main/experiments/notebooks/01_finite_mdp.ipynb) |
| 02 | `02_bellman_value_iteration.ipynb` | Bellman expectation and optimality equations; value iteration; policy extraction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bayanasar/membraneclaw/blob/main/experiments/notebooks/02_bellman_value_iteration.ipynb) |
| 03 | `03_monte_carlo.ipynb` | Monte Carlo policy evaluation from sampled episodes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bayanasar/membraneclaw/blob/main/experiments/notebooks/03_monte_carlo.ipynb) |
| 04 | `04_reinforce_baseline.ipynb` | REINFORCE; variance reduction with a baseline and a learned critic | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bayanasar/membraneclaw/blob/main/experiments/notebooks/04_reinforce_baseline.ipynb) |
| 05 | `05_ppo_clipping.ipynb` | PPO's probability ratio and clipping behaviour | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bayanasar/membraneclaw/blob/main/experiments/notebooks/05_ppo_clipping.ipynb) |
| 06 | `06_td_sarsa_qlearning.ipynb` | bootstrapping; TD(0) against Monte Carlo; SARSA and Q-learning, and the different fixed points they converge to | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bayanasar/membraneclaw/blob/main/experiments/notebooks/06_td_sarsa_qlearning.ipynb) |
| 07 | `07_model_based_dyna.ipynb` | learning $P$ and $R$ from data; certainty equivalence; Dyna-Q; what a plan does where it has no data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bayanasar/membraneclaw/blob/main/experiments/notebooks/07_model_based_dyna.ipynb) |

Read them in order the first time. Each one re-defines the environment in its
first code cell, so they can also be opened individually later.

### A note on the Colab links

The badges point at `main`. **This repository is private**, so Colab cannot open
them anonymously — a click gives a 404 until you connect Colab to GitHub once,
via *File → Open notebook → GitHub* and the "Include private repos" checkbox.
After that the badges work normally for anyone with repo access. If the
repository is ever made public, they work for everyone with no setup.

Colab supplies `numpy` and `matplotlib` out of the box, so nothing needs
installing there — including the optional heatmap in notebook 01, which will
render rather than fall back to text.

Runtimes: 01 and 02 are instant, 03 about 6s, 04 about 9s, 05 about 13s, 06 about
11s, 07 about 9s. The slow cells are multi-seed experiments that deliberately
re-run training many times.

## Checking they still run

`test_notebooks.py` executes every notebook end to end and fails on any cell that
errors. It also compares the MDP across notebooks: each one redefines the problem
in its own cells so it can stand alone in Colab, which means one copy per
notebook, all free to drift apart silently. The test pins them together, along
with `GAMMA` and the exact `v*` that 03 onwards all grade themselves against, and
pins the headline claim of the two newest notebooks — that SARSA and Q-learning
converge to different fixed points, and that a plan built from expert-only data
knows nothing about the states that data never entered.

```
pip install matplotlib nbformat nbclient ipykernel pytest
python3 -m pytest
```

The repository `.venv` already has all of these:

```
cd experiments/notebooks && ../../.venv/bin/python -m pytest
```

It runs every notebook once, so the suite takes about as long as reading them all
does. Regenerate the committed outputs the same way they were made:

```
../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace [0-9][0-9]_*.ipynb
```

If you change the MDP on purpose, `EXPECTED_V_STAR` in `test_notebooks.py` is
the single place the new answer key has to be written down.

## The example problem

Reverse-osmosis permeate is off spec. An agent can check feed salinity and
fouling indicators before proposing a fix, and can either validate the fix in a
simulator or submit it blind. Evidence is cheap; unsupported fixes are
expensive.

Six states, four actions, two of them terminal — small enough to print in full
and solve exactly, which is the point. **Every probability and reward is a
hand-picked teaching parameter.** Nothing here comes from plant data, and this is
not a process simulator.

## Method

Each notebook alternates markdown and code: the prose states what is about to
happen and why, the code does it, and the following prose interprets the actual
output.

Where a claim is checkable, it is checked. Several sections test a prediction and
report that it failed — the discount factor never changes the optimal policy in
this MDP, and the baseline's variance reduction does not produce a statistically
significant learning speedup on a problem this small. Those negative results are
kept deliberately, along with the diagnosis of why. Notebook 06 adds two more:
TD(0) is *beaten* by Monte Carlo for the first hundred episodes rather than
beating it as advertised, and SARSA and Q-learning produce the same policy here
despite converging to demonstrably different values. Notebook 07 ends on a plan
built from two thousand episodes of expert data that scores perfectly by the
usual metric and is worth $-5.0$ against an optimal $7.1$ one state off the
expert's path. Notebooks 03, 04 and 06 also show why single-seed conclusions
about stochastic methods are unreliable, then apply that lesson to their own
experiments.

Notebook 02's exact `v*` serves as the answer key for everything after it: 03
through 06 are all model-free and are graded against it, and 07 grades a policy
planned from a *learned* model against it too. Notebook 06 adds a second exact
target, `q_eps`, because SARSA provably does *not* converge to `v*` and grading it
against `v*` alone would misrepresent what it is doing.

## Structure of the argument

1. **01** — write the problem down as an MDP.
2. **02** — with the model, solve it exactly: Bellman backups, no sampling.
3. **03** — drop the model; estimate values by averaging sampled returns.
4. **04** — skip values; push on the policy directly, and fight the variance.
5. **05** — reuse each batch safely by constraining how far the policy may move.
6. **06** — stop waiting for the episode to end; bootstrap, and learn action
   values so control needs no model at all.
7. **07** — stop throwing the transitions away; estimate the model itself, plan
   on the estimate, and find out where that quietly fails.
