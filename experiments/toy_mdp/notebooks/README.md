# RL from scratch — five notebooks

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

## The notebooks

| # | File | Covers | Colab |
| --- | --- | --- | --- |
| 01 | `01_finite_mdp.ipynb` | finite MDPs; states, actions, transitions, rewards, terminal states; MDP diagram and transition matrices | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bayanasar/membraneclaw/blob/main/experiments/notebooks/01_finite_mdp.ipynb) |
| 02 | `02_bellman_value_iteration.ipynb` | Bellman expectation and optimality equations; value iteration; policy extraction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bayanasar/membraneclaw/blob/main/experiments/notebooks/02_bellman_value_iteration.ipynb) |
| 03 | `03_monte_carlo.ipynb` | Monte Carlo policy evaluation from sampled episodes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bayanasar/membraneclaw/blob/main/experiments/notebooks/03_monte_carlo.ipynb) |
| 04 | `04_reinforce_baseline.ipynb` | REINFORCE; variance reduction with a baseline and a learned critic | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bayanasar/membraneclaw/blob/main/experiments/notebooks/04_reinforce_baseline.ipynb) |
| 05 | `05_ppo_clipping.ipynb` | PPO's probability ratio and clipping behaviour | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bayanasar/membraneclaw/blob/main/experiments/notebooks/05_ppo_clipping.ipynb) |

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

Runtimes on a laptop: 01 and 02 are instant, 03 about 30s, 04 about 40s, 05
about 60s. The slow cells are multi-seed experiments that deliberately re-run
training many times.

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
kept deliberately, along with the diagnosis of why. Notebooks 03 and 04 also
show why single-seed conclusions about stochastic methods are unreliable, then
apply that lesson to their own experiments.

Notebook 02's exact `v*` serves as the answer key for everything after it: 03,
04, and 05 are all model-free and are graded against it.

## Structure of the argument

1. **01** — write the problem down as an MDP.
2. **02** — with the model, solve it exactly: Bellman backups, no sampling.
3. **03** — drop the model; estimate values by averaging sampled returns.
4. **04** — skip values; push on the policy directly, and fight the variance.
5. **05** — reuse each batch safely by constraining how far the policy may move.
