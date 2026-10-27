# rlaif — PPO with Gemini 4 Argon as the judge

**Status: planned.** No code, runs or results yet. The design below is a proposal still under discussion.

On tasks without a verifier, such as the real-scenario questions the MembraneClaw benchmark is drawn from, an RL
reward can only come from a judge. Notebook 07's task in [`../notebooks/smoke_test35`](../notebooks/smoke_test35) has
an exact verifier (`score()`), so it can measure three things: whether a reward from `gemini-4-argon` as judge can
replace the verifier's, what the replacement costs, and whether the student learns to exploit the judge.

**Arm J** is identical to 07's control PPO (`ppo-n300-maskend-turn-lr2e6-err0-s0`: same seed, same config, 200
steps) except that the reward comes from the judge. The verifier only evaluates and monitors; it never enters the
gradient. The control at step 200: dev 0.721, test 0.714, holdout_shift 0.608 (T=1 x2, err0).

## Judge

- **Sees** the question, the tool descriptions and the student's whole trajectory: its calls, the simulator's real
  results and its answer. No gold answer and no verifier score: reference-free, as on real tasks.
- **Returns** structured JSON with categorical verdicts, not 1–10 scores:
  - each call turn: inputs match the question; units correct; call necessary;
  - the answer: the judge's own pass/fail from the evidence; whether the student's decision, violations and values
    are right.
- **Reward:** `score()`'s formula and weights, with the gold-dependent components replaced by the judge's verdicts.
  Format is still checked by code; efficiency becomes the share of calls judged necessary. The per-turn credit is the
  control's, so the two arms differ only in who grades.
- One judge call per episode. The judge prompt is written on one half of G1's questions and frozen before it is
  measured on the other half.

## Gates

Named before running; a failed gate stops the experiment. The bars are proposals.

| Gate | What | Bar |
|---|---|---|
| G0 | structured-output calls work; tokens and latency measured; thinking low vs medium | — |
| G1 | offline, on saved dev episodes of the seed and PPO-200 (one per question; tool results rebuilt by replay): 120 questions to write the prompt, 120 to measure | decision agreement with the verifier ≥ 0.95, wrong answers judged right ≤ 0.05; episode-reward Spearman ≥ 0.85; test–retest agreement ≥ 0.95 on 100 episodes; beats a rule baseline ("answered, no tool error, JSON closed") by ≥ 0.10 on decision agreement; ranks PPO-200 above the seed by a margin like the verifier's (0.721 vs 0.457) |
| G2a | the first 40 steps | the training samples' verifier reward rises; judge − verifier gap within G1's bias ± 0.05 |
| G2b | 200 steps | stop if the gap grows more than 0.10 over G1's bias in any 25-step window (the student is exploiting the judge); at step 200, dev T=1 x2 err0 paired vs the seed: CI excludes 0 |

**Main readout:** J − control at step 200, paired. Near 0 means the judge can stand in for the verifier; a negative
difference is the price of having no verifier. Then test, holdout_shift and the forced probe. Route readout as in 07:
does the judge favour converting first or simulating first?

## Later

Keep the verifier's outcome reward and add the judge's per-turn verdicts as potential-based shaping, which leaves the
optimal policy unchanged. That targets 07's credit-assignment problem rather than replacing the verifier.

## Rules

- Training questions come only from the generator. The benchmark (`../auto-evaluate/benchmarks`) is evaluation-only
  and is never sent to the judge.
- The judge never sees gold answers or verifier scores.
- Every judge response is stored as received. The API ignores temperature, so nothing else makes a run replayable.

## Open

- Which trainer runs it: 07's PyTorch trainer (comparable with the control today) or the Rust training stack once it
  can train.
- Where the judge client lives: beside the reward, or behind a worker protocol as WaterTAP is.
- The layout of code and runs in this directory.

Cost estimate at Argon's introductory price: ~600 judge calls for G1 and 1,600 for 200 steps (8 episodes a step),
$0.02–0.05 each, ~$50–110 in all. Judge calls are synchronous, to keep the PPO identical to the control, and add
~30–60 s to each ~200 s step.
