# opd — distilling Gemini 4 Argon into Qwen3.5-0.8B

**Status: planned.** No code, runs or results yet. The design below is a proposal still under discussion.

Teacher `gemini-4-argon` (Gemini API), student `Qwen/Qwen3.5-0.8B`. Task, tools, grader and splits are those of
notebook 07 in [`../notebooks/smoke_test35`](../notebooks/smoke_test35): natural-language WaterTAP questions,
graded by code (`score()`). The student starts from 07's seed, so every result compares with 07's PPO at equal
steps.

## Why turn level, not token level

Token-level on-policy distillation (GKD; Thinking Machines' recipe) needs the teacher's log-probability of every
token the student sampled. The public Gemini API cannot give it:

- `responseLogprobs` / `logprobs` cover only tokens the model generates itself. Google's reference marks both
  deprecated for Gemini 3.x, Argon's capability table does not list them, and the Interactions API has no such field.
- Argon does not accept prefilled model turns, so the student's text cannot be put where the teacher would continue it.

The tokenizer difference is not the blocker: cross-tokenizer OPD (arXiv 2610.08448) handles it, but assumes a
white-box teacher. G0 settles the API question with one call of each kind.

## Method: turn-level on-policy distillation (DAgger form)

1. The student rolls out at T=1.
2. At every turn boundary of each rollout (t=0 included), Argon continues the episode to the end, with the same
   tools and the same fixed system prompt.
3. `score()` filters the continuations: decision correct plus evidence, the check 07's reference episodes pass.
4. The student is fine-tuned on the kept continuations. Its own prefix turns are masked, as 07 masks a slipped turn;
   the teacher's turns are trained.
5. Repeat with the updated student, aggregating the data of all rounds.

In GKD's terms the states are on-policy and the divergence is a forward KL estimated with one teacher sample per
state. It keeps OPD's on-policy state distribution and gives up the per-token reverse KL.

| Arm | Data |
|---|---|
| A — on-policy | all branch points |
| B — off-policy distillation | Argon's own episodes from t=0 only, cost-matched to A |
| control | 07's PPO from the same seed (`ppo-n300-maskend-turn-lr2e6-err0-s0`), step 200: dev 0.721, test 0.714, holdout_shift 0.608 (T=1 x2, err0) |

A against B tests OPD's central claim, on-policy beats off-policy distillation, on this task.

## Gates

Named before running; a failed gate stops the experiment.

| Gate | What | Bar |
|---|---|---|
| G0 | our tool declarations accepted; the student's turns injectable into Argon's history; tokens and latency measured; one `responseLogprobs` call and one prefill call | all work, or a documented fallback |
| G1 | Argon from scratch on dev (240 questions) and the forced probe (144), paired with PPO-200 | dev reward ≥ PPO-200 + 0.10, paired CI excludes 0 |
| G2 | Argon from the student's saved dev states, paired with the student's own continuation from the same state | clearly higher, above all after an erroring call; verifier yield ≥ 50% |
| round r | SFT on rounds 1..r; dev T=1 x2 err0, paired | round 1 vs seed: CI excludes 0; each later round beats the previous one |

At the end: test and holdout_shift (T=1 x2 and greedy) and the forced probe, paired with the seed and PPO-200.

## Later

- **Online.** Each training step labels the current student's states. Teacher targets do not go stale as the policy
  moves (only the state distribution does), so teacher calls can run one step behind the rollouts. Branches at t=0
  do not depend on the student and are computed once.
- **Token level through a white-box teacher.** SFT a same-tokenizer Qwen3.5-4B/9B on verified Argon trajectories,
  then per-token reverse-KL OPD into the 0.8B; or a Gemma teacher with 2610.08448's strict top-16.

## Rules

- Training questions come only from the generator. The MembraneClaw benchmark (`../auto-evaluate/benchmarks`) is
  evaluation-only and is never sent to the teacher.
- The task prompt is fixed and sent to the teacher verbatim, so the teacher acts in the context the student is
  trained in.
- Every teacher response is stored as received. The API ignores temperature, so nothing else makes a run replayable.

## Open

- Which trainer runs it: 07's PyTorch trainer (comparable with the control today) or the Rust training stack once it
  can train.
- The layout of code and runs in this directory.

Cost estimate at Argon's introductory price: ~$0.06–0.14 per Argon episode, ~$500–1,000 for G0 to round 3 with arm B.
