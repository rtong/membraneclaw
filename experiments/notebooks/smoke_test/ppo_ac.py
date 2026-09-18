"""Actor-critic PPO written out by hand, as the counterpart to GRPO.

`../../membrane_grpo/grpo_scratch.py` drops the value network: the other G-1
samples of the same prompt *are* the baseline. This file keeps the value
network and drops the group. Same data, same reward, same model, same clipped
surrogate -- the only thing that moves is where the baseline comes from and how
finely credit is assigned.

## The two methods side by side

| | GRPO | actor-critic PPO |
| --- | --- | --- |
| baseline | mean reward of G samples of one prompt | learned `V_phi(s_t)` |
| advantage | one scalar per sequence, broadcast to every token | one per token, via GAE |
| rollouts per prompt | G (8 in the reference run) | 1 is enough |
| degenerate case | all G rewards equal -> advantages exactly 0, no gradient | none: `R - V` is nonzero unless the critic is exactly right |
| extra parameters | none | `hidden_size + 1` |

The last two rows are the reasons to bother. 16% of GRPO's groups at the frozen
baseline are degenerate and contribute exactly nothing; a critic has no such
hole. What it has instead is a second thing that can be wrong, and a wrong
critic corrupts the *sign* of the advantage rather than merely zeroing it.

## The token-level MDP

The task is a contextual bandit -- one prompt, one completion, one terminal
scalar -- but PPO is written for a sequence of states, so the completion is read
as a trajectory:

    s_t = (prompt, a_0 .. a_{t-1})      the state before emitting token t
    a_t                                 the token emitted
    r_t = 0 for t < L-1,  r_{L-1} = R   reward arrives only at the last token
    gamma = 1                           completions are ~110 tokens; nothing
                                        here justifies preferring early tokens

`L` is the length of the *active* span: up to and including the first EOS, per
`build_mask` in `grpo_scratch.py`, which this file imports rather than
reimplements. Deciding to stop is an action and is scored; padding is not.

## GAE

With `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)` and `V(s_L) = 0` (terminal):

    A_t = delta_t + (gamma * lam) * A_{t+1},      A_t = 0 for t >= L

Two limits are worth holding on to, and `01_actor_critic_and_gae.ipynb` checks
both numerically rather than taking them on faith:

* `lam = 1` collapses to `A_t = R - V(s_t)`. Unbiased, and the whole trajectory's
  noise lands on every token.
* `lam = 0` collapses to `A_t = V(s_{t+1}) - V(s_t)` for t < L-1. Low variance,
  and completely dependent on the critic being right.

The critic's regression target is the GAE return `G_t = A_t + V(s_t)`, which at
`lam = 1` is just `R` at every position.

## The actor gradient, by hand

Identical in form to the GRPO surrogate -- only `A` is now indexed by t:

    rho_t = exp(logp_t - logp_old_t)
    obj_t = min(rho_t * A_t, clip(rho_t, 1-eps, 1+eps) * A_t)

    d obj_t / d logp_t  =  rho_t * A_t   inside the trust region
                           0             when A_t > 0 and rho_t > 1+eps
                           0             when A_t < 0 and rho_t < 1-eps

At `rho = 1` this reduces to `A_t`, so with one inner epoch the clip is inert by
construction and this is vanilla actor-critic policy gradient. That is not a
defect: it is the same property `grpo_scratch` documents, and it makes the two
runs comparable at `--inner-epochs 1`.

## Two choices worth arguing about

**The value head does not backpropagate into the trunk** (`value_detach=True`).
A shared trunk lets the value loss reshape the representations the policy reads,
which is standard in RLHF and is also a confound in an experiment whose question
is what the *policy* loss did. Detached, the critic is a linear probe on the
policy's own hidden states: weaker, and clean. `--no-value-detach` restores the
shared-trunk version.

**Advantages are not whitened** (`whiten=False`). Whitening across a batch of
mixed prompts would re-introduce a batch-level baseline and paper over whether
the critic works at all -- which is the one thing being measured here. It also
happens to hide the pathology below, and that pathology is worth seeing.

## The pathology to watch for

The reward is non-negative. A value head initialised to zero therefore produces
`A_t = R >= 0` for every token of every completion, and the first updates
reinforce *everything*, garbage included. GRPO cannot do this: group centring
makes advantages zero-mean by construction. The default here initialises the
head's bias to the frozen policy's own held-out reward -- looked up in
`FROZEN_BASELINE` per `(model, weight set)`, because the reward is a weighted sum
and the same frozen Qwen3-1.7B scores 0.3015 under MAIN, 0.4500 under PROBE and
0.2615 under ABLATE. So the critic starts approximately calibrated instead of
catastrophically low. `--value-init-bias 0` reproduces the failure on purpose,
and the failure is now larger than it was on the 0.5B: the gap between a cold
head and a calibrated one is 0.30 here against 0.086 before.
"""
from __future__ import annotations

import contextlib
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

SMOKE_DIR = Path(__file__).resolve().parent
GRPO_DIR = SMOKE_DIR.parents[1] / "membrane_grpo"
if str(GRPO_DIR) not in sys.path:
    sys.path.insert(0, str(GRPO_DIR))

# Imported, never reimplemented: one tokenizer convention, one reward, one
# prompt, shared with the GRPO run so the comparison is not between two
# subtly different tasks.
from eval import supports_thinking_toggle  # noqa: E402
from task.decision_table import CAUSES  # noqa: E402
from grpo_scratch import build_mask, selective_logprobs  # noqa: E402
from reward import ABLATE, MAIN, PROBE, Weights, score  # noqa: E402
from task.prompt import PROMPT_VERSION, build_messages  # noqa: E402

DATA = GRPO_DIR / "data"

#: Held-out reward of the *frozen* policy on dev, greedy, prompt v2 -- one
#: number per weight set, because the reward is a weighted sum and the same
#: policy scores differently under each. This is what the value head's bias is
#: initialised to, so it must match the weights the run is actually using;
#: initialising to another weight set's baseline is the pathology in the module
#: docstring, only quieter.
#:
#: Qwen3-1.7B, from `membrane_grpo/runs/q3-{main,probe,ablate}-s0/eval.jsonl`
#: step 0, which is the same `eval.py` code path that produced the frozen
#: baselines. `STAGE_ONLY` is derived rather than measured: the MAIN evaluation
#: reports `components.stage = 0.0291` against a weight of 0.03, so the frozen
#: policy already copies the stage on 97% of cases.
#: The same measurement under v3, where the harness supplies the arithmetic.
#: From `runs/paired/base_v3_{main,ablate}.json`, the frozen policy on the full
#: dev split, greedy, identical settings to `runs/paired/base.json`. It is far
#: higher than the v2 figure for a mechanical reason -- `numeric` is 1.000 by
#: construction -- so a value head initialised to the v2 number would start a v3
#: run badly biased and spend the early steps unlearning it.
FROZEN_BASELINE_V3: dict[str, dict[str, float]] = {
    "Qwen/Qwen3-1.7B": {"MAIN": 0.5151, "ABLATE": 0.6561},
}

#: The tool line's warm start. Not the raw model's reward under the tool prompt
#: (0.2343) -- PPO on this line starts from the tool seed, and a head warm-started
#: at the raw model's level sits 0.39 below every return it will ever see.
#: `12`'s first gate run did exactly that: it resolved `FROZEN_BASELINE`'s v2
#: 0.2615, inherited it through `resume_from`, and spent 40 steps climbing
#: 0.295 -> 0.412 while the lam = 0.95 targets, bootstrapped off that same V,
#: climbed with it. This is `runs/tool/sft_raw_dev.json` -> overall.reward, the
#: seed's held-out reward under ABLATE; MAIN was never measured on it.
FROZEN_BASELINE_TOOL: dict[str, dict[str, float]] = {
    "Qwen/Qwen3-1.7B": {"ABLATE": 0.6521},
}

FROZEN_BASELINE: dict[str, dict[str, float]] = {
    "Qwen/Qwen3-1.7B": {
        "MAIN": 0.3015,
        "PROBE": 0.4500,
        "ABLATE": 0.2615,
        "STAGE_ONLY": 0.970,
    },
    # Kept so the 0.5B runs already in `runs/` stay reproducible from this file.
    # MAIN from runs/baseline-0.5b-v2/eval_dev_greedy.json -> overall.reward;
    # PROBE from MEMO.md's probe run at step 0. ABLATE is absent rather than
    # guessed -- it was never run on this model, and an unmeasured pair should
    # fall through to the loud 0.0 rather than borrow a neighbour's number.
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "MAIN": 0.086,
        "PROBE": 0.024,
        "STAGE_ONLY": 0.0,
    },
}


def value_init_bias_for(model: str, weights: str) -> float:
    """The warm start for the value head, or 0.0 if this pair was never measured.

    0.0 is the *documented* failure -- the reward is non-negative, so a critic at
    zero makes every advantage non-negative and the first updates reinforce
    everything, garbage included. Falling back to it is loud rather than
    plausible, which is what a fallback for a missing measurement should be.
    """
    return FROZEN_BASELINE.get(model, {}).get(weights, 0.0)


#: The default run's warm start: Qwen3-1.7B under MAIN. Named separately because
#: `01_actor_critic_and_gae.ipynb` contrasts it against a cold head at 0.0.
VALUE_INIT_BIAS = FROZEN_BASELINE["Qwen/Qwen3-1.7B"]["MAIN"]

#: The go/no-go weighting. Every point is on `stage`, which the prompt states
#: outright -- `task/generate.py` puts `anomaly_stage` in the record and the
#: answer copies it, so the correct output is a literal already on screen. No
#: arithmetic, no table lookup, no vocabulary the model has to infer.
#:
#: The question this asks is deliberately the easiest one available: can RL move
#: this model on a pure copy task when copying is the *only* thing that pays?
#: Under MAIN weights `stage` is worth 0.03 and 200 steps left it at 0.000. If
#: it does not move even here, the ceiling is not a tuning problem.
#:
#: That question was asked of the 0.5B and answered (`runs/gonogo-stage-s0`).
#: **It is not a live probe on Qwen3-1.7B**, which already copies the stage on
#: 97% of dev cases before any training -- there is 0.03 of headroom, so a
#: STAGE_ONLY run here measures a ceiling effect and nothing else. Kept for
#: reproducing the 0.5B result, not for rerunning on the new model.
STAGE_ONLY = Weights(format=0.0, numeric=0.0, flags=0.0, stage=1.0, root_cause=0.0, action=0.0)

#: The weight sets a run may be trained under, resolved by name in one place so
#: `train`, `evaluate` and the CLI cannot drift apart. `ABLATE` comes from the
#: GRPO side: it holds `numeric` at PROBE's 0.35 while leaving `root_cause`
#: substantial at 0.25, and on Qwen3-1.7B it settled which of the two weights
#: was doing the work (`membrane_grpo/runs/q3-ablate-s0`).
WEIGHT_SETS: dict[str, Weights] = {
    "MAIN": MAIN,
    "PROBE": PROBE,
    "ABLATE": ABLATE,
    "STAGE_ONLY": STAGE_ONLY,
}

#: Steps averaged when deciding which policy was "best". A single step is 8
#: samples and far too noisy to checkpoint on; a trailing mean is not.
#:
#: This is a *training* reward, and a training reward is not evidence: the
#: 0.5B stage run held 1.0000 here for 30 steps while it was worth 0.000 on
#: dev. With `eval_every` on, prefer `eval.jsonl` for that judgement; this
#: checkpoints often enough not to lose the policy in between.
BEST_WINDOW = 10

__all__ = [
    "ABLATE",
    "DATA",
    "FROZEN_BASELINE",
    "PROMPT_VERSION",
    "VALUE_INIT_BIAS",
    "WEIGHT_SETS",
    "MAIN",
    "PROBE",
    "STAGE_ONLY",
    "Weights",
    "ValueHead",
    "build_mask",
    "build_messages",
    "build_messages_v3",
    "build_messages_v4",
    "v3_prompts",
    "compute_changes",
    "CALCULATOR_TOOL",
    "build_messages_tool",
    "calculate",
    "gold_expressions",
    "tool_call_target",
    "tool_generate",
    "tool_rollout",
    "critic_fit",
    "evaluate",
    "value_init_bias_for",
    "gae",
    "load_cases",
    "ppo_actor_loss",
    "score",
    "selective_logprobs",
    "terminal_rewards",
    "dense_rewards",
    "field_credits",
    "token_entropy",
    "value_loss",
    "whiten",
]


#: Magnitude of the privileged one-hot -- derivation at its use site.
PRIVILEGED_SCALE = 64.0


# --- v3: the arithmetic, computed by the harness ------------------------------
#
# `06`'s appendix and `07`'s closing section measured the same thing from two
# directions: Qwen3-1.7B does not do this task's arithmetic. Two of the three
# percent changes involve no domain formula at all -- a division and a sum, then
# a percent change -- and it misses those at the same rate as the one that uses
# `TCF`. Over the 1000 case-evaluations in `runs/paired/` the count of numbers
# landing inside the 0.5 pp tolerance is 844 zeros, 140 ones, 15 twos and one
# three. Showing the working in a 2-shot prompt moved it by 0.000.
#
# The readings are already structured in `record["t0"]` / `record["t1"]`, so the
# harness can do the arithmetic exactly and hand the model the result. That is
# what a calculator tool would achieve, minus the tool-call protocol and minus
# the extraction step, because there is nothing to extract -- the numbers are
# already parsed. `compute_changes` reproduces every one of the 600 answers in
# `data/{train,dev}.jsonl` exactly, which is the check that licenses this.
#
# v3 is v2 with Step 1 replaced, by surgery on v2's own rendered text rather
# than by re-rendering it. Everything outside the Step 1 block and the closing
# instruction is byte-identical to v2, so a v2/v3 comparison has one variable.


def compute_changes(record: dict[str, Any]) -> dict[str, float]:
    """The three percent changes, from the structured readings. Exact."""

    def tcf(temp_c: float) -> float:
        return 1.03 ** (25 - temp_c)

    def normalized_flow(t: dict[str, float]) -> float:
        return t["permeate_flow_m3_h"] * tcf(t["feed_temp_C"])

    def salt_passage(t: dict[str, float]) -> float:
        return t["permeate_conductivity_uS_cm"] / t["feed_conductivity_uS_cm"] * 100

    def dp(t: dict[str, float]) -> float:
        return t["dp_lead_bar"] + t["dp_tail_bar"]

    t0, t1 = record["t0"], record["t1"]

    def pct(before: float, after: float) -> float:
        return round((after - before) / before * 100, 1)

    return {
        "normalized_flow_change_pct": pct(normalized_flow(t0), normalized_flow(t1)),
        "salt_passage_change_pct": pct(salt_passage(t0), salt_passage(t1)),
        "dp_change_pct": pct(dp(t0), dp(t1)),
    }


V3_STEP1 = """Step 1 -- the three percent changes, already computed for you.

  normalized_flow_change_pct = {normalized_flow_change_pct}
  salt_passage_change_pct    = {salt_passage_change_pct}
  dp_change_pct              = {dp_change_pct}

  Copy these three values into the JSON unchanged. Do not recompute them.

"""

V3_CLOSING = """The three percent changes are given above -- copy them into the JSON
unchanged. Use at most one short line of plain reasoning for the flags -- no
prose, no LaTeX, no headings.

Then end your reply with this JSON object and nothing after it:"""


def build_user_prompt_v3(record: dict[str, Any]) -> str:
    """v2's user turn with the arithmetic supplied instead of demanded."""
    from task.prompt import CLOSINGS, build_user_prompt

    text = build_user_prompt(record, "v2")
    start, end = text.index("Step 1 -- "), text.index("Step 2 -- ")
    text = text[:start] + V3_STEP1.format(**compute_changes(record)) + text[end:]
    old = CLOSINGS["v2"]
    if old not in text:  # pragma: no cover - guards a membrane_grpo edit
        raise RuntimeError("v2 closing not found; task/prompt.py changed under v3")
    return text.replace(old, V3_CLOSING)


@contextlib.contextmanager
def v3_prompts(v4: bool = False):
    """Swap the prompt builder `eval.generate_hf` closes over, for one call.

    `generate_hf` differs from what v3 needs in exactly one line -- the
    `build_messages` it calls -- and rebinding that name in its own module
    namespace leaves every other line of the generation and scoring path
    literally theirs. That is the property the whole series rests on: the frozen
    baseline, the GRPO curve and every PPO curve are produced by the same
    function, so a v2/v3 difference is a difference in the prompt and nothing
    else. Nothing is written into that tree; the binding is restored on exit.
    """
    import eval as membrane_eval

    original = membrane_eval.build_messages
    builder = build_messages_v4 if v4 else build_messages_v3
    membrane_eval.build_messages = lambda record, version=None: builder(record)
    try:
        yield
    finally:
        membrane_eval.build_messages = original


# --- v4: say the flat band out loud ---------------------------------------------
#
# `09`'s diagnosis, on the frozen policy and on the trained v3 one alike: **the
# model emits `flat` exactly zero times in 600 flag slots**, and `flat` is the
# right answer in 184 of them. Every other value is nearly perfect once the
# arithmetic is supplied -- 399/416 = 0.959 on the non-flat slots -- and
# 0.959 * 416/600 = 0.665 is `flags_acc` to three decimals. The entire remaining
# deficit is one value the policy will not say.
#
# That matters beyond the accuracy, because it is the one shape RL provably
# cannot fix: a policy gradient reweights behaviour that appears in a sample, and
# `flat` never appears. Four prompt variants were measured on the frozen policy,
# greedy, full dev:
#
#   control v3                          flat emitted   0 / 184   all-three 0.130
#   enumerate the values in the schema  flat emitted   1 / 184   all-three 0.210
#   *state the flat band first*         flat emitted  26 / 184   all-three 0.245
#   both of the above                   flat emitted   3 / 184   all-three 0.155
#
# v4 is the third. It restates Step 2 with the flat band as its own leading
# condition rather than a trailing `else`, and says how often `flat` is right.
# 26 is still far below 184 -- telling the model directly barely moves it, which
# is a fact about the model -- but 26 is not 0, and that is the difference
# between a gradient existing and not existing.
V4_STEP2 = """Step 2 -- turn each change into a flag. Check the flat band first:

  flow          : flat if strictly between -10 and +10; else down if <= -10, up if >= +10
  salt_passage  : flat if strictly between -15 and +15; else down if <= -15, \
sharp_up if >= +50, up if >= +15
  dp            : flat if strictly between -15 and +15; else down if <= -15, up if >= +15

  Roughly a third of all flags are flat. Do not avoid it.

"""


def build_user_prompt_v4(record: dict[str, Any]) -> str:
    """v3's user turn with Step 2's flat band promoted out of the `else`."""
    text = build_user_prompt_v3(record)
    start, end = text.index("Step 2 -- "), text.index("Step 3 -- ")
    return text[:start] + V4_STEP2 + text[end:]


def build_messages_v4(record: dict[str, Any]) -> list[dict[str, str]]:
    from task.prompt import SYSTEM_PROMPTS

    return [
        {"role": "system", "content": SYSTEM_PROMPTS["v2"]},
        {"role": "user", "content": build_user_prompt_v4(record)},
    ]


def build_messages_v3(record: dict[str, Any]) -> list[dict[str, str]]:
    from task.prompt import SYSTEM_PROMPTS

    return [
        {"role": "system", "content": SYSTEM_PROMPTS["v2"]},
        {"role": "user", "content": build_user_prompt_v3(record)},
    ]


# --- the calculator: the arithmetic goes back to the model, with a tool --------
#
# v3 took the arithmetic away from the model and printed the three answers in the
# prompt. That made `numeric_acc` 1.000 by construction and it made the task a
# different task: `08` said as much -- "what a calculator tool would achieve,
# minus the tool-call protocol and minus the extraction step". This puts both of
# those back, and changes no word of the task to do it.
#
# **The prompt is v2's, unchanged.** System turn and user turn are exactly
# `task.prompt.build_messages(record, "v2")` -- same formulas, same closing, same
# demand to compute -- and `12` checks that byte for byte on every case. The one
# addition is the tool declaration Qwen3's chat template writes into the system
# turn when a tool is offered, and `CALCULATOR_TOOL` describes a generic
# calculator with no example that could hint at the task's expressions. Nothing
# in the prompt says to use it: that the tool exists is the prompt's business,
# when to call it is what the SFT teaches. The model still has to read twelve
# numbers off the record and write three expressions from the stated formulas;
# the harness only evaluates what it is handed.
#
# One layout is used everywhere -- SFT, rollout, evaluation -- and it is the
# layout the model actually generates in: prompt, then the policy's turn (ending
# in `<|im_end|>`), then the tool responses and a fresh assistant header, then
# the policy's next turn. Re-rendering a finished episode through the chat
# template is *not* the same token sequence: the template drops the empty
# `<think></think>` block from an earlier assistant turn, which the model did see
# when it wrote that turn. Building by concatenation keeps the trained tokens
# identical to the sampled ones.

CALCULATOR_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Evaluate an arithmetic expression and return the result. "
            "Supports + - * / ** and parentheses, plus round(x, n) and abs(x)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "the arithmetic expression"}
            },
            "required": ["expression"],
        },
    },
}

#: Tool calls honoured per assistant turn. Three are needed; the cap only stops a
#: policy that has learned to spam calls from blowing the sequence length.
MAX_CALLS_PER_TURN = 8
#: Longest expression evaluated. The gold flow expression is about 90 characters.
MAX_EXPRESSION_CHARS = 400


def calculate(expression: str) -> str:
    """Evaluate one expression and return what the tool sends back, as text.

    Walks the AST rather than calling `eval`: numbers, the five binary operators,
    unary signs, and `round` / `abs` / `min` / `max`, nothing else. Errors come
    back as `error: ...` text rather than raising -- the policy sees them, and a
    malformed call has to cost a turn, not crash a rollout.
    """
    import ast
    import math
    import operator

    binary = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    }
    unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    functions = {"round": round, "abs": abs, "min": min, "max": max}

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
                raise ValueError("exponent out of range")
            return binary[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary:
            return unary[type(node.op)](walk(node.operand))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in functions and not node.keywords):
            args = [walk(a) for a in node.args]
            if node.func.id == "round" and len(args) == 2:
                args[1] = int(args[1])
            return functions[node.func.id](*args)
        raise ValueError(f"unsupported syntax: {type(node).__name__}")

    if not isinstance(expression, str) or not expression.strip():
        return "error: empty expression"
    if len(expression) > MAX_EXPRESSION_CHARS:
        return "error: expression too long"
    try:
        value = walk(ast.parse(expression.strip(), mode="eval"))
        value = float(value)
        if not math.isfinite(value):
            return "error: result is not finite"
    except ZeroDivisionError:
        return "error: division by zero"
    except (SyntaxError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        return f"error: {str(exc) or type(exc).__name__}"
    # Ten significant figures: enough that nothing the task needs is lost, few
    # enough that float noise (-22.400000000000002) never reaches the policy.
    return format(value, ".10g")


def parse_tool_calls(turn: str) -> list[dict[str, Any]]:
    """Every `<tool_call>...</tool_call>` block in one assistant turn, in order.

    A block whose body is not a valid call is still returned, with `error` set,
    so it is answered with an error rather than silently skipped.
    """
    import json
    import re

    calls = []
    for body in re.findall(r"<tool_call>(.*?)</tool_call>", turn, flags=re.S):
        try:
            obj = json.loads(body.strip())
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if obj.get("name") != "calculator":
                raise ValueError(f"unknown tool {obj.get('name')!r}")
            calls.append({"expression": args["expression"]})
        except (json.JSONDecodeError, ValueError, TypeError, KeyError, AttributeError) as exc:
            calls.append({"error": f"error: malformed tool call ({type(exc).__name__})"})
    return calls


def run_tool_call(call: dict[str, Any]) -> str:
    return call["error"] if "error" in call else calculate(call["expression"])


def build_messages_tool(record: dict[str, Any]) -> list[dict[str, str]]:
    """v2's messages, untouched. The tool is passed to the chat template, not here."""
    return build_messages(record, "v2")


def gold_expressions(record: dict[str, Any]) -> dict[str, str]:
    """The three calculator expressions a correct policy would write.

    Built from the numbers *as the prompt prints them* (`task.prompt._num`), since
    those are the only numbers the policy can see, and in `compute_changes`'
    operation order, so Python evaluates them to the same float. Whether that
    reproduces the answer key is checked, not assumed: `12` runs it over every
    case of every split.
    """
    from task.prompt import _num

    t0, t1 = record["t0"], record["t1"]

    def nf(t):
        return f"{_num(t['permeate_flow_m3_h'])} * 1.03 ** (25 - {_num(t['feed_temp_C'])})"

    def sp(t):
        return f"{_num(t['permeate_conductivity_uS_cm'])} / {_num(t['feed_conductivity_uS_cm'])} * 100"

    def dp(t):
        return f"({_num(t['dp_lead_bar'])} + {_num(t['dp_tail_bar'])})"

    def pct(before: str, after: str) -> str:
        return f"round(({after} - {before}) / ({before}) * 100, 1)"

    return {
        "normalized_flow_change_pct": pct(nf(t0), nf(t1)),
        "salt_passage_change_pct": pct(sp(t0), sp(t1)),
        "dp_change_pct": pct(dp(t0), dp(t1)),
    }


def tool_call_target(record: dict[str, Any]) -> str:
    """The first assistant turn, as SFT supervises it: three calls and nothing else.

    Exactly the text Qwen3's template writes for an assistant message carrying
    three `tool_calls`. No answer field, no flag, no label appears in it -- the
    numbers come back from the tool, and everything after the tool response is
    left to the model.
    """
    import json

    blocks = [
        "<tool_call>\n"
        + json.dumps({"name": "calculator", "arguments": {"expression": expr}})
        + "\n</tool_call>"
        for expr in gold_expressions(record).values()
    ]
    return "\n".join(blocks)


def generation_prefix(tokenizer, template_kwargs: dict[str, Any]) -> str:
    """What `add_generation_prompt` appends: the assistant header, and under
    `enable_thinking=False` the empty think block."""
    probe = [{"role": "user", "content": "x"}]
    with_prompt = tokenizer.apply_chat_template(
        probe, tokenize=False, add_generation_prompt=True, **template_kwargs)
    without = tokenizer.apply_chat_template(
        probe, tokenize=False, add_generation_prompt=False, **template_kwargs)
    return with_prompt[len(without):]


def tool_response_text(results: list[str], prefix: str) -> str:
    """Everything between the policy's `<|im_end|>` and its next turn.

    The tool block is Qwen3's template output for consecutive `tool` messages;
    `12` checks it against the template character for character.
    """
    body = "".join(f"\n<tool_response>\n{r}\n</tool_response>" for r in results)
    return "\n<|im_start|>user" + body + "<|im_end|>\n" + prefix


@dataclass
class Episode:
    """One multi-turn completion. `ids` is everything after the prompt."""

    prompt_ids: list[int]
    ids: list[int] = field(default_factory=list)
    #: Per entry of `ids`: True if the policy sampled it, False if the tool wrote it.
    policy: list[bool] = field(default_factory=list)
    #: The policy's own tokens, decoded. This is what gets scored.
    text: str = ""
    calls: list[dict[str, Any]] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    #: Calls written in the last allowed turn, which were never answered.
    unanswered: int = 0
    truncated: bool = False
    #: Compacted index of the first policy token of each turn.
    turn_starts: list[int] = field(default_factory=list)
    #: The harness's own verdict, taken as each call returned (only when it is
    #: handed the answer key): numeric field -> compacted index of the
    #: `</tool_call>` token of the first call whose result matched that field.
    #: The key never reaches the policy's context -- it scores, it does not reply.
    call_hits: dict[str, int] = field(default_factory=dict)

    @property
    def policy_tokens(self) -> int:
        return sum(self.policy)


def _sample_round(policy, inputs: list[list[int]], budgets: list[int], *, pad: int,
                  temperature: float, device: str) -> list[list[int]]:
    """One `generate` over a round's rows, left-padded; each row cut to its budget.

    Shared by both harnesses, so a round is sampled identically whichever one
    is orchestrating it.
    """
    width = max(len(x) for x in inputs)
    input_ids = torch.tensor([[pad] * (width - len(x)) + x for x in inputs], device=device)
    attention = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in inputs], device=device)
    sample = temperature > 0
    out = policy.generate(
        input_ids=input_ids,
        attention_mask=attention,
        do_sample=sample,
        temperature=temperature if sample else None,
        top_p=1.0 if sample else None,
        max_new_tokens=max(budgets),
        pad_token_id=pad,
    )
    return [row[:b] for row, b in zip(out[:, width:].tolist(), budgets)]


@dataclass
class _Turn:
    """What one sampled turn asked for. `calls` is empty when the episode ends."""

    calls: list[dict[str, Any]]
    #: Offset, within the turn, of each call's `</tool_call>` token.
    closes: list[int]
    #: Compacted index of the turn's first token.
    start: int
    text: str


def _take_turn(ep: Episode, row: list[int], tokenizer, *, last_round: bool) -> _Turn:
    """Append one sampled turn to `ep` and read the calls out of it."""
    eos_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    close_call = tokenizer.convert_tokens_to_ids("</tool_call>")
    cut = next((k for k, t in enumerate(row) if t in eos_ids), None)
    take = row if cut is None else row[: cut + 1]
    start = ep.policy_tokens
    ep.turn_starts.append(start)
    ep.ids += take
    ep.policy += [True] * len(take)
    text = tokenizer.decode(take, skip_special_tokens=True)
    if cut is None:
        ep.truncated = True
        return _Turn([], [], start, text)
    calls = parse_tool_calls(text)
    if not calls or row[cut] != im_end:
        return _Turn([], [], start, text)
    if last_round:
        ep.unanswered += len(calls)
        return _Turn([], [], start, text)
    # Where each call ends, in the compacted sequence. `</tool_call>` is one
    # added token, so its k-th occurrence closes the k-th block; if the counts
    # disagree (the string spelled out of sub-word pieces), every call in the
    # turn falls back to the turn's last token.
    closes = [k for k, t in enumerate(take) if t == close_call]
    if len(closes) != len(calls):
        closes = [len(take) - 1] * len(calls)
    return _Turn(calls[:MAX_CALLS_PER_TURN], closes[:MAX_CALLS_PER_TURN], start, text)


def _record_results(ep: Episode, turn: _Turn, results: list[str], tokenizer, prefix: str,
                    answer: dict[str, Any] | None) -> None:
    """Score the returned calls (if the harness holds the key), then append the
    tool's turn to the episode."""
    from reward import NUMERIC_TOLERANCE_PP
    from task.schema import NUMERIC_KEYS

    if answer is not None:
        for result, close in zip(results, turn.closes):
            try:
                value = float(result)
            except ValueError:
                continue
            for key in NUMERIC_KEYS:
                if key not in ep.call_hits and abs(value - answer[key]) <= NUMERIC_TOLERANCE_PP:
                    ep.call_hits[key] = turn.start + close
                    break
    ep.calls += turn.calls
    ep.results += results
    env = tokenizer(tool_response_text(results, prefix), add_special_tokens=False).input_ids
    ep.ids += env
    ep.policy += [False] * len(env)


def _render_tool_prompt(tokenizer, messages, template_kwargs) -> str:
    return tokenizer.apply_chat_template(
        messages, tools=[CALCULATOR_TOOL], tokenize=False, add_generation_prompt=True,
        **template_kwargs,
    )


@torch.no_grad()
def tool_generate(
    policy,
    tokenizer,
    messages: list[list[dict[str, str]]],
    *,
    temperature: float,
    max_new_tokens: int,
    max_rounds: int,
    template_kwargs: dict[str, Any],
    device: str,
    answers: list[dict[str, Any]] | None = None,
) -> list[Episode]:
    """Decode with the calculator in the loop, for a batch of prompts.

    `max_new_tokens` caps the policy's tokens summed over all its turns, so a
    tool episode never gets more generation budget than a single-turn one.
    `max_rounds` is how many times the tool may answer; a turn that still calls
    it after that ends the episode.

    `answers`, when given, lets the harness score each call the moment it
    returns (`Episode.call_hits`). That is for the learner only; what the
    policy sees next is the calculator's result and nothing else.
    """
    pad = tokenizer.pad_token_id
    prefix = generation_prefix(tokenizer, template_kwargs)
    episodes = [
        Episode(prompt_ids=tokenizer(_render_tool_prompt(tokenizer, m, template_kwargs),
                                     add_special_tokens=False).input_ids)
        for m in messages
    ]
    live = list(range(len(episodes)))
    for rnd in range(max_rounds + 1):
        budget = {i: max_new_tokens - episodes[i].policy_tokens for i in live}
        live = [i for i in live if budget[i] > 0]
        if not live:
            break
        rows = _sample_round(
            policy, [episodes[i].prompt_ids + episodes[i].ids for i in live],
            [budget[i] for i in live], pad=pad, temperature=temperature, device=device,
        )
        still = []
        for row, i in zip(rows, live):
            turn = _take_turn(episodes[i], row, tokenizer, last_round=rnd == max_rounds)
            if not turn.calls:
                continue
            results = [run_tool_call(c) for c in turn.calls]
            _record_results(episodes[i], turn, results, tokenizer, prefix,
                            None if answers is None else answers[i])
            still.append(i)
        live = still
    for ep in episodes:
        ep.text = tokenizer.decode(
            [t for t, p in zip(ep.ids, ep.policy) if p], skip_special_tokens=True)
    return episodes


# --- 13b: Qwen-Agent as the harness ----------------------------------------------
#
# `membraneclaw-agent` runs on qwen-agent 0.0.34 (`agent/core.py`): an agent over a
# vLLM endpoint with `use_raw_api`, so tools reach the chat template natively and
# the server parses the calls. This puts the policy under training behind that
# same orchestration. qwen-agent owns the loop -- it takes the calls the model
# made, runs them through its tool map, appends the results and asks again --
# and `PolicyChat` stands where vLLM stands: it renders, decodes and parses. What
# it adds is what a text API cannot give a trainer: the exact token ids of every
# turn, kept per episode and never re-rendered.
#
# Each episode's agent runs in its own thread. `_RoundBatcher` holds every
# request until each live episode has either asked for its next turn or
# finished, then samples them as one batch with `_sample_round` -- the same rows,
# in the same order, padded the same way as `tool_generate`. So the two harnesses
# are one decoding procedure with two orchestrators, which is what `13`'s
# equivalence gate checks.
#
# qwen-agent is imported read-only from the repository's own `.venv`, appended
# after this venv on `sys.path`. Nothing is installed.

AGENT_SITE = (SMOKE_DIR.parents[2] / ".venv" / "lib"
              / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")

#: Live agent runs, by id. qwen-agent builds its model object from a config dict,
#: so the run is handed over by name rather than by reference.
_AGENT_RUNS: dict[str, "_AgentRun"] = {}


def _qwen_agent():
    if str(AGENT_SITE) not in sys.path:
        sys.path.append(str(AGENT_SITE))
    import logging

    import qwen_agent

    logging.getLogger("qwen_agent_logger").setLevel(logging.WARNING)
    return qwen_agent


class _RoundBatcher:
    """Collects one turn request per live episode and samples them as one batch.

    Every `generate` runs on the batcher's own worker thread, never on an agent's
    thread: CUDA is then touched from one thread only, which is how
    `tool_generate` touches it. (`13b`'s first training attempt died with
    `CUDA error: unknown error`, together with WSL's dxg driver logging failed
    ioctls, while rounds were being sampled on whichever agent thread happened
    to complete them.)
    """

    def __init__(self, policy, n: int, *, pad: int, temperature: float, device: str):
        import threading

        self.policy, self.pad, self.temperature, self.device = policy, pad, temperature, device
        self.cv = threading.Condition()
        self.live = set(range(n))
        self.pending: dict[int, tuple[list[int], int]] = {}
        self.done: dict[int, list[int]] = {}
        self.error: BaseException | None = None
        self.worker = threading.Thread(target=self._serve, daemon=True)
        self.worker.start()

    def _serve(self) -> None:
        while True:
            with self.cv:
                while not (self.pending and self.live <= set(self.pending)) and self.live:
                    self.cv.wait()
                if not self.live:
                    return
                order = sorted(self.pending)
                batch = [self.pending[j] for j in order]
                self.pending.clear()
            try:
                rows = _sample_round(
                    self.policy, [ids for ids, _ in batch], [b for _, b in batch],
                    pad=self.pad, temperature=self.temperature, device=self.device,
                )
            except BaseException as exc:  # handed to every waiting episode
                with self.cv:
                    self.error = exc
                    self.cv.notify_all()
                return
            with self.cv:
                self.done.update(zip(order, rows))
                self.cv.notify_all()

    def request(self, i: int, ids: list[int], budget: int) -> list[int]:
        with self.cv:
            self.pending[i] = (ids, budget)
            self.cv.notify_all()
            while i not in self.done:
                if self.error is not None:
                    raise RuntimeError("sampling failed on the batcher thread") from self.error
                self.cv.wait()
            return self.done.pop(i)

    def finish(self, i: int) -> None:
        with self.cv:
            self.live.discard(i)
            self.cv.notify_all()


class _AgentRun:
    """The token-level state of a batch of agent episodes."""

    def __init__(self, policy, tokenizer, messages, *, temperature, max_new_tokens, max_rounds,
                 template_kwargs, device, answers):
        self.tokenizer, self.max_new_tokens, self.max_rounds = tokenizer, max_new_tokens, max_rounds
        self.template_kwargs, self.answers = template_kwargs, answers
        self.expected = [_render_tool_prompt(tokenizer, m, template_kwargs) for m in messages]
        self.prefix = generation_prefix(tokenizer, template_kwargs)
        self.episodes = [Episode(prompt_ids=[]) for _ in messages]
        self.rounds = [0] * len(messages)
        self.open_turn: list[_Turn | None] = [None] * len(messages)
        self.batcher = _RoundBatcher(policy, len(messages), pad=tokenizer.pad_token_id,
                                     temperature=temperature, device=device)

    def turn(self, i: int, messages, tools) -> list:
        from qwen_agent.llm.schema import ASSISTANT, FUNCTION, FunctionCall, Message

        ep = self.episodes[i]
        if not ep.prompt_ids:
            plain = [{"role": m.role, "content": m.content} for m in messages]
            text = self.tokenizer.apply_chat_template(
                plain, tools=tools, tokenize=False, add_generation_prompt=True, **self.template_kwargs)
            if text != self.expected[i]:
                raise RuntimeError("qwen-agent handed over a prompt that is not the task's prompt")
            ep.prompt_ids = self.tokenizer(text, add_special_tokens=False).input_ids
        else:
            turn = self.open_turn[i]
            results = [m.content for m in messages if m.role == FUNCTION][len(ep.results):]
            if turn is None or len(results) != len(turn.calls):
                raise RuntimeError(f"episode {i}: expected {0 if turn is None else len(turn.calls)} "
                                   f"tool results, got {len(results)}")
            _record_results(ep, turn, results, self.tokenizer, self.prefix,
                            None if self.answers is None else self.answers[i])
            self.open_turn[i] = None
        budget = self.max_new_tokens - ep.policy_tokens
        if budget <= 0:
            return [Message(ASSISTANT, "")]
        row = self.batcher.request(i, ep.prompt_ids + ep.ids, budget)
        turn = _take_turn(ep, row, self.tokenizer, last_round=self.rounds[i] == self.max_rounds)
        if not turn.calls:
            return [Message(ASSISTANT, turn.text)]
        self.rounds[i] += 1
        self.open_turn[i] = turn
        import json
        import re

        out = []
        content = re.sub(r"<tool_call>.*?</tool_call>", "", turn.text, flags=re.S).strip()
        if content:
            out.append(Message(ASSISTANT, content))
        for k, call in enumerate(turn.calls):
            # A malformed block still becomes a call, answered by the tool with
            # the same error text `tool_generate` sends, so the two harnesses
            # show the policy the same thing.
            args = ({"__error__": call["error"]} if "error" in call
                    else {"expression": call["expression"]})
            out.append(Message(ASSISTANT, "", function_call=FunctionCall(
                name="calculator", arguments=json.dumps(args)), extra={"function_id": str(k)}))
        return out


def _agent_classes():
    _qwen_agent()
    from qwen_agent.llm.base import LLM_REGISTRY, register_llm
    from qwen_agent.llm.function_calling import BaseFnCallModel
    from qwen_agent.tools.base import BaseTool

    if "ppo_policy" not in LLM_REGISTRY:

        @register_llm("ppo_policy")
        class PolicyChat(BaseFnCallModel):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.run_id, self.episode = cfg["run"], cfg["episode"]

            def _chat_stream(self, messages, delta_stream, generate_cfg):
                yield _AGENT_RUNS[self.run_id].turn(self.episode, messages, generate_cfg.get("tools"))

            def _chat_no_stream(self, messages, generate_cfg):
                return _AGENT_RUNS[self.run_id].turn(self.episode, messages, generate_cfg.get("tools"))

    class CalculatorTool(BaseTool):
        name = CALCULATOR_TOOL["function"]["name"]
        description = CALCULATOR_TOOL["function"]["description"]
        parameters = CALCULATOR_TOOL["function"]["parameters"]

        def call(self, params, **kwargs) -> str:
            import json

            args = json.loads(params) if isinstance(params, str) else params
            if "__error__" in args:
                return args["__error__"]
            return calculate(args.get("expression", ""))

    return CalculatorTool


@torch.no_grad()
def agent_generate(
    policy,
    tokenizer,
    messages: list[list[dict[str, str]]],
    *,
    temperature: float,
    max_new_tokens: int,
    max_rounds: int,
    template_kwargs: dict[str, Any],
    device: str,
    answers: list[dict[str, Any]] | None = None,
) -> list[Episode]:
    """`tool_generate`, with qwen-agent's `FnCallAgent` running each episode."""
    import threading
    import uuid

    CalculatorTool = _agent_classes()
    import qwen_agent.agents.fncall_agent as fncall_agent
    from qwen_agent.agents import FnCallAgent

    # One call per round plus the final answer, and never more: a turn past
    # `max_rounds` is closed by `_take_turn` before the agent could run its calls.
    fncall_agent.MAX_LLM_CALL_PER_RUN = max_rounds + 1
    run_id = uuid.uuid4().hex
    run = _AGENT_RUNS[run_id] = _AgentRun(
        policy, tokenizer, messages, temperature=temperature, max_new_tokens=max_new_tokens,
        max_rounds=max_rounds, template_kwargs=template_kwargs, device=device, answers=answers)
    errors: list[BaseException] = []

    def drive(i: int) -> None:
        try:
            system, *rest = messages[i]
            agent = FnCallAgent(
                function_list=[CalculatorTool()],
                llm={"model": "ppo-policy", "model_type": "ppo_policy", "run": run_id, "episode": i,
                     "generate_cfg": {"use_raw_api": True}},
                system_message=system["content"],
            )
            for _ in agent.run(messages=rest):
                pass
        except BaseException as exc:  # surfaced after join, never swallowed
            errors.append(exc)
        finally:
            run.batcher.finish(i)

    threads = [threading.Thread(target=drive, args=(i,), daemon=True) for i in range(len(messages))]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        _AGENT_RUNS.pop(run_id, None)
    if errors:
        raise errors[0]
    for ep in run.episodes:
        ep.text = tokenizer.decode(
            [t for t, p in zip(ep.ids, ep.policy) if p], skip_special_tokens=True)
    return run.episodes


def load_cases(split: str = "train") -> list[dict[str, Any]]:
    import json

    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines()]


# --- the value head -----------------------------------------------------------


class ValueHead(nn.Module):
    """A scalar read off the policy's own last hidden state.

    Kept in float32 regardless of the trunk's dtype. The head is the only part
    of this that is a regression rather than a classification, and bf16's
    ~3 decimal digits of mantissa are not enough to keep a squared error stable
    when the target sits in [0, 1] and the differences being learned are ~0.01.

    `normalize=True` standardises the features first (LayerNorm, no affine
    parameters). It is **off by default, because it was tested and did not
    help.** The argument for it was that the head reads raw hidden states whose
    L1 norm is set by a handful of enormous outlier dimensions -- Qwen3-1.7B's
    largest averages 87x its median -- so a few features decide the prediction
    and `lr` is hostage to them.

    Measured on real Qwen3-1.7B hidden states, fitting a linear functional of
    those states over 25 batches:

        normalised,     8 steps/batch    value_ev +0.947
        raw,            8 steps/batch    value_ev +0.993   <- the default
        raw,            1 step /batch    value_ev +0.345   <- what the first runs did

    Normalising is slightly *worse*, and there is a reason: LayerNorm discards
    each token's mean and scale, and on this task those carry signal.

    **Those three numbers have no held-out split, and the conclusion drawn from
    them was wrong.** With `gamma = lam = 1` and a terminal reward the return is
    constant along a sequence, so 25 batches of 8 is 200 *independent* targets
    against this head's 2049 parameters -- enough to interpolate. Re-measured in
    `07_the_critic_cannot_work.ipynb` on frozen rollouts, held out by sequence,
    the same fit gives **+0.031**; splitting by token instead (which leaks the
    target, since a sequence's tokens share it) gives +0.101, and no split at all
    gives +0.311. The "representation is not the constraint" claim these numbers
    were used to support does not survive a split, and `--no-value-detach` was
    chasing it.

    The flag stays so the input-scaling question can be re-checked on a different
    model rather than taken on my word.
    """

    def __init__(
        self,
        hidden_size: int,
        init_bias: float = VALUE_INIT_BIAS,
        *,
        normalize: bool = True,
        extra_features: int = 0,
    ):
        super().__init__()
        # Normalisation covers the hidden part only; the privileged one-hot is
        # already on a sane scale and LayerNorm across a concatenation of the two
        # would let the 2048 hidden dims set the statistics for all of it.
        self.hidden_size = hidden_size
        self.norm: nn.Module = (
            nn.LayerNorm(hidden_size, elementwise_affine=False, dtype=torch.float32)
            if normalize
            else nn.Identity()
        )
        self.v = nn.Linear(hidden_size + extra_features, 1, dtype=torch.float32)
        # Zero weight so V starts at exactly `init_bias` whether or not the
        # features are normalised -- the two configurations begin identically.
        nn.init.zeros_(self.v.weight)
        nn.init.constant_(self.v.bias, init_bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden[..., : self.hidden_size].float())
        want = self.v.in_features - self.hidden_size
        if want:
            extra = hidden[..., self.hidden_size :].float()
            if extra.shape[-1] != want:
                # `forward_policy_value` hands over the trunk's hidden states with
                # no privileged block attached. Those callers only need a value to
                # exist; the loop recomputes `old_values` on the full feature.
                extra = x.new_zeros((*x.shape[:-1], want))
            x = torch.cat([x, extra], dim=-1)
        return self.v(x).squeeze(-1)


# --- advantages ---------------------------------------------------------------


def terminal_rewards(rewards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Place each sequence's scalar reward on its own last active token.

    `rewards` is (batch,), `mask` is (batch, tokens). Returns (batch, tokens),
    zero everywhere except one position per row. A row with an empty mask -- a
    completion that was pure padding -- gets no reward anywhere rather than a
    reward at position -1, which is why the index is clamped and then masked
    again.
    """
    lengths = mask.sum(dim=-1).long()
    out = torch.zeros_like(mask)
    idx = (lengths - 1).clamp(min=0)
    out.scatter_(1, idx.unsqueeze(-1), rewards.unsqueeze(-1).to(out.dtype))
    return out * (lengths > 0).float().unsqueeze(-1)


#: The output fields in the order the schema lists them, paired with the regex
#: that finds the *end* of that field's value in a completion. `flags` is nested,
#: so its three keys are matched inside the flags object rather than in the whole
#: text -- "flow" also occurs inside "normalized_flow_change_pct".
_FIELD_PATTERNS: tuple[tuple[str, str], ...] = (
    ("normalized_flow_change_pct", r'"normalized_flow_change_pct"\s*:\s*(-?\d+(?:\.\d+)?)'),
    ("salt_passage_change_pct", r'"salt_passage_change_pct"\s*:\s*(-?\d+(?:\.\d+)?)'),
    ("dp_change_pct", r'"dp_change_pct"\s*:\s*(-?\d+(?:\.\d+)?)'),
    ("flags.flow", r'"flow"\s*:\s*"[a-z_]*"'),
    ("flags.salt_passage", r'"salt_passage"\s*:\s*"[a-z_]*"'),
    ("flags.dp", r'"dp"\s*:\s*"[a-z_]*"'),
    ("stage", r'"stage"\s*:\s*"[a-z_]*"'),
    ("root_cause", r'"root_cause"\s*:\s*"[a-z_]*"'),
    ("action", r'"action"\s*:\s*"[a-z_]*"'),
)


#: Multiplier on the credit for a flag whose *true* value is `flat`. See
#: `field_credits`. 1.0 reproduces the flat-blind behaviour of every run before
#: `09`.
FLAT_CREDIT_SCALE = 1.0


def field_credits(
    completion: str,
    answer: dict[str, Any],
    weights: Weights,
    flat_scale: float = FLAT_CREDIT_SCALE,
) -> tuple[dict[str, float], float]:
    """Split one completion's reward into per-field credits and a terminal rest.

    The pieces sum to `score(completion, answer, weights).total` exactly, so the
    objective the actor optimises is unchanged -- only *when* the credit arrives
    moves. `numeric` and `flags` are scored by `reward.py` as the mean of three
    hits, so each of the three carries a third of that component's weight.

    `format` is schema validity, which is not known until the object closes, and
    anything whose field cannot be located goes to the same place: the last
    active token, exactly where `terminal_rewards` would have put all of it.
    """
    import re

    from reward import _flag_hits, _numeric_hits
    from task.schema import FLAG_KEYS, NUMERIC_KEYS, parse_answer, validate

    parsed = parse_answer(completion)
    obj = parsed.obj if isinstance(parsed.obj, dict) else None
    if obj is None:
        return {}, 0.0

    numeric, flags = _numeric_hits(obj, answer), _flag_hits(obj, answer)
    # `09`: the policy emits `flat` on 0.5% of flag slots at temperature 1.0 and
    # `flat` is the right answer on 30.7% of them -- a 60x deficit, and the whole
    # of the remaining `flags_acc` gap (0.959 x 416/600 = 0.665 = the measured
    # value). Saying anything but `flat` is locally optimal, because the non-flat
    # values are 96% reliable, so the mode-seeking update suppresses the rare
    # correct answer further; `runs/v4-stopped-at-68-regressing` is that happening.
    #
    # This pays more for a `flat` slot the policy got right. It is *not* inverse
    # class frequency -- `flat` is already the most common true value, so standard
    # class balancing does nothing here. The imbalance is in the policy's prior,
    # not in the labels. Non-flat credit is left alone rather than scaled down:
    # the 96% is the part that works and there is no reason to make it cheaper.
    def _flag_credit(key: str, hit: bool) -> float:
        scale = flat_scale if answer["flags"][key] == "flat" else 1.0
        return weights.flags / 3 * scale * hit

    earned = {
        **{k: weights.numeric / 3 * hit for k, hit in zip(NUMERIC_KEYS, numeric)},
        **{f"flags.{k}": _flag_credit(k, hit) for k, hit in zip(FLAG_KEYS, flags)},
        "stage": weights.stage * (obj.get("stage") == answer["stage"]),
        "root_cause": weights.root_cause * (obj.get("root_cause") == answer["root_cause"]),
        "action": weights.action * (obj.get("action") == answer["action"]),
    }
    terminal = weights.format * float(validate(obj).ok)

    flags_span = re.search(r'"flags"\s*:\s*\{[^}]*\}', completion)
    placed: dict[str, float] = {}
    for field, pattern in _FIELD_PATTERNS:
        credit = float(earned.get(field, 0.0))
        if not credit:
            continue
        if field.startswith("flags."):
            if flags_span is None:
                terminal += credit
                continue
            hit = None
            for hit in re.finditer(pattern, flags_span.group()):
                pass
            if hit is None:
                terminal += credit
                continue
            placed[field] = flags_span.start() + hit.end()
        else:
            hit = None
            for hit in re.finditer(pattern, completion):
                pass
            if hit is None:
                terminal += credit
                continue
            placed[field] = hit.end()
        placed[field] = (placed[field], credit)

    return {f: v for f, v in placed.items()}, terminal


#: The three flag values, with the value itself captured so its *span* can be
#: located rather than just its end. `field_credits` deliberately keeps its own
#: end-of-match offsets: those are what every committed run was trained on.
_FLAG_VALUE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("flow", r'"flow"\s*:\s*"([a-z_]*)"'),
    ("salt_passage", r'"salt_passage"\s*:\s*"([a-z_]*)"'),
    ("dp", r'"dp"\s*:\s*"([a-z_]*)"'),
)


def flag_value_spans(completion: str) -> list[tuple[int, int]]:
    """Character spans of the three flag values inside the flags object."""
    import re

    block = re.search(r'"flags"\s*:\s*\{[^}]*\}', completion)
    if block is None:
        return []
    spans = []
    for _, pattern in _FLAG_VALUE_PATTERNS:
        hit = None
        for hit in re.finditer(pattern, block.group()):
            pass
        if hit is not None:
            spans.append((block.start() + hit.start(1), block.start() + hit.end(1)))
    return spans


def flag_token_mask(
    completions: list[str], completion_ids: torch.Tensor, mask: torch.Tensor, tokenizer
) -> torch.Tensor:
    """1.0 on the tokens that spell the three flag values, 0 elsewhere.

    `09` measured the failure this exists for. Held out on dev, the trained
    policy calls a non-flat flag correctly on 97% of slots that sit within 2-5 pp
    of a threshold -- it can do the comparison, and boundary proximity is not the
    problem -- while `flat` is **0 for 41, 0 for 87 and 0 for 56** across every
    distance bucket. The token is not produced at all, on easy cases and hard
    ones alike. That is a prior, not a capability.

    A reward that never sees the action cannot reweight it, which is why paying
    8x for a correct `flat` (`runs/v3-flat8-stopped-at-78`) moved `flat` recall
    0.000 -> 0.304 and cost non-flat 0.97 -> 0.84 in the same buckets: an 8x
    gradient monopolises the trust region. Credit was the wrong instrument, and
    the giveaway is that credit was already neutral -- every flag pays the same
    `weights.flags / 3` whether or not it is flat.

    The right instrument for an action that is never sampled is exploration, and
    the dense-credit machinery already localises *where* the choice is made. This
    mask is that: an entropy bonus can then be applied to the three decision
    tokens and nowhere else, which is what `entropy_coef` cannot do -- `05`
    measured the global coefficient blowing completion length up at 0.020.
    """
    out = torch.zeros_like(mask, dtype=torch.float32)
    lengths = mask.sum(dim=-1).long()
    for b, text in enumerate(completions):
        length = int(lengths[b])
        if length == 0:
            continue
        spans = flag_value_spans(text)
        if not spans:
            continue
        ids = completion_ids[b, :length].tolist()
        bounds = [len(tokenizer.decode(ids[: i + 1], skip_special_tokens=True)) for i in range(length)]
        for start, end in spans:
            for i, stop in enumerate(bounds):
                begin = bounds[i - 1] if i else 0
                if begin < end and stop > start:
                    out[b, i] = 1.0
    return out


def dense_rewards(
    completions: list[str],
    completion_ids: torch.Tensor,
    mask: torch.Tensor,
    cases: list[dict[str, Any]],
    weights: Weights,
    tokenizer,
    flat_scale: float = FLAT_CREDIT_SCALE,
    call_hits: list[dict[str, int]] | None = None,
) -> torch.Tensor:
    """Per-token rewards that land where each field is written, not at the end.

    Why this exists. With `gamma = lam = 1` and one terminal reward the return is
    constant along a sequence, so `V(s_t)` has nothing to track and the per-token
    credit assignment actor-critic PPO offers has nothing to assign -- which is
    the finding `07` closes on. Paying each field at the token that decides it
    makes the return fall as the answer is written, which is the one thing that
    gives a causal value function a job. It only bites with `lam < 1`; at
    `lam = 1` GAE ignores `V` and this changes nothing but the metrics.

    Row sums equal `Rollout.rewards` up to float error, so the sequence-level
    objective is untouched.

    `call_hits` (tool episodes, `13`) moves the numeric credit earlier still: a
    field's third of `weights.numeric` is paid on the `</tool_call>` of the call
    whose result matched it, as the harness judged when that call returned, and
    *settled* where the JSON writes the field -- plus the copy's own credit,
    minus what the call was already paid. A right call copied wrongly nets zero;
    a right number never computed by a call is paid at the copy, as before. The
    row sum is still exactly `score(...).total`.
    """
    out = torch.zeros_like(mask, dtype=torch.float32)
    lengths = mask.sum(dim=-1).long()
    for b, (text, case) in enumerate(zip(completions, cases)):
        length = int(lengths[b])
        if length == 0:
            continue
        ids = completion_ids[b, :length].tolist()
        # Cumulative decoded length after each token, so a character offset can
        # be turned into the index of the token that completed it. Prefix decodes
        # rather than per-token decodes: byte-level BPE splits multi-byte
        # characters across tokens, and their lengths do not add up.
        bounds = [len(tokenizer.decode(ids[: i + 1], skip_special_tokens=True)) for i in range(length)]

        try:
            placed, terminal = field_credits(text, case["answer"], weights, flat_scale)
        except (OverflowError, ValueError, TypeError, KeyError):
            placed, terminal = {}, 0.0

        for offset, credit in placed.values():
            index = next((i for i, end in enumerate(bounds) if end >= offset), length - 1)
            out[b, index] += credit

        if call_hits is not None and call_hits[b]:
            import re

            unit = weights.numeric / 3
            for key, pattern in _FIELD_PATTERNS[:3]:
                if key not in call_hits[b]:
                    continue
                out[b, min(call_hits[b][key], length - 1)] += unit
                copy = None
                for copy in re.finditer(pattern, text):
                    pass
                if copy is None:
                    terminal -= unit
                else:
                    index = next((i for i, end in enumerate(bounds) if end >= copy.end()), length - 1)
                    out[b, index] -= unit
        out[b, length - 1] += terminal
    return out


def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    gamma: float = 1.0,
    lam: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over the completion span.

    Shapes are all (batch, tokens). `rewards` is the per-token reward from
    `terminal_rewards`; `values` is `V(s_t)`, the value of the state *before*
    token t, so it lines up index-for-index with the log-probs.

    Returns `(advantages, returns)`, both zero outside the mask. `returns` is
    the GAE return `A_t + V(s_t)`, which is what the critic regresses against.

    The bootstrap past the last active token is zero, not `V(s_L)`: the episode
    genuinely ends there. Bootstrapping off a padded position would let the
    critic's own error leak in as if it were future reward.
    """
    batch, tokens = mask.shape
    advantages = torch.zeros_like(values)
    running = torch.zeros(batch, dtype=values.dtype, device=values.device)

    for t in range(tokens - 1, -1, -1):
        m = mask[:, t]
        # Zero past the end of the span, so V(s_{t+1}) is 0 on the last token.
        next_value = values[:, t + 1] * mask[:, t + 1] if t + 1 < tokens else torch.zeros_like(running)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        running = delta + gamma * lam * running
        # An inactive position contributes nothing and must not carry the
        # accumulator backwards across the padding boundary either.
        running = running * m
        advantages[:, t] = running

    return advantages * mask, (advantages + values) * mask


def critic_fit(
    values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    """How much of the return the critic actually explains, over one batch.

    `value_mean` and `value_mae` cannot tell a working critic from one that has
    collapsed to a constant, and the 0.5B runs in `runs/` are the demonstration:
    on `ppo-ac-ie2-s0` the learned value function posted a respectable MAE of
    0.089 and an **explained variance of +0.07** -- worse than predicting the
    mean of the returns, which is what a critic exists to beat. `value_std` says
    the same thing from the other side: on 125 steps of `gonogo-stage-s0` where
    every sequence in the batch drew the identical reward, V had mean 0.95 and a
    within-batch spread of 0.065, so V(s) was barely a function of s at all.

    Computed over the whole batch from the values the advantages were actually
    built from, not per micro-batch -- the per-micro-batch stats in `value_loss`
    are overwritten by whichever slice happens to run last.
    """
    sel = mask > 0
    v, g = values[sel], returns[sel]
    if v.numel() < 2:
        return {"value_ev": None, "value_std": 0.0}
    std = round(float(v.std(unbiased=False)), 6)
    g_var = float(g.var(unbiased=False))
    # When every sequence in a batch draws nearly the same reward the returns have
    # almost no variance, and 1 - var(G-V)/var(G) divides by roughly zero. A
    # `g_var > 0` guard is not enough: 1e-14 passes it, and one step of the first
    # fixed run recorded -1e13 that way. That is the metric being *undefined* on
    # that batch, not the critic failing on it, and the two must not be confused --
    # explained variance is bounded above by 1 but unbounded below, so a genuine
    # -3 is a genuinely bad critic and has to survive the guard.
    #
    # 15% of batches were degenerate this way over 200 steps, which is a fact
    # about the task worth reporting rather than filtering into silence. Hence
    # None, which JSON writes as null, rather than a plausible-looking 0.0.
    if g_var < 1e-6:
        return {"value_ev": None, "value_std": std, "value_ev_position": None}
    # With `dense_rewards` the return falls as the answer is written, so a head
    # that learned nothing but "how far along am I" would already score well.
    # This is that head, fitted for free: the mean return at each *relative*
    # position, in twenty bins, scored on the same batch. `value_ev` above it is
    # the part of the critic that is reading the sequence rather than counting
    # tokens; `value_ev` at or below it means the critic is a clock.
    position = torch.zeros_like(returns)
    index = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0).expand_as(mask)
    lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    binned = (index / lengths * 20).floor().clamp(max=19).long()
    for b in range(20):
        hit = (binned == b) & sel
        if hit.any():
            position[hit] = returns[hit].mean()
    ev_pos = round(1.0 - float((g - position[sel]).var(unbiased=False)) / g_var, 6)
    return {
        "value_ev": round(1.0 - float((g - v).var(unbiased=False)) / g_var, 6),
        "value_std": std,
        "value_ev_position": ev_pos,
    }


def whiten(advantages: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero-mean, unit-variance over active tokens only. Off by default."""
    active = mask.sum().clamp(min=1.0)
    mean = (advantages * mask).sum() / active
    var = (((advantages - mean) * mask) ** 2).sum() / active
    return ((advantages - mean) / (var.sqrt() + eps)) * mask


# --- the losses ---------------------------------------------------------------


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of the policy's next-token distribution, per position.

    `H = logsumexp(z) - sum(softmax(z) * z)`, in float32. This is the honest
    distribution entropy, and it is a *different quantity* from the
    `entropy_proxy` the diagnostics already carry: that one is `-logp` of the
    single token that was sampled, a one-sample estimate of the cross entropy.
    The proxy is fine as a monotone collapse detector and costs nothing; a bonus
    term in the loss needs the real thing.

    Materialises one logits-sized tensor for `softmax(z)`, kept for the backward
    pass, so `forward_policy_value` computes it only when it is asked to -- never
    in the old-value no-grad pass. It is built over the whole `logits_to_keep`
    span, not the active mask, so at Qwen3-1.7B's 151,936-token vocabulary and a
    640-token generation window that is ~390 MiB in float32 at micro_batch=1.
    """
    logits = logits.float()
    logz = torch.logsumexp(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return logz - (probs * logits).sum(dim=-1)


def ppo_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    normalize: str = "token",
    entropy: torch.Tensor | None = None,
    entropy_coef: float = 0.0,
    focus_mask: torch.Tensor | None = None,
    focus_entropy_coef: float = 0.0,
    entropy_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The clipped surrogate, with a per-token advantage.

    Differs from `grpo_surrogate` in exactly one respect: `advantages` is
    (batch, tokens) here and (batch,) there. The clipping, the normalisation and
    the diagnostics are deliberately identical so the two runs' numbers can be
    read against each other.

    `entropy` is the per-token `token_entropy` of the *current* policy, passed in
    when `entropy_coef > 0`. It enters as `-entropy_coef * mean(H)`, the standard
    PPO entropy bonus -- a reward-free force that resists the near-greedy
    collapse `inner_epochs > 1` drives here. `entropy_coef=0.0` reproduces every
    run in `runs/` exactly: the term is not merely small, it is absent.
    """
    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    objective = torch.min(unclipped, clipped)

    per_token = -objective * mask
    if normalize == "sequence":
        lengths = mask.sum(dim=-1).clamp(min=1.0)
        loss = (per_token.sum(dim=-1) / lengths).mean()
    elif normalize == "token":
        loss = per_token.sum() / mask.sum().clamp(min=1.0)
    else:
        raise ValueError(f"unknown normalize={normalize!r}")

    active = mask.sum().clamp(min=1.0)
    surrogate = float(loss.detach())  # the clipped surrogate alone, as in every prior run
    entropy_mean = None
    if entropy is not None:
        if entropy_mask is None:
            entropy_mean = (entropy * mask).sum() / active
        else:
            # `12`'s 200-step run: the bonus, averaged over a sequence whose calls
            # are 72% of it, pushed on tokens the SFT had made near-deterministic
            # and no reward lands on. Scoped, the call tokens drop out of the sum
            # and the denominator stays every active token, so each answer token
            # is pushed exactly as hard as it was unscoped -- one change, not two.
            # (`13b`'s first scoped run divided by the answer tokens instead, a
            # 3.7x stronger push on them, and its answer entropy doubled in 25
            # steps; it was stopped at step 50.)
            entropy_mean = (entropy * entropy_mask * mask).sum() / active
        if entropy_coef:
            loss = loss - entropy_coef * entropy_mean
        if focus_entropy_coef and focus_mask is not None:
            # The same bonus, restricted to the tokens `focus_mask` marks -- the
            # three flag values. Normalised over those tokens only, so the
            # coefficient means the same thing whatever fraction of the sequence
            # they are (about 3 of 95).
            focused = focus_mask * mask
            focus_active = focused.sum().clamp(min=1.0)
            focus_entropy = (entropy * focused).sum() / focus_active
            loss = loss - focus_entropy_coef * focus_entropy

    with torch.no_grad():
        binding = ((unclipped > clipped) & (mask > 0)).float().sum() / active
        stats = {
            "actor_loss": surrogate,
            "ratio_mean": float((ratio * mask).sum() / active),
            "clip_frac": float(binding),
            "adv_mean": float((advantages * mask).sum() / active),
            "adv_std": float(
                (((advantages - (advantages * mask).sum() / active) * mask) ** 2).sum().div(active).sqrt()
            ),
            "entropy_proxy": float(-(logprobs * mask).sum() / active),
        }
        if entropy_mean is not None:
            stats["policy_entropy"] = float(entropy_mean)
            if entropy_mask is not None:
                scoped = entropy_mask * mask
                stats["answer_entropy"] = float((entropy * scoped).sum() / scoped.sum().clamp(min=1.0))
        if focus_mask is not None and entropy is not None:
            focused = focus_mask * mask
            stats["flag_entropy"] = float(
                (entropy * focused).sum() / focused.sum().clamp(min=1.0)
            )
            stats["flag_tokens"] = float(focused.sum() / mask.shape[0])
    return loss, stats


def value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_eps: float | None = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped squared error, the standard PPO critic loss.

    The clip is a trust region for the *critic*: the value is not allowed to
    move more than `clip_eps` from the estimate the batch was collected under,
    which stops one unlucky batch from dragging the baseline somewhere the
    advantages of the next batch cannot recover from. Taking the `max` of the
    two errors -- rather than the `min`, as the actor does -- is what makes it a
    penalty instead of a licence.

    `clip_eps=None` gives plain MSE, which is the honest thing to compare
    against and is what `01_actor_critic_and_gae.ipynb` checks the clip against.
    """
    active = mask.sum().clamp(min=1.0)
    plain = (values - returns) ** 2
    if clip_eps is None:
        loss = (plain * mask).sum() / active
        clipped_frac = 0.0
    else:
        moved = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps)
        clipped_err = (moved - returns) ** 2
        both = torch.max(plain, clipped_err)
        loss = (both * mask).sum() / active
        with torch.no_grad():
            clipped_frac = float(((clipped_err > plain) & (mask > 0)).float().sum() / active)

    with torch.no_grad():
        err = (values - returns) * mask
        stats = {
            "value_loss": float(loss.detach()),
            "value_mean": float((values * mask).sum() / active),
            "return_mean": float((returns * mask).sum() / active),
            "value_mae": float(err.abs().sum() / active),
            "value_clip_frac": clipped_frac,
        }
    return loss, stats


@dataclass
class Config:
    """Defaults chosen to sit alongside `grpo_scratch.Config`, not to beat it."""

    # Qwen3-1.7B, chosen on the GRPO side by measurement rather than argument
    # (`membrane_grpo/runs/sel-*`). The reason that matters here: its schema
    # validity is already 0.970, so a reward rise cannot be explained away as
    # format learning -- which is exactly what 52% of the 0.5B's gain turned out
    # to be. It also starts at cause_acc 0.255 against a 1/7 = 0.143 chance
    # floor, so there is something partial for RL to sharpen; the 0.5B sat
    # *exactly* at chance and had nothing.
    model: str = "Qwen/Qwen3-1.7B"
    steps: int = 200
    prompts_per_step: int = 8
    # 1 is the point: unlike GRPO, the algorithm needs no second sample of the
    # same prompt. Raise it only to fill the batch, never for the baseline.
    samples_per_prompt: int = 1
    max_new_tokens: int = 640
    temperature: float = 1.0
    lr: float = 1e-5
    # Derived, not guessed, and re-derived for this model rather than carried
    # over. Adam moves every one of the head's `hidden_size` weights by about
    # `lr` per step, and those steps are summed through the features, so the
    # value itself moves by roughly `lr * ||h||_1`. On the 0.5B that meant
    # lr=1e-3 moved V by ~4.2 against a reward living in [0, 1], and the first
    # CPU smoke run did exactly that: V went 0.086 -> -1.868 in one step.
    #
    # The constant does not survive the model change, and not in the direction
    # the parameter count suggests. Measured in `01_actor_critic_and_gae.ipynb`:
    #
    #     Qwen2.5-0.5B   hidden  896   E|h| 4.824   ||h||_1 4323
    #     Qwen3-1.7B     hidden 2048   E|h| 1.127   ||h||_1 2308
    #
    # 2.3x the dimensions but 4.3x smaller per-dimension activations -- Qwen3
    # simply does not carry the huge outlier features the 0.5B does -- so
    # `||h||_1` *falls* by 47%. Keeping 5e-6 would have quietly cut the critic's
    # step from 0.022 to 0.012 on a model where the critic was already the
    # bottleneck. 8.5e-6 puts it at 0.020, which is the target the old default
    # was chosen for; `value_step` in the metrics confirms it per run.
    #
    # That target is itself the open question, and it is deliberately *not*
    # touched here so that the model is the only thing that changed. The 0.5B
    # runs moved V by a median of 0.004-0.009 per step against a regression
    # target whose own batch-to-batch swing reaches 1.0 -- a ~100-step time
    # constant chasing a signal that changes every step, which is why
    # `value_ev` came out at +0.07. `value_ev` is now in the metrics so the
    # next round argues from a number instead of from a reward curve.
    # Raised from 8.5e-6, but this is the smaller of the two critic changes and
    # the derivation is no longer what sets it. Measured on real Qwen3-1.7B
    # hidden states over 25 batches at 8 steps each: 8.5e-6 reaches value_ev
    # +0.950 and 3e-5 reaches +0.993, so the higher rate helps but the step
    # count is what matters (1 step per batch reaches only +0.345 at either
    # rate). `value_clip_eps` caps the per-batch movement at 0.2 regardless.
    value_lr: float = 3e-5
    weight_decay: float = 0.0  # same reasoning as grpo_scratch: no reward-free force
    gamma: float = 1.0
    lam: float = 1.0
    clip_eps: float = 0.2
    value_clip_eps: float | None = 0.2
    value_coef: float = 0.5
    # Entropy bonus on the policy's next-token distribution, added to the actor
    # objective as `-entropy_coef * mean(H)`. 0.0 by default, which is not a
    # small term but no term -- every run in `runs/` was trained without it, and
    # `weight_decay` carries the same "no reward-free force" reasoning. It is on
    # the table because with `inner_epochs=4` the entropy proxy fell 0.169 ->
    # 0.073 over 200 steps on MAIN while `clip_frac` held near 0.008: four
    # unrestrained updates per batch and nothing resisting a near-greedy
    # collapse. `03_critic_and_trust_region.ipynb` runs a nonzero value here
    # against lowering `inner_epochs` as the two candidate levers.
    entropy_coef: float = 0.0
    value_detach: bool = True
    # Off, because it was tested and did not help -- see `ValueHead` for the
    # measurement. Kept as a flag rather than deleted so the claim is falsifiable
    # on another model.
    normalize_value: bool = False
    # Dedicated critic steps per rollout batch, on the hidden states cached
    # during the old-value pass. The critic is `hidden_size + 1` parameters on
    # features that are already computed and detached, so these cost one small
    # matmul each and no forward through the trunk at all. This is the knob the
    # first two runs did not have: the critic got exactly `inner_epochs` updates
    # per batch, which at this learning rate is a ~100-step time constant
    # chasing a target that changes every step.
    critic_epochs: int = 8
    # How many recent rollout batches the critic fits over. 1 reproduces every run
    # in `runs/`: the head sees one batch, and with `gamma = lam = 1.0` and the
    # reward on the last active token only, `returns[b, t] = R_b` for every t --
    # so one batch is `prompts_per_step` distinct target values (8) replicated
    # across ~770 masked positions. `critic_fit` pools tokens, so the token axis
    # inflates n without adding target variance, and a regressor shown 8 numbers
    # per step whose mean moves every step converges to the running mean. That is
    # what the metrics say happened: `value_mean` tracks `return_mean` to three
    # decimals with a spread of 0.04, and `value_ev` came out *negative*.
    #
    # The representation is not the constraint. `ValueHead`'s docstring records
    # this same head, on these same real hidden states, at these same 8 steps per
    # batch, reaching `value_ev` +0.993 offline -- the difference is that the
    # offline fit accumulates across batches. 25 restores that condition in the
    # loop: 200 distinct targets rather than 8.
    #
    # Only the active positions are kept, so the window is flat (n_active, hidden)
    # and batches of different completion lengths concatenate without padding.
    # ~3 MiB per batch at the observed 96-token mean, 21 MiB at the 640 cap.
    # Which hidden layer the value head reads. -1 (the last) is what every run in
    # `runs/` used. Measured on frozen rollouts, 48 prompts x 8 samples, held out
    # by prompt against the prompt's mean reward:
    #
    #     layer 28 (-1, the default)   test EV  +0.263
    #     layer 27 (-2)                test EV  +0.456
    #     layer 20                     test EV  +0.398
    #     layer 14                     test EV  +0.106
    #     layer  0                     test EV  +0.001
    #
    # The last layer is specialised for predicting the next token. The reward is
    # a property of the finished answer, and the feature that carries it peaks
    # one layer earlier.
    # Asymmetric actor-critic: give the value head the case's true `root_cause`
    # as a one-hot alongside the hidden state. The actor never sees it.
    #
    # This is unbiased, and the reason is the standard one -- a baseline that is
    # any function of the *state* leaves the policy gradient's expectation alone,
    # because E[grad log pi(a|s) b(s)] = 0. The label is determined by the prompt,
    # so it is part of s and not of a. Sim-to-real robotics uses exactly this
    # (the critic reads privileged simulator state the actor cannot observe).
    #
    # It is here because `07` measured what the reward actually depends on:
    # regressing the prompt's mean reward on the task's *designed* difficulty
    # variables (`tier`, `margin_pp`, `severe`) gives leave-one-out R^2 = -0.081,
    # while regressing it on `root_cause` alone gives **+0.902**. A prompt is easy
    # exactly when its answer is one of the labels the policy already emits. So
    # the one thing that would let V(s) beat a constant is the label -- which the
    # actor must not see, and the critic may.
    privileged_cause: bool = False
    value_layer: int = -1
    #: Hand the model the three percent changes instead of demanding them.
    #: `07`'s closing section is the argument: the arithmetic is a capability
    #: this model does not have, at the level of elementary decimal arithmetic
    #: rather than the domain formula, so a reward that pays for it pays for
    #: nothing. The readings are already structured, so the harness computes the
    #: changes exactly and the model does steps 2-4. See `build_messages_v3`.
    prompt_v3: bool = False

    #: v4 implies v3, and additionally restates Step 2 with the flat band as its
    #: own leading condition. See `V4_STEP2` for the measurement that motivates it.
    prompt_v4: bool = False

    #: Multiplier on the dense credit for a correctly-called `flat` flag. Only
    #: meaningful with `dense_rewards`. 1.0 is every run before `09`.
    flat_credit_scale: float = 1.0

    #: v2's prompt, unchanged and with nothing precomputed, plus a calculator the
    #: policy may call (`CALCULATOR_TOOL`). Replaces v3 rather than stacking on
    #: it: with this on, `prompt_v3` / `prompt_v4` are ignored.
    prompt_tool: bool = False
    #: How many times the tool may answer within one episode.
    tool_rounds: int = 2
    #: Steps at the start of a run on which the critic is fitted and the actor is
    #: not stepped. `13a`'s first run collapsed because the policy moved before
    #: `V` had learned what a new reward placement does to the return; this lets
    #: the critic see it first. 0 is every run before `13`.
    critic_warmup: int = 0
    #: Where the entropy bonus applies on a tool episode: "all" policy tokens (every
    #: run before `13`) or only the "answer" turn. See `ppo_actor_loss`.
    entropy_scope: str = "all"
    #: Stop the run (saving `adapter-stopped/`) once the sampled `numeric_acc`,
    #: averaged over the last 10 steps, falls below this. 0 disables it. The
    #: arithmetic has to hold every step; a run that has lost it is not worth
    #: continuing, and the policy at the point of failure is worth keeping.
    numeric_stop: float = 0.0
    #: Who orchestrates a tool episode: "native" (`tool_generate`) or
    #: "qwen_agent" (`agent_generate`, `13b`). Same decoding either way.
    harness: str = "native"
    #: `13`: pay the numeric credit on the call that computed it, as the harness
    #: judges each call on return, settled at the JSON copy. Needs `prompt_tool`
    #: and `dense_rewards`. See `dense_rewards`.
    call_credit: bool = False
    #: SFT on `tool_call_target` instead of an answer: the loss covers the three
    #: calls and nothing after them. Uses every case of `split`, shuffled.
    sft_tool: bool = False
    #: With `resume_from`: load the adapter but not the value head, which starts
    #: fresh at `value_init_bias`. For a seed whose head was never trained -- an
    #: SFT pass does not touch it -- the saved head is only a bias, and that bias
    #: was resolved for whatever prompt the seed was run under, not for the
    #: policy PPO is about to start from.
    value_head_reset: bool = False
    #: No training: one greedy tool pass over `eval_split`, written to `--out`
    #: in `runs/paired/`'s envelope, with every episode beside it.
    eval_only: int = 0

    #: A finished run directory to continue from -- its `adapter/` and
    #: `value_head.pt` are loaded instead of starting from the frozen model. The
    #: step counter restarts at 0, so `eval.jsonl` in the new directory is the
    #: continuation's own curve and has to be read with the offset in mind.
    resume_from: str = ""

    #: SFT seeding. `10`'s finding: two output labels have *exactly* zero
    #: probability at temperature 1.0, in the frozen policy and in the 575-step
    #: PPO policy alike -- `organic_fouling` 0/600 and the severe action 0/600.
    #: A policy gradient reweights behaviour that appears in a sample, so zero is
    #: the one number it cannot move, and `09` said so before this run hit it.
    #: These seed the labels by maximum likelihood on a *supplied* target, which
    #: needs no sample of it. `sft_seed_from` is an adapter directory.
    sft_seed_from: str = ""
    sft_epochs: int = 1
    sft_lr: float = 1e-5
    #: Normal cases per dead-label case in the seed set. 0.0 would seed only the
    #: dead labels, which teaches "say organic_fouling" rather than "this label
    #: exists here" -- see `build_sft_examples`.
    sft_balance: float = 1.0
    #: Run the seeding instead of a PPO run and exit.
    sft_only: int = 0
    #: A *seed*, in the sense the argument for it actually requires: the loss is
    #: masked to the tokens spelling the two dead label values and nothing else,
    #: so the pass moves those strings off zero probability and supervises no
    #: other field. Without this the pass is a full supervised fine-tune of the
    #: whole answer -- which works, but answers a different question, and cannot
    #: be described as "PPO learned the task".
    #: 0 = full-answer supervision. 1 = the dead label strings only (collapses
    #: the `action` slot -- see `slot_value_spans`). 2 = the `root_cause` and
    #: `action` values on every seed case, which is the one that holds.
    sft_labels_only: int = 0
    #: `11`'s axis. `sft_labels_only = 2` supervises both lookup slots on every
    #: seed case, which keeps the slot's alternatives represented but leaves
    #: `organic_fouling` at 0/600 -- the prior is simply stronger than 57 cases
    #: of evidence. `sft_labels_only = 1` supervises the dead values alone and
    #: collapses the `action` slot to 600/600. This weights the loss of a
    #: dead-label case by `w` under mode 2, so 1.0 is mode 2 and the limit is
    #: mode 1: the two failures of `10` are the two ends of one axis.
    sft_dead_weight: float = 1.0
    #: A JSONL of cases to seed from, instead of a slice of `train.jsonl`. This
    #: is the whole point of `11`: a seed drawn from the task's own training
    #: split cannot be told apart from supervised fine-tuning on the task, so
    #: "the seed only installed the vocabulary" stays an assertion. These records
    #: are off-distribution by plant scale -- flows, conductivities, pressures,
    #: temperatures and recoveries outside every band the four splits occupy --
    #: while the decision table, which reads percentage *changes*, applies
    #: unchanged. Whether the labels then transfer to real dev cases is then a
    #: measurement rather than a story. Composition is taken as given: these
    #: files are built to a brief, not sampled.
    sft_cases: str = ""

    #: An entropy bonus applied to the three flag-value tokens and nowhere else.
    #: See `flag_token_mask` for the measurement that motivates it. 0.0 is every
    #: run before `09`; it is independent of `entropy_coef`, which stays global.
    flag_entropy_coef: float = 0.0

    #: Pay each output field at the token that decides it, instead of paying the
    #: whole sequence reward at the last one. `07` closes on why this matters:
    #: with `gamma = lam = 1` and a terminal reward the return is constant along
    #: a sequence, so `V(s_t)` has nothing to track and the per-token credit
    #: assignment this method offers over GRPO has nothing to assign. Row sums
    #: are unchanged, so the sequence-level objective is the same one; only the
    #: timing of the credit moves. **It only bites with `lam < 1`** -- at
    #: `lam = 1` GAE never consults `V` and this changes the metrics and nothing
    #: else. See `dense_rewards`.
    dense_rewards: bool = False

    critic_window: int = 1
    # Rebuild the advantages from the critic's *post-fit* values. Off reproduces
    # every existing run, where `fitted_values` was computed and then used only
    # for metrics (`value_ev_fit`) -- the actor trained on advantages built from
    # `old_values`, so the critic's dedicated steps did nothing for the batch that
    # paid for them and only reached the actor through the next batch.
    #
    # Andrychowicz et al. 2020 (C5) is the one improvement the paper proposes
    # itself: stale advantages hurt, and recomputing them at the start of each
    # pass beat every other variant they tried. It is one line here because
    # `lam = 1.0` makes `returns` the plain Monte-Carlo return, independent of V.
    recompute_advantages: bool = False
    whiten_advantages: bool = False
    # -1.0 means "look it up in FROZEN_BASELINE for this model and weight set"
    # and is resolved in __post_init__, so config.json records the number that
    # was actually used. 0.0 still reproduces the documented failure on purpose.
    value_init_bias: float = -1.0
    # 4, not 1. With one inner epoch the only gradient pass has logp == logp_old
    # by construction, so rho is identically 1 and the clipped surrogate reduces
    # to plain `A_t` -- `ratio_mean` was exactly 1.000000000 and `clip_frac`
    # exactly 0.0 at all 400 steps of the first two runs. That is not PPO, it is
    # vanilla policy gradient with a baseline. From the second pass onward the
    # policy has moved off the sampling policy and the trust region has
    # something to do; `clip_frac` is the metric that says whether it did.
    inner_epochs: int = 4
    # 1, not 2: 1.7B in bf16 is 3.4 GiB of weights before activations, and the
    # value head needs `output_hidden_states=True`, which keeps all 29 layers'
    # hidden states alive through the backward pass. Same choice GRPO made when
    # it moved to this model.
    micro_batch: int = 1
    normalize: str = "token"
    lora_r: int = 16
    weights: str = "MAIN"
    seed: int = 0
    dtype: str = "bfloat16"
    split: str = "train"
    prompt_version: str = PROMPT_VERSION
    # Held-out evaluation during training, taken from `grpo_scratch.Config`.
    # The reason to have it is on this side of the fence, not GRPO's: this
    # method has already produced a run (`runs/gonogo-stage-s0`) whose *training*
    # reward sat at 1.0000 for 30 steps while the policy it was checkpointing
    # was worth 0.000 on dev. A reward curve alone cannot tell learning from
    # collapse, and `adapter-best` picked the wreck.
    eval_every: int = 25
    # The full dev split, so step 0 is directly comparable to the frozen
    # baseline rather than to a subset of it with a different denominator.
    eval_cases: int = 200
    eval_batch: int = 32
    eval_split: str = "dev"
    eval_max_tokens: int = 640
    metrics: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.weights not in WEIGHT_SETS:
            raise ValueError(f"unknown weights {self.weights!r}; have {sorted(WEIGHT_SETS)}")
        if self.critic_epochs < 1:
            raise ValueError("critic_epochs must be >= 1")
        if self.critic_window < 1:
            raise ValueError("critic_window must be >= 1")
        # A negative value means "no value clipping", the same sentinel convention
        # `value_init_bias` uses, and for the same reason: the generated CLI types
        # every flag from its default, so there is no way to pass None. It must not
        # be spelled 0.0 -- that clamps the critic's movement to nothing, which is
        # strictly worse than clipping at 0.2 rather than equivalent to disabling it.
        if self.value_clip_eps is not None and self.value_clip_eps < 0:
            self.value_clip_eps = None
        if self.value_init_bias < 0:
            if self.prompt_tool:
                table = FROZEN_BASELINE_TOOL
            elif self.prompt_v3 or self.prompt_v4:
                table = FROZEN_BASELINE_V3
            else:
                table = FROZEN_BASELINE
            self.value_init_bias = table.get(self.model, {}).get(self.weights, 0.0)


# --- the model side -----------------------------------------------------------
#
# One seam here that GRPO never meets. A GRPO group is G samples of a *single*
# prompt, so every sequence in the batch has the same prompt length and there is
# no padding to get wrong. This method takes one sample each from many different
# prompts, so the batch is left-padded -- and a left-padded batch fed through a
# plain `model(input_ids=...)` is silently wrong twice over: the model attends to
# the pad tokens, and RoPE reads position 0 at the pad rather than at the first
# real token. Both are fixed below, and `01_actor_critic_and_gae.ipynb` checks
# the fix by scoring the same sequence padded and unpadded and comparing.


def padded_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Positions that start at 0 on the first *real* token of each row."""
    return (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)


def forward_policy_value(
    model,
    value_head,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_len: int,
    *,
    value_detach: bool = True,
    return_hidden: bool = False,
    return_entropy: bool = False,
    value_layer: int = -1,
) -> tuple[torch.Tensor, ...]:
    """Per-token log-probs and values over the completion span, in one forward.

    `return_hidden` also hands back the (detached) hidden states the value head
    read. Caching those is what makes extra critic steps free: the critic is a
    linear probe on features that have already been computed, so fitting it
    again needs no second pass through the trunk.

    `return_entropy` hands back the per-token `token_entropy` instead, computed
    from the same logits slice that produced `logprobs`. The two flags are not
    combined -- `return_hidden` is the no-grad old-value pass, `return_entropy`
    the actor pass -- and asking for both is a bug.

    `logits_to_keep` restricts the (batch, positions, 151936) output head to the
    span that is actually scored -- the single largest tensor in the step, and
    what exhausted the card on the GRPO run's first attempt. The hidden states
    come back full length regardless, so `V(s_t)` is sliced from them at the
    same positions whose logits produced token t.
    """
    if return_hidden and return_entropy:
        raise ValueError("return_hidden and return_entropy are not combined")
    out = model(
        input_ids=sequences,
        attention_mask=attention_mask,
        position_ids=padded_position_ids(attention_mask),
        logits_to_keep=completion_len + 1,
        output_hidden_states=True,
    )
    logprobs = selective_logprobs(out.logits[:, :-1], sequences[:, -completion_len:])
    entropy = token_entropy(out.logits[:, :-1]) if return_entropy else None

    # -1 is the last layer, which is what every run in `runs/` used and what the
    # critic was measured on. It is not the best layer for this, and that was
    # never checked: probing all 29 on frozen rollouts, held out *by prompt*
    # against the prompt's mean reward, the last layer explains +0.263 of the
    # variance and layer 27 explains **+0.456**. The last layer is specialised
    # for next-token prediction; the reward is a property of the whole answer,
    # and the representation that carries it best sits a little earlier.
    hidden = out.hidden_states[value_layer]
    if hidden.shape[1] == sequences.shape[1]:
        hidden = hidden[:, -(completion_len + 1) : -1]
    else:  # a transformers version that also truncates the hidden states
        hidden = hidden[:, :-1]
    detached = hidden.detach()
    values = value_head(detached if value_detach else hidden)
    if return_hidden:
        return logprobs, values, detached
    if return_entropy:
        return logprobs, values, entropy
    return logprobs, values


def load_policy(cfg: "Config", device: str):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=getattr(torch, cfg.dtype))
    policy = get_peft_model(
        base,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=2 * cfg.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    ).to(device)
    value_head = ValueHead(
        base.config.hidden_size, cfg.value_init_bias, normalize=cfg.normalize_value,
        extra_features=len(CAUSES) if cfg.privileged_cause else 0,
    ).to(device)

    if cfg.resume_from:
        # Continue an existing run rather than start one. `09` is the reason this
        # exists: a 400-step run's `flags_acc` was still climbing at its last
        # three evaluation points (0.883 -> 0.900 -> 0.927), and re-running with
        # more steps is *not* a continuation -- the same command diverges from
        # itself at step 3, because bf16 matmuls are not bitwise deterministic
        # and one flipped logit at `temperature = 1.0` is enough. Extending a run
        # therefore has to start from its weights, not from its seed.
        #
        # Both halves are restored. The value head matters as much as the policy:
        # `critic_window` means it is fitted over the last K batches, and a head
        # re-initialised to `value_init_bias` would spend the first K steps of the
        # continuation relearning what it already knew.
        source = Path(cfg.resume_from)
        policy.load_adapter(str(source / "adapter"), adapter_name="default")
        head_state = source / "value_head.pt"
        if cfg.value_head_reset:
            pass  # the head keeps the fresh init at cfg.value_init_bias
        elif head_state.exists():
            value_head.load_state_dict(torch.load(head_state, map_location=device))
        else:  # pragma: no cover - a run killed before it saved
            raise FileNotFoundError(f"no value_head.pt under {source}")

    return policy, value_head, tokenizer


@dataclass
class Rollout:
    cases: list[dict[str, Any]]
    sequences: torch.Tensor
    attention_mask: torch.Tensor
    mask: torch.Tensor
    completions: list[str]
    rewards: list[float]
    # Tool episodes only; all None on the single-turn path, which is unchanged.
    #
    # A tool episode's completion span interleaves the policy's tokens with the
    # tool's, and the tool's are not actions: they get no log-prob term, no
    # reward, no value target. So everything downstream of the forward pass runs
    # on the policy's tokens *compacted* -- `mask`, `completions` and
    # `policy_ids` describe that compacted sequence, and `policy_index` maps each
    # of its positions back to a column of the completion span. GAE then runs
    # straight from the last token of one turn to the first of the next, which is
    # the MDP: the tool response is part of the transition, and the next state
    # already contains it. Without compaction `gae` would read the tool block as
    # padding, cut the return there, and pay the tool call nothing for the answer
    # it made possible.
    policy_index: torch.Tensor | None = None
    policy_ids: torch.Tensor | None = None
    completion_len: int | None = None
    episodes: list[Episode] | None = None
    #: Compacted: 1 on the policy's last turn -- the answer -- and 0 on the turns
    #: before it, which are calls.
    answer_mask: torch.Tensor | None = None


def take_policy(x: torch.Tensor, index: torch.Tensor | None) -> torch.Tensor:
    """Gather the policy's positions out of a (batch, completion[, d]) tensor."""
    if index is None:
        return x
    if x.dim() == 3:
        return torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    return torch.gather(x, 1, index)


@torch.no_grad()
def rollout(
    policy,
    tokenizer,
    cases: list[dict[str, Any]],
    *,
    samples_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    weights: Weights,
    device: str,
    template_kwargs: dict[str, Any] | None = None,
    prompt_v3: bool = False,
    prompt_v4: bool = False,
) -> Rollout:
    """Sample one batch: `samples_per_prompt` completions for each case."""
    expanded = [c for c in cases for _ in range(samples_per_prompt)]
    messages = build_messages_v4 if prompt_v4 else (build_messages_v3 if prompt_v3 else build_messages)
    texts = [
        tokenizer.apply_chat_template(
            messages(c["record"]),
            tokenize=False,
            add_generation_prompt=True,
            **(template_kwargs or {}),
        )
        for c in expanded
    ]
    encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    prompt_len = encoded["input_ids"].shape[1]

    out = policy.generate(
        **encoded,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    if out.shape[1] < prompt_len + max_new_tokens:
        pad = out.new_full(
            (out.shape[0], prompt_len + max_new_tokens - out.shape[1]), tokenizer.pad_token_id
        )
        out = torch.cat([out, pad], dim=1)

    completion_ids = out[:, prompt_len:]
    mask = build_mask(completion_ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    def _reward(c: str, case: dict[str, Any]) -> float:
        # A completion the scorer cannot process earned no reward. `reward.score`
        # assumes well-formed numeric fields, and an exploring policy can emit a
        # literal outside float range (`OverflowError` in `reward._numeric_hits`)
        # or otherwise break parsing; that is a 0, not a dead run.
        try:
            return score(c, case["answer"], weights).total
        except (OverflowError, ValueError, TypeError, KeyError):
            return 0.0

    rewards = [_reward(c, case) for c, case in zip(completions, expanded)]

    return Rollout(
        cases=expanded,
        sequences=out,
        # The prompt's own padding, then the completion's active span. A token
        # after the first EOS was never chosen and must not be attended to.
        attention_mask=torch.cat([encoded["attention_mask"], mask.long()], dim=1),
        mask=mask,
        completions=completions,
        rewards=rewards,
    )


@torch.no_grad()
def tool_rollout(
    policy,
    tokenizer,
    cases: list[dict[str, Any]],
    *,
    samples_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    weights: Weights,
    device: str,
    template_kwargs: dict[str, Any] | None = None,
    max_rounds: int = 2,
    score_calls: bool = False,
    harness: str = "native",
) -> Rollout:
    """`rollout` with the calculator in the loop. See `Rollout` for the layout."""
    expanded = [c for c in cases for _ in range(samples_per_prompt)]
    generate = agent_generate if harness == "qwen_agent" else tool_generate
    episodes = generate(
        policy, tokenizer, [build_messages_tool(c["record"]) for c in expanded],
        temperature=temperature, max_new_tokens=max_new_tokens, max_rounds=max_rounds,
        template_kwargs=template_kwargs or {}, device=device,
        answers=[c["answer"] for c in expanded] if score_calls else None,
    )
    pad = tokenizer.pad_token_id
    prompt_width = max(len(ep.prompt_ids) for ep in episodes)
    span = max(1, max(len(ep.ids) for ep in episodes))
    compact = max(1, max(ep.policy_tokens for ep in episodes))

    sequences, attention, index, ids, mask, answer = [], [], [], [], [], []
    for ep in episodes:
        left = prompt_width - len(ep.prompt_ids)
        right = span - len(ep.ids)
        sequences.append([pad] * left + ep.prompt_ids + ep.ids + [pad] * right)
        attention.append([0] * left + [1] * (len(ep.prompt_ids) + len(ep.ids)) + [0] * right)
        where = [k for k, mine in enumerate(ep.policy) if mine]
        fill = compact - len(where)
        index.append(where + [0] * fill)
        ids.append([ep.ids[k] for k in where] + [pad] * fill)
        mask.append([1.0] * len(where) + [0.0] * fill)
        last = ep.turn_starts[-1] if ep.turn_starts else 0
        answer.append([0.0] * last + [1.0] * (len(where) - last) + [0.0] * fill)

    def _reward(c: str, case: dict[str, Any]) -> float:
        try:
            return score(c, case["answer"], weights).total
        except (OverflowError, ValueError, TypeError, KeyError):
            return 0.0

    completions = [ep.text for ep in episodes]
    return Rollout(
        cases=expanded,
        sequences=torch.tensor(sequences, device=device),
        attention_mask=torch.tensor(attention, device=device),
        mask=torch.tensor(mask, device=device),
        completions=completions,
        rewards=[_reward(c, case) for c, case in zip(completions, expanded)],
        policy_index=torch.tensor(index, device=device),
        policy_ids=torch.tensor(ids, device=device),
        completion_len=span,
        episodes=episodes,
        answer_mask=torch.tensor(answer, device=device),
    )


def numeric_rate(completions: list[str], cases: list[dict[str, Any]]) -> float:
    """Mean of `numeric_correct / 3` over a batch, as `summarise` computes it."""
    hits = []
    for text, case in zip(completions, cases):
        try:
            hits.append((score(text, case["answer"]).diagnostics.get("numeric_correct", 0) or 0) / 3)
        except (OverflowError, ValueError, TypeError, KeyError):
            hits.append(0.0)
    return sum(hits) / max(1, len(hits))


def region_entropy(logprobs: torch.Tensor, mask: torch.Tensor, answer: torch.Tensor) -> dict[str, float]:
    """`entropy_proxy` split into the calls and the answer.

    `12`'s 200-step run collapsed with the aggregate going 0.003 -> 2.1; this is
    the early warning, per region, so a drift in the calls is visible before it
    reaches the arithmetic.
    """
    out = {}
    for name, region in (("calls", (1 - answer) * mask), ("answer", answer * mask)):
        n = float(region.sum())
        if n:
            out[f"entropy_proxy_{name}"] = round(float(-(logprobs * region).sum()) / n, 6)
    return out


def turn_values(episodes: list[Episode], values: torch.Tensor, returns: torch.Tensor) -> dict[str, float]:
    """What the critic says, and what it is regressed to, at the start of each turn.

    Turn 1 starts on the bare prompt; turn 2 starts right after the tool has
    answered. The gap between the two is what the calls were worth, in the
    critic's own estimate.
    """
    out: dict[str, float] = {}
    for turn in (0, 1):
        rows = [(b, ep.turn_starts[turn]) for b, ep in enumerate(episodes) if len(ep.turn_starts) > turn]
        if not rows:
            continue
        v = [float(values[b, t]) for b, t in rows]
        g = [float(returns[b, t]) for b, t in rows]
        out[f"value_turn{turn + 1}"] = round(sum(v) / len(v), 6)
        out[f"return_turn{turn + 1}"] = round(sum(g) / len(g), 6)
    return out


def tool_stats(episodes: list[Episode]) -> dict[str, float]:
    """How the calculator was used, over a batch of episodes."""
    n = max(1, len(episodes))
    calls = sum(len(ep.calls) for ep in episodes)
    errors = sum(r.startswith("error") for ep in episodes for r in ep.results)
    return {
        "tool_use_rate": sum(bool(ep.calls) for ep in episodes) / n,
        "tool_calls_mean": calls / n,
        "tool_error_rate": errors / calls if calls else 0.0,
        "tool_unanswered": sum(ep.unanswered for ep in episodes) / n,
        "tool_env_tokens": sum(len(ep.ids) - ep.policy_tokens for ep in episodes) / n,
        "truncated_rate": sum(ep.truncated for ep in episodes) / n,
    }


# --- held-out evaluation ------------------------------------------------------


def evaluate(policy, tokenizer, cfg: "Config", cases: list[dict], step: int) -> dict[str, Any]:
    """Greedy pass over a fixed held-out slice, through `eval.py`'s own code path.

    Lifted from `grpo_scratch.evaluate` unchanged in substance, and that is the
    point: the PPO curve, the GRPO curve and the frozen baseline are then all
    produced by literally the same function, so the three are comparable without
    an argument about whose harness was fairer.

    The value head is not involved. This measures the policy, and the critic is
    scaffolding for training it.
    """
    from eval import generate_hf, summarise

    # generate_hf still moves the encoded batch onto a device, so it needs the
    # one the live policy is already on rather than a re-derived guess.
    device = str(next(policy.parameters()).device)
    if cfg.prompt_tool:
        # `generate_hf` is one `generate` call and cannot hand a turn to a tool,
        # so decoding is `tool_generate`'s. Scoring is still `summarise`, on the
        # policy's own text, so the metrics mean what they meant for every run.
        results, episodes = generate_tool_results(policy, tokenizer, cfg, cases)
        metrics = summarise(results, WEIGHT_SETS[cfg.weights])
        return {**_eval_record(step, metrics), **tool_stats(episodes)}
    with v3_prompts(cfg.prompt_v4) if (cfg.prompt_v3 or cfg.prompt_v4) else contextlib.nullcontext():
        results = generate_hf(
            cases,
            model=cfg.model,
            device=device,
            dtype=cfg.dtype,
            n=1,
            temperature=0.0,
            max_tokens=cfg.eval_max_tokens,
            seed=cfg.seed,
            batch_size=cfg.eval_batch,
            adapter=None,
            prompt_version=cfg.prompt_version,
            loaded=(policy, tokenizer),
        )
    metrics = summarise(results, WEIGHT_SETS[cfg.weights])
    return _eval_record(step, metrics)


def _eval_record(step: int, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": step,
        "reward": metrics["reward"],
        "exact_match": metrics["exact_match"],
        "validity_gate": metrics["validity_gate"],
        "schema_ok": metrics["schema_ok"],
        "cause_acc": metrics["cause_acc"],
        "flags_acc": metrics["flags_acc"],
        "numeric_acc": metrics["numeric_acc"],
        "action_acc": metrics["action_acc"],
        "completion_tokens": metrics["completion_tokens_mean"],
    }


def generate_tool_results(policy, tokenizer, cfg: "Config", cases: list[dict]):
    """Greedy tool episodes over `cases`, wrapped as `eval.CaseResult`s.

    Same seed, batch size and token budget as `generate_hf` is given in
    `evaluate`. Returns the episodes too, so a caller can see the calls.
    """
    from eval import CaseResult, Sample

    device = str(next(policy.parameters()).device)
    was_training = policy.training
    policy.eval()
    torch.manual_seed(cfg.seed)
    template_kwargs = {"enable_thinking": False} if supports_thinking_toggle(tokenizer) else {}
    results, episodes = [], []
    for start in range(0, len(cases), cfg.eval_batch):
        chunk = cases[start : start + cfg.eval_batch]
        generate = agent_generate if cfg.harness == "qwen_agent" else tool_generate
        batch = generate(
            policy, tokenizer, [build_messages_tool(c["record"]) for c in chunk],
            temperature=0.0, max_new_tokens=cfg.eval_max_tokens,
            max_rounds=cfg.tool_rounds, template_kwargs=template_kwargs, device=device,
        )
        for case, ep in zip(chunk, batch):
            results.append(CaseResult(case, [Sample(case["id"], ep.text, ep.policy_tokens)]))
        episodes += batch
        print(f"  {len(results)}/{len(cases)} cases", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    if was_training:
        policy.train()
    return results, episodes


# --- training -----------------------------------------------------------------


def train(cfg: Config, out_dir: Path, device: str, *, progress=print) -> list[dict[str, Any]]:
    """One PPO actor-critic run. Returns the per-step metrics it also writes.

    Deliberately the same shape as `grpo_scratch.train`: same rollout-then-update
    order, same micro-batching, same `metrics.jsonl` next to a `config.json`, so
    `02_ppo_actor_critic_1.7b.ipynb` can plot the two runs on one axis without
    special-casing either.
    """
    import json
    import random
    import time
    from dataclasses import asdict

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    cases = load_cases(cfg.split)
    weights = WEIGHT_SETS[cfg.weights]

    policy, value_head, tokenizer = load_policy(cfg, device)

    # A Qwen3 chat template defaults thinking *on*, and on this task that is
    # fatal rather than merely verbose: the policy spends its whole budget
    # inside <think> and returns nothing to score, so every rollout fails the
    # gate at 0.0 and the run trains on noise. Qwen2.5 templates have no such
    # variable, so this is detected rather than switched on by model name --
    # the same fix `eval.py` and `grpo_scratch.py` already carry.
    template_kwargs: dict[str, Any] = {}
    if supports_thinking_toggle(tokenizer):
        template_kwargs["enable_thinking"] = False
        progress("  chat template honours enable_thinking; set to False")

    actor_params = [p for p in policy.parameters() if p.requires_grad]
    critic_params = list(value_head.parameters())
    # Two optimisers, not two param groups in one. With `value_detach` the value
    # loss never reaches the trunk anyway, so the critic can be fitted on cached
    # features without a forward pass -- and that only works if stepping it does
    # not also step the actor. `--no-value-detach` puts the value term back into
    # the joint loss, because there the gradient must flow through the trunk.
    optimizer = torch.optim.AdamW(actor_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    critic_optimizer = torch.optim.AdamW(
        critic_params, lr=cfg.value_lr, weight_decay=cfg.weight_decay
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("")
    eval_path = out_dir / "eval.jsonl"
    eval_path.write_text("")

    eval_cases = [
        json.loads(line)
        for line in (DATA / f"{cfg.eval_split}.jsonl").read_text().splitlines()
    ][: cfg.eval_cases]

    def run_eval(step: int) -> dict[str, Any]:
        record = evaluate(policy, tokenizer, cfg, eval_cases, step)
        with eval_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        # The latest policy, every evaluation, so a run the machine takes down
        # can be picked up with `--resume-from <run>/last` rather than lost:
        # `13b`'s first full run died at step 34 when WSL shut down with it.
        if step:
            policy.save_pretrained(out_dir / "last" / "adapter")
            torch.save(value_head.state_dict(), out_dir / "last" / "value_head.pt")
            (out_dir / "last" / "step.json").write_text(json.dumps({"step": step}) + "\n")
        if record["reward"] > best["reward"]:
            best.update(reward=record["reward"], step=step, cause_acc=record["cause_acc"])
            policy.save_pretrained(out_dir / "adapter-best")
            torch.save(value_head.state_dict(), out_dir / "value_head-best.pt")
            (out_dir / "best.json").write_text(json.dumps(best, indent=2) + "\n")
        progress(
            f"    eval @{step:<4} reward {record['reward']:.4f} "
            f"cause {record['cause_acc']:.3f} flags {record['flags_acc']:.3f} "
            f"EM {record['exact_match']:.3f} valid {record['validity_gate']:.3f}"
            f" numeric {record['numeric_acc']:.3f}"
            + (f" tool {record['tool_use_rate']:.3f}" if "tool_use_rate" in record else "")
        )
        return record

    per_step = cfg.prompts_per_step * cfg.samples_per_prompt
    progress(
        f"{cfg.model} | {cfg.steps} steps | {cfg.prompts_per_step}x{cfg.samples_per_prompt}"
        f"={per_step} seq/step | lam={cfg.lam} | {device}"
    )

    history: list[dict[str, Any]] = []
    # (hidden, old_values, returns) for the last `critic_window` batches, active
    # positions only, held on CPU so the window costs host RAM rather than the
    # VRAM the rollout needs. Concatenated onto the device once per step, not
    # once per critic epoch.
    critic_history: deque = deque(maxlen=max(1, cfg.critic_window))
    # An RL run that has solved its task can still destroy itself: this one's
    # reward sat at 1.0000 from step 170 and collapsed to 0.0000 at 196 with the
    # gradient norm going from 2.7 to 109. Saving only the final policy saved the
    # wreck, and every measurement afterwards was of the wreck. Checkpoint the
    # best trailing mean as well, and say so if the two differ.
    # Selected on the *held-out* curve, not on training reward. Measured on the
    # first fixed run: corr(trailing-10 training reward, held-out cause_acc) was
    # only +0.472, training reward peaked at step 100 while held-out was already
    # falling, and at the step where held-out cause_acc collapsed to 0.135 the
    # training trailing mean was still 0.2930 -- higher than at step 25. Training
    # reward barely reacts to the collapse it is supposed to protect against.
    # Defined before the step-0 eval because `run_eval` closes over it.
    best: dict[str, Any] = {"reward": float("-inf"), "step": None, "criterion": "held-out reward"}

    if cfg.eval_every:
        run_eval(0)  # step 0 must reproduce the frozen baseline

    for step in range(cfg.steps):
        started = time.perf_counter()
        batch = [rng.choice(cases) for _ in range(cfg.prompts_per_step)]

        policy.eval()
        if cfg.prompt_tool:
            roll = tool_rollout(
                policy,
                tokenizer,
                batch,
                samples_per_prompt=cfg.samples_per_prompt,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                weights=weights,
                device=device,
                template_kwargs=template_kwargs,
                max_rounds=cfg.tool_rounds,
                score_calls=cfg.call_credit,
                harness=cfg.harness,
            )
        else:
            roll = rollout(
                policy,
                tokenizer,
                batch,
                samples_per_prompt=cfg.samples_per_prompt,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                weights=weights,
                device=device,
                template_kwargs=template_kwargs,
                prompt_v3=cfg.prompt_v3 or cfg.prompt_v4,
                prompt_v4=cfg.prompt_v4,
            )
        gen_seconds = time.perf_counter() - started

        sequences, mask = roll.sequences, roll.mask
        attn = roll.attention_mask
        # On the single-turn path the completion span *is* the policy's sequence
        # and `index` is None, so every `take_policy` below is the identity and
        # the runs already in `runs/` are untouched. See `Rollout`.
        index = roll.policy_index
        completion_len = roll.completion_len or mask.shape[1]
        completion_ids = roll.policy_ids if index is not None else sequences[:, -completion_len:]
        rewards = torch.tensor(roll.rewards, device=device, dtype=torch.float32)
        if cfg.dense_rewards:
            per_token_rewards = dense_rewards(
                roll.completions,
                completion_ids,
                mask,
                roll.cases,
                weights,
                tokenizer,
                cfg.flat_credit_scale,
                call_hits=[ep.call_hits for ep in roll.episodes] if cfg.call_credit else None,
            )
        else:
            per_token_rewards = terminal_rewards(rewards, mask)
        if cfg.call_credit and not (cfg.prompt_tool and cfg.dense_rewards):
            raise ValueError("call_credit needs prompt_tool and dense_rewards")
        call_record: dict[str, float] = {}
        if cfg.call_credit:
            # The invariant the change rests on, checked live: moving the credit
            # onto the calls must leave every row's total exactly where it was.
            plain = dense_rewards(
                roll.completions, completion_ids, mask, roll.cases, weights, tokenizer,
                cfg.flat_credit_scale,
            )
            call_record = {
                "call_numeric_acc": sum(len(ep.call_hits) for ep in roll.episodes) / (3 * len(roll.episodes)),
                "call_credit_sum_err": float((per_token_rewards.sum(1) - plain.sum(1)).abs().max()),
            }

        flag_mask = None
        if cfg.flag_entropy_coef:
            flag_mask = flag_token_mask(roll.completions, completion_ids, mask, tokenizer)

        policy.train()
        with torch.no_grad():
            old_lp, old_v, old_h = [], [], []
            for i in range(0, len(sequences), cfg.micro_batch):
                sl = slice(i, i + cfg.micro_batch)
                lp, v, h = forward_policy_value(
                    policy, value_head, sequences[sl], attn[sl], completion_len,
                    value_detach=cfg.value_detach, return_hidden=True,
                    value_layer=cfg.value_layer,
                )
                if index is not None:
                    at = index[sl]
                    lp, v, h = take_policy(lp, at), take_policy(v, at), take_policy(h, at)
                old_lp.append(lp)
                old_v.append(v)
                old_h.append(h)
            old_logprobs = torch.cat(old_lp)
            old_values = torch.cat(old_v)
            # The features the critic reads, kept so it can be refitted without
            # touching the trunk again. (batch, tokens, hidden) in the trunk's
            # own dtype -- 21 MiB for 8 x 640 x 2048 in bf16.
            hidden_cache = torch.cat(old_h)
            if cfg.privileged_cause:
                # One row per sequence, broadcast along the token axis: the label
                # is a property of the prompt, so it does not vary within a
                # sequence. `roll.cases` is expanded to match `samples_per_prompt`
                # by `rollout`, so it lines up index-for-index with the batch.
                # Scaled, and the scale is derived rather than picked. This file's
                # own relation is |dV| ~ lr * ||h||_1: the hidden block moves V by
                # 3e-5 * 1605 = 0.048 per critic step, while a one-hot has
                # ||.||_1 = 1 and would move it by 3e-5 -- 1600x slower. Over a
                # run's 640 critic steps the privileged weights would travel 0.019
                # against cause offsets of +-0.25, i.e. stay frozen. 64 puts one
                # step at 1.9e-3 and the run's total travel above 1.0.
                onehot = torch.zeros(len(roll.cases), len(CAUSES),
                                     device=hidden_cache.device, dtype=hidden_cache.dtype)
                for i, case in enumerate(roll.cases):
                    onehot[i, CAUSES.index(case["answer"]["root_cause"])] = PRIVILEGED_SCALE
                hidden_cache = torch.cat(
                    [hidden_cache, onehot.unsqueeze(1).expand(-1, hidden_cache.shape[1], -1)],
                    dim=-1,
                )
                # old_values was computed by the head *before* the one-hot existed
                # in this batch's tensor, so recompute it on the full feature.
                old_values = value_head(hidden_cache)

            advantages, returns = gae(
                per_token_rewards, old_values, mask, gamma=cfg.gamma, lam=cfg.lam
            )
            if cfg.whiten_advantages:
                advantages = whiten(advantages, mask)

        # --- the critic, on cached features -----------------------------------
        # No forward through the trunk: `hidden_cache` is already computed and
        # detached, so each of these steps is one (batch, tokens, hidden) x
        # (hidden, 1) matmul. That is what makes `critic_epochs` affordable, and
        # the critic needs it -- one update per batch at this learning rate is a
        # ~100-step time constant chasing a target that changes every step.
        critic_stats: dict[str, float] = {}
        if cfg.value_detach:
            # The window keeps active positions only. Flattening to (n_active,
            # hidden) is what lets batches of different completion lengths sit in
            # one tensor without padding, and it drops nothing: every masked
            # position contributes zero to the loss anyway.
            with torch.no_grad():
                sel = mask > 0
                critic_history.append((
                    hidden_cache[sel].to("cpu"),
                    old_values[sel].to("cpu"),
                    returns[sel].to("cpu"),
                ))
                if len(critic_history) > 1:
                    fit_h = torch.cat([h for h, _, _ in critic_history]).to(device)
                    fit_v = torch.cat([v for _, v, _ in critic_history]).to(device)
                    fit_r = torch.cat([r for _, _, r in critic_history]).to(device)
                    fit_m = torch.ones_like(fit_r)
                else:
                    # critic_window=1 stays on the original tensors rather than a
                    # flattened copy of them, so the existing runs reproduce
                    # bit-for-bit rather than merely equivalently.
                    fit_h, fit_v, fit_r, fit_m = hidden_cache, old_values, returns, mask

            for _ in range(cfg.critic_epochs):
                critic_optimizer.zero_grad(set_to_none=True)
                v = value_head(fit_h)
                v_loss, critic_stats = value_loss(
                    v, fit_v, fit_r, fit_m, clip_eps=cfg.value_clip_eps,
                )
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic_params, 1.0)
                critic_optimizer.step()
            with torch.no_grad():
                # On the *current* batch, always: this feeds `value_ev_fit` and the
                # recomputed advantages, both of which are about this batch.
                fitted_values = value_head(hidden_cache)
        else:
            fitted_values = old_values

        # C5. Without this the critic's dedicated steps reach the actor only
        # through the next batch's `old_values`. `returns` is the Monte-Carlo
        # return at lam=1, so it does not move when V does, and the subtraction is
        # the whole update. Re-masked because `fitted_values` is nonzero at padded
        # positions where `gae` left the advantage at exactly zero.
        if cfg.recompute_advantages:
            with torch.no_grad():
                advantages = (returns - fitted_values) * mask
                if cfg.whiten_advantages:
                    advantages = whiten(advantages, mask)

        # --- the actor --------------------------------------------------------
        actor_stats: dict[str, float] = {}
        for _ in range(cfg.inner_epochs):
            optimizer.zero_grad(set_to_none=True)
            if not cfg.value_detach:
                critic_optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            # Weighted by each micro-batch's share, rather than letting whichever
            # slice happens to run last overwrite the lot. With micro_batch=1
            # that bug made every reported adv_mean / adv_std / ratio_mean /
            # clip_frac describe a single sequence rather than the batch.
            epoch_stats: dict[str, float] = {}
            for i in range(0, len(sequences), cfg.micro_batch):
                sl = slice(i, i + cfg.micro_batch)
                share = min(cfg.micro_batch, len(sequences) - i) / len(sequences)
                want_entropy = cfg.entropy_coef > 0
                fwd = forward_policy_value(
                    policy, value_head, sequences[sl], attn[sl], completion_len,
                    value_detach=cfg.value_detach,
                    return_entropy=want_entropy,
                    value_layer=cfg.value_layer,
                )
                if want_entropy:
                    logprobs, values, entropy = fwd
                else:
                    logprobs, values = fwd
                    entropy = None
                if index is not None:
                    at = index[sl]
                    logprobs, values = take_policy(logprobs, at), take_policy(values, at)
                    entropy = None if entropy is None else take_policy(entropy, at)
                a_loss, stats = ppo_actor_loss(
                    logprobs, old_logprobs[sl], advantages[sl], mask[sl],
                    clip_eps=cfg.clip_eps, normalize=cfg.normalize,
                    entropy=entropy, entropy_coef=cfg.entropy_coef,
                    focus_mask=None if flag_mask is None else flag_mask[sl],
                    focus_entropy_coef=cfg.flag_entropy_coef,
                    entropy_mask=(roll.answer_mask[sl]
                                  if cfg.entropy_scope == "answer" and roll.answer_mask is not None
                                  else None),
                )
                loss = a_loss
                if not cfg.value_detach:
                    # Shared trunk: the value loss has to travel through it, so
                    # it stays in the joint objective and `value_coef` applies.
                    v_loss, critic_stats = value_loss(
                        values, old_values[sl], returns[sl], mask[sl],
                        clip_eps=cfg.value_clip_eps,
                    )
                    loss = loss + cfg.value_coef * v_loss
                (loss * share).backward()
                accumulated += float(loss.detach()) * share
                for k, val in stats.items():
                    epoch_stats[k] = epoch_stats.get(k, 0.0) + val * share
            params = actor_params if cfg.value_detach else actor_params + critic_params
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if step >= cfg.critic_warmup:
                optimizer.step()
            if not cfg.value_detach:
                critic_optimizer.step()
            actor_stats = epoch_stats  # the last epoch: where the policy ended up

        record = {
            "step": step,
            "reward_mean": sum(roll.rewards) / len(roll.rewards),
            "reward_max": max(roll.rewards),
            # GRPO's failure mode, measured on this method for comparison: the
            # fraction of tokens whose advantage is exactly zero. A critic makes
            # this ~0 by construction, which is the claim being checked.
            "adv_zero_frac": float(
                (((advantages == 0) & (mask > 0)).float().sum() / mask.sum().clamp(min=1)).item()
            ),
            # Whether the critic is a critic, measured two ways because they
            # answer different questions.
            #
            # `value_ev` uses `old_values` -- the head as it stood *before* this
            # batch was fitted, scored on data it had not seen. That is the
            # honest, out-of-sample number, and it is also the operative one:
            # `old_values` is what GAE actually built this step's advantages
            # from.
            #
            # `value_ev_fit` uses the head *after* `critic_epochs` steps on this
            # same batch, so it is in-sample and optimistic. It is here to
            # separate two failures that look identical from the outside: a head
            # that cannot fit the batch at all (both near zero) from one that
            # fits it and then fails to generalise to the next batch (fit high,
            # out-of-sample near zero).
            **critic_fit(old_values, returns, mask),
            **{f"{k}_fit": v for k, v in critic_fit(fitted_values, returns, mask).items()},
            "completion_tokens": float(mask.sum() / mask.shape[0]),
            "unique_completions": float(len(set(roll.completions))),
            "loss": accumulated,
            "grad_norm": float(grad_norm),
            "gen_seconds": round(gen_seconds, 2),
            "step_seconds": round(time.perf_counter() - started, 2),
            **{k: round(v, 6) for k, v in actor_stats.items()},
            **{k: round(v, 6) for k, v in critic_stats.items()},
            **({k: round(v, 6) for k, v in tool_stats(roll.episodes).items()}
               if roll.episodes is not None else {}),
            # The arithmetic gate, on every step rather than every `eval_every`:
            # the share of the batch's three numbers inside tolerance, sampled.
            **({"train_numeric_acc": round(numeric_rate(roll.completions, roll.cases), 6)}
               if roll.episodes is not None else {}),
            **{k: round(v, 8) for k, v in call_record.items()},
            **(turn_values(roll.episodes, old_values, returns) if roll.episodes is not None else {}),
            **(region_entropy(old_logprobs, mask, roll.answer_mask) if roll.answer_mask is not None else {}),
        }
        # How far the critic actually moved this step. The knob that sets it is
        # `value_lr`, and the first CPU smoke run had it two orders of magnitude
        # too high; this is the number that makes that visible without waiting
        # for the reward curve to look wrong.
        record["value_step"] = round(
            abs(record["value_mean"] - history[-1]["value_mean"]) if history else 0.0, 6
        )
        # What the dedicated critic steps bought within this batch alone. None on a
        # degenerate batch, where `critic_fit` leaves both explained variances undefined
        # rather than reporting a plausible-looking 0.0.
        ev, ev_fit = record["value_ev"], record["value_ev_fit"]
        record["value_ev_gain"] = round(ev_fit - ev, 6) if ev is not None and ev_fit is not None else None
        history.append(record)
        # Kept as a diagnostic; no longer used to choose a checkpoint.
        window = [r["reward_mean"] for r in history[-BEST_WINDOW:]]
        record["reward_trailing"] = round(sum(window) / len(window), 6)
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        ev_txt = "  undef  " if record["value_ev"] is None else (
            f"{record['value_ev']:+.3f}/{record['value_ev_fit']:+.3f}"
        )
        progress(
            f"  step {step:>4} reward {record['reward_mean']:.4f} "
            f"V {record['value_mean']:.3f} ev {ev_txt} "
            f"adv {record['adv_mean']:+.3f}+-{record['adv_std']:.3f} "
            f"clip {record['clip_frac']:.3f} "
            f"tok {record['completion_tokens']:.0f} "
            f"|g| {record['grad_norm']:.3f} {record['step_seconds']:.1f}s"
            + (f" num {record['train_numeric_acc']:.3f} calls {record['tool_calls_mean']:.2f}"
               if "train_numeric_acc" in record else "")
            + (f" call-num {record['call_numeric_acc']:.3f} sum-err {record['call_credit_sum_err']:.1e}"
               if "call_numeric_acc" in record else "")
        )

        if cfg.eval_every and (step + 1) % cfg.eval_every == 0:
            run_eval(step + 1)

        if cfg.numeric_stop and "train_numeric_acc" in record and len(history) >= 10:
            recent = [r["train_numeric_acc"] for r in history[-10:]]
            if sum(recent) / 10 < cfg.numeric_stop:
                policy.save_pretrained(out_dir / "adapter-stopped")
                torch.save(value_head.state_dict(), out_dir / "value_head-stopped.pt")
                (out_dir / "stopped.json").write_text(json.dumps({
                    "step": step, "numeric_last10": recent, "threshold": cfg.numeric_stop,
                }, indent=2) + "\n")
                progress(f"STOPPED at step {step}: sampled numeric_acc over the last 10 steps "
                         f"{sum(recent) / 10:.3f} < {cfg.numeric_stop}; adapter-stopped/ written")
                break

    policy.save_pretrained(out_dir / "adapter")
    torch.save(value_head.state_dict(), out_dir / "value_head.pt")
    progress(f"\nwrote {out_dir}")
    if best["step"] is not None and best["step"] < cfg.steps:
        progress(
            f"NOTE: best held-out reward {best['reward']:.4f} was at step "
            f"{best['step']}, not at the end. "
            f"adapter-best/ holds that policy; adapter/ holds the final one."
        )
    return history


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def eval_only(cfg: Config, out: Path, device: str, *, progress=print) -> dict[str, Any]:
    """One greedy tool pass over `eval_split`, from `resume_from` or the raw model.

    Writes `out` in the envelope `runs/paired/*.json` uses -- `paired_test.py`
    reads it unmodified -- and `out` with `.episodes.jsonl` beside it: the
    policy's text, every call it made and what the tool said back.
    """
    import hashlib
    import json

    from eval import summarise

    if not cfg.prompt_tool:
        raise ValueError("eval_only is the tool path; the single-turn path has eval.py")
    policy, _, tokenizer = load_policy(cfg, device)
    source = DATA / f"{cfg.eval_split}.jsonl"
    cases = load_cases(cfg.eval_split)[: cfg.eval_cases]
    results, episodes = generate_tool_results(policy, tokenizer, cfg, cases)
    overall = summarise(results, WEIGHT_SETS[cfg.weights])
    overall.update(tool_stats(episodes))
    envelope = {
        "model": cfg.model,
        "backend": "hf+calculator" + ("+qwen_agent" if cfg.harness == "qwen_agent" else ""),
        "split": cfg.eval_split,
        "split_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "prompt_version": "v2",
        "tools": [CALCULATOR_TOOL],
        "tool_rounds": cfg.tool_rounds,
        "weights": cfg.weights,
        "mode": "greedy",
        "k": 1,
        "temperature": 0.0,
        "max_tokens": cfg.eval_max_tokens,
        "seed": cfg.seed,
        "adapter": cfg.resume_from or None,
        "overall": overall,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(envelope, indent=2) + "\n")
    with out.with_suffix(".episodes.jsonl").open("w") as fh:
        for case, ep in zip(cases, episodes):
            fh.write(json.dumps({
                "id": case["id"], "text": ep.text, "calls": ep.calls, "results": ep.results,
                "unanswered": ep.unanswered, "truncated": ep.truncated,
                "policy_tokens": ep.policy_tokens, "env_tokens": len(ep.ids) - ep.policy_tokens,
            }) + "\n")
    progress(
        f"  {cfg.eval_split}: EM {overall['exact_match']:.3f} numeric {overall['numeric_acc']:.3f} "
        f"flags {overall['flags_acc']:.3f} cause {overall['cause_acc']:.3f} "
        f"schema {overall['schema_ok']:.3f} | tool use {overall['tool_use_rate']:.3f} "
        f"calls {overall['tool_calls_mean']:.2f} errors {overall['tool_error_rate']:.3f}"
    )
    progress(f"  wrote {out}")
    return envelope


# --- SFT seeding ---------------------------------------------------------------
#
# Why this exists, and why it is not "more training". Measured at temperature
# 1.0 over 600 train samples, on the frozen policy and on the 575-step PPO
# policy alike:
#
#     organic_fouling                    0 / 600
#     isolate_and_evaluate_replacement   0 / 600
#
# Both labels are recoverable from fields the policy already produces correctly.
# `organic_fouling` is the only row in `COVERED` with `salt_passage = "down"`,
# and on all 29 dev cases where it is the answer the policy writes that flag
# correctly and then names `compaction` -- a cause whose own rows require
# `salt_passage` to be `flat` or `up`. It contradicts evidence it just wrote
# down. The severe action is `flow_pct <= -30` and `numeric_acc` is 1.000.
#
# So the knowledge is present and the token is not. That is the one shape a
# policy gradient provably cannot fix and maximum likelihood trivially can:
# `grad log pi(y*|x)` is computable for a *given* `y*` whether or not the policy
# would ever sample it. This puts mass on the labels so that PPO -- which does
# the actual learning of when to use them -- has something to reweight. The
# precedent is `flat`: 3/600 = 0.5% was enough for `flat_credit_scale` to drive
# recall 0.000 -> 0.984. RL handles rare. It does not handle absent.


def sft_target(answer: dict[str, Any]) -> str:
    """The gold completion, in the shape v3's closing asks for.

    One short line naming the three flags, then the JSON object and nothing
    after it. Matching the instructed format matters: seeding on bare JSON would
    also teach the policy to drop the reasoning line, which is a second change
    riding along with the one being tested.
    """
    import json

    flags = answer["flags"]
    line = f"Flow {flags['flow']}, salt passage {flags['salt_passage']}, dP {flags['dp']}."
    return line + "\n" + json.dumps(answer, indent=2, sort_keys=True)


def build_sft_examples(
    cases: list[dict[str, Any]], balance: float, seed: int
) -> tuple[list[dict[str, Any]], int, int]:
    """Every dead-label case, plus `balance` x as many normal ones.

    The normal cases are not padding. Seeding on the dead labels alone teaches
    "say organic_fouling", which is `09`'s `compaction` over-emission reproduced
    in the other direction -- the policy already over-produces one cause at the
    lookup step, and a one-sided seed hands it another. The mixed set teaches
    that the label exists in one situation, which is the thing actually missing.
    """
    import random

    from task.decision_table import SEVERE_ACTION

    dead, rest = [], []
    for case in cases:
        answer = case["answer"]
        target = dead if (
            answer["root_cause"] == "organic_fouling" or answer["action"] == SEVERE_ACTION
        ) else rest
        target.append(case)

    rng = random.Random(seed)
    # Drawn round-robin over the remaining causes rather than flat, so the seed
    # does not also reweight whichever cause happens to be commonest in train.
    buckets: dict[str, list[dict[str, Any]]] = {}
    for case in rest:
        buckets.setdefault(case["answer"]["root_cause"], []).append(case)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    want, normal = int(round(len(dead) * balance)), []
    while len(normal) < want and any(buckets.values()):
        for bucket in buckets.values():
            if bucket and len(normal) < want:
                normal.append(bucket.pop())

    examples = dead + normal
    rng.shuffle(examples)
    return examples, len(dead), len(normal)


SEVERE_ACTION_NAME = "isolate_and_evaluate_replacement"


def dead_label_spans(text: str, offset: int) -> list[tuple[int, int]]:
    """Character spans of the two dead label values, searched after `offset`.

    The value only, not the key and not the quotes: the key is already produced
    correctly on every case, and it is the value token that has zero mass.
    """
    from task.decision_table import SEVERE_ACTION

    spans = []
    for literal in ("organic_fouling", SEVERE_ACTION):
        start = text.find(literal, offset)
        while start != -1:
            spans.append((start, start + len(literal)))
            start = text.find(literal, start + 1)
    return spans


def slot_value_spans(text: str, offset: int, slots=("root_cause", "action")) -> list[tuple[int, int]]:
    """Character spans of the *values* of the named slots, whatever they are.

    `sft_labels_only` masked to the dead label strings alone and that collapsed
    the `action` slot: supervised toward `isolate_and_evaluate_replacement` and
    never toward anything else, the policy learned to emit it on 600/600 samples.
    A slot needs its negatives. This supervises the correct value of two slots on
    every seed case, which is still two fields out of eight -- it cannot teach
    the arithmetic, the flags or the stage, all of which the policy already gets
    right -- while leaving the slot's alternatives represented.
    """
    spans = []
    for slot in slots:
        key = f'"{slot}": "'
        start = text.find(key, offset)
        while start != -1:
            value_start = start + len(key)
            value_end = text.find('"', value_start)
            if value_end == -1:
                break
            spans.append((value_start, value_end))
            start = text.find(key, value_end)
    return spans


def sft_seed(cfg: "Config", out_dir: Path, device: str, *, progress=print) -> dict[str, Any]:
    """One seeding pass. Writes an adapter PPO can `--resume-from`."""
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    policy, value_head, tokenizer = load_policy(cfg, device)
    if cfg.sft_seed_from:
        policy.load_adapter(cfg.sft_seed_from, adapter_name="default")
        # The critic is carried across unchanged. Seeding does not touch it, and
        # re-initialising it would make the continuation spend its first
        # `critic_window` steps relearning what it already knew -- the same
        # argument `resume_from` makes, for the same reason.
        source = Path(cfg.sft_seed_from)
        head = source.parent / (
            "value_head-best.pt" if source.name == "adapter-best" else "value_head.pt"
        )
        if head.exists():
            value_head.load_state_dict(torch.load(head, map_location=device))
        else:
            progress(f"  no value head at {head}; PPO will resume with a fresh one")

    if cfg.sft_tool:
        import random

        if cfg.sft_labels_only or cfg.sft_cases:
            raise ValueError("sft_tool supervises the calls only; it takes no label mask or seed file")
        examples = load_cases(cfg.split)
        random.Random(cfg.seed).shuffle(examples)
        n_dead, n_normal = 0, len(examples)
        progress(f"  tool seed: {len(examples)} {cfg.split} cases, loss on the three calls only, "
                 f"{cfg.sft_epochs} epoch(s), lr {cfg.sft_lr}")
    elif cfg.sft_cases:
        import random

        source = Path(cfg.sft_cases)
        if not source.is_absolute():
            source = SMOKE_DIR / source
        cases = [json.loads(l) for l in source.read_text().splitlines() if l.strip()]
        examples = list(cases)
        random.Random(cfg.seed).shuffle(examples)
        n_dead = sum(
            c["answer"]["root_cause"] == "organic_fouling"
            or c["answer"]["action"] == SEVERE_ACTION_NAME
            for c in examples
        )
        n_normal = len(examples) - n_dead
        progress(f"  seed set: {source.name}, {len(examples)} cases "
                 f"({n_dead} carry a dead label, {n_normal} do not), "
                 f"{cfg.sft_epochs} epoch(s), lr {cfg.sft_lr}")
    else:
        examples, n_dead, n_normal = build_sft_examples(
            cases := load_cases(cfg.split), cfg.sft_balance, cfg.seed)
        progress(
            f"  seed set: {n_dead} dead-label + {n_normal} normal = {len(examples)}"
            f" of {len(cases)} {cfg.split} cases, {cfg.sft_epochs} epoch(s), lr {cfg.sft_lr}"
        )

    template_kwargs: dict[str, Any] = {}
    if supports_thinking_toggle(tokenizer):
        template_kwargs["enable_thinking"] = False

    policy.train()
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.sft_lr)
    accum = max(1, cfg.prompts_per_step)

    losses: list[dict[str, Any]] = []
    n_supervised = 0
    for epoch in range(cfg.sft_epochs):
        for i, case in enumerate(examples):
            if cfg.sft_tool:
                # The prompt is the one `tool_generate` decodes from, and the
                # target is the text a correct first turn would sample -- so the
                # supervised tokens are the tokens PPO will later score.
                prompt = tokenizer.apply_chat_template(
                    build_messages_tool(case["record"]),
                    tools=[CALCULATOR_TOOL],
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
                target = tool_call_target(case["record"])
            else:
                prompt = tokenizer.apply_chat_template(
                    build_messages_v3(case["record"]),
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
                target = sft_target(case["answer"])
            prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            full = prompt + target + (tokenizer.eos_token or "")
            encoded = tokenizer(full, return_tensors="pt", return_offsets_mapping=True)
            ids = encoded["input_ids"].to(device)
            labels = ids.clone()
            labels[:, :prompt_len] = -100  # loss on the completion only
            if cfg.sft_labels_only:
                # Mask everything the policy already produces correctly. Only the
                # characters spelling a dead label carry gradient, so this cannot
                # teach the arithmetic, the flags, the stage, or any cause the
                # policy already emits -- it can only move two strings off zero.
                spans = (
                    slot_value_spans(full, len(prompt))
                    if cfg.sft_labels_only == 2
                    else dead_label_spans(full, len(prompt))
                )
                keep = torch.zeros_like(labels, dtype=torch.bool)
                offsets = encoded["offset_mapping"][0].tolist()
                for t, (a, b) in enumerate(offsets):
                    if a == b:
                        continue
                    if any(a < end and b > start for start, end in spans):
                        keep[0, t] = True
                labels = torch.where(keep.to(device), labels, torch.full_like(labels, -100))
                n_supervised += int(keep.sum())
                if int(keep.sum()) == 0:
                    continue  # nothing to learn from this case under this mask

            loss = policy(input_ids=ids, labels=labels).loss
            weight = 1.0
            if cfg.sft_labels_only == 2 and cfg.sft_dead_weight != 1.0:
                answer = case["answer"]
                if answer["root_cause"] == "organic_fouling" or answer["action"] == SEVERE_ACTION_NAME:
                    weight = cfg.sft_dead_weight
            ((loss * weight) / accum).backward()
            if (i + 1) % accum == 0 or i + 1 == len(examples):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            losses.append({"epoch": epoch, "i": i, "loss": float(loss.detach())})
        window = [r["loss"] for r in losses if r["epoch"] == epoch]
        if window:
            progress(f"  epoch {epoch}: loss {window[0]:.4f} -> {window[-1]:.4f}")
    if cfg.sft_labels_only:
        progress(f"  supervised tokens total: {n_supervised} over {len(losses)} examples")

    policy.save_pretrained(out_dir / "adapter")
    torch.save(value_head.state_dict(), out_dir / "value_head.pt")
    (out_dir / "sft_losses.jsonl").write_text(
        "\n".join(json.dumps(r) for r in losses) + "\n"
    )
    from dataclasses import asdict

    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    progress(f"  wrote {out_dir}/adapter and value_head.pt")
    return {"n_examples": len(examples), "n_dead": n_dead, "n_normal": n_normal}


def main() -> None:
    import argparse
    from dataclasses import asdict

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name, value in asdict(Config()).items():
        if name == "metrics":
            continue
        if isinstance(value, bool):
            parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=int(value))
        else:
            parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    # The sentinel, not the resolved value. `asdict(Config())` has already run
    # __post_init__, so without this every run would inherit MAIN's baseline as
    # a hard-coded default -- including `--weights PROBE`, whose frozen policy
    # scores 0.45 rather than 0.30. A wrong bias on a critic this slow to move
    # is not a detail: it is most of the baseline for the whole run.
    parser.set_defaults(value_init_bias=-1.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.weights not in WEIGHT_SETS:
        parser.error(f"--weights must be one of {sorted(WEIGHT_SETS)}")

    fields = {k: v for k, v in vars(args).items() if k in asdict(Config()) and k != "metrics"}
    fields["value_detach"] = bool(fields["value_detach"])
    fields["whiten_advantages"] = bool(fields["whiten_advantages"])
    fields["prompt_v3"] = bool(fields["prompt_v3"])
    fields["dense_rewards"] = bool(fields["dense_rewards"])
    fields["prompt_v4"] = bool(fields["prompt_v4"])
    fields["prompt_tool"] = bool(fields["prompt_tool"])
    fields["sft_tool"] = bool(fields["sft_tool"])
    fields["value_head_reset"] = bool(fields["value_head_reset"])
    fields["call_credit"] = bool(fields["call_credit"])
    if fields["entropy_scope"] not in ("all", "answer"):
        parser.error("--entropy-scope must be all or answer")
    if fields["harness"] not in ("native", "qwen_agent"):
        parser.error("--harness must be native or qwen_agent")
    cfg = Config(**fields)
    tag = cfg.model.rsplit("/", 1)[-1].replace(".", "").lower()
    out = args.out or SMOKE_DIR / "runs" / f"ppo-{tag}-{cfg.weights.lower()}-s{cfg.seed}"
    if cfg.eval_only:
        eval_only(cfg, out, resolve_device(args.device))
    elif cfg.sft_only:
        sft_seed(cfg, out, resolve_device(args.device))
    else:
        train(cfg, out, resolve_device(args.device))


if __name__ == "__main__":
    main()
