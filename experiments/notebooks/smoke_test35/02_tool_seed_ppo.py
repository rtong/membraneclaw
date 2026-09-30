"""Actor-critic PPO on Qwen3.5-0.8B with a calculator, from a seed that only learned to call it.

This file belongs to `02_tool_seed_ppo.ipynb` and to nothing else. It copies
what it needs from `01_raw_dense_critic.py` rather than importing it: the series
in this directory does not share code between notebooks.

## What is being reproduced

`../smoke_test/13b`, the configuration on which the v2 prompt first had a critic
that beat a position-only clock, moved to Qwen3.5-0.8B:

* **The prompt is v2's, byte for byte.** The one addition is the tool
  declaration the chat template writes into the system turn when a tool is
  offered, and `CALCULATOR_TOOL` describes a generic calculator with no example
  that could hint at the task's formulas. Nothing tells the model to use it.
* **A tool seed from the raw model.** One supervised pass over the 400 training
  cases in which the loss covers the three calculator calls and nothing else --
  no flag, no label, no answer field. The numbers come back from the calculator;
  everything after the tool's reply is left to the model.
* **PPO from that seed**, with dense per-field credit on the answer, `lam =
  0.95`, the entropy bonus on the answer turn only, and a stop once sampled
  arithmetic falls below 0.90 over ten steps.

`01` measured why the calculator is needed: on the raw model the numbers are
right 1% of the time, the policy's state says nothing about the return beyond
position, and the critic stays a clock. Every run of the 1.7B series whose
critic got ahead of the clock had the numbers right first.

## Qwen3.5's tool protocol

Calls are XML, not JSON::

    <tool_call>
    <function=calculator>
    <parameter=expression>
    ...
    </parameter>
    </function>
    </tool_call>

and the calculator's replies come back in one user turn as `<tool_response>`
blocks. An episode is built by concatenating tokens -- prompt, the policy's
turn up to its `<|im_end|>`, the tool's turn and the next assistant header, the
policy's next turn -- which is also what the chat template renders for the same
conversation; the notebook checks that character for character.
"""
from __future__ import annotations

import sys

# Importing from membrane_grpo must leave no trace in that tree.
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from dataclasses import asdict, dataclass, field, fields  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

HERE = Path(__file__).resolve().parent
PREFIX = "02_tool_seed_ppo"
RUNS = HERE / f"{PREFIX}_runs"
GRPO_DIR = HERE.parents[1] / "membrane_grpo"
if str(GRPO_DIR) not in sys.path:
    sys.path.insert(0, str(GRPO_DIR))

from eval import CaseResult, Sample, summarise, supports_thinking_toggle  # noqa: E402
from grpo_scratch import selective_logprobs  # noqa: E402
from reward import ABLATE, Weights, _flag_hits, _numeric_hits, score  # noqa: E402
from task.prompt import _num, build_messages  # noqa: E402
from task.schema import FLAG_KEYS, NUMERIC_KEYS, parse_answer, validate  # noqa: E402

DATA = GRPO_DIR / "data"
MODEL = "Qwen/Qwen3.5-0.8B"
PROMPT_VERSION = "v2"
LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
)
WEIGHT_SETS: dict[str, Weights] = {"ABLATE": ABLATE}
TEMPLATE_KWARGS = {"enable_thinking": False}


def load_cases(split: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines()]


def safe_score(completion: str, answer: dict[str, Any], weights: Weights):
    try:
        return score(completion, answer, weights)
    except (OverflowError, ValueError, TypeError, KeyError):
        return None


# --- the calculator -----------------------------------------------------------

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
            "properties": {"expression": {"type": "string", "description": "the arithmetic expression"}},
            "required": ["expression"],
        },
    },
}

#: Calls honoured per turn; three are needed. The cap only stops a policy that
#: has learned to spam calls from blowing up the sequence.
MAX_CALLS_PER_TURN = 8
MAX_EXPRESSION_CHARS = 400


def calculate(expression: str) -> str:
    """Evaluate one expression by walking its AST -- numbers, + - * / ** %, unary
    signs, round/abs/min/max, nothing else -- and return what the tool replies.
    Errors come back as `error: ...` text: a bad call costs a turn, not a run."""
    import ast
    import math
    import operator

    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
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
        value = float(walk(ast.parse(expression.strip(), mode="eval")))
        if not math.isfinite(value):
            return "error: result is not finite"
    except ZeroDivisionError:
        return "error: division by zero"
    except (SyntaxError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        return f"error: {str(exc) or type(exc).__name__}"
    # Ten significant figures: nothing the task needs is lost, and float noise
    # (-22.400000000000002) never reaches the policy.
    return format(value, ".10g")


def parse_tool_calls(turn: str) -> list[dict[str, Any]]:
    """Every `<tool_call>` block in one assistant turn, in Qwen3.5's XML form.
    A block that is not a valid calculator call is returned with `error` set, so
    it is answered with an error rather than skipped."""
    calls = []
    for body in re.findall(r"<tool_call>(.*?)</tool_call>", turn, flags=re.S):
        fn = re.fullmatch(r"\s*<function=([^>\s]+)>(.*?)</function>\s*", body, flags=re.S)
        if fn is None:
            calls.append({"error": "error: malformed tool call"})
            continue
        if fn.group(1) != "calculator":
            calls.append({"error": f"error: unknown tool {fn.group(1)!r}"})
            continue
        params = dict(re.findall(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", fn.group(2), flags=re.S))
        if "expression" not in params:
            calls.append({"error": "error: missing parameter 'expression'"})
            continue
        calls.append({"expression": params["expression"].strip()})
    return calls


def run_tool_call(call: dict[str, Any]) -> str:
    return call["error"] if "error" in call else calculate(call["expression"])


def gold_expressions(record: dict[str, Any]) -> dict[str, str]:
    """The three expressions a correct policy would write, from the numbers as
    the prompt prints them. The notebook checks that they reproduce the key."""
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


def render_tool_prompt(tokenizer, record: dict[str, Any]) -> str:
    """v2's two messages, the calculator declared, thinking off."""
    return tokenizer.apply_chat_template(
        build_messages(record, PROMPT_VERSION), tools=[CALCULATOR_TOOL], tokenize=False,
        add_generation_prompt=True, **TEMPLATE_KWARGS,
    )


def generation_prefix(tokenizer) -> str:
    """What `add_generation_prompt` appends: the assistant header and, with
    thinking off, the empty think block."""
    probe = [{"role": "user", "content": "x"}]
    on = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True, **TEMPLATE_KWARGS)
    off = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=False, **TEMPLATE_KWARGS)
    return on[len(off):]


def call_block(expression: str) -> str:
    return (f"<tool_call>\n<function=calculator>\n<parameter=expression>\n{expression}\n"
            "</parameter>\n</function>\n</tool_call>")


def tool_call_target(record: dict[str, Any]) -> str:
    """The first assistant turn as the seed supervises it: three calls, nothing
    else, exactly as the chat template writes an assistant message carrying them."""
    return "\n".join(call_block(e) for e in gold_expressions(record).values())


def tool_response_text(results: list[str], prefix: str) -> str:
    """Everything between the policy's `<|im_end|>` and its next turn."""
    body = "".join(f"\n<tool_response>\n{r}\n</tool_response>" for r in results)
    return "\n<|im_start|>user" + body + "<|im_end|>\n" + prefix


# --- tool episodes ------------------------------------------------------------


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
    unanswered: int = 0
    truncated: bool = False
    #: The episode ended because its answer's JSON object closed (`stop_at_json`).
    answer_closed: bool = False
    #: Index, among the policy's own tokens, of the first token of each turn.
    turn_starts: list[int] = field(default_factory=list)

    @property
    def policy_tokens(self) -> int:
        return sum(self.policy)


# --- the turn protocol ----------------------------------------------------------
#
# The chat template ends every turn with `<|im_end|>`, and that is the only token
# that ends a turn here. Qwen3.5 has a second end token, `<|endoftext|>` -- it
# ends a document, and the model writes it after `<|im_end|>\n` -- and
# `generation_config` stops on that one alone. Left available, it is a second way
# to end a turn that the harness cannot hand to the tool: `base-s0`'s first run
# moved P(`<|endoftext|>`) after the third call from 0.0000 to 0.998 in four PPO
# steps, and every episode then ended before the calculator ran.
#
# So the policy is the model with `<|endoftext|>` removed from its vocabulary --
# in sampling, in the log-probabilities PPO trains on, and in the entropy -- and
# generation stops at `<|im_end|>`. It carries no task content and changes no
# reward. `suppressed_mass` in the metrics is how much probability the raw model
# would have put on it, so a drift towards it stays visible.


def end_tokens(tokenizer) -> tuple[int, int]:
    """(`<|im_end|>`, `<|endoftext|>`)."""
    return tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")


#: What a suppressed logit is set to. Finite, so `0 * logit` in the entropy stays 0.
SUPPRESSED = -1e9


# --- the answer ends where its JSON object does -----------------------------------
#
# The prompt asks for one JSON object. With `stop_at_json`, a turn ends the moment
# the first top-level object in it closes, the way JSON-mode decoding ends: the
# policy never gets to write after its answer. `base-s0`'s runs showed what it
# writes there when it can -- the answer again after a `</think>`, then prose, to
# the end of the budget -- and nothing in the reward says not to, because text
# after the graded object earns exactly nothing either way.
#
# "Closes" is decided by the scorer's own brace scanner (`task.schema.
# _json_object_spans`, mirrored character for character in `JsonScanner`): braces
# inside strings do not count, and the nested `flags` object does not end it.


class JsonScanner:
    """`task.schema._json_object_spans`'s state machine, fed incrementally: reports
    when the first top-level `{...}` has closed."""

    def __init__(self) -> None:
        self.depth, self.in_string, self.escaped, self.closed = 0, False, False, False

    def feed(self, text: str) -> bool:
        for ch in text:
            if self.closed:
                break
            if self.in_string:
                if self.escaped:
                    self.escaped = False
                elif ch == "\\":
                    self.escaped = True
                elif ch == '"':
                    self.in_string = False
                continue
            if ch == '"':
                self.in_string = True
            elif ch == "{":
                self.depth += 1
            elif ch == "}" and self.depth:
                self.depth -= 1
                self.closed = self.depth == 0
        return self.closed


def _token_text(tokenizer, token: int, cache: dict[int, str]) -> str:
    text = cache.get(token)
    if text is None:
        text = cache[token] = tokenizer.decode([token])
    return text


def json_close_index(tokens: list[int], tokenizer, cache: dict[int, str]) -> int | None:
    """Index of the token that closes the first top-level JSON object, or None."""
    scanner = JsonScanner()
    for k, t in enumerate(tokens):
        if scanner.feed(_token_text(tokenizer, t, cache)):
            return k
    return None


class _StopAtJsonClose:
    """A `generate` stopping criterion: a row is done once its first JSON object closes."""

    def __init__(self, tokenizer, width: int, rows: int, cache: dict[int, str]):
        self.tokenizer, self.seen, self.cache = tokenizer, width, cache
        self.scanners = [JsonScanner() for _ in range(rows)]

    def __call__(self, input_ids: torch.Tensor, scores, **kwargs) -> torch.Tensor:
        new = input_ids[:, self.seen:].tolist()
        self.seen = input_ids.shape[1]
        for scanner, tokens in zip(self.scanners, new):
            for t in tokens:
                if scanner.feed(_token_text(self.tokenizer, t, self.cache)):
                    break
        return torch.tensor([s.closed for s in self.scanners], device=input_ids.device)


def _sample_round(policy, inputs, budgets, *, pad: int, temperature: float, device,
                  im_end: int, suppress: int, stop_at_json: bool = False, tokenizer=None,
                  cache: dict[int, str] | None = None) -> list[list[int]]:
    from transformers import StoppingCriteriaList

    width = max(len(x) for x in inputs)
    input_ids = torch.tensor([[pad] * (width - len(x)) + x for x in inputs], device=device)
    attention = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in inputs], device=device)
    sample = temperature > 0
    stopping = (StoppingCriteriaList([_StopAtJsonClose(tokenizer, width, len(inputs), cache)])
                if stop_at_json else None)
    out = policy.generate(
        input_ids=input_ids, attention_mask=attention, do_sample=sample,
        temperature=temperature if sample else None, top_p=1.0 if sample else None,
        max_new_tokens=max(budgets), pad_token_id=pad, eos_token_id=im_end, suppress_tokens=[suppress],
        stopping_criteria=stopping,
    )
    return [row[:b] for row, b in zip(out[:, width:].tolist(), budgets)]


@torch.no_grad()
def tool_generate(policy, tokenizer, records, *, temperature: float, max_new_tokens: int,
                  max_rounds: int = 2, stop_at_json: bool = False) -> list[Episode]:
    """Decode a batch with the calculator in the loop.

    `max_new_tokens` caps the policy's tokens summed over its turns. The tool may
    answer `max_rounds` times; a turn that still calls it after that ends the
    episode with its calls unanswered. A turn without calls ends the episode.
    With `stop_at_json`, a turn whose first JSON object closes before any
    `<|im_end|>` ends there, and so does the episode: that object is the answer.
    """
    device = next(policy.parameters()).device
    pad = tokenizer.pad_token_id
    im_end, suppress = end_tokens(tokenizer)
    prefix = generation_prefix(tokenizer)
    cache: dict[int, str] = {}
    episodes = [Episode(tokenizer(render_tool_prompt(tokenizer, r), add_special_tokens=False).input_ids)
                for r in records]
    live = list(range(len(episodes)))
    for rnd in range(max_rounds + 1):
        budget = {i: max_new_tokens - episodes[i].policy_tokens for i in live}
        live = [i for i in live if budget[i] > 0]
        if not live:
            break
        rows = _sample_round(policy, [episodes[i].prompt_ids + episodes[i].ids for i in live],
                             [budget[i] for i in live], pad=pad, temperature=temperature, device=device,
                             im_end=im_end, suppress=suppress, stop_at_json=stop_at_json,
                             tokenizer=tokenizer, cache=cache)
        still = []
        for row, i in zip(rows, live):
            ep = episodes[i]
            cut = next((k for k, t in enumerate(row) if t == im_end), None)
            closed = json_close_index(row, tokenizer, cache) if stop_at_json else None
            if closed is not None and (cut is None or closed < cut):
                ep.turn_starts.append(ep.policy_tokens)
                ep.ids += row[: closed + 1]
                ep.policy += [True] * (closed + 1)
                ep.answer_closed = True
                continue
            take = row if cut is None else row[: cut + 1]
            ep.turn_starts.append(ep.policy_tokens)
            ep.ids += take
            ep.policy += [True] * len(take)
            if cut is None:
                ep.truncated = True
                continue
            calls = parse_tool_calls(tokenizer.decode(take, skip_special_tokens=True))
            if not calls:
                continue
            if rnd == max_rounds:
                ep.unanswered += len(calls)
                continue
            calls = calls[:MAX_CALLS_PER_TURN]
            results = [run_tool_call(c) for c in calls]
            ep.calls += calls
            ep.results += results
            env = tokenizer(tool_response_text(results, prefix), add_special_tokens=False).input_ids
            ep.ids += env
            ep.policy += [False] * len(env)
            still.append(i)
        live = still
    for ep in episodes:
        ep.text = tokenizer.decode([t for t, p in zip(ep.ids, ep.policy) if p], skip_special_tokens=True)
    return episodes


def tool_stats(episodes: list[Episode]) -> dict[str, float]:
    n = max(1, len(episodes))
    calls = sum(len(ep.calls) for ep in episodes)
    errors = sum(r.startswith("error") for ep in episodes for r in ep.results)
    return {
        "tool_use_rate": sum(bool(ep.calls) for ep in episodes) / n,
        "tool_calls_mean": calls / n,
        "tool_error_rate": errors / calls if calls else 0.0,
        "tool_unanswered": sum(ep.unanswered for ep in episodes) / n,
        "truncated_rate": sum(ep.truncated for ep in episodes) / n,
        "answer_closed_rate": sum(ep.answer_closed for ep in episodes) / n,
        "policy_tokens_mean": sum(ep.policy_tokens for ep in episodes) / n,
    }


def turn_record(episodes: list[Episode], advantages: torch.Tensor, logprobs: torch.Tensor,
                turn_end_mask: torch.Tensor) -> dict[str, float | None]:
    """How long each turn ran; the advantage on the token that ended it -- the first
    turn's (the hand-over to the tool) and the last's (the end of the answer); and,
    where that token is `<|im_end|>`, the probability the policy gave it."""
    first, last, adv_first, adv_last, turns, p_first, p_last = [], [], [], [], [], [], []
    for b, ep in enumerate(episodes):
        if not ep.turn_starts:
            continue
        bounds = ep.turn_starts + [ep.policy_tokens]
        lengths = [bounds[k + 1] - bounds[k] for k in range(len(ep.turn_starts))]
        turns.append(len(lengths))
        first.append(lengths[0])
        last.append(lengths[-1])
        if len(lengths) > 1:
            adv_first.append(float(advantages[b, bounds[1] - 1]))
            if turn_end_mask[b, bounds[1] - 1]:
                p_first.append(float(logprobs[b, bounds[1] - 1].exp()))
        adv_last.append(float(advantages[b, ep.policy_tokens - 1]))
        if turn_end_mask[b, ep.policy_tokens - 1]:
            p_last.append(float(logprobs[b, ep.policy_tokens - 1].exp()))

    def mean(x):
        return round(sum(x) / len(x), 6) if x else None

    return {"turns_mean": mean(turns), "first_turn_tokens": mean(first), "last_turn_tokens": mean(last),
            "adv_first_turn_end": mean(adv_first), "adv_last_turn_end": mean(adv_last),
            "p_first_turn_end": mean(p_first), "p_last_turn_end": mean(p_last)}


# --- dense per-field credit ---------------------------------------------------

FIELD_PATTERNS: tuple[tuple[str, str], ...] = (
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


def graded_span(completion: str, obj: dict[str, Any]) -> tuple[int, int] | None:
    """Character span of the *first* JSON object in `completion` whose content is
    the object `parse_answer` graded.

    `parse_answer` grades the object with the most answer keys, a later one winning
    ties -- so an answer written twice is graded on its second copy. But the values
    were decided where they were first written; the copy repeats them. Credit placed
    on the copy is credit moved later for free: in `base-s0-duplicate-at-120` the same
    0.540 left 0.380 still to be earned after the first object closed, so the token
    that started a second copy drew an advantage the answer itself had already
    earned, and PPO learned to write every answer twice."""
    from task.schema import _json_object_spans

    stripped = completion.strip()
    lead = len(completion) - len(completion.lstrip())
    try:
        if json.loads(stripped) == obj:
            return lead, lead + len(stripped)
    except (json.JSONDecodeError, ValueError):
        pass
    pos = 0
    for span in _json_object_spans(completion):
        start = completion.find(span, pos)
        pos = start + len(span)
        try:
            if json.loads(span) == obj:
                return start, start + len(span)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def field_credits(completion: str, answer: dict[str, Any], weights: Weights,
                  flat_scale: float = 1.0) -> tuple[dict[str, tuple[int, float]], float, int | None]:
    """One completion's reward as `{field: (char offset, credit)}`, a rest, and the
    offset the rest is paid at. With `flat_scale = 1` the pieces sum to `reward.score`.

    Every piece is placed inside `graded_span`: the fields where they are written,
    the rest (schema validity, and anything not located) on the brace that closes
    the object. Text after it earns nothing. Offset None means no such span was
    found; the rest then goes to the last token."""
    parsed = parse_answer(completion)
    obj = parsed.obj if isinstance(parsed.obj, dict) else None
    if obj is None:
        return {}, 0.0, None
    span = graded_span(completion, obj)
    region, base0 = (completion[span[0]: span[1]], span[0]) if span else (completion, 0)
    numeric, flags = _numeric_hits(obj, answer), _flag_hits(obj, answer)

    def flag_credit(key: str, hit: bool) -> float:
        return weights.flags / 3 * (flat_scale if answer["flags"][key] == "flat" else 1.0) * hit

    earned = {
        **{k: weights.numeric / 3 * hit for k, hit in zip(NUMERIC_KEYS, numeric)},
        **{f"flags.{k}": flag_credit(k, hit) for k, hit in zip(FLAG_KEYS, flags)},
        "stage": weights.stage * (obj.get("stage") == answer["stage"]),
        "root_cause": weights.root_cause * (obj.get("root_cause") == answer["root_cause"]),
        "action": weights.action * (obj.get("action") == answer["action"]),
    }
    rest = weights.format * float(validate(obj).ok)
    flags_block = re.search(r'"flags"\s*:\s*\{[^}]*\}', region)
    placed: dict[str, tuple[int, float]] = {}
    for name, pattern in FIELD_PATTERNS:
        credit = float(earned.get(name, 0.0))
        if not credit:
            continue
        if name.startswith("flags."):
            if flags_block is None:
                rest += credit
                continue
            where, base = flags_block.group(), base0 + flags_block.start()
        else:
            where, base = region, base0
        hit = None
        for hit in re.finditer(pattern, where):
            pass
        if hit is None:
            rest += credit
            continue
        placed[name] = (base + hit.end(), credit)
    return placed, rest, (span[1] if span else None)


def dense_rewards(completions, completion_ids, mask, cases, weights, tokenizer, flat_scale=1.0) -> torch.Tensor:
    """Per-token rewards on the policy's own tokens, each field's credit on the
    token that completes it."""
    out = torch.zeros_like(mask, dtype=torch.float32)
    lengths = mask.sum(dim=-1).long()
    for b, (text, case) in enumerate(zip(completions, cases)):
        length = int(lengths[b])
        if length == 0:
            continue
        ids = completion_ids[b, :length].tolist()
        bounds = [len(tokenizer.decode(ids[: i + 1], skip_special_tokens=True)) for i in range(length)]
        try:
            placed, rest, rest_at = field_credits(text, case["answer"], weights, flat_scale)
        except (OverflowError, ValueError, TypeError, KeyError):
            placed, rest, rest_at = {}, 0.0, None
        for offset, credit in placed.values():
            index = next((i for i, end in enumerate(bounds) if end >= offset), length - 1)
            out[b, index] += credit
        rest_index = length - 1 if rest_at is None else next(
            (i for i, end in enumerate(bounds) if end >= rest_at), length - 1)
        out[b, rest_index] += rest
    return out


# --- GAE, the critic's scorecard, the losses -----------------------------------


def gae(rewards, values, mask, *, gamma: float = 1.0, lam: float = 1.0):
    batch, tokens = mask.shape
    advantages = torch.zeros_like(values)
    running = torch.zeros(batch, dtype=values.dtype, device=values.device)
    for t in range(tokens - 1, -1, -1):
        next_value = values[:, t + 1] * mask[:, t + 1] if t + 1 < tokens else torch.zeros_like(running)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        running = (delta + gamma * lam * running) * mask[:, t]
        advantages[:, t] = running
    return advantages * mask, (advantages + values) * mask


def position_clock(returns, mask, bins: int = 20):
    """Mean return at each relative position, fitted on the batch itself."""
    sel = mask > 0
    out = torch.zeros_like(returns)
    index = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0).expand_as(mask)
    lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    binned = (index / lengths * bins).floor().clamp(max=bins - 1).long()
    for b in range(bins):
        hit = (binned == b) & sel
        if hit.any():
            out[hit] = returns[hit].mean()
    return out


def critic_fit(values, returns, mask) -> dict[str, float | None]:
    sel = mask > 0
    v, g = values[sel], returns[sel]
    g_var = float(g.var(unbiased=False)) if g.numel() > 1 else 0.0
    std = round(float(v.std(unbiased=False)), 6) if v.numel() > 1 else 0.0
    if g_var < 1e-6:
        return {"value_ev": None, "value_ev_position": None, "value_std": std}
    clock = position_clock(returns, mask)[sel]
    return {
        "value_ev": round(1.0 - float((g - v).var(unbiased=False)) / g_var, 6),
        "value_ev_position": round(1.0 - float((g - clock).var(unbiased=False)) / g_var, 6),
        "value_std": std,
    }


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    return torch.logsumexp(logits, dim=-1) - (torch.softmax(logits, dim=-1) * logits).sum(dim=-1)


def ppo_actor_loss(logprobs, old_logprobs, advantages, mask, *, clip_eps, entropy, entropy_coef, entropy_mask):
    """Clipped surrogate per token, averaged over the policy's tokens, minus
    `entropy_coef` times the entropy summed over `entropy_mask` (the answer turn)
    and divided by *all* the policy's tokens -- `13b`'s scoping, which leaves each
    answer token pushed exactly as hard as an unscoped bonus would push it."""
    active = mask.sum().clamp(min=1.0)
    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -(torch.min(unclipped, clipped) * mask).sum() / active
    stats: dict[str, float] = {"actor_loss": float(loss.detach())}
    if entropy is not None:
        scoped = entropy_mask * mask
        loss = loss - entropy_coef * (entropy * scoped).sum() / active
        stats["answer_entropy"] = float((entropy * scoped).sum().detach() / scoped.sum().clamp(min=1.0))
    with torch.no_grad():
        mean_adv = (advantages * mask).sum() / active
        stats.update({
            "ratio_mean": float((ratio * mask).sum() / active),
            "clip_frac": float(((unclipped > clipped) & (mask > 0)).float().sum() / active),
            "adv_mean": float(mean_adv),
            "adv_std": float(((((advantages - mean_adv) * mask) ** 2).sum() / active).sqrt()),
            "entropy_proxy": float(-(logprobs * mask).sum() / active),
        })
    return loss, stats


def value_loss(values, old_values, returns, *, clip_eps: float | None):
    plain = (values - returns) ** 2
    if clip_eps is None:
        loss = plain.mean()
    else:
        moved = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps)
        loss = torch.max(plain, (moved - returns) ** 2).mean()
    with torch.no_grad():
        stats = {"value_loss": float(loss), "value_mean": float(values.mean()),
                 "return_mean": float(returns.mean()), "value_mae": float((values - returns).abs().mean())}
    return loss, stats


# --- the model side -----------------------------------------------------------


class ValueHead(nn.Module):
    """`V(s_t) = w . h_t + b` on one detached hidden layer, in float32; zero
    weight at init, so V starts at exactly `init_bias` everywhere."""

    def __init__(self, hidden_size: int, init_bias: float):
        super().__init__()
        self.v = nn.Linear(hidden_size, 1, dtype=torch.float32)
        nn.init.zeros_(self.v.weight)
        nn.init.constant_(self.v.bias, init_bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.v(hidden.float()).squeeze(-1)


def padded_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    return (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)


def take(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather the policy's positions out of a (batch, span[, d]) tensor."""
    if x.dim() == 3:
        return torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    return torch.gather(x, 1, index)


def forward_policy(model, sequences, attention_mask, span: int, index, *, suppress: int,
                   hidden_layer: int | None = None, want_entropy: bool = False):
    """Log-probs (and the critic's features, or the entropy) at the policy's own
    tokens, under the turn protocol: `suppress` is not in the policy's vocabulary.
    The tool's tokens are context: no log-prob, reward or value.

    With `hidden_layer` (the no-grad pass) it also returns the probability the raw
    model puts on `suppress` at each of those positions."""
    out = model(input_ids=sequences, attention_mask=attention_mask,
                position_ids=padded_position_ids(attention_mask), logits_to_keep=span + 1,
                output_hidden_states=hidden_layer is not None)
    raw = out.logits[:, :-1].float()
    mass = None
    if hidden_layer is not None:
        mass = take(torch.exp(raw[..., suppress] - torch.logsumexp(raw, dim=-1)), index)
    logits = raw.index_fill(-1, torch.tensor([suppress], device=raw.device), SUPPRESSED)
    del raw
    logprobs = take(selective_logprobs(logits, sequences[:, -span:]), index)
    entropy = take(token_entropy(logits), index) if want_entropy else None
    hidden = None
    if hidden_layer is not None:
        hidden = take(out.hidden_states[hidden_layer][:, -(span + 1): -1].detach(), index)
    return logprobs, hidden, entropy, mass


def load_policy(model_id: str, lora_r: int, device: str, adapter: str | Path | None = None):
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    policy = get_peft_model(
        base, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, target_modules=list(LORA_TARGETS), task_type="CAUSAL_LM"),
    ).to(device)
    if adapter:
        set_peft_model_state_dict(policy, load_file(str(Path(adapter) / "adapter_model.safetensors"), device=str(device)))
    return policy, tokenizer


# --- rollouts and evaluation --------------------------------------------------


@dataclass
class Rollout:
    cases: list[dict[str, Any]]
    episodes: list[Episode]
    sequences: torch.Tensor  # (batch, prompt + span): left-padded prompt, the episode, right padding
    attention_mask: torch.Tensor
    span: int  # the episode columns at the right end of `sequences`
    index: torch.Tensor  # (batch, n): column in the span of the k-th policy token
    policy_ids: torch.Tensor  # (batch, n): the policy's tokens, compacted
    mask: torch.Tensor  # (batch, n): 1 on real policy tokens
    answer_mask: torch.Tensor  # (batch, n): 1 on the last turn's tokens
    #: (batch, n): 1 on the `<|im_end|>` that ends a turn -- the hand-over to the tool
    #: after the calls, and the stop after the answer.
    turn_end_mask: torch.Tensor
    completions: list[str]
    rewards: list[float]


@torch.no_grad()
def tool_rollout(policy, tokenizer, cases, *, max_new_tokens, temperature, weights, max_rounds=2,
                 stop_at_json: bool = False) -> Rollout:
    device = next(policy.parameters()).device
    episodes = tool_generate(policy, tokenizer, [c["record"] for c in cases], temperature=temperature,
                             max_new_tokens=max_new_tokens, max_rounds=max_rounds, stop_at_json=stop_at_json)
    pad = tokenizer.pad_token_id
    im_end, _ = end_tokens(tokenizer)
    width = max(len(ep.prompt_ids) for ep in episodes)
    span = max(1, max(len(ep.ids) for ep in episodes))
    n = max(1, max(ep.policy_tokens for ep in episodes))
    seqs, attn, index, ids, mask, answer, turn_end = [], [], [], [], [], [], []
    for ep in episodes:
        left, right = width - len(ep.prompt_ids), span - len(ep.ids)
        seqs.append([pad] * left + ep.prompt_ids + ep.ids + [pad] * right)
        attn.append([0] * left + [1] * (len(ep.prompt_ids) + len(ep.ids)) + [0] * right)
        where = [k for k, mine in enumerate(ep.policy) if mine]
        fill = n - len(where)
        index.append(where + [0] * fill)
        mine = [ep.ids[k] for k in where]
        ids.append(mine + [pad] * fill)
        mask.append([1.0] * len(where) + [0.0] * fill)
        last = ep.turn_starts[-1] if ep.turn_starts else 0
        answer.append([0.0] * last + [1.0] * (len(where) - last) + [0.0] * fill)
        ends = [0.0] * n
        for k in (ep.turn_starts[1:] + [len(where)]):
            if k > 0 and mine[k - 1] == im_end:
                ends[k - 1] = 1.0
        turn_end.append(ends)
    completions = [ep.text for ep in episodes]
    scored = [safe_score(c, case["answer"], weights) for c, case in zip(completions, cases)]
    t = lambda x, **kw: torch.tensor(x, device=device, **kw)  # noqa: E731
    return Rollout(
        cases=list(cases), episodes=episodes, sequences=t(seqs), attention_mask=t(attn), span=span,
        index=t(index), policy_ids=t(ids), mask=t(mask), answer_mask=t(answer), turn_end_mask=t(turn_end),
        completions=completions, rewards=[s.total if s is not None else 0.0 for s in scored],
    )


def sampled_numeric(completions, cases) -> float:
    hits = []
    for text, case in zip(completions, cases):
        s = safe_score(text, case["answer"], ABLATE)
        hits.append(((s.diagnostics.get("numeric_correct", 0) or 0) / 3) if s is not None else 0.0)
    return sum(hits) / max(1, len(hits))


def evaluate(policy, tokenizer, cases, *, weights: Weights, batch: int, max_tokens: int,
             temperature: float = 0.0, samples: int = 1, stop_at_json: bool = False):
    """Tool episodes over `cases` (greedy by default), scored by `eval.summarise`
    on the policy's own text -- the same scorer every run in the repository uses."""
    was_training = policy.training
    policy.eval()
    results, episodes = [], []
    expanded = [c for c in cases for _ in range(samples)]
    for start in range(0, len(expanded), batch):
        chunk = expanded[start: start + batch]
        eps = tool_generate(policy, tokenizer, [c["record"] for c in chunk], temperature=temperature,
                            max_new_tokens=max_tokens, stop_at_json=stop_at_json)
        episodes += eps
    for i, case in enumerate(cases):
        mine = episodes[i * samples: (i + 1) * samples]
        results.append(CaseResult(case, [Sample(case["id"], ep.text, ep.policy_tokens) for ep in mine]))
    if was_training:
        policy.train()
    return results, episodes, {**summarise(results, weights), **tool_stats(episodes)}


EVAL_KEYS = ("reward", "exact_match", "validity_gate", "schema_ok", "cause_acc", "flags_acc", "numeric_acc",
             "action_acc", "cause_given_flags", "predicted_cause_hist", "tool_use_rate", "tool_calls_mean",
             "tool_error_rate", "tool_unanswered", "truncated_rate", "answer_closed_rate", "policy_tokens_mean")


def write_eval(path: Path, metrics: dict[str, Any], episodes: list[Episode], cases, meta: dict[str, Any]) -> None:
    """The envelope `../smoke_test/runs/tool/*.json` uses, and every episode beside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"model": MODEL, "backend": "hf+calculator", "prompt_version": PROMPT_VERSION,
                "tools": [CALCULATOR_TOOL], **meta, "overall": metrics}
    path.write_text(json.dumps(envelope, indent=2, default=str) + "\n")
    with path.with_suffix(".episodes.jsonl").open("w") as fh:
        samples = len(episodes) // max(1, len(cases))
        for i, ep in enumerate(episodes):
            fh.write(json.dumps({"id": cases[i // samples]["id"], "text": ep.text, "calls": ep.calls,
                                 "results": ep.results, "truncated": ep.truncated,
                                 "policy_tokens": ep.policy_tokens}) + "\n")


# --- the tool seed ------------------------------------------------------------


#: The first token of the answer turn: the `{` that opens the JSON. The same on
#: every case, so it carries nothing about any case's answer.
ANSWER_OPEN = "{"


def seed_example(case, tokenizer, anchor_open: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Prompt, the three gold calls, `<|im_end|>`; the loss on the calls and the
    `<|im_end|>` that hands the turn to the tool.

    With `anchor_open` the example continues the way the episode does -- the
    calculator's replies to the gold calls and the next assistant header, as
    context -- and ends on the answer turn's first token, which is labelled too.
    Trained on the calls alone, the seed's second turn repeated the calls instead
    of answering (4.9 calls per dev episode, 1 answer in 100): the second
    assistant header looked like the first, and the first had only ever been
    followed by a call. `../smoke_test/14` met the same thing and anchored the
    same token."""
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    record = case["record"]
    prompt_ids = tokenizer(render_tool_prompt(tokenizer, record), add_special_tokens=False).input_ids
    calls_ids = tokenizer(tool_call_target(record), add_special_tokens=False).input_ids + [im_end]
    ids = prompt_ids + calls_ids
    labelled = [False] * len(prompt_ids) + [True] * len(calls_ids)
    if anchor_open:
        results = [calculate(e) for e in gold_expressions(record).values()]
        env = tokenizer(tool_response_text(results, generation_prefix(tokenizer)), add_special_tokens=False).input_ids
        open_ids = tokenizer(ANSWER_OPEN, add_special_tokens=False).input_ids
        ids += env + open_ids
        labelled += [False] * len(env) + [True] * len(open_ids)
    ids = torch.tensor([ids])
    labels = torch.where(torch.tensor([labelled]), ids, torch.full_like(ids, -100))
    return ids, labels


def sft_loss(policy, ids, labels) -> torch.Tensor:
    """Mean next-token cross-entropy, with the 248k-wide head applied only where
    a label is."""
    model = policy.get_base_model()
    hidden = model.model(input_ids=ids).last_hidden_state[:, :-1]
    targets = labels[:, 1:]
    where = targets != -100
    return torch.nn.functional.cross_entropy(model.lm_head(hidden[where]).float(), targets[where])


def train_seed(out_dir: Path, *, epochs: int = 3, lr: float = 1e-4, accum: int = 8, lora_r: int = 16,
               seed: int = 0, split: str = "train", anchor_open: bool = False, progress=print) -> None:
    """`../smoke_test/runs/tool-sft-raw`'s recipe: every case of `split`, shuffled,
    loss on the three calls only (and, with `anchor_open`, the answer turn's first
    token), `epochs` passes at `lr`, one optimiser step per `accum` cases. Each
    epoch's adapter is kept."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    policy, tokenizer = load_policy(MODEL, lora_r, device)
    examples = load_cases(split)
    random.Random(seed).shuffle(examples)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {"model": MODEL, "split": split, "epochs": epochs, "lr": lr, "accum": accum, "lora_r": lora_r,
         "seed": seed, "lora_targets": list(LORA_TARGETS), "anchor_open": anchor_open,
         "loss": "the three gold calls and <|im_end|>" + (", and the answer turn's first token" if anchor_open else "")},
        indent=2) + "\n")
    policy.train()
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    with (out_dir / "losses.jsonl").open("w") as fh:
        for epoch in range(epochs):
            first = last = None
            for i, case in enumerate(examples):
                ids, labels = (x.to(device) for x in seed_example(case, tokenizer, anchor_open))
                loss = sft_loss(policy, ids, labels)
                (loss / accum).backward()
                if (i + 1) % accum == 0 or i + 1 == len(examples):
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                fh.write(json.dumps({"epoch": epoch, "i": i, "loss": float(loss.detach())}) + "\n")
                first = float(loss.detach()) if first is None else first
                last = float(loss.detach())
            policy.save_pretrained(out_dir / f"epoch{epoch + 1}")
            progress(f"  epoch {epoch + 1}: loss {first:.4f} -> {last:.4f}")


# --- the PPO run --------------------------------------------------------------


@dataclass
class Config:
    model: str = MODEL
    #: The adapter PPO starts from: the tool seed.
    start_from: str = ""
    steps: int = 2000
    prompts_per_step: int = 8
    max_new_tokens: int = 640
    tool_rounds: int = 2
    temperature: float = 1.0
    lr: float = 1e-5
    weights: str = "ABLATE"
    gamma: float = 1.0
    lam: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    #: On the answer turn only (`13b`); the calls are left to the seed.
    entropy_coef: float = 0.005
    inner_epochs: int = 4
    micro_batch: int = 1
    lora_r: int = 16
    flat_credit_scale: float = 3.0
    value_layer: int = -1
    value_lr: float = 0.0
    value_init_bias: float = -1.0
    critic_epochs: int = 8
    critic_window: int = 25
    seed: int = 0
    split: str = "train"
    eval_every: int = 25
    eval_split: str = "dev"
    eval_cases: int = 200
    eval_batch: int = 32
    eval_max_tokens: int = 640
    #: Arithmetic gate: stop once sampled numeric_acc, averaged over 10 steps, is below this.
    numeric_stop: float = 0.90
    #: Convergence: stop after this many evaluations without a new best held-out reward.
    patience: int = 8
    #: No-regression gate: held-out reward below step 0's by more than this, twice in a row.
    regress_margin: float = 0.05
    #: Critic gate: from this step, at every evaluation, the median of value_ev minus
    #: the clock over the last 25 steps must be above 0; two failures in a row stop.
    critic_gate_from: int = 40
    #: Every this many steps the batch's episodes go to `samples.jsonl`.
    sample_every: int = 5
    #: Steps at the start on which only the critic is fitted; the actor does not move.
    #: Without it, the head starts at one constant (the mean return over all tokens),
    #: which is far above what is left to earn late in the answer, and the first
    #: steps charge that gap to every token there. In `base-s0`'s run the advantage on
    #: the answer's closing token went -0.40 -> -0.05 over the first eleven steps while
    #: the actor was already moving, and the opening quote of the `action` value fell
    #: to 0.825x its seed probability: unquoted values, unparseable answers, and the
    #: arithmetic gate stopped the run at step 12. `../smoke_test/13` used the same knob.
    critic_warmup: int = 0
    #: 1: the `<|im_end|>` that ends each turn is left out of the actor's loss (and its
    #: entropy term); the critic, the rewards and the episode are unchanged. The token
    #: has no alternative in the protocol -- a finished call is handed to the tool, a
    #: finished answer stops -- and it earns nothing itself, so its advantage is the
    #: critic's own difference across the turn boundary. With warm-up, `base-s0`'s
    #: hand-over token drew a negative advantage on 37 of 38 steps from step 35, and
    #: P(it) after the third call went 1.0000 -> 0.0006 by step 75: the turn never
    #: ended, the calculator never ran. (An int, not a bool: argparse reads "0" as True.)
    freeze_turn_ends: int = 0
    #: 1: a turn ends where its first JSON object closes (see `JsonScanner`), in training and
    #: in every evaluation. (An int for the same argparse reason.)
    stop_at_json: int = 0
    #: KL penalty towards the seed, in the reward: every policy token pays
    #: `kl_coef * (log pi(a) - log pi_seed(a))`, the InstructGPT form, so the critic learns
    #: it with the rest of the return. The seed is near-certain on the answer's scaffolding --
    #: quotes, braces, the `<|im_end|>` that hands over to the tool -- and uncertain on its
    #: content, so leaving the scaffolding is expensive and moving the content is not. Five
    #: times in `base-s0`'s runs a critic that was slightly wrong at one structural token
    #: charged it a small advantage of one sign on almost every step (the closing brace:
    #: negative on 38 of 40 steps, about -0.01), Adam turned that into full-size steps, and
    #: the token was replaced -- `<|endoftext|>`, missing quotes, no hand-over, a second
    #: answer, an object that never closes. Each targeted fix only closed that one exit.
    #: The reference is the adapter in `start_from`. 0 disables it.
    kl_coef: float = 0.0

    def __post_init__(self) -> None:
        if self.weights not in WEIGHT_SETS:
            raise ValueError(f"unknown weights {self.weights!r}")
        if self.value_lr <= 0 or self.value_init_bias < 0:
            raise ValueError("value_lr and value_init_bias are measured in the notebook; pass them")


#: Longer than any healthy step, evaluation included (~3 minutes).
WATCHDOG_SECONDS = 600


def _critic_window_stats(history, n: int = 25) -> dict[str, float | None]:
    rows = [r for r in history[-n:] if r.get("value_ev") is not None and r.get("value_ev_position") is not None]
    if not rows:
        return {"critic_residual_median": None, "critic_ahead_frac": None}
    res = sorted(r["value_ev"] - r["value_ev_position"] for r in rows)
    mid = len(res) // 2
    median = res[mid] if len(res) % 2 else (res[mid - 1] + res[mid]) / 2
    return {"critic_residual_median": round(median, 6), "critic_ahead_frac": round(sum(x > 0 for x in res) / len(res), 4)}


def _save_state(path: Path, policy, value_head, optimizer, critic_optimizer, critic_history, rng, gate, step) -> None:
    """Everything a continuation needs: adapter, head, both optimisers' moments,
    the critic's window and every random state."""
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(path / "adapter")
    torch.save({
        "value_head": value_head.state_dict(), "optimizer": optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(), "critic_history": list(critic_history),
        "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "gate": gate, "step": step,
    }, path / "state.pt")


def train(cfg: Config, run_dir: Path, *, resume: bool = False, progress=print) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = WEIGHT_SETS[cfg.weights]
    cases = load_cases(cfg.split)
    eval_cases = load_cases(cfg.eval_split)[: cfg.eval_cases]
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    adapter = (run_dir / "last" / "adapter") if resume else (HERE / cfg.start_from if cfg.start_from else None)
    policy, tokenizer = load_policy(cfg.model, cfg.lora_r, device, adapter)
    reference = None
    if cfg.kl_coef:
        # The seed, frozen: the same base and adapter the run started from, unmerged, so
        # its forward is the policy's own at step 0 and the penalty starts at exactly 0.
        reference, _ = load_policy(cfg.model, cfg.lora_r, device, HERE / cfg.start_from)
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)
    _, suppress = end_tokens(tokenizer)
    value_head = ValueHead(policy.get_base_model().config.hidden_size, cfg.value_init_bias).to(device)
    actor_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(actor_params, lr=cfg.lr, weight_decay=0.0)
    critic_optimizer = torch.optim.AdamW(value_head.parameters(), lr=cfg.value_lr, weight_decay=0.0)
    critic_history: deque = deque(maxlen=cfg.critic_window)

    metrics_path, eval_path = run_dir / "metrics.jsonl", run_dir / "eval.jsonl"
    history: list[dict[str, Any]] = []
    gate: dict[str, Any] = {"best": None, "best_step": None, "since_best": 0, "regress": 0,
                            "critic_fail": 0, "step0_reward": None}
    start = 0
    if resume:
        state = torch.load(run_dir / "last" / "state.pt", map_location="cpu", weights_only=False)
        value_head.load_state_dict(state["value_head"])
        optimizer.load_state_dict(state["optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
        critic_history.extend(state["critic_history"])
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        gate, start = state["gate"], state["step"]
        history = [r for r in map(json.loads, metrics_path.read_text().splitlines()) if r["step"] < start]
        evals = [r for r in map(json.loads, eval_path.read_text().splitlines()) if r["step"] <= start]
        metrics_path.write_text("".join(json.dumps(r) + "\n" for r in history))
        eval_path.write_text("".join(json.dumps(r) + "\n" for r in evals))
        samples_path = run_dir / "samples.jsonl"
        if samples_path.exists():
            kept = [r for r in map(json.loads, samples_path.read_text().splitlines()) if r["step"] < start]
            samples_path.write_text("".join(json.dumps(r) + "\n" for r in kept))
        progress(f"resumed {run_dir.name} at step {start}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
        metrics_path.write_text("")
        eval_path.write_text("")

    def run_eval(step: int) -> str | None:
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        _, _, m = evaluate(policy, tokenizer, eval_cases, weights=weights, batch=cfg.eval_batch,
                           max_tokens=cfg.eval_max_tokens, stop_at_json=bool(cfg.stop_at_json))
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        record = {"step": step, **{k: m.get(k) for k in EVAL_KEYS}, **_critic_window_stats(history)}
        reason = None
        if step == 0:
            gate["step0_reward"] = record["reward"]
        if gate["best"] is None or record["reward"] > gate["best"]:
            gate.update(best=record["reward"], best_step=step, since_best=0)
            policy.save_pretrained(run_dir / "adapter-best")
            torch.save(value_head.state_dict(), run_dir / "value_head-best.pt")
            (run_dir / "best.json").write_text(json.dumps(record, indent=2) + "\n")
        else:
            gate["since_best"] += 1
            if gate["since_best"] >= cfg.patience:
                reason = (f"converged: {cfg.patience} evaluations without beating held-out reward "
                          f"{gate['best']:.4f} (step {gate['best_step']})")
        gate["regress"] = gate["regress"] + 1 if record["reward"] < gate["step0_reward"] - cfg.regress_margin else 0
        if gate["regress"] >= 2:
            reason = f"no-regression gate: held-out reward {record['reward']:.4f} below step 0 twice"
        if step >= cfg.critic_gate_from and record["critic_residual_median"] is not None:
            gate["critic_fail"] = gate["critic_fail"] + 1 if record["critic_residual_median"] <= 0 else 0
            if gate["critic_fail"] >= 2:
                reason = "critic gate: value_ev not above the position clock at two evaluations"
        record["gate"] = dict(gate)
        with eval_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        if step:
            _save_state(run_dir / "last", policy, value_head, optimizer, critic_optimizer, critic_history, rng, gate, step)
        crit = record["critic_residual_median"]
        progress(f"    eval @{step:<5} reward {record['reward']:.4f} EM {record['exact_match']:.3f} "
                 f"cause {record['cause_acc']:.3f} flags {record['flags_acc']:.3f} numeric {record['numeric_acc']:.3f} "
                 f"tool {record['tool_use_rate']:.2f} critic {'-' if crit is None else f'{crit:+.4f}'} "
                 f"| best {gate['best']:.4f}@{gate['best_step']}")
        return reason

    progress(f"{cfg.model} from {cfg.start_from or 'the raw model'} | ABLATE | gamma {cfg.gamma} lam {cfg.lam} | "
             f"value layer {cfg.value_layer} lr {cfg.value_lr:.3g} bias {cfg.value_init_bias:.4f} | {device}")
    stop = run_eval(0) if (start == 0 and cfg.eval_every) else None

    # A step that runs past WATCHDOG_SECONDS writes every thread's Python stack to
    # stderr (the run's log) and exits, so `02_tool_seed_ppo_run_resume.sh` can pick the
    # run up from its last checkpoint. Under WSL the card has stalled mid-`generate` twice
    # (once for 50 minutes, once for hours: GPU idle, one core spinning on the wait,
    # `dxgkio_escape: Ioctl failed` in dmesg) and crashed once with a CUDA unknown error.
    # SIGUSR1 dumps the stacks on demand.
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    step = start
    while stop is None and step < cfg.steps:
        faulthandler.dump_traceback_later(WATCHDOG_SECONDS, exit=True)
        t0 = time.perf_counter()
        batch = [rng.choice(cases) for _ in range(cfg.prompts_per_step)]
        policy.eval()
        roll = tool_rollout(policy, tokenizer, batch, max_new_tokens=cfg.max_new_tokens,
                            temperature=cfg.temperature, weights=weights, max_rounds=cfg.tool_rounds,
                            stop_at_json=bool(cfg.stop_at_json))
        gen_seconds = time.perf_counter() - t0
        seqs, attn, span, index, mask = roll.sequences, roll.attention_mask, roll.span, roll.index, roll.mask
        rewards = dense_rewards(roll.completions, roll.policy_ids, mask, roll.cases, weights, tokenizer,
                                cfg.flat_credit_scale)

        policy.train()
        with torch.no_grad():
            old_lp, old_h, old_mass = [], [], []
            for i in range(0, len(seqs), cfg.micro_batch):
                sl = slice(i, i + cfg.micro_batch)
                lp, h, _, mass = forward_policy(policy, seqs[sl], attn[sl], span, index[sl], suppress=suppress,
                                                hidden_layer=cfg.value_layer)
                old_lp.append(lp)
                old_h.append(h)
                old_mass.append(mass)
            old_logprobs, hidden, suppressed = torch.cat(old_lp), torch.cat(old_h), torch.cat(old_mass)
            kl = torch.zeros_like(old_logprobs)
            if reference is not None:
                ref_lp = torch.cat([
                    forward_policy(reference, seqs[i:i + cfg.micro_batch], attn[i:i + cfg.micro_batch], span,
                                   index[i:i + cfg.micro_batch], suppress=suppress)[0]
                    for i in range(0, len(seqs), cfg.micro_batch)
                ])
                kl = (old_logprobs - ref_lp) * mask
                rewards = rewards - cfg.kl_coef * kl
            old_values = value_head(hidden) * mask
            advantages, returns = gae(rewards, old_values, mask, gamma=cfg.gamma, lam=cfg.lam)

        sel = mask > 0
        critic_history.append((hidden[sel].cpu(), old_values[sel].cpu(), returns[sel].cpu()))
        fit_h = torch.cat([h for h, _, _ in critic_history]).to(device)
        fit_v = torch.cat([v for _, v, _ in critic_history]).to(device)
        fit_r = torch.cat([r for _, _, r in critic_history]).to(device)
        critic_stats: dict[str, float] = {}
        for _ in range(cfg.critic_epochs):
            critic_optimizer.zero_grad(set_to_none=True)
            v_loss, critic_stats = value_loss(value_head(fit_h), fit_v, fit_r, clip_eps=cfg.value_clip_eps)
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_head.parameters(), 1.0)
            critic_optimizer.step()
        with torch.no_grad():
            fitted_values = value_head(hidden) * mask
        del fit_h, fit_v, fit_r

        # The tokens the actor's loss covers: every policy token, or all but the
        # `<|im_end|>` that ends each turn (`freeze_turn_ends`).
        actor_mask = mask * (1 - roll.turn_end_mask) if cfg.freeze_turn_ends else mask
        actor_stats: dict[str, float] = {}
        grad_norm = torch.zeros(())
        warming = step < cfg.critic_warmup
        if warming:
            active = mask.sum().clamp(min=1.0)
            mean_adv = (advantages * mask).sum() / active
            actor_stats = {
                "adv_mean": float(mean_adv),
                "adv_std": float(((((advantages - mean_adv) * mask) ** 2).sum() / active).sqrt()),
                "clip_frac": 0.0, "ratio_mean": 1.0, "actor_loss": 0.0,
                "entropy_proxy": float(-(old_logprobs * mask).sum() / active),
            }
        for _ in range(0 if warming else cfg.inner_epochs):
            optimizer.zero_grad(set_to_none=True)
            epoch_stats: dict[str, float] = {}
            for i in range(0, len(seqs), cfg.micro_batch):
                sl = slice(i, i + cfg.micro_batch)
                share = min(cfg.micro_batch, len(seqs) - i) / len(seqs)
                lp, _, ent, _ = forward_policy(policy, seqs[sl], attn[sl], span, index[sl], suppress=suppress,
                                               want_entropy=cfg.entropy_coef > 0)
                loss, stats = ppo_actor_loss(lp, old_logprobs[sl], advantages[sl], actor_mask[sl], clip_eps=cfg.clip_eps,
                                             entropy=ent, entropy_coef=cfg.entropy_coef,
                                             entropy_mask=roll.answer_mask[sl])
                (loss * share).backward()
                for k, v in stats.items():
                    epoch_stats[k] = epoch_stats.get(k, 0.0) + v * share
            grad_norm = torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
            optimizer.step()
            actor_stats = epoch_stats

        calls_region = (1 - roll.answer_mask) * mask
        answer_region = roll.answer_mask * mask
        record = {
            "step": step,
            "critic_warmup": warming,
            "reward_mean": sum(roll.rewards) / len(roll.rewards),
            "kl_mean": round(float(kl.sum() / mask.sum().clamp(min=1)), 6),
            "kl_answer": round(float((kl * roll.answer_mask).sum() / (roll.answer_mask * mask).sum().clamp(min=1)), 6),
            "kl_calls": round(float((kl * (1 - roll.answer_mask)).sum() / ((1 - roll.answer_mask) * mask).sum().clamp(min=1)), 6),
            "kl_max": round(float(kl.max()), 4),
            "kl_penalty_per_episode": round(float(cfg.kl_coef * kl.sum() / kl.shape[0]), 6),
            "train_numeric_acc": round(sampled_numeric(roll.completions, roll.cases), 6),
            **{k: round(v, 6) for k, v in tool_stats(roll.episodes).items()},
            **turn_record(roll.episodes, advantages, old_logprobs, roll.turn_end_mask),
            "suppressed_mass_mean": float((suppressed * mask).sum() / mask.sum().clamp(min=1)),
            "suppressed_mass_max": float((suppressed * mask).max()),
            **critic_fit(old_values, returns, mask),
            **{f"{k}_fit": v for k, v in critic_fit(fitted_values, returns, mask).items()},
            **{k: round(v, 6) for k, v in actor_stats.items()},
            **{k: round(v, 6) for k, v in critic_stats.items()},
            "entropy_proxy_calls": round(float(-(old_logprobs * calls_region).sum() / calls_region.sum().clamp(min=1)), 6),
            "entropy_proxy_answer": round(float(-(old_logprobs * answer_region).sum() / answer_region.sum().clamp(min=1)), 6),
            "grad_norm": float(grad_norm),
            "gen_seconds": round(gen_seconds, 2),
            "step_seconds": round(time.perf_counter() - t0, 2),
        }
        record["value_step"] = round(abs(record["value_mean"] - history[-1]["value_mean"]), 6) if history else 0.0
        history.append(record)
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        if step % cfg.sample_every == 0:
            with (run_dir / "samples.jsonl").open("a") as fh:
                for ep, case, reward in zip(roll.episodes, roll.cases, roll.rewards):
                    fh.write(json.dumps({"step": step, "id": case["id"], "reward": reward, "text": ep.text,
                                         "calls": ep.calls, "results": ep.results, "truncated": ep.truncated,
                                         "turn_starts": ep.turn_starts, "policy_tokens": ep.policy_tokens}) + "\n")
        ev, clock = record["value_ev"], record["value_ev_position"]
        progress(f"  step {step:>5} reward {record['reward_mean']:.4f} num {record['train_numeric_acc']:.3f} "
                 f"calls {record['tool_calls_mean']:.2f} V {record['value_mean']:.3f} "
                 f"ev {'undef' if ev is None else f'{ev:+.3f}'} clock {'undef' if clock is None else f'{clock:+.3f}'} "
                 f"adv {record['adv_mean']:+.4f} clip {record['clip_frac']:.3f} tok {record['policy_tokens_mean']:.0f} "
                 f"|g| {record['grad_norm']:.2f} {record['step_seconds']:.1f}s")
        step += 1
        if cfg.eval_every and step % cfg.eval_every == 0:
            stop = run_eval(step)
        if stop is None and cfg.numeric_stop and len(history) >= 10:
            recent = sum(r["train_numeric_acc"] for r in history[-10:]) / 10
            if recent < cfg.numeric_stop:
                stop = f"arithmetic gate: sampled numeric_acc over the last 10 steps {recent:.3f} < {cfg.numeric_stop}"

    faulthandler.cancel_dump_traceback_later()
    policy.save_pretrained(run_dir / "adapter")
    torch.save(value_head.state_dict(), run_dir / "value_head.pt")
    ending = {"stopped_at": step, "reason": stop or f"step budget {cfg.steps} reached",
              "best_step": gate["best_step"], "best_reward": gate["best"]}
    (run_dir / "ending.json").write_text(json.dumps(ending, indent=2) + "\n")
    progress(f"END {ending}")
    return ending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_seed = sub.add_parser("seed", help="train the tool seed")
    p_seed.add_argument("--out", required=True, help=f"directory under {RUNS.name}/")
    p_seed.add_argument("--epochs", type=int, default=3)
    p_seed.add_argument("--lr", type=float, default=1e-4)
    p_seed.add_argument("--anchor-open", action="store_true", help="also label the answer turn's first token")
    p_eval = sub.add_parser("eval", help="tool episodes on a split, greedy unless --temperature")
    p_eval.add_argument("--adapter", default="", help="adapter directory, relative to this file; empty = raw")
    p_eval.add_argument("--split", default="dev")
    p_eval.add_argument("--out", required=True, help="json path, relative to this file")
    p_eval.add_argument("--temperature", type=float, default=0.0)
    p_eval.add_argument("--samples", type=int, default=1)
    p_eval.add_argument("--cases", type=int, default=0)
    p_eval.add_argument("--stop-at-json", type=int, default=0, help="1: the answer turn ends where its JSON closes")
    p_train = sub.add_parser("train", help="the PPO run")
    p_train.add_argument("--run", required=True, help=f"run name; written to {RUNS.name}/<run>")
    p_train.add_argument("--resume", action="store_true")
    for f in fields(Config):
        p_train.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    for p in (p_seed, p_eval, p_train):
        p.add_argument("--max-vram-mib", type=int, default=0, help="cap this process's CUDA allocator")
    args = parser.parse_args()
    if args.max_vram_mib and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_vram_mib * 2**20 / total))
    say = lambda s: print(s, flush=True)  # noqa: E731

    if args.cmd == "seed":
        train_seed(RUNS / args.out, epochs=args.epochs, lr=args.lr, anchor_open=args.anchor_open, progress=say)
    elif args.cmd == "eval":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(0)
        policy, tokenizer = load_policy(MODEL, 16, device, (HERE / args.adapter) if args.adapter else None)
        cases = load_cases(args.split)
        cases = cases[: args.cases] if args.cases else cases
        _, episodes, m = evaluate(policy, tokenizer, cases, weights=ABLATE, batch=32, max_tokens=640,
                                  temperature=args.temperature, samples=args.samples,
                                  stop_at_json=bool(args.stop_at_json))
        write_eval(HERE / args.out, m, episodes, cases,
                   {"split": args.split, "adapter": args.adapter or None, "temperature": args.temperature,
                    "samples": args.samples, "weights": "ABLATE", "stop_at_json": bool(args.stop_at_json)})
        say(json.dumps({k: m.get(k) for k in EVAL_KEYS}, default=str))
    else:
        run_dir = RUNS / args.run
        if args.resume:
            cfg = Config(**json.loads((run_dir / "config.json").read_text()))
        else:
            if run_dir.exists() and any(run_dir.iterdir()):
                raise SystemExit(f"{run_dir} exists; pick another --run or pass --resume")
            cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
        train(cfg, run_dir, resume=args.resume, progress=say)


if __name__ == "__main__":
    main()
