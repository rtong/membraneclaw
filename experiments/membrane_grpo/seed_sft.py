"""A short supervised pass that writes a LoRA adapter for `grpo_scratch.py` to start from.

Why it exists. Part 3 of the memo found that, with the arithmetic and the flags
handed over, greedy decoding of the frozen Qwen3-1.7B never names four of the
seven causes, and never applies the severity override in 37 chances. A policy
gradient can only reweight what appears in a sample, so a label with no
probability mass is out of reach of GRPO however the reward is set. A *seed* --
a small supervised pass before RL -- moves such labels off zero.

What it cannot tell apart on its own. The seed supervises the values of two
slots, `root_cause` and `action`, and nothing else. But the target is
teacher-forced: by the time those tokens are scored, the context already holds
the record, the supplied percent changes and the *correct* flags. So a seed with
correct labels trains "these flags -> this row", which is the lookup itself, not
just "these strings exist". Whether GRPO then needs that mapping, or only the
vocabulary, is the open question.

`--shuffle-labels` is the control that separates them. Every record keeps its
own reading, supplied numbers and correct flags, but its `(root_cause, action)`
pair is taken from another record by one permutation of the whole set. Each
label occurs exactly as often as in the correct seed, and so does each pair, so
the vocabulary installed -- and the number of cases the dead-label weight
applies to -- is identical; the mapping from a record's flags to its labels is
destroyed. A seed that only needed to install vocabulary works as well shuffled
as unshuffled.

The recipe -- two-slot mask, dead-label weight 4, 3 epochs, lr 1e-4, gradient
accumulated over 8 cases, the policy's own LoRA shape -- is fixed before the run
and recorded in `seed_manifest.json` next to the adapter.

    python3 seed_sft.py --model Qwen/Qwen3-1.7B --shuffle-labels 1 --out runs/seed-shuffled
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from task.decision_table import SEVERE_ACTION
from task.prompt import build_messages

HERE = Path(__file__).resolve().parent
SLOTS = ("root_cause", "action")


def target_text(answer: dict[str, Any]) -> str:
    """The gold completion: one line naming the flags, then the JSON object.

    The line matches what the prompt's closing permits, so the seed does not
    also teach the policy to drop or change its working. Keys are sorted, which
    puts `action` before `flags` and `root_cause` after it.
    """
    flags = answer["flags"]
    line = f"Flow {flags['flow']}, salt passage {flags['salt_passage']}, dP {flags['dp']}."
    return line + "\n" + json.dumps(answer, indent=2, sort_keys=True)


def value_spans(text: str, offset: int = 0, slots: tuple[str, ...] = SLOTS) -> list[tuple[int, int]]:
    """Character spans of the *values* of the named slots, searched after `offset`.

    The value only: the key and the quotes are already produced on every case,
    and it is the value that has no mass.
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
    return sorted(spans)


def is_dead(answer: dict[str, Any]) -> bool:
    """A label the frozen policy was measured never to produce."""
    return answer["root_cause"] == "organic_fouling" or answer["action"] == SEVERE_ACTION


def shuffle_labels(cases: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """One permutation of the `(root_cause, action)` pairs across the set.

    The pair moves together rather than each slot on its own permutation. Shuffled
    independently, the overlap between `organic_fouling` and the severe action
    changes, and with it how many cases `is_dead` weights -- 172 correct against
    161 shuffled on this data, found by the test that checks it -- which would
    have made the control's total loss weight differ from the seed it controls
    for. Everything else about each case stays its own. Returns new dicts; the
    input is not modified.
    """
    rng = random.Random(seed)
    pairs = [(c["answer"]["root_cause"], c["answer"]["action"]) for c in cases]
    rng.shuffle(pairs)
    out = []
    for case, (cause, action) in zip(cases, pairs):
        answer = {**case["answer"], "flags": dict(case["answer"]["flags"])}
        answer["root_cause"], answer["action"] = cause, action
        out.append({**case, "answer": answer})
    return out


@dataclass
class SeedConfig:
    model: str = "Qwen/Qwen3-1.7B"
    cases: str = "seed/seed_cases.jsonl"
    shuffle_labels: int = 0
    shuffle_seed: int = 0
    dead_weight: float = 4.0
    epochs: int = 3
    lr: float = 1e-4
    # AdamW's default. The RL loop sets weight decay to zero for a reason that
    # does not apply to a supervised pass, and this matches the recipe under test.
    weight_decay: float = 0.01
    accum: int = 8
    lora_r: int = 16
    prompt_version: str = "v2-oracle-num"
    seed: int = 0
    dtype: str = "bfloat16"


def run(cfg: SeedConfig, out_dir: Path, device: str) -> dict[str, Any]:
    import torch

    from eval import supports_thinking_toggle
    from grpo_scratch import Config, load_policy

    source = HERE / cfg.cases
    raw = source.read_bytes()
    cases = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    targets = shuffle_labels(cases, cfg.shuffle_seed) if cfg.shuffle_labels else cases

    torch.manual_seed(cfg.seed)
    policy, tokenizer = load_policy(
        Config(model=cfg.model, lora_r=cfg.lora_r, dtype=cfg.dtype), device
    )
    template_kwargs: dict[str, Any] = {}
    if supports_thinking_toggle(tokenizer):
        template_kwargs["enable_thinking"] = False

    order = list(range(len(cases)))
    random.Random(cfg.seed).shuffle(order)

    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    policy.train()

    losses, supervised_tokens = [], 0
    for epoch in range(cfg.epochs):
        for i, index in enumerate(order):
            prompt = tokenizer.apply_chat_template(
                build_messages(cases[index]["record"], cfg.prompt_version),
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            answer = targets[index]["answer"]
            full = prompt + target_text(answer) + (tokenizer.eos_token or "")
            encoded = tokenizer(full, return_tensors="pt", return_offsets_mapping=True)
            ids = encoded["input_ids"].to(device)

            spans = value_spans(full, offset=len(prompt))
            keep = torch.tensor(
                [
                    a != b and any(a < end and b > start for start, end in spans)
                    for a, b in encoded["offset_mapping"][0].tolist()
                ],
                device=device,
            ).unsqueeze(0)
            labels = torch.where(keep, ids, torch.full_like(ids, -100))
            supervised_tokens += int(keep.sum())

            loss = policy(input_ids=ids, labels=labels).loss
            weight = cfg.dead_weight if is_dead(answer) else 1.0
            (loss * weight / cfg.accum).backward()
            if (i + 1) % cfg.accum == 0 or i + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append({"epoch": epoch, "i": i, "loss": float(loss.detach())})
        window = [r["loss"] for r in losses if r["epoch"] == epoch]
        print(f"  epoch {epoch}: loss {window[0]:.4f} -> {window[-1]:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir / "adapter")
    (out_dir / "seed_losses.jsonl").write_text("\n".join(json.dumps(r) for r in losses) + "\n")

    unchanged = {
        slot: sum(t["answer"][slot] == c["answer"][slot] for t, c in zip(targets, cases)) / len(cases)
        for slot in SLOTS
    }
    manifest = {
        "config": asdict(cfg),
        "cases_sha256": hashlib.sha256(raw).hexdigest(),
        "n_cases": len(cases),
        "n_dead_weighted": sum(is_dead(t["answer"]) for t in targets),
        "supervised_tokens_per_epoch": supervised_tokens // cfg.epochs,
        "labels_unchanged_by_shuffle": unchanged if cfg.shuffle_labels else None,
    }
    (out_dir / "seed_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, value in asdict(SeedConfig()).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = SeedConfig(**{k: v for k, v in vars(args).items() if k in asdict(SeedConfig())})
    run(cfg, args.out, args.device)


if __name__ == "__main__":
    main()
