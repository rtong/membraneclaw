"""SFT warm-start continuing the @400 GRPO adapter.

Why this exists: 400 steps of GRPO collapsed 3 of 7 cause rows
(mechanical_leak 0/28, scaling 0/29, compaction 14/28) to ~0%. With entropy
~0.05 the policy puts no probability mass on them, so continued GRPO has no
gradient signal to recover. Supervised cross-entropy on the correct labels
gives direct gradient on exactly those rows.

What it does: loads base + the GRPO adapter, keeps training the SAME LoRA
(flags/numeric/format behavior preserved), 2 epochs over train.jsonl with the
3 confused classes upsampled 2x. Then GRPO can resume from the result.

Usage:
    ./.venv/bin/python sft_warmstart.py \
        --from-adapter runs/grpo-qwen35-08b-s0-cont100-s3/adapter \
        --out runs/sft-warmstart-01/adapter \
        --epochs 2 --lr 1e-5 --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

from task.prompt import PROMPT_VERSION, build_messages
from grpo_scratch import _lora_targets
from task.schema import canonical

# The three cause rows GRPO collapsed; everything else the adapter already does well.
CONFUSED = {"mechanical_leak", "scaling", "compaction"}
ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--from-adapter", default=None, help="GRPO adapter dir to continue from (warm-start)")
    ap.add_argument("--cold-start", action="store_true", help="fresh LoRA from base model")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--out", required=True, help="where to save the SFT adapter")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--upsample", type=int, default=2,
                    help="replication factor for the confused classes")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    if args.cold_start:
        policy = get_peft_model(
            base,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=2 * args.lora_r,
                target_modules=_lora_targets(args.model),
                task_type="CAUSAL_LM",
            ),
        )
        print("cold-start: fresh LoRA from base")
    else:
        assert args.from_adapter, "--from-adapter required for warm-start"
        policy = PeftModel.from_pretrained(base, args.from_adapter)
        # from_pretrained loads the adapter frozen; re-enable training (same fix as grpo resume).
        for name, param in policy.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"trainable params: {n_train:,}")

    rows: list[dict] = []
    for line in open(ROOT / "data" / "train.jsonl"):
        case = json.loads(line)
        rep = 1 if args.cold_start else (args.upsample if case["answer"]["root_cause"] in CONFUSED else 1)
        rows.extend([case] * rep)
    random.Random(args.seed).shuffle(rows)

    def to_text(case: dict) -> dict:
        prompt = tok.apply_chat_template(
            build_messages(case["record"], PROMPT_VERSION),
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "completion": canonical(case["answer"])}

    ds = Dataset.from_list([to_text(c) for c in rows])
    n_conf = sum(1 for c in rows if c["answer"]["root_cause"] in CONFUSED)
    print(f"sft examples: {len(ds)} (confused-class examples: {n_conf})")

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        seed=args.seed,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        max_length=4096,
        packing=False,
    )
    trainer = SFTTrainer(
        model=policy,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    policy.save_pretrained(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
