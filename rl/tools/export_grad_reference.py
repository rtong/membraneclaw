"""PyTorch's gradient of the update's loss on one minibatch, for the Rust gradient's test (tests/grad.rs).

    python rl/tools/export_grad_reference.py batch.json adapter_dir out.safetensors {bf16|fp32}

Reads a minibatch written by `grad_batch` (each episode's sequence, its actor tokens' positions,
targets and old log-probs, its z-scored advantage and loss weight), loads Qwen3.5-0.8B with the PEFT
adapter on CUDA in the given dtype, its LoRA factors f32 and trainable as the Rust update keeps them,
and records the gradient of

    loss = sum_e weight_e * sum_t -min(r z_e, clip(r, 1-eps, 1+eps) z_e),  r = exp(lp - old)

with respect to every LoRA factor, lp with `<|endoftext|>` out of the vocabulary. Episodes run whole,
one at a time, each layer recomputed in the backward (transformers' gradient checkpointing; the
adapter's dropout is 0, so training mode changes nothing else). bf16 runs transformers' fast path (flash-linear-attention's kernels, as 07 trained);
fp32 hides those kernels so that transformers' own PyTorch code runs.
"""

import json
import re
import sys

mode = sys.argv[4]
if mode == "fp32":
    for name in ("fla", "causal_conv1d"):
        sys.modules[name] = None

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

MODEL = "Qwen/Qwen3.5-0.8B"
NAME = re.compile(r"layers\.(\d+)\.(.+)\.lora_([AB])\.")


def main() -> None:
    path, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
    dtype = torch.bfloat16 if mode == "bf16" else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    batch = json.load(open(path))
    eps_clip = batch["clip_eps"]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).cuda()
    model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    assert model.peft_config["default"].lora_dropout == 0.0
    model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    factors = {}
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()
            m = NAME.search(name)
            factors[f"model.layers.{m[1]}.{m[2]}.lora_{m[3]}"] = p
        else:
            p.requires_grad_(False)
    total = 0.0
    for i, ep in enumerate(batch["episodes"]):
        if ep["weight"] == 0.0:
            continue
        at = ep["at"]
        seq = torch.tensor([ep["seq"][: max(at) + 1]], device="cuda")
        pos = torch.tensor(at, device="cuda")
        tgt = torch.tensor(ep["targets"], device="cuda")
        old = torch.tensor(ep["old"], device="cuda", dtype=torch.float32)
        logits = model(input_ids=seq, use_cache=False, logits_to_keep=pos).logits[0].float()
        logits[:, batch["suppress"]] = float("-inf")
        lp = torch.log_softmax(logits, dim=-1).gather(-1, tgt[:, None])[:, 0]
        ratio = torch.exp(lp - old)
        z = ep["z"]
        surrogate = torch.minimum(ratio * z, ratio.clamp(1 - eps_clip, 1 + eps_clip) * z)
        loss = -surrogate.sum() * ep["weight"]
        loss.backward()
        total += loss.item()
        print(f"episode {i}: {len(at)} actor tokens of {seq.shape[1]}, loss {loss.item():+.6e}", flush=True)
        del logits, lp, ratio, surrogate, loss
        torch.cuda.empty_cache()
    grads = {k: (p.grad if p.grad is not None else torch.zeros_like(p)).float().cpu().contiguous() for k, p in factors.items()}
    save_file(grads, out, metadata={"mode": mode, "adapter": adapter, "loss": repr(total)})
    print(f"loss {total:+.8e}; wrote {len(grads)} gradients to {out}")


if __name__ == "__main__":
    main()
