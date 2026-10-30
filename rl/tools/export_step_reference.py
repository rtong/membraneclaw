"""PyTorch's version of one whole nb10 update step on a recorded batch, for tests/step.rs.

    python rl/tools/export_step_reference.py batch.json policy_dir reference_dir out.safetensors {bf16|fp32}

Reads a batch written by `grad_batch`: each episode's sequence, every policy token's position and
target, which of them the actor's loss takes, and its score; the groups, each epoch's order of
them, and the run's update settings. Loads Qwen3.5-0.8B on CUDA in the given dtype with two PEFT
adapters, the policy (its LoRA factors f32 and trainable) and the frozen reference, and runs the
step from the formulas (notebook 10's config.json and rl/src/grpo.rs's notes):

    old, ref    every policy token's log-prob under the policy and the reference
    return      score - kl_coef * sum(old - ref)
    advantage   return - its group's mean
    for each epoch, for the groups in its order, minibatches of groups / minibatches of them:
        z       the members' advantages on their actor tokens, z-scored over those tokens
                (token-weighted mean, Bessel standard deviation plus 1e-8)
        loss    sum_e 1 / (actor tokens of e * members) * sum_t -min(r z_e, clip(r, 1 -+ eps) z_e),
                r = exp(lp - old)
        its gradient clipped to max_grad_norm (torch.nn.utils.clip_grad_norm_), one torch.optim.AdamW step

Log-probs take `<|endoftext|>` out of the vocabulary. Episodes run whole, one at a time, each
layer recomputed in the backward (the adapter's dropout is 0); the head runs on the read positions
only, in chunks. bf16 runs transformers' fast path (flash-linear-attention's kernels); fp32 hides
those kernels so that transformers' own PyTorch code runs.

Writes the policy's factors after the step (`model.layers.N.<module>.lora_A` / `lora_B`), and under
`step.` the old log-probs (concatenated), each episode's KL, return and advantage, and each
minibatch's gradient norm before clipping, mean ratio and approx KL (k3) before its step.
"""

import json
import re
import sys

mode = sys.argv[5]
if mode == "fp32":
    for name in ("fla", "causal_conv1d"):
        sys.modules[name] = None

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

MODEL = "Qwen/Qwen3.5-0.8B"
NAME = re.compile(r"layers\.(\d+)\.(.+)\.lora_([AB])\.")
CHUNK = 512


def main() -> None:
    path, policy, reference, out = sys.argv[1:5]
    dtype = torch.bfloat16 if mode == "bf16" else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    batch = json.load(open(path))
    s = batch["settings"]
    eps_clip = s["clip_eps"]
    suppress = batch["suppress"]
    episodes = batch["episodes"]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).cuda()
    model = PeftModel.from_pretrained(model, policy, adapter_name="default", is_trainable=True)
    model.load_adapter(reference, adapter_name="reference", is_trainable=False)
    for name in ("default", "reference"):
        assert model.peft_config[name].lora_dropout == 0.0
    inner = model.get_base_model()
    inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # The model returns the final norm's output at the read positions; the head runs in chunks.
    head = inner.lm_head
    inner.lm_head = torch.nn.Identity()
    model.train()
    factors = {}
    for name, p in model.named_parameters():
        if "lora_" in name and ".default." in name:
            p.data = p.data.float()
            m = NAME.search(name)
            factors[f"model.layers.{m[1]}.{m[2]}.lora_{m[3]}"] = p
        else:
            if "lora_" in name:
                p.data = p.data.float()
            p.requires_grad_(False)
    params = list(factors.values())
    assert len(params) == 228, len(params)

    def chunk_lp(h, tgt):
        logits = head(h).float()
        logits[:, suppress] = float("-inf")
        return torch.log_softmax(logits, dim=-1).gather(-1, tgt[:, None])[:, 0]

    def logprobs(ep, ks, grad):
        """Log-probs of episode `ep`'s policy tokens `ks` under the active adapter."""
        at = [ep["reads_at"][k] for k in ks]
        tgt = torch.tensor([ep["reads_targets"][k] for k in ks], device="cuda")
        seq = torch.tensor([ep["seq"][: max(at) + 1]], device="cuda")
        pos = torch.tensor(at, device="cuda")
        with torch.set_grad_enabled(grad):
            h = model(input_ids=seq, use_cache=False, logits_to_keep=pos).logits[0]
            parts = []
            for i in range(0, len(at), CHUNK):
                if grad:
                    parts.append(torch.utils.checkpoint.checkpoint(chunk_lp, h[i : i + CHUNK], tgt[i : i + CHUNK], use_reentrant=False))
                else:
                    parts.append(chunk_lp(h[i : i + CHUNK], tgt[i : i + CHUNK]))
            return torch.cat(parts)

    # The step's start: old and reference log-probs, returns, advantages.
    every = [list(range(len(ep["reads_at"]))) for ep in episodes]
    actor = [[k for k, a in enumerate(ep["actor"]) if a] for ep in episodes]
    old, ref = [], []
    for name, into in (("default", old), ("reference", ref)):
        model.set_adapter(name)
        for e, ep in enumerate(episodes):
            into.append(logprobs(ep, every[e], False) if every[e] else torch.zeros(0, device="cuda"))
    model.set_adapter("default")
    # set_adapter leaves only the active adapter trainable; the policy's factors stay ours.
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    kl = [float((o.double() - r.double()).sum()) for o, r in zip(old, ref)]
    returns = [ep["score"] - s["kl_coef"] * k for ep, k in zip(episodes, kl)]
    adv = [0.0] * len(episodes)
    for g in batch["groups"]:
        mean = sum(returns[e] for e in g) / len(g)
        for e in g:
            adv[e] = returns[e] - mean
    print("advantages", " ".join(f"{a:+.4f}" for a in adv), flush=True)

    optim = torch.optim.AdamW(params, lr=s["lr"], betas=tuple(s["adam_betas"]), eps=s["adam_eps"], weight_decay=s["weight_decay"], foreach=False)
    groups = batch["groups"]
    per_mb = len(groups) // s["minibatches"]
    norms, ratios, kls = [], [], []
    for epoch, order in enumerate(batch["orders"]):
        for i in range(0, len(order), per_mb):
            members = [e for g in order[i : i + per_mb] for e in groups[g]]
            counts = {e: len(actor[e]) for e in members}
            n = sum(counts.values())
            a = torch.tensor([adv[e] for e in members], dtype=torch.float64)
            c = torch.tensor([counts[e] for e in members], dtype=torch.float64)
            mu = (a * c).sum() / n
            std = (((a - mu) ** 2 * c).sum() / (n - 1)).sqrt()
            z = {e: float((adv[e] - mu) / (std + 1e-8)) for e in members}
            optim.zero_grad(set_to_none=True)
            r_means, k_means = [], []
            for e in members:
                if counts[e] == 0:
                    continue
                ks = actor[e]
                lp = logprobs(episodes[e], ks, True)
                o = old[e][ks]
                logratio = lp - o
                ratio = torch.exp(logratio)
                ze = torch.tensor(z[e], device="cuda", dtype=torch.float32)
                surrogate = torch.minimum(ratio * ze, ratio.clamp(1 - eps_clip, 1 + eps_clip) * ze)
                loss = -surrogate.sum() / (counts[e] * len(members))
                loss.backward()
                with torch.no_grad():
                    r_means.append(float(ratio.mean()))
                    k_means.append(float((ratio - 1 - logratio).mean()))
                del lp, ratio, logratio, surrogate, loss
                torch.cuda.empty_cache()
            norm = torch.nn.utils.clip_grad_norm_(params, s["max_grad_norm"])
            optim.step()
            norms.append(float(norm))
            ratios.append(sum(r_means) / len(r_means))
            kls.append(sum(k_means) / len(k_means))
            print(f"epoch {epoch} groups {order[i : i + per_mb]}: grad norm {norms[-1]:.6f}, ratio {ratios[-1]:.6f}, approx kl {kls[-1]:.3e}", flush=True)

    tensors = {k: p.detach().float().cpu().contiguous() for k, p in factors.items()}
    tensors["step.old"] = torch.cat([o.float().cpu() for o in old]).contiguous()
    for key, v in (("kl", kl), ("returns", returns), ("advantages", adv), ("grad_norm", norms), ("ratio", ratios), ("approx_kl", kls)):
        tensors[f"step.{key}"] = torch.tensor(v, dtype=torch.float64)
    save_file(tensors, out, metadata={"mode": mode, "policy": policy, "reference": reference})
    print(f"wrote {len(factors)} factors after {len(norms)} updates to {out}")


if __name__ == "__main__":
    main()
