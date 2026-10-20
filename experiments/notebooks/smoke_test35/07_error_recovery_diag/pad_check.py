"""Do left padding (the trainer's batch) or sequence length change a token's log-prob? Route tokens and all tokens of
A's step-200/201 episodes: (a) the episode sliced from the padded batch (as the trainer), (b) the episode alone,
unpadded, (c) the prompt + first-turn prefix alone; in eval and in train mode."""
import json, sys, torch
from diag_common import *
cfg = {**nb.PPO_DEFAULTS, **json.loads((R / "ppo-n300-turn-lr2e6-center-s0" / "config.json").read_text())}
torch.cuda.set_per_process_memory_fraction(6000 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
policy, tok = nb.load_trainable(R / "ppo-n300-turn-lr2e6-center-s0" / "last" / "adapter")
SUP = tok.convert_tokens_to_ids("<|endoftext|>"); IDS = route_ids(tok)
prefix = nb.generation_prefix(tok); im_end = tok.convert_tokens_to_ids("<|im_end|>")
rows = [json.loads(l) for l in (R / "ppo-n300-turn-lr2e6-center-s0-to400" / "samples.jsonl").read_text().splitlines()]
batch = [r for r in rows if r["step"] == 200]
pool = CachedPool(6)
eps_ = [rebuild(r, tok, pool, prefix, im_end) for r in batch]
B = nb.build_batch(eps_, tok, "cuda")
base = policy.get_base_model()
for mode in ("eval", "train"):
    getattr(policy, mode)()
    with torch.no_grad():
        for i, ep in enumerate(eps_):
            a = policy_logprobs(policy, B["seqs"][i:i + 1], B["attn"][i:i + 1], B["span"], B["index"][i:i + 1], suppress=SUP)[0][0]
            n = int(B["mask"][i].sum())
            full = torch.tensor([ep.prompt_ids + ep.ids], device="cuda")
            hs = base.model(input_ids=full, use_cache=False).last_hidden_state[0]
            n0 = len(ep.prompt_ids)
            pol = [n0 + k for k, p in enumerate(ep.policy) if p]
            parts = []
            for s0 in range(0, len(pol), 256):
                q = pol[s0:s0 + 256]
                lg = base.lm_head(hs[[x - 1 for x in q]]).float()
                lg[..., SUP] = nb.SUPPRESSED
                parts.append(torch.log_softmax(lg, -1).gather(-1, full[0, q].unsqueeze(-1)).squeeze(-1))
            b_ = torch.cat(parts)
            route = route_positions(ep, tok, IDS)
            c_ = None
            if route:
                pre = torch.tensor([ep.prompt_ids + ep.ids[:route[-1] + 1]], device="cuda")
                h2 = base.model(input_ids=pre, use_cache=False).last_hidden_state[0]
                lg2 = base.lm_head(h2[[n0 + k - 1 for k in route]]).float()
                lg2[..., SUP] = nb.SUPPRESSED
                l2 = torch.log_softmax(lg2, -1)
                c_ = float(sum(l2[j, pre[0, n0 + k]] for j, k in enumerate(route)))
            pad = int((B["attn"][i] == 0).sum())
            print(f"{mode} ep{i} pad {pad:4} len {full.shape[1]:5} | all tokens |a-b| mean {float((a[:n] - b_[:n]).abs().mean()):.4f} max {float((a[:n] - b_[:n]).abs().max()):.3f}"
                  + (f" | route a {float(sum(a[k] for k in route)):+.3f} b {float(sum(b_[k] for k in route)):+.3f} c {c_:+.3f}" if route else ""), flush=True)
