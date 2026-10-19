"""Step 2b: the route token's share of attribution.py, per episode, so the push on "convert first" can be split by
the question the episode was on (L0 / L1 one unit / L1 2+ units / L2) and the route it took. Same first-order
model (step-200 state, its Adam preconditioner, whole eventual effect of each training step's gradient), but only
the route tokens' gradient, taken on the prompt + first-turn prefix (causal: the same log-probs as the full
sequence), dotted with four probe directions: J on the 64 train L1/L2 prompts (attribution.py's), and J restricted
to train L1 one-unit, train multi-unit (L1 2+ and L2), and dev L0 prompts. Advantages as the trainer's (critic,
GAE, KL to the seed, centring constant). One JSON line per episode.
usage: python route_attr.py <out.jsonl> <state dir> <samples dir> <first step> <last step (exclusive)>"""
import json, math, os, sys, time
import torch
from diag_common import *

out_path, state_dir, samples_dir = Path(sys.argv[1]), R / sys.argv[2], R / sys.argv[3]
lo, hi = int(sys.argv[4]), int(sys.argv[5])
cfg = {**nb.PPO_DEFAULTS, **json.loads((state_dir / "config.json").read_text())}
torch.cuda.set_per_process_memory_fraction(min(1.0, int(os.environ.get("CAP_MIB", "12000")) * 2**20 / torch.cuda.get_device_properties(0).total_memory))
state = torch.load(state_dir / "last" / "state.pt", map_location="cpu", weights_only=False)
assert state["step"] == 200, state["step"]
policy, tok = nb.load_trainable(state_dir / "last" / "adapter")
reference, _ = nb.load_trainable(HERE / cfg["start_from"])
reference.eval()
for p in reference.parameters():
    p.requires_grad_(False)
SUP = tok.convert_tokens_to_ids("<|endoftext|>")
IDS = route_ids(tok)
prefix = nb.generation_prefix(tok); im_end = tok.convert_tokens_to_ids("<|im_end|>")
head = nb._value_head(policy.get_base_model().config.get_text_config().hidden_size, 0.0).cuda()
head.load_state_dict(state["head"])
actor = [p for p in policy.parameters() if p.requires_grad]
ost = state["opt"]["state"]
b1, b2 = state["opt"]["param_groups"][0]["betas"]
eps, lr = state["opt"]["param_groups"][0]["eps"], state["opt"]["param_groups"][0]["lr"]
precond = [1.0 / ((ost[i]["exp_avg_sq"].cuda() / (1 - b2 ** float(ost[i]["step"]))).sqrt() + eps) for i in range(len(actor))]
mom = [ost[i]["exp_avg"].cuda() for i in range(len(actor))]
policy.train()
base = policy.get_base_model()
dot = lambda a, b: float(sum((x.float() * y.float()).sum() for x, y in zip(a, b)))

P = probe_set()
train = [c for c in P if c.get("probe_split")]
PROBES = {"J": train[:64], "J one unit": [c for c in train if route_group(c["id"]) == "L1 one unit"],
          "J multi": [c for c in train if route_group(c["id"]) in ("L1 2+ units", "L2")],
          "J dev L0": [c for c in P if not c.get("probe_split") and route_group(c["id"]) == "L0"]}


def J_of(cases, grad: bool):
    total = 0.0
    for c in cases:
        ids = torch.tensor([tok(nb.render_prompt(tok, c) + OPEN, add_special_tokens=False).input_ids], device="cuda")
        with torch.set_grad_enabled(grad):
            lg = base.lm_head(base.model(input_ids=ids, use_cache=False).last_hidden_state[:, -2:]).float()
            lg[..., SUP] = nb.SUPPRESSED
            l = torch.log_softmax(lg, -1)[0]
            j = (l[0, IDS["eq"]] + l[1, IDS["convert"]] - l[0, IDS["eqs"]]) / len(cases)
            if grad:
                j.backward()
        total += float(j.detach())
    return total


U, J0, MOM = {}, {}, {}
for name, cases in PROBES.items():
    policy.zero_grad(set_to_none=True)
    J0[name] = J_of(cases, True)
    U[name] = [(p.grad.detach() if p.grad is not None else torch.zeros_like(p)) * pc for p, pc in zip(actor, precond)]
    MOM[name] = -lr * b1 / (1 - b1) * dot(U[name], mom)
    policy.zero_grad(set_to_none=True)
print(f"{state_dir.name}: " + "  ".join(f"{k} {J0[k]:+.2f} (n {len(PROBES[k])}, momentum tail {MOM[k]:+.2f})" for k in PROBES), flush=True)

rows = [json.loads(l) for l in (samples_dir / "samples.jsonl").read_text().splitlines()]
pool = CachedPool(6)
scale = None
t0 = time.time()
with out_path.open("w") as fh:
    fh.write(json.dumps({"_meta": True, "state": str(state_dir.relative_to(R)), "J_200": J0, "momentum_tail": MOM,
                         "n_probe": {k: len(v) for k, v in PROBES.items()}, "centred_run": bool(cfg.get("adv_center", 0))}) + "\n")
    for step in range(lo, hi):
        batch = [r for r in rows if r["step"] == step]
        if not batch:
            continue
        eps_ = [rebuild(r, tok, pool, prefix, im_end) for r in batch]
        nb.ERR_CALL_CREDIT = bool(cfg.get("err_call_credit", 1))
        B = nb.build_batch(eps_, tok, "cuda")
        for r, sc in zip(batch, B["scores"]):
            assert abs(sc["total"] - r["reward"]) < 1e-6, (r["id"], sc["total"], r["reward"])
        mask = B["mask"]
        with torch.no_grad():
            lp_old, hid, lp_ref = [], [], []
            for i in range(len(eps_)):
                a, h = policy_logprobs(policy, B["seqs"][i:i + 1], B["attn"][i:i + 1], B["span"], B["index"][i:i + 1], suppress=SUP, hidden_layer=cfg["value_layer"])
                lp_old.append(a); hid.append(h)
                lp_ref.append(policy_logprobs(reference, B["seqs"][i:i + 1], B["attn"][i:i + 1], B["span"], B["index"][i:i + 1], suppress=SUP)[0])
            lp_old, hidden, lp_ref = torch.cat(lp_old), torch.cat(hid), torch.cat(lp_ref)
            kl = (lp_old - lp_ref) * mask
            values = nb._values(head, hidden) * mask
            adv, _ = nb.gae_turn(B["rewards"] - cfg["kl_coef"] * kl, values, mask, B["turn"], gamma=cfg["gamma"], lam=cfg["lam"])
            adv_task, _ = nb.gae_turn(B["rewards"], values, mask, B["turn"], gamma=cfg["gamma"], lam=cfg["lam"])
            am = mask * (1 - B["turn_end"]) if cfg["freeze_turn_ends"] else mask
            if cfg["freeze_answer_structure"]:
                am = am * (1 - B["answer"] + B["content"])
            T = am.sum(-1)
            has = T > 0
            c = float(((adv * am).sum(-1)[has] / T[has]).mean())
        scale = lr * cfg["inner_epochs"] / len(eps_)
        for i, (r, ep) in enumerate(zip(batch, eps_)):
            route = route_positions(ep, tok, IDS)
            first = next((x["name"] for x in ep.calls), None)
            rec = {"step": step, "id": r["id"], "group": route_group(r["id"]), "reward": r["reward"],
                   "first": "convert" if first == nb.CONVERT else "simulate" if first and first != nb.COMPARE else first,
                   "T": float(T[i]), "c": c, "kl_ep": float(kl[i].sum()), "value0": float(values[i, 0])}
            if route and T[i] > 0 and all(am[i, k] > 0 for k in route):
                k = route[0]
                rec |= {"adv_task": float(adv_task[i, k]), "adv_kl": float(adv[i, k] - adv_task[i, k]),
                        "lp_route": float(sum(lp_old[i, q] for q in route))}
                ids = torch.tensor([ep.prompt_ids + ep.ids[:route[-1] + 1]], device="cuda")
                n0 = len(ep.prompt_ids)
                lg = base.lm_head(base.model(input_ids=ids, use_cache=False).last_hidden_state[0, [n0 + q - 1 for q in route]]).float()
                lg[..., SUP] = nb.SUPPRESSED
                lp = torch.log_softmax(lg, -1).gather(-1, ids[0, [n0 + q for q in route]].unsqueeze(-1)).sum()
                rec["lp_route_check"] = float(lp)
                gr = torch.autograd.grad(lp, actor, allow_unused=True)
                gr = [g if g is not None else torch.zeros_like(p) for g, p in zip(gr, actor)]
                w = {"task": rec["adv_task"] / float(T[i]), "kl": rec["adv_kl"] / float(T[i]), "centre": -c / float(T[i])}
                d = {name: dot(U[name], gr) for name in U}
                rec["dJ"] = {name: {part: scale * wp * d[name] for part, wp in w.items()} for name in U}
                del gr
            fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(f"  step {step}: {time.time() - t0:.0f}s", flush=True)

nxt = samples_dir / f"adapter-step{hi}" / "adapter_model.safetensors"
if nxt.exists():
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(policy, load_file(str(nxt)))
    J1 = {name: J_of(cases, False) for name, cases in PROBES.items()}
    with out_path.open("a") as fh:
        fh.write(json.dumps({"_meta": True, f"J_{hi}": J1}) + "\n")
    print("actual change: " + "  ".join(f"{k} {J1[k] - J0[k]:+.2f}" for k in J1))
