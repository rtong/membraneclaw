"""Step 2 of the A diagnosis: what pushes the route decision. From a run's exact step-200 state (adapter, critic
head, Adam moments) and its real batches of the next steps (samples.jsonl, rebuilt), the first-order change each
part of the actor's update makes to J = the mean log-odds of "convert first" over 64 fixed train L1/L2 prompts:
    dJ = g_J . dtheta,  dtheta = -lr (b1 m + (1 - b1) g) / (sqrt(v_hat) + eps)    (Adam, the saved v and m)
so a gradient g moves J by -lr (1 - b1) u . g at once, u = g_J / (sqrt(v_hat) + eps), and through the momentum by
-lr u . g in all (the geometric tail sums to 1): its whole effect, used here. The momentum carried in from before
the window moves J by -lr b1 / (1 - b1) u . m over its tail, reported apart. g is split by the token it comes from
(the route tokens, the rest of the first turn, later local-tool turns, simulate turns, the answer's trainable
content) and by the part of the advantage it carries: the task reward's (GAE of the credits with the critic), the
KL penalty's (GAE of -kl_coef * kl), and the centring constant (-c on every trainable token; applied in a centred
run, a counterfactual in the others). Values are per training step (4 inner epochs), in log-odds units.
usage: python attribution.py <out.json> <state dir> <samples dir> <first step> <last step (exclusive)>"""
import json, math, os, sys, time
from collections import defaultdict
import torch
from diag_common import *

out_path, state_dir, samples_dir = Path(sys.argv[1]), R / sys.argv[2], R / sys.argv[3]
lo, hi = int(sys.argv[4]), int(sys.argv[5])
cfg = json.loads((state_dir / "config.json").read_text())
cfg = {**nb.PPO_DEFAULTS, **cfg}
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
assert len(ost) == len(actor)
b1, b2 = state["opt"]["param_groups"][0]["betas"]
eps, lr = state["opt"]["param_groups"][0]["eps"], state["opt"]["param_groups"][0]["lr"]
precond, mom = [], []
for i, p in enumerate(actor):
    s = ost[i]
    t = float(s["step"])
    assert s["exp_avg_sq"].shape == p.shape
    precond.append(1.0 / ((s["exp_avg_sq"].cuda() / (1 - b2 ** t)).sqrt() + eps))
    mom.append(s["exp_avg"].cuda())
print(f"state {state_dir.name} step {state['step']}: lr {lr} eps {eps} betas {b1},{b2} adam step {t:.0f}; centred run: {cfg.get('adv_center', 0)}", flush=True)
policy.train()  # dropout is 0 in these adapters; train() keeps gradient checkpointing active

# The probe J and u = g_J / (sqrt(v_hat) + eps)
probes = [c for c in probe_set() if c.get("probe_split")][:64]
base = policy.get_base_model()


def J_of(model, grad: bool):
    total = 0.0
    bm = model.get_base_model()
    for c in probes:
        ids = torch.tensor([tok(nb.render_prompt(tok, c) + OPEN, add_special_tokens=False).input_ids], device="cuda")
        with torch.set_grad_enabled(grad):
            h = bm.model(input_ids=ids, use_cache=False).last_hidden_state[:, -2:]
            lg = bm.lm_head(h).float()
            lg[..., SUP] = nb.SUPPRESSED
            l = torch.log_softmax(lg, -1)[0]
            j = (l[0, IDS["eq"]] + l[1, IDS["convert"]] - l[0, IDS["eqs"]]) / len(probes)
            if grad:
                j.backward()
        total += float(j.detach())
    return total


policy.zero_grad(set_to_none=True)
J0 = J_of(policy, True)
gJ = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in actor]
policy.zero_grad(set_to_none=True)
u = [g * pc for g, pc in zip(gJ, precond)]
dot = lambda a, b: float(sum((x.float() * y.float()).sum() for x, y in zip(a, b)))
mom_tail = -lr * b1 / (1 - b1) * dot(u, mom)
print(f"J at step 200 {J0:+.3f}; |g_J| {math.sqrt(dot(gJ, gJ)):.3e}; momentum carried in from before step 200: {mom_tail:+.3f} over its tail", flush=True)

KINDS = ["route", "first turn, rest", "local tools turns", "simulate turns", "answer content", "other"]
PARTS = ["task", "kl", "centre"]
acc = defaultdict(float)            # (kind, part, episode route) -> summed dJ over episodes
nstep = 0
rows = [json.loads(l) for l in (samples_dir / "samples.jsonl").read_text().splitlines()]
pool = CachedPool(6)
t_start = time.time()
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
        adv_kl = adv - adv_task
        am = mask * (1 - B["turn_end"]) if cfg["freeze_turn_ends"] else mask
        if cfg["freeze_answer_structure"]:
            am = am * (1 - B["answer"] + B["content"])
        T = am.sum(-1)
        has = T > 0
        c = float(((adv * am).sum(-1)[has] / T[has]).mean())
    scale = lr * cfg["inner_epochs"] / len(eps_)   # the whole eventual dJ of one training step's gradients
    for i, ep in enumerate(eps_):
        if T[i] <= 0:
            continue
        route = set(route_positions(ep, tok, IDS))
        n = int(mask[i].sum())
        kind = []
        for k in range(n):
            j = int(B["turn"][i][k])
            tk = turn_kind(ep, j)
            kind.append("route" if k in route else "first turn, rest" if j == 0 else
                        "answer content" if tk == "answer" else
                        {"local tools": "local tools turns", "simulate": "simulate turns"}.get(tk, "other"))
        first = next((x["name"] for x in ep.calls), None)
        er = "convert-first" if first == nb.CONVERT else "simulate-first" if first else "no call"
        lp, _ = policy_logprobs(policy, B["seqs"][i:i + 1], B["attn"][i:i + 1], B["span"], B["index"][i:i + 1], suppress=SUP)
        lp = lp[0, :n]
        w_all = {"task": adv_task[i, :n] * am[i, :n] / T[i], "kl": adv_kl[i, :n] * am[i, :n] / T[i], "centre": -c * am[i, :n] / T[i]}
        jobs = []
        for kd in KINDS:
            sel = torch.tensor([x == kd for x in kind], device="cuda")
            if not bool((sel & (am[i, :n] > 0)).any()):
                continue
            for part in PARTS:
                jobs.append((kd, part, (w_all[part] * sel).detach()))
        for q, (kd, part, w) in enumerate(jobs):
            S = (w * lp).sum()
            gr = torch.autograd.grad(S, actor, retain_graph=q < len(jobs) - 1, allow_unused=True)
            acc[(kd, part, er)] += scale * dot(u, [g if g is not None else torch.zeros_like(p) for g, p in zip(gr, actor)])
        del lp
    nstep += 1
    print(f"  step {step}: {len(eps_)} episodes, centring constant {c:+.4f}, {time.time() - t_start:.0f}s", flush=True)

# The actual change of J over the window: J at the adapter saved at `hi` (if any)
from safetensors.torch import load_file
from peft import set_peft_model_state_dict
J1 = None
nxt = samples_dir / f"adapter-step{hi}" / "adapter_model.safetensors"
if nxt.exists():
    set_peft_model_state_dict(policy, load_file(str(nxt)))
    J1 = J_of(policy, False)
res = {"state": str(state_dir.relative_to(R)), "samples": str(samples_dir.relative_to(R)), "steps": [lo, hi], "n_steps": nstep,
       "J_200": J0, f"J_{hi}": J1, "momentum_tail_from_before": mom_tail, "centred_run": bool(cfg.get("adv_center", 0)),
       "per_step": {f"{kd} | {part} | {er}": v / max(1, nstep) for (kd, part, er), v in sorted(acc.items())}}
out_path.write_text(json.dumps(res, indent=1) + "\n")
print(f"\nJ: step 200 {J0:+.3f}" + (f", step {hi} {J1:+.3f} (actual change {J1 - J0:+.3f} over {hi - 200} steps)" if J1 is not None else ""))
print(f"predicted dJ per training step, by token kind x advantage part (sum over episode routes):")
print(f"   {'':22}" + "".join(f"{p:>12}" for p in PARTS))
for kd in KINDS:
    vals = [sum(v for (k, p, e), v in acc.items() if k == kd and p == part) / max(1, nstep) for part in PARTS]
    print(f"   {kd:22}" + "".join(f"{x:+12.2e}" for x in vals))
tot = {part: sum(v for (k, p, e), v in acc.items() if p == part) / max(1, nstep) for part in PARTS}
print(f"   {'total':22}" + "".join(f"{tot[p]:+12.2e}" for p in PARTS) + f"   | applied in this run: "
      f"{tot['task'] + tot['kl'] + (tot['centre'] if res['centred_run'] else 0):+.2e} per step; x {nstep} steps = "
      f"{nstep * (tot['task'] + tot['kl'] + (tot['centre'] if res['centred_run'] else 0)):+.3f}, + momentum carried in {mom_tail:+.3f}")
