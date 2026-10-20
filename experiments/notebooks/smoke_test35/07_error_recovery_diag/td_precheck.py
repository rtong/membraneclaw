"""Gate 0 of the first-turn TD experiment: can the critic carry the convert-or-simulate decision's credit?
From the n300-maskend run's exact step-200 state (adapter + critic head) and its continuation's training samples of
steps 200-249 (extended to 299 when a cell below has fewer than 10 episodes), each episode's first-turn advantage
under the trainer's gae_turn three ways: MC (lam 0.95, as trained), TD0 (first_lam 0: r_0 + V(s_1) - V(s_0)) and
TD0.5 (first_lam 0.5). Task rewards only (the KL term is ~1e-4 per episode); the old call credit, which this critic
was trained on. Per (question group, first call): n, mean, sd; contrast = convert-first minus simulate-first.
Pass (pre-registered), for a variant: in L0 and L1 2+ units the contrast has the reference's sign and at least half
its size (references from 3000 episodes of the run: L0 -0.189, L1 2+ units +0.299), and the pooled within-cell sd
over those four cells is below MC's. Decision: TD0 if it passes, else TD0.5 if it passes, else neither.
usage: python td_precheck.py [out stem, default td_precheck]   (DEVICE=cpu LIMIT=n: a smoke test on n episodes)"""
import json, math, os, statistics as st, sys, time
from collections import defaultdict
import torch
from diag_common import *

stem = sys.argv[1] if len(sys.argv) > 1 else "td_precheck"
device = os.environ.get("DEVICE", "cuda")
limit = int(os.environ.get("LIMIT", "0"))
STATE = R / "ppo-n300-maskend-turn-lr2e6-err0-s0"
SAMPLES = R / "ppo-n300-maskend-turn-lr2e6-err0-s0-to400"
REF = {"L0": -0.189, "L1 2+ units": 0.299}
CELLS = [(g, ro) for g in REF for ro in ("convert", "simulate")]
VARIANTS = {"MC": None, "TD0": 0.0, "TD0.5": 0.5}

if device == "cuda":
    torch.cuda.set_per_process_memory_fraction(min(1.0, 6000 * 2**20 / torch.cuda.get_device_properties(0).total_memory))
cfg = {**nb.PPO_DEFAULTS, **json.loads((STATE / "config.json").read_text())}
state = torch.load(STATE / "last" / "state.pt", map_location="cpu", weights_only=False)
assert state["step"] == 200, state["step"]
policy, tok = nb.load_trainable(STATE / "last" / "adapter", device=device)
policy.eval()
head = nb._value_head(policy.get_base_model().config.get_text_config().hidden_size, 0.0).to(device)
head.load_state_dict(state["head"])
SUP = tok.convert_tokens_to_ids("<|endoftext|>")
prefix = nb.generation_prefix(tok); im_end = tok.convert_tokens_to_ids("<|im_end|>")
nb.ERR_CALL_CREDIT = bool(cfg.get("err_call_credit", 1))
nb.CALL_CREDIT = "call"
pool = CachedPool(6)
rows_all = [json.loads(l) for l in (SAMPLES / "samples.jsonl").read_text().splitlines()]


def route_of(calls):
    first = next((c["name"] for c in calls), None)
    return "convert" if first == nb.CONVERT else "simulate" if first and first not in nb.LOCAL_TOOLS else "other"


out, t0 = [], time.time()


def run(lo, hi):
    for step in range(lo, hi):
        batch = [r for r in rows_all if r["step"] == step]
        if limit:
            batch = batch[:max(0, limit - len(out))]
        if not batch:
            continue
        eps = [rebuild(r, tok, pool, prefix, im_end) for r in batch]
        B = nb.build_batch(eps, tok, device)
        for r, sc in zip(batch, B["scores"]):
            assert abs(sc["total"] - r["reward"]) < 1e-6, (r["id"], sc["total"], r["reward"])
        mask = B["mask"]
        values = torch.zeros_like(mask)
        with torch.no_grad():
            for i in range(len(eps)):
                _, h = policy_logprobs(policy, B["seqs"][i:i + 1], B["attn"][i:i + 1], B["span"], B["index"][i:i + 1],
                                       suppress=SUP, hidden_layer=cfg["value_layer"])
                values[i] = nb._values(head, h)[0] * mask[i]
        adv = {k: nb.gae_turn(B["rewards"], values, mask, B["turn"], gamma=cfg["gamma"], lam=cfg["lam"], first_lam=fl)
               for k, fl in VARIANTS.items()}
        for i, (r, ep) in enumerate(zip(batch, eps)):
            n = int(mask[i].sum())
            tt = B["turn"][i, :n]
            first = tt == tt[0]
            k1 = int(torch.nonzero(~first)[0]) if bool((~first).any()) else None
            starts = [k for k in range(n) if k == 0 or tt[k] != tt[k - 1]]
            out.append({"id": r["id"], "step": step, "group": route_group(r["id"]), "route": route_of(r["calls"]),
                        "erred0": any(c["error"] for c in r["calls"] if c["turn"] == 0),
                        **{f"A_{k}": float(a[0][i, 0]) for k, a in adv.items()},
                        "V0": float(values[i, 0]), "V1": None if k1 is None else float(values[i, k1]),
                        "R0": float(B["rewards"][i, :n][first].sum()), "G": float(B["rewards"][i, :n].sum()),
                        "turn_starts": [(float(values[i, k]), float(adv["MC"][1][i, k])) for k in starts]})
        print(f"  step {step}: {len(out)} episodes, {time.time() - t0:.0f}s", flush=True)
        if limit and len(out) >= limit:
            return


def cells(rows):
    c = defaultdict(list)
    for r in rows:
        c[(r["group"], r["route"])].append(r)
    return c


run(200, 250)
if not limit and min(len(cells(out)[k]) for k in CELLS) < 10:
    print("a cell has fewer than 10 episodes: extending to step 299", flush=True)
    run(250, 300)

c = cells(out)
table = {}
for (g, ro), rs in sorted(c.items(), key=lambda x: (str(x[0][0]), x[0][1])):
    table[f"{g} | {ro}"] = {"n": len(rs), **{v: {"mean": st.mean(r[f"A_{v}"] for r in rs),
                                                  "sd": st.pstdev([r[f"A_{v}"] for r in rs]) if len(rs) > 1 else None}
                                              for v in VARIANTS}}
res = {"episodes": len(out), "steps": [min(r["step"] for r in out), max(r["step"] for r in out)], "cells": table, "variants": {}}
for v in VARIANTS:
    con = {g: (st.mean(r[f"A_{v}"] for r in c[(g, "convert")]) - st.mean(r[f"A_{v}"] for r in c[(g, "simulate")]))
           if c[(g, "convert")] and c[(g, "simulate")] else None for g in REF}
    dev = [r[f"A_{v}"] - st.mean(x[f"A_{v}"] for x in c[k]) for k in CELLS for r in c[k]]
    pooled = math.sqrt(sum(d * d for d in dev) / max(1, len(dev) - len(CELLS)))
    res["variants"][v] = {"contrast": con, "pooled_sd": pooled,
                          "sign_and_size": all(con[g] is not None and con[g] * REF[g] > 0 and abs(con[g]) >= 0.5 * abs(REF[g]) for g in REF)}
mc_sd = res["variants"]["MC"]["pooled_sd"]
for v in ("TD0", "TD0.5"):
    res["variants"][v]["lower_sd_than_MC"] = res["variants"][v]["pooled_sd"] < mc_sd
    res["variants"][v]["passes"] = res["variants"][v]["sign_and_size"] and res["variants"][v]["lower_sd_than_MC"]
res["decision"] = "TD0" if res["variants"]["TD0"]["passes"] else "TD0.5" if res["variants"]["TD0.5"]["passes"] else "neither"
vs = [(v, g) for r in out for v, g in r["turn_starts"]]
gv = st.pvariance([g for _, g in vs]) if len(vs) > 1 else 0.0
res["critic_ev_at_turn_starts"] = 1 - st.pvariance([g - v for v, g in vs]) / gv if gv else None
res["V1_by_first_response"] = {k: st.mean([r["V1"] for r in out if r["V1"] is not None and r["erred0"] == e])
                               for k, e in (("erred", True), ("clean", False))
                               if any(r["V1"] is not None and r["erred0"] == e for r in out)}
(SC / f"{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
(SC / f"{stem}.json").write_text(json.dumps(res, indent=1) + "\n")
print(f"\n{len(out)} episodes, steps {res['steps'][0]}-{res['steps'][1]}; critic EV at turn starts {res['critic_ev_at_turn_starts']}")
print(f"{'cell':26} {'n':>4}" + "".join(f"{v + ' mean/sd':>18}" for v in VARIANTS))
for k, t in table.items():
    print(f"{k:26} {t['n']:4}" + "".join(f"{t[v]['mean']:+9.3f}/{t[v]['sd'] if t[v]['sd'] is not None else float('nan'):.3f}" for v in VARIANTS))
for v, x in res["variants"].items():
    print(f"{v:6} contrast " + ", ".join(f"{g} {x['contrast'][g]:+.3f}" if x["contrast"][g] is not None else f"{g} -" for g in REF)
          + f" | pooled sd {x['pooled_sd']:.3f}" + (f" | passes {x['passes']}" if "passes" in x else ""))
print(f"V(s_1) after an erring / a clean first response: {res['V1_by_first_response']}")
print(f"decision: {res['decision']}")
