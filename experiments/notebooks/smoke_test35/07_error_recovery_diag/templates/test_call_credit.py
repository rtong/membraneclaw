"""CPU check of --call-credit delta (per-turn increments of the calls component, r_call[j] = C[j] - C[j-1]).
  1. synthetic episodes: right then failed simulate call in two turns -> [+1, -0.5] before weighting (old: [0.5, 0]);
     and a few other patterns, against C[j] computed by hand.
  2. saved trajectories (the new run's training samples, steps 200-224, err0 reward as trained): the episode's total
     and its summed per-token rewards are the same under both credits (and equal the stored reward); under delta each
     turn's summed reward equals its increment computed independently (+ the answer credits on the last turn)."""
import json, sys
from collections import defaultdict
sys.path.insert(0, "/home/bayan/MembraneClaw/experiments/notebooks/smoke_test35/07_error_recovery_diag")
from diag_common import *   # nb, R, CASES, CachedPool, rebuild
import torch
from transformers import AutoTokenizer

nb.ERR_CALL_CREDIT = False
W = nb.WEIGHTS["calls"]


def credits_of(case, calls, mode, final="{}"):
    nb.CALL_CREDIT = mode
    sc = nb.score(case, calls, final)
    z = sum(nb.WEIGHTS[k] for k in sc["components"])
    return [a / (W / z) for (anchor, a) in sc["credits"] if anchor[0] == "call"], sc


# 1. synthetic
case = next(c for c in nb.load_split("dev") if c["level"] == 0)
pt = case["points"][0]["args"]
ok, bad = '{"ok": 1}', '{"error": "rejected"}'
sim = lambda res, turn, args=pt: {"name": case["tool"], "arguments": dict(args), "result": res, "turn": turn}
conv = lambda turn: {"name": nb.CONVERT, "arguments": {}, "result": ok, "turn": turn}
assert nb.call_fidelity(case, sim(ok, 0))[0] == 1.0
patterns = {
    "right (turn 0), failed (turn 1)": ([sim(ok, 0), sim(bad, 1)], [1.0, -0.5]),
    "failed, right": ([sim(bad, 0), sim(ok, 1)], [0.0, 0.5]),
    "right, failed, right": ([sim(ok, 0), sim(bad, 1), sim(ok, 2)], [1.0, -0.5, 2 / 3 - 0.5]),
    "convert (turn 0), right (turn 1), failed (turn 2)": ([conv(0), sim(ok, 1), sim(bad, 2)], [0.0, 1.0, -0.5]),
    "right + failed in one turn, then right": ([sim(ok, 0), sim(bad, 0), sim(ok, 1)], [0.0, 0.5, 2 / 3 - 0.5]),
}
for name, (calls, want) in patterns.items():
    new, sc_new = credits_of(case, calls, "delta")
    old, sc_old = credits_of(case, calls, "call")
    assert sc_new["total"] == sc_old["total"]
    assert all(abs(a - b) < 1e-12 for a, b in zip(new, want)) and len(new) == len(want), (name, new, want)
    assert abs(sum(new) - sum(old)) < 1e-12
    print(f"  {name:50} delta {[round(x, 4) for x in new]}   call {[round(x, 4) for x in old]}   total {sc_new['total']:.4f}")

# 2. saved trajectories
tok = AutoTokenizer.from_pretrained(nb.MODEL, padding_side="left")
prefix = nb.generation_prefix(tok); im_end = tok.convert_tokens_to_ids("<|im_end|>")
pool = CachedPool(6)
rows = [r for r in map(json.loads, (R / "ppo-n300-maskend-turn-lr2e6-err0-s0-to400" / "samples.jsonl").read_text().splitlines())
        if 200 <= r["step"] < 225]
worst = defaultdict(float)
n_diff = n_neg = 0
example = None
for b in range(0, len(rows), 8):
    eps = [rebuild(r, tok, pool, prefix, im_end) for r in rows[b:b + 8]]
    nb.CALL_CREDIT = "call"; Bo = nb.build_batch(eps, tok, "cpu")
    nb.CALL_CREDIT = "delta"; Bn = nb.build_batch(eps, tok, "cpu")
    for i, (r, ep) in enumerate(zip(rows[b:b + 8], eps)):
        so, sn = Bo["scores"][i], Bn["scores"][i]
        worst["total: delta vs call"] = max(worst["total: delta vs call"], abs(sn["total"] - so["total"]))
        worst["total vs stored reward"] = max(worst["total vs stored reward"], abs(sn["total"] - r["reward"]))
        ro, rn = float(Bo["rewards"][i].double().sum()), float(Bn["rewards"][i].double().sum())
        worst["token rewards summed: delta vs call"] = max(worst["token rewards summed: delta vs call"], abs(rn - ro))
        worst["token rewards summed vs total"] = max(worst["token rewards summed vs total"], abs(rn - sn["total"]))
        # each turn's reward under delta against C[j] - C[j-1] computed here from call_fidelity
        z = sum(nb.WEIGHTS[k] for k in sn["components"])
        T = len(ep.turns)
        want = [0.0] * T
        s = k = 0; prev = 0.0
        for j in range(T):
            mine = [c for c in ep.calls if c["turn"] == j and c.get("name") not in nb.LOCAL_TOOLS]
            for c in mine:
                f = 0.0 if nb.result_is_error(c["result"]) else nb.call_fidelity(ep.case, c)[0]
                s, k = s + f, k + 1
            if mine:
                cj = W * s / k / z
                want[j], prev = cj - prev, cj
        rest = sum(a for (anchor, a) in sn["credits"] if anchor[0] != "call")
        n = int(Bn["mask"][i].sum())
        last_turn = int(Bn["turn"][i][n - 1])
        want[last_turn] += rest
        got = [0.0] * T
        for kk in range(n):
            got[int(Bn["turn"][i][kk])] += float(Bn["rewards"][i][kk])
        worst["per-turn reward vs C[j] - C[j-1]"] = max(worst["per-turn reward vs C[j] - C[j-1]"], max(abs(x - y) for x, y in zip(got, want)))
        old_turn = [0.0] * T
        for kk in range(n):
            old_turn[int(Bo["turn"][i][kk])] += float(Bo["rewards"][i][kk])
        if max(abs(x - y) for x, y in zip(old_turn, got)) > 1e-6:
            n_diff += 1
        incs = [a for (anchor, a) in sn["credits"] if anchor[0] == "call" and a < -1e-9]
        if incs:
            n_neg += 1
            if example is None or len(ep.turns) < len(example[1]):
                example = (r["id"], [round(x, 4) for x in old_turn], [round(x, 4) for x in got])
print(f"\n{len(rows)} saved episodes (steps 200-224); max absolute difference:")
for k, v in worst.items():
    print(f"  {k:40} {v:.2e}")
print(f"episodes whose per-turn rewards change: {n_diff}; with a negative call increment: {n_neg}")
if example:
    print(f"shortest with a negative increment: {example[0]}\n  per-turn reward, call : {example[1]}\n  per-turn reward, delta: {example[2]}")
nb.CALL_CREDIT = "call"
try:
    nb.train_ppo({**nb.PPO_DEFAULTS, "call_credit": "turn"}, None)
except SystemExit as e:
    print("bad value:", e)
