"""Does the reward still tell the two routes apart late in training? For the PPO runs' training samples, per question
group and 50-step phase: the reward of convert-first minus simulate-first episodes. Every run draws the same questions
at the same steps, so the contrast is also taken within question: questions answered by both routes in the phase (any
run), each question's convert-first mean minus its simulate-first mean, averaged over those questions (95% interval
by resampling questions). Simulate-first episodes are split by their first simulate call: rejected, accepted with the
right inputs (the conversion done in the head), or accepted with wrong inputs; with the reward of each.
The same split for the dev probes (T=1 x2, err0) at step 200, the four runs pooled, and per probe.
usage: python route_contrast.py [key=run dir ...]"""
import json, random, statistics as st, sys
from collections import defaultdict
from diag_common import *

RUNS = dict(a.split("=", 1) for a in sys.argv[1:]) or {
    "new": "ppo-n300-maskend-turn-lr2e6-err0-s0-to400", "td0": "ppo-n300-maskend-turn-lr2e6-err0-delta-td0-s0",
    "td05": "ppo-n300-maskend-turn-lr2e6-err0-delta-td05-s0", "alam05": "ppo-n300-maskend-turn-lr2e6-err0-delta-alam05-s0"}
G = ["L0", "L1 one unit", "L1 2+ units", "L2"]
PH = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 350), (350, 400)]


def load(run):
    rows = [json.loads(l) for l in (R / run / "samples.jsonl").read_text().splitlines()]
    cut = max([0] + [i for i in range(1, len(rows)) if rows[i]["step"] < rows[i - 1]["step"]])
    return rows[cut:]                    # a fresh restart appends to samples.jsonl: keep the last attempt only


def first_sim(r):
    """The route (convert / sim / other) and, for simulate-first, how its first simulate call went."""
    calls, case = r["calls"], CASES[r["id"]]
    if not calls:
        return "none", None
    if calls[0]["name"] == nb.CONVERT:
        return "convert", None
    if calls[0]["name"] != case["tool"]:
        return "other", None
    c = calls[0]
    if c.get("error"):
        return "sim", "rejected"
    a = c["arguments"] if isinstance(c["arguments"], dict) else {}
    right = all(any(k in p["args"] and nb._same_arg(a.get(k), p["args"][k]) for p in case["points"]) for k in NEED[r["id"]])
    return "sim", "right" if right else "wrong"


def within(eps, rng, B=2000):
    """Per question with both routes: mean(convert) - mean(sim); the mean over questions and a 95% bootstrap interval."""
    by = defaultdict(lambda: {"convert": [], "sim": []})
    for e in eps:
        by[e["id"]][e["route"]].append(e["reward"])
    d = [st.mean(v["convert"]) - st.mean(v["sim"]) for v in by.values() if v["convert"] and v["sim"]]
    if len(d) < 5:
        return None, len(d)
    bs = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(B))
    return (st.mean(d), bs[int(.025 * B)], bs[int(.975 * B)]), len(d)


eps = []
for key, run in RUNS.items():
    for r in load(run):
        g = route_group(r["id"])
        route, how = first_sim(r)
        if g and route in ("convert", "sim"):
            eps.append({"run": key, "step": r["step"], "id": r["id"], "group": g, "route": route, "how": how, "reward": r["reward"]})
rng = random.Random(0)
f = lambda v: f"{st.mean(v):.2f}" if v else "  - "
for g in G:
    print(f"== {g}: per phase, all runs pooled | within question: convert - sim (95% interval, questions with both routes)"
          f" | raw means convert/sim (n) | sim-first by first simulate call: share and reward, rejected / right / wrong")
    for lo, hi in PH:
        w = [e for e in eps if e["group"] == g and lo <= e["step"] < hi]
        if not w:
            continue
        runs = sorted({e["run"] for e in w}, key=list(RUNS).index)
        res, nq = within(w, rng)
        cv = [e["reward"] for e in w if e["route"] == "convert"]
        sm = [e for e in w if e["route"] == "sim"]
        wq = f"{res[0]:+.2f} ({res[1]:+.2f} to {res[2]:+.2f}) q{nq:3}" if res else f"    -              q{nq:3}"
        parts = []
        for how in ("rejected", "right", "wrong"):
            x = [e["reward"] for e in sm if e["how"] == how]
            parts.append(f"{len(x) / max(1, len(sm)):.2f} {f(x)}")
        print(f"  {lo:3}-{hi - 1:3} {','.join(runs):22} | {wq} | {f(cv)}/{f([e['reward'] for e in sm])} ({len(cv):3}/{len(sm):3})"
              f" | " + " / ".join(parts))
    print()
print("== per run, L1 (one unit and 2+ pooled): within question, convert - sim")
for key in RUNS:
    cells = []
    for lo, hi in PH:
        w = [e for e in eps if e["run"] == key and e["group"] in ("L1 one unit", "L1 2+ units") and lo <= e["step"] < hi]
        if not w:
            continue
        res, nq = within(w, rng)
        cells.append(f"{lo}-{hi - 1} " + (f"{res[0]:+.2f} ({res[1]:+.2f},{res[2]:+.2f}) q{nq}" if res else f"- q{nq}"))
    print(f"  {key:6} " + " | ".join(cells))

print()
DEV = {"new200": "ppo-n300-maskend-turn-lr2e6-err0-s0/step200", "td0-200": "ppo-n300-maskend-turn-lr2e6-err0-delta-td0-s0/step200",
       "td05-200": "ppo-n300-maskend-turn-lr2e6-err0-delta-td05-s0/step200",
       "alam05-200": "ppo-n300-maskend-turn-lr2e6-err0-delta-alam05-s0/step200"}
dev = []
for key, d in DEV.items():
    for line in (R / d / "dev-T1-x2-stop-err0.episodes.jsonl").read_text().splitlines():
        r = json.loads(line)
        g = route_group(r["id"])
        route, how = first_sim(r)
        if g and route in ("convert", "sim"):
            dev.append({"run": key, "id": r["id"], "group": g, "route": route, "how": how, "reward": r["score"]["total"]})
print("== dev T=1 x2 (err0) at step 200: within question, convert - sim | raw convert/sim (n) | sim-first: rejected / right / wrong")
for g in G:
    for keys in (list(DEV), *[[k] for k in DEV]):
        w = [e for e in dev if e["group"] == g and e["run"] in keys]
        res, nq = within(w, rng)
        cv = [e["reward"] for e in w if e["route"] == "convert"]
        sm = [e for e in w if e["route"] == "sim"]
        wq = f"{res[0]:+.2f} ({res[1]:+.2f} to {res[2]:+.2f}) q{nq:3}" if res else f"    -              q{nq:3}"
        parts = []
        for how in ("rejected", "right", "wrong"):
            x = [e["reward"] for e in sm if e["how"] == how]
            parts.append(f"{len(x) / max(1, len(sm)):.2f} {f(x)}")
        name = "pooled" if len(keys) > 1 else keys[0]
        print(f"  {g:12} {name:10} | {wq} | {f(cv)}/{f([e['reward'] for e in sm])} ({len(cv):3}/{len(sm):3}) | " + " / ".join(parts))
    print()
