"""The provenance run's pre-registered primary readout (prov_ppo.sh), from decisions.jsonl and its training samples.
Primary (user 2026-10-23, set before the run's checkpoints were probed), both against the control:
  (1) the route on dev L1 one unit: the teacher-forced route log-odds, mean over the checkpoints 100-200, higher than
      the control's (where the old reward paid both routes alike and the route went back and forth);
  (2) the route swings, steps 25-200: mean |change of the route log-odds| between checkpoints, train and dev L1 2+
      units, both below the control's.
Reported, not a criterion: on its own samples (reward err0 + provenance, as trained), per question group and 50-step
phase to step 199, the within-question reward of converting first minus simulating first (route_contrast.py's
measure, 95% interval by resampling questions), the control's samples re-scored the same way beside it -- planned as
the primary (L1 at 150-199 at least +0.15), dropped because one run pairs too few questions (1-10 per phase; the
earlier estimates pooled four runs that drew the same questions); and dev L0 p(convert) over the checkpoints (a
secondary readout: no higher than the control's).
Writes <run>/primary.json.
usage: python prov_primary.py <run> [decisions key, default prov] [control run, default ...-err0-s0-to400] [its key, default new]"""
import json, random, statistics as st, sys
sys.argv, ARGS = sys.argv[:1], sys.argv[1:]   # route_contrast reads its runs from argv
from diag_common import *

ns = {}
exec((SC / "route_contrast.py").read_text().split("\neps = []")[0], ns)
first_sim, within, G, PH, load = (ns[k] for k in ("first_sim", "within", "G", "PH", "load"))
run, key = ARGS[0], ARGS[1] if len(ARGS) > 1 else "prov"
ctrl, ckey = ARGS[2] if len(ARGS) > 2 else "ppo-n300-maskend-turn-lr2e6-err0-s0-to400", ARGS[3] if len(ARGS) > 3 else "new"
L1 = ("L1 one unit", "L1 2+ units")
PH = [p for p in PH if p[1] <= 200]


def result(c):
    """The tool result the harness gave (as rescore_prov.py)."""
    a = c["arguments"] if isinstance(c["arguments"], dict) else {}
    if c.get("error"):
        return '{"error": "rejected"}'
    if c["name"] == nb.CONVERT:
        return nb.convert_units(a.get("value"), str(a.get("from_unit", "")), str(a.get("to_unit", "")))
    return '{"ok": 1}'


def prov_reward(r):
    nb.ERR_CALL_CREDIT, nb.PROVENANCE = False, True
    x = nb.score(CASES[r["id"]], [dict(c, result=result(c)) for c in r["calls"]], r["final_text"])["total"]
    nb.PROVENANCE = False
    return x


def episodes(run):
    out, same = [], 0
    rows = [r for r in load(run) if r["step"] < 200]
    for r in rows:
        x = prov_reward(r)
        same += abs(x - r["reward"]) < 1e-12
        g = route_group(r["id"]); route, _ = first_sim(r)
        if g and route in ("convert", "sim"):
            out.append({"id": r["id"], "step": r["step"], "group": g, "route": route, "reward": x})
    return out, same, len(rows)


rng = random.Random(0)
mine, same, n = episodes(run)
theirs, _, _ = episodes(ctrl)
print(f"(reported) training samples, steps 0-199, reward err0 + provenance; {run}'s stored reward reproduced on {same}/{n}")
print("    within question: convert-first minus simulate-first (95% interval, questions with both routes)")
cell = lambda res, nq: f"{res[0]:+.2f} ({res[1]:+.2f},{res[2]:+.2f}) q{nq:2}" if res else f"      -          q{nq:2}"
out = {}
for g in G + ["L1 pooled"]:
    for label, eps in ((key, mine), (f"{ckey} re-scored", theirs)):
        cells = []
        for lo, hi in PH:
            w = [e for e in eps if (e["group"] in L1 if g == "L1 pooled" else e["group"] == g) and lo <= e["step"] < hi]
            res, nq = within(w, rng)
            cells.append(cell(res, nq))
            if label == key and g == "L1 pooled" and (lo, hi) == (150, 200):
                out["l1_150_199"] = None if res is None else {"mean": res[0], "lo": res[1], "hi": res[2], "questions": nq}
        print(f"    {g:12} {label:14} " + " | ".join(f"{lo}-{hi - 1} {c}" for (lo, hi), c in zip(PH, cells)))
    print()
out["contrast_at_least_0.15"] = bool(out.get("l1_150_199")) and out["l1_150_199"]["mean"] >= 0.15   # reported only

rows = [json.loads(l) for l in (SC / "decisions.jsonl").read_text().splitlines()]
fmt = lambda x, p: "-" if x is None else f"{x:.{p}f}"  # noqa: E731
LEVEL = (100, 125, 150, 175, 200)
print(f"(1) dev L1 one unit route log-odds at the checkpoints {LEVEL} and their mean")
lv = {}
for k in (key, ckey):
    xs = {r["step"]: r["route"]["dev L1 one unit"]["log_odds"] for r in rows if r["run"] == k and r["step"] in LEVEL}
    lv[k] = st.mean(xs.values()) if len(xs) == len(LEVEL) else None
    print(f"    {k:8} " + " ".join(f"{s}:{xs[s]:+.2f}" for s in sorted(xs)) + f"  mean {fmt(lv[k], 2)}")
out["level_dev_l1_one_unit"] = lv
out["level_passed"] = None not in lv.values() and lv[key] > lv[ckey]
print("(2) route swings, steps 25-200: mean |change of route log-odds| between checkpoints (sign changes)")
sw = {}
for k in (key, ckey):
    sel = sorted((r for r in rows if r["run"] == k and 25 <= r["step"] <= 200), key=lambda r: r["step"])
    sw[k] = {}
    for g in ("train L1 2+ units", "dev L1 2+ units"):
        xs = [r["route"][g]["log_odds"] for r in sel]
        d = [b - a for a, b in zip(xs, xs[1:])]
        sw[k][g] = st.mean(abs(x) for x in d) if d else None
        flips = sum(x * y < 0 for x, y in zip(d, d[1:]))
        print(f"    {k:8} {g:18} {fmt(sw[k][g], 2)} ({flips})  checkpoints {len(sel)}  "
              + " ".join(f"{r['step']}:{x:+.2f}" for r, x in zip(sel, xs)))
    sw[k]["dev L0 p(convert)"] = st.mean(r["route"]["dev L0"]["p_convert"] for r in sel) if sel else None
    print(f"    {k:8} dev L0 p(convert), mean over the checkpoints {fmt(sw[k]['dev L0 p(convert)'], 3)}")
out["swings"] = sw
out["swings_passed"] = all(sw[key][g] is not None and sw[ckey][g] is not None and sw[key][g] < sw[ckey][g]
                           for g in ("train L1 2+ units", "dev L1 2+ units"))
out["l0_p_convert_no_higher"] = None if None in (sw[key]["dev L0 p(convert)"], sw[ckey]["dev L0 p(convert)"]) else \
    sw[key]["dev L0 p(convert)"] <= sw[ckey]["dev L0 p(convert)"]
out["passed"] = out["level_passed"] and out["swings_passed"]
(R / run / "primary.json").write_text(json.dumps(out) + "\n")
print("primary:", json.dumps({k: out[k] for k in ("level_dev_l1_one_unit", "level_passed", "swings_passed", "passed")}))
print("reported:", json.dumps({k: out[k] for k in ("l1_150_199", "contrast_at_least_0.15", "l0_p_convert_no_higher")}))
