"""Why the first action drifts: L1/L2 training episodes by window and route (first call convert_units vs the simulate
tool), with each route's mean return under the old reward and under err0 -- what PPO's first-turn gradient follows.
usage: python route_returns.py <run dir name> [window]"""
import importlib.util, json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
HERE = Path("/home/bayan/MembraneClaw/experiments/notebooks/smoke_test35")
spec = importlib.util.spec_from_file_location("nb07", HERE / "07_error_recovery.py")
nb = importlib.util.module_from_spec(spec); sys.modules["nb07"] = nb; spec.loader.exec_module(nb)
nb.result_is_error = lambda r: r == "ERR"
cases = {c["id"]: c for c in nb.load_split("train")}
need = {q: {t[1] for p in nb.convert_plan(c, limits=False) for t in p["targets"] if t[0] == "arg"} for q, c in cases.items()}
old_fid = nb.call_fidelity
def new_fid(case, call):
    return (0.0, None) if nb.result_is_error(call.get("result")) else old_fid(case, call)
def both(r):
    case = cases[r["id"]]
    calls = [dict(c, result="ERR" if c.get("error") else "OK") for c in r["calls"]]
    nb.call_fidelity = old_fid; a = nb.score(case, calls, r["final_text"])["total"]
    nb.call_fidelity = new_fid; b = nb.score(case, calls, r["final_text"])["total"]
    nb.call_fidelity = old_fid
    return a, b
def first_sim_right(r):
    c = cases[r["id"]]; calls = r["calls"]
    i = next((j for j, x in enumerate(calls) if x["name"] == c["tool"] and isinstance(x["arguments"], dict)), None)
    if i is None: return False
    a = calls[i]["arguments"]
    return all(any(k in p["args"] and nb._same_arg(a.get(k), p["args"][k]) for p in c["points"]) for k in need[r["id"]])
run = sys.argv[1]; W = int(sys.argv[2]) if len(sys.argv) > 2 else 25
rows = [json.loads(l) for l in (HERE / "07_error_recovery_runs" / run / "samples.jsonl").read_text().splitlines()]
first = lambda r: next((x["name"] for x in r["calls"]), None)
groups = {"L1 one unit": lambda c, q: c["level"] == 1 and len(need[q]) == 1, "L1 2+ units": lambda c, q: c["level"] == 1 and len(need[q]) >= 2,
          "L2": lambda c, q: c["level"] == 2 and need[q]}
print(f"{run}: per window, route share conv-first; mean return conv | sim (old reward / err0); sim-first: first simulate right")
for lo in range(0, max(r["step"] for r in rows) + 1, W):
    w = [r for r in rows if lo <= r["step"] < lo + W]
    cells = []
    for g, f in groups.items():
        ws = [r for r in w if f(cases[r["id"]], r["id"])]
        cv = [r for r in ws if first(r) == nb.CONVERT]; sm = [r for r in ws if first(r) == cases[r["id"]]["tool"]]
        if not ws: continue
        sc = lambda rs: (st.mean(both(r)[0] for r in rs), st.mean(both(r)[1] for r in rs)) if rs else (float("nan"),) * 2
        a, b = sc(cv), sc(sm)
        right = sum(first_sim_right(r) for r in sm) / len(sm) if sm else float("nan")
        cells.append(f"{g}: n{len(ws):3} cf {len(cv)/len(ws):.2f} | {a[0]:.2f}/{a[1]:.2f} vs {b[0]:.2f}/{b[1]:.2f} sim-right {right:.2f}")
    print(f"  {lo:3}-{lo+W-1:3}  " + "   ".join(cells))
