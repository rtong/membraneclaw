"""Per 25-step window of a run's training samples: by question group, the share of episodes whose first call is
convert_units, and the mean reward of convert-first vs simulate-first episodes; with the change of the teacher-forced
route log-odds (decisions.jsonl, train probe groups) over the same window.
usage: python window_mix.py <decisions key> <run dir> [first step] [last step]"""
import json, statistics as st, sys
from collections import defaultdict
from diag_common import *

key, run = sys.argv[1], sys.argv[2]
lo0, hi0 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 400)
rows = [json.loads(l) for l in (R / run / "samples.jsonl").read_text().splitlines()]
dec = {j["step"]: j for j in map(json.loads, (SC / "decisions.jsonl").read_text().splitlines()) if j["run"] == key}
first = lambda r: next((x["name"] for x in r["calls"]), None)
J = lambda s: st.mean(dec[s]["route"][g]["log_odds"] for g in ("train L1 2+ units", "train L1 one unit", "train L2")) if s in dec else None
G = ["L0", "L1 one unit", "L1 2+ units", "L2"]
print(f"{key} ({run}): window | dJ (train L1/L2 mean log-odds) | per group: n, convert-first share, reward convert-first / simulate-first")
for lo in range(lo0, hi0, 25):
    w = [r for r in rows if lo <= r["step"] < lo + 25]
    if not w:
        continue
    j0, j1 = J(lo if lo else 25), J(lo + 25)
    cells = []
    for g in G:
        ws = [r for r in w if route_group(r["id"]) == g]
        cv = [r["reward"] for r in ws if first(r) == nb.CONVERT]
        sm = [r["reward"] for r in ws if first(r) and first(r) != nb.CONVERT and first(r) != nb.COMPARE]
        f = lambda v: f"{st.mean(v):.2f}" if v else "  - "
        cells.append(f"{g[:6]:6} {len(ws):3} cf {len(cv) / max(1, len(ws)):.2f} {f(cv)}/{f(sm)} (sim n {len(sm):2})")
    d = f"{j1 - j0:+5.2f}" if j0 is not None and j1 is not None else "   - "
    print(f"  {lo:3}-{lo + 24:3} dJ {d} | " + " | ".join(cells))
