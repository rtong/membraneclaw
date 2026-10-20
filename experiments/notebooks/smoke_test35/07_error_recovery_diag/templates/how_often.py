"""How often --call-credit delta changes a turn's reward, from the stored samples alone (calls' arguments, turns and
error flags: the err0 fidelity needs nothing else). Per run window, and by the episode's first call."""
import json, sys
from collections import Counter
sys.path.insert(0, "/home/bayan/MembraneClaw/experiments/notebooks/smoke_test35/07_error_recovery_diag")
from diag_common import *
nb.ERR_CALL_CREDIT = False
W = nb.WEIGHTS["calls"]
for run, lo, hi in [("ppo-n300-maskend-turn-lr2e6-err0-s0-to400", 200, 400), ("ppo-n300-maskend-turn-lr2e6-err0-s0", 0, 200)]:
    rows = [r for r in map(json.loads, (R / run / "samples.jsonl").read_text().splitlines()) if lo <= r["step"] < hi]
    cnt, by_first, multi = Counter(), Counter(), 0
    for r in rows:
        case = CASES[r["id"]]
        calls = [dict(c, result='{"error": 1}' if c["error"] else '{"ok": 1}') for c in r["calls"]]
        per = {}
        for mode in ("call", "delta"):
            nb.CALL_CREDIT = mode
            sc = nb.score(case, calls, r["final_text"])
            t = Counter()
            for (anchor, a) in sc["credits"]:
                if anchor[0] == "call":
                    t[calls[anchor[1]]["turn"]] += a
            per[mode] = t
        turns = {c["turn"] for c in calls if c["name"] not in nb.LOCAL_TOOLS}
        multi += len(turns) > 1
        diff = any(abs(per["call"][j] - per["delta"][j]) > 1e-9 for j in set(per["call"]) | set(per["delta"]))
        neg = any(v < -1e-9 for v in per["delta"].values())
        first = next((c["name"] for c in calls), None)
        f = "convert first" if first == nb.CONVERT else "no call" if first is None else "check first" if first == nb.COMPARE else "simulate first"
        cnt["changed"] += diff; cnt["negative"] += neg
        by_first[(f, "n")] += 1; by_first[(f, "changed")] += diff
    print(f"{run} steps {lo}-{hi - 1}: {len(rows)} episodes; simulate calls in 2+ turns {multi / len(rows):.3f}; "
          f"per-turn call reward changed {cnt['changed'] / len(rows):.3f}; a negative increment {cnt['negative'] / len(rows):.3f}")
    for f in sorted({k for k, _ in by_first}):
        n = by_first[(f, "n")]
        print(f"   {f:16} n={n:5}  changed {by_first[(f, 'changed')] / n:.3f}")
