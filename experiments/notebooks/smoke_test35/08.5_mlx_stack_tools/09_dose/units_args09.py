"""Per target unit: share of simulate calls with that argument right, and question-level success; plus overall reward
and how often the first simulate call already errs. Usage: units_args09.py <label>=<episodes.jsonl> ..."""
import importlib.util, json, sys
from collections import defaultdict
from pathlib import Path

ST = Path.home() / "Public/membraneclaw/experiments/notebooks/smoke_test35"
sys.path.insert(0, str(ST)); argv = sys.argv[1:]; sys.argv = ["08.5_mlx_stack.py"]
spec = importlib.util.spec_from_file_location("m85", ST / "08.5_mlx_stack.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
cases = {c["id"]: c for c in m.load_split("dev")}
TARGETS = [("feed_flow_mass_kg_s", "t/h"), ("feed_flow_m3_s", "m3/h"), ("A_comp", "LMH/bar"), ("B_comp", "LMH"),
           ("feed_temperature_c", "F"), ("feed_flow_mass_kg_s", "kg/h"), ("membrane_area_m2", "ft2"), ("feed_tds_g_L", "mg/L")]
for item in argv:
    label, path = item.split("=", 1)
    arg = defaultdict(lambda: [0, 0]); ok_q = defaultdict(lambda: [0, 0]); rewards = []; first_err = [0, 0]
    for line in open(path):
        ep = json.loads(line); c = cases[ep["id"]]; s = ep["score"]
        rewards.append(s["total"])
        u = {(k, v.get("unit")) for k, v in c["inputs"].items()}
        for pt in c["points"]:
            u |= {(k, v.get("unit")) for k, v in (pt.get("inputs") or {}).items()}
        sims = [x for x in ep["calls"] if x.get("name") == c["tool"] and isinstance(x.get("arguments"), dict)]
        if sims:
            first_err[1] += 1; first_err[0] += bool(sims[0].get("error"))
        for t in TARGETS:
            if t not in u:
                continue
            ok_q[t][1] += 1; ok_q[t][0] += bool(s.get("decision_correct")) and bool(s.get("evidence"))
            d = m.DEFAULTS[c["tool"]]
            for x in sims:
                best = max(c["points"], key=lambda pt: sum(m._same_arg(x["arguments"].get(k, d[k]), v) for k, v in pt["args"].items()))
                if t[0] in best["args"]:
                    arg[t][1] += 1; arg[t][0] += m._same_arg(x["arguments"].get(t[0], d[t[0]]), best["args"][t[0]])
    print(f"== {label}: n={len(rewards)} mean reward {sum(rewards)/len(rewards):.3f}; first simulate call errs {first_err[0]}/{first_err[1]} ({first_err[0]/max(1,first_err[1]):.2f})")
    print("   " + "  ".join(f"{t[1]}: arg {arg[t][0]}/{arg[t][1]} q {ok_q[t][0]}/{ok_q[t][1]}" for t in TARGETS))
