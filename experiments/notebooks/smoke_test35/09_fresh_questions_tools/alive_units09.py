"""Stricter whole-question alive check on the 8 target units (09's dose sweep found argument-level alive in
units_args09.py is the right gate at k=2). Usage: alive_units09.py <episodes.jsonl>"""
import importlib.util, json, sys
from collections import defaultdict
from pathlib import Path

ST = Path(__file__).resolve().parent.parent
ep_path = Path(sys.argv[1])
sys.path.insert(0, str(ST)); sys.argv = ["09_fresh_questions.py"]
spec = importlib.util.spec_from_file_location("m09", ST / "09_fresh_questions.py")
m = importlib.util.module_from_spec(spec); sys.modules["m09"] = m; spec.loader.exec_module(m)
cases = {c["id"]: c for c in m.load_split("dev")}
TARGETS = [("feed_flow_mass_kg_s", "t/h"), ("feed_flow_m3_s", "m3/h"), ("A_comp", "LMH/bar"), ("B_comp", "LMH"),
           ("feed_temperature_c", "F"), ("feed_flow_mass_kg_s", "kg/h"), ("membrane_area_m2", "ft2"), ("feed_tds_g_L", "mg/L")]
hit = defaultdict(lambda: [0, 0])
for line in open(ep_path):
    ep = json.loads(line); c = cases[ep["id"]]; s = ep["score"]
    u = {(k, v.get("unit")) for k, v in c["inputs"].items()}
    for pt in c["points"]:
        u |= {(k, v.get("unit")) for k, v in (pt.get("inputs") or {}).items()}
    ok = bool(s.get("decision_correct")) and bool(s.get("evidence"))
    for t in TARGETS:
        if t in u:
            hit[t][1] += 1; hit[t][0] += ok
dead = [f"{p} [{u}]" for p, u in TARGETS if hit[(p, u)][0] == 0]
for p, u in TARGETS:
    r, n = hit[(p, u)]
    print(f"   {p + ' [' + u + ']':34s} right {r}/{n}{'  DEAD' if r == 0 else ''}")
print("TARGETS ALIVE" if not dead else f"NOT YET ALIVE: {dead}")
sys.exit(0 if not dead else 1)
