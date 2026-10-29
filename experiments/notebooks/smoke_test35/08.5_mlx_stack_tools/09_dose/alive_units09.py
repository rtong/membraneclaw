"""09: is a dose alive on its targets? Each target unit right (decision + proving simulations) at least once in a dev
T=1 probe. Exit 0 if alive. Usage: alive_units09.py <probe .episodes.jsonl>"""
import json, sys
from collections import defaultdict
from pathlib import Path

ST = Path.home() / "Public/membraneclaw/experiments/notebooks/smoke_test35"
TARGETS = [("feed_flow_mass_kg_s", "t/h"), ("feed_flow_m3_s", "m3/h"), ("A_comp", "LMH/bar"), ("B_comp", "LMH"),
           ("feed_temperature_c", "F"), ("feed_flow_mass_kg_s", "kg/h"), ("membrane_area_m2", "ft2"), ("feed_tds_g_L", "mg/L")]
cases = {c["id"]: c for c in map(json.loads, (ST / "06_direction_and_compare_data/dev.jsonl").read_text().splitlines())}
hit = defaultdict(lambda: [0, 0])
for line in open(sys.argv[1]):
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
